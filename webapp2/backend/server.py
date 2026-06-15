# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
WebApp v2 backend — a thin FastAPI adapter over the Task v2 / cli_estimate SDK.

This prototype exposes just enough to drive the interactive perf-breakdown view:
  GET  /api/databases   -> supported system/backend/version (drives the cascading form)
  POST /api/estimate    -> static-mode perf breakdown (replaces CLI ASCII bars)

No business logic lives here. Everything is delegated to the SDK; this file only
shapes inputs/outputs into JSON for the frontend.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk import perf_database
from aiconfigurator.sdk.common import get_default_models
from aiconfigurator.sdk.models import check_is_moe

from . import llm, settings, sweep

logger = logging.getLogger("webapp2")

app = FastAPI(title="aiconfigurator webapp2", version="0.0.1")

# Dev convenience: the Vite dev server runs on a different port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/databases")
def databases() -> dict:
    """Supported {system: {backend: [versions]}} — drives the cascading selectors."""
    return perf_database.get_supported_databases()


# Computed once at import: model_path -> is_moe for all built-in models.
def _build_model_list() -> list[dict]:
    models = []
    for path in sorted(get_default_models()):
        try:
            is_moe = bool(check_is_moe(path))
        except Exception:
            is_moe = False
        models.append({"path": path, "is_moe": is_moe})
    return models


_MODELS = _build_model_list()


@app.get("/api/models")
def models() -> list[dict]:
    """Built-in supported models, each flagged is_moe (drives the model selector)."""
    return _MODELS


class EstimateRequest(BaseModel):
    # One-shot estimate across modes. AFD intentionally not exposed yet.
    mode: str = "static"  # static | agg | disagg
    model_path: str
    system_name: str
    backend_name: str = "trtllm"
    backend_version: str | None = None
    database_mode: str = "SILICON"
    isl: int = 1024
    osl: int = 1024
    # static / agg worker config
    batch_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    attention_dp_size: int = 1  # "MoE DP" — attention data parallelism
    moe_tp_size: int | None = None
    moe_ep_size: int | None = None
    # disagg prefill worker config
    prefill_tp_size: int = 1
    prefill_pp_size: int = 1
    prefill_attention_dp_size: int = 1
    prefill_moe_tp_size: int | None = None
    prefill_moe_ep_size: int | None = None
    prefill_batch_size: int = 1
    prefill_num_workers: int = 1
    # disagg decode worker config
    decode_tp_size: int = 1
    decode_pp_size: int = 1
    decode_attention_dp_size: int = 1
    decode_moe_tp_size: int | None = None
    decode_moe_ep_size: int | None = None
    decode_batch_size: int = 32
    decode_num_workers: int = 1


def _dicts_to_rows(latency: dict, source: dict) -> list[dict]:
    """Zip a {op: ms} dict with its parallel {op: source} dict into chart rows."""
    rows = [
        {"op": op, "ms": round(float(ms), 4), "source": source.get(op, "unknown")}
        for op, ms in latency.items()
        if isinstance(ms, (int, float)) and ms  # drop zero / non-numeric (scheduling math)
    ]
    rows.sort(key=lambda r: r["ms"], reverse=True)
    return rows


# Per-mode mapping of "phase label -> source key" for the per-op breakdown.
# static reads the InferenceSummary dicts; agg/disagg read per_ops_data.
_AGG_PHASES = [("Mixed step (prefill+decode)", "mix_step"), ("Gen-only step", "genonly_step")]
_DISAGG_PHASES = [("Prefill", "prefill"), ("Decode", "decode")]


def _build_phases(result, mode: str) -> list[dict]:
    if mode == "static":
        s = result.summary
        return [
            {"name": "Prefill", "rows": _dicts_to_rows(s.get_context_latency_dict(), s.get_context_source_dict())},
            {"name": "Decode", "rows": _dicts_to_rows(s.get_generation_latency_dict(), s.get_generation_source_dict())},
        ]
    pod = result.per_ops_data or {}
    pos = result.per_ops_source or {}
    spec = _AGG_PHASES if mode == "agg" else _DISAGG_PHASES
    phases = []
    for label, key in spec:
        if pod.get(key):
            phases.append({"name": label, "rows": _dicts_to_rows(pod[key], pos.get(key, {}))})
    return phases


def _build_cards(result, mode: str) -> list[dict]:
    raw = result.raw
    cards = [
        {"label": "TTFT", "value": result.ttft, "unit": "ms"},
        {"label": "TPOT", "value": result.tpot, "unit": "ms"},
    ]
    if mode == "disagg":
        cards += [
            {"label": "Tokens/s/GPU", "value": raw.get("tokens/s/gpu"), "unit": ""},
            {"label": "Tokens/s/user", "value": raw.get("tokens/s/user"), "unit": ""},
            {"label": "Prefill mem/GPU", "value": raw.get("(p)memory"), "unit": "GiB"},
            {"label": "Decode mem/GPU", "value": raw.get("(d)memory"), "unit": "GiB"},
            {"label": "Prefill workers", "value": raw.get("(p)workers"), "unit": ""},
            {"label": "Decode workers", "value": raw.get("(d)workers"), "unit": ""},
        ]
    else:
        cards += [
            {"label": "Memory / GPU", "value": result.summary.get_memory().get("total"), "unit": "GiB"},
            {"label": "Power / GPU", "value": result.power_w, "unit": "W"},
        ]
        if mode == "agg":
            cards += [
                {"label": "Tokens/s/GPU", "value": raw.get("tokens/s/gpu"), "unit": ""},
                {"label": "Tokens/s/user", "value": raw.get("tokens/s/user"), "unit": ""},
            ]
    return [c for c in cards if c["value"] is not None]


def _memory_group(summary, label: str) -> dict:
    """Build a memory-breakdown group (rows + per-token KV) from an InferenceSummary."""
    mem = summary.get_memory()
    rows = [{"name": k, "gib": round(float(v), 4)} for k, v in mem.items() if k != "total" and v]
    kv_bytes_per_seq, seq_len_used = summary.get_kv_per_seq()
    kv = kv_bytes_per_seq / seq_len_used if kv_bytes_per_seq and seq_len_used else None
    return {"label": label, "rows": rows, "kv_bytes_per_token": kv}


@app.post("/api/estimate")
def estimate(req: EstimateRequest) -> dict:
    """Run a one-shot estimate (static / agg / disagg) and return scalars + breakdown."""
    common = dict(
        mode=req.mode,
        backend_name=req.backend_name,
        backend_version=req.backend_version,
        database_mode=req.database_mode,
        isl=req.isl,
        osl=req.osl,
    )
    try:
        if req.mode == "disagg":
            result = cli_estimate(
                req.model_path,
                req.system_name,
                **common,
                prefill_tp_size=req.prefill_tp_size,
                prefill_pp_size=req.prefill_pp_size,
                prefill_attention_dp_size=req.prefill_attention_dp_size,
                prefill_moe_tp_size=req.prefill_moe_tp_size,
                prefill_moe_ep_size=req.prefill_moe_ep_size,
                prefill_batch_size=req.prefill_batch_size,
                prefill_num_workers=req.prefill_num_workers,
                decode_tp_size=req.decode_tp_size,
                decode_pp_size=req.decode_pp_size,
                decode_attention_dp_size=req.decode_attention_dp_size,
                decode_moe_tp_size=req.decode_moe_tp_size,
                decode_moe_ep_size=req.decode_moe_ep_size,
                decode_batch_size=req.decode_batch_size,
                decode_num_workers=req.decode_num_workers,
            )
        else:
            result = cli_estimate(
                req.model_path,
                req.system_name,
                **common,
                batch_size=req.batch_size,
                tp_size=req.tp_size,
                pp_size=req.pp_size,
                attention_dp_size=req.attention_dp_size,
                moe_tp_size=req.moe_tp_size,
                moe_ep_size=req.moe_ep_size,
            )
    except Exception as e:  # surface SDK/validation errors to the UI verbatim
        logger.exception("estimate failed")
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Memory breakdown groups. static/agg have a single InferenceSummary. Disagg has
    # no combined summary, but its (p)/(d) totals come from internal static_ctx /
    # static_gen runs — so we reproduce those two runs here to recover per-worker
    # weights/kvcache/activations/nccl/others (totals match raw["(p)memory"]/["(d)memory"]).
    memory_groups: list[dict] = []
    if result.summary is not None:
        memory_groups.append(_memory_group(result.summary, "Memory / GPU"))
    elif req.mode == "disagg":
        shared = dict(
            backend_name=req.backend_name,
            backend_version=req.backend_version,
            database_mode=req.database_mode,
            isl=req.isl,
            osl=req.osl,
        )
        try:
            p = cli_estimate(
                req.model_path,
                req.system_name,
                mode="static_ctx",
                **shared,
                batch_size=req.prefill_batch_size,
                tp_size=req.prefill_tp_size,
                pp_size=req.prefill_pp_size,
                attention_dp_size=req.prefill_attention_dp_size,
                moe_tp_size=req.prefill_moe_tp_size,
                moe_ep_size=req.prefill_moe_ep_size,
            )
            memory_groups.append(_memory_group(p.summary, "Prefill worker"))
            g = cli_estimate(
                req.model_path,
                req.system_name,
                mode="static_gen",
                **shared,
                batch_size=req.decode_batch_size,
                tp_size=req.decode_tp_size,
                pp_size=req.decode_pp_size,
                attention_dp_size=req.decode_attention_dp_size,
                moe_tp_size=req.decode_moe_tp_size,
                moe_ep_size=req.decode_moe_ep_size,
            )
            memory_groups.append(_memory_group(g.summary, "Decode worker"))
        except Exception:
            logger.exception("disagg per-worker memory breakdown failed; skipping")
            memory_groups = []

    return {
        "mode": req.mode,
        "cards": _build_cards(result, req.mode),
        "phases": _build_phases(result, req.mode),
        "memory_groups": memory_groups,
        "meta": {
            "model_path": req.model_path,
            "system_name": req.system_name,
            "backend_name": req.backend_name,
            "backend_version": result.backend_version,
            "isl": req.isl,
            "osl": req.osl,
        },
    }


# --- LLM config (provider / model / API key) ---


class ConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None  # omit to keep the existing key
    base_url: str | None = None  # for the "custom" (OpenAI-compatible) provider


@app.get("/api/config")
def get_config() -> dict:
    return settings.public_status()


@app.post("/api/config")
def set_config(req: ConfigRequest) -> dict:
    settings.save(req.provider, req.model, req.api_key, req.base_url)
    return settings.public_status()


# --- LLM-driven Task YAML generation ---


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/api/llm/generate-task")
def generate_task(req: GenerateRequest) -> dict:
    try:
        return llm.generate_task_yaml(req.prompt)
    except Exception as e:
        logger.exception("llm generate-task failed")
        raise HTTPException(status_code=400, detail=str(e)) from e


# --- Sweep (Task.run() as a background job) ---


class SweepRequest(BaseModel):
    task: dict | None = None  # flat Task v2 fields
    yaml: str | None = None  # or a YAML string (parsed server-side)


@app.post("/api/sweep")
def start_sweep(req: SweepRequest) -> dict:
    import yaml as _yaml

    data = req.task
    if data is None and req.yaml:
        try:
            data = _yaml.safe_load(req.yaml)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="task must be a mapping of fields")
    # Unwrap a single named wrapper (e.g. `my_exp: {serving_mode: ...}`).
    if "serving_mode" not in data and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict):
            data = inner
    return {"job_id": sweep.start(data)}


@app.get("/api/sweep/{job_id}")
def sweep_status(job_id: str) -> dict:
    job = sweep.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job
