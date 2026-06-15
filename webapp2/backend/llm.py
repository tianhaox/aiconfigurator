# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LLM-driven Task v2 YAML generation.

Natural language -> Task YAML -> validate via Task.from_yaml().validate() ->
feed validation errors back to the model for self-correction. The strict Task
schema + friendly validation errors are what make this accurate.

Provider-agnostic: Anthropic and OpenAI via their official SDKs (no shims).
"""

from __future__ import annotations

import dataclasses
import logging
import re

import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_v2 import Task

from . import settings

logger = logging.getLogger("webapp2.llm")

_MAX_ATTEMPTS = 4

_EXAMPLE_AGG = """serving_mode: agg
model_path: Qwen/Qwen3-32B
system_name: h200_sxm
backend_name: trtllm
total_gpus: 8
isl: 4000
osl: 1000
ttft: 1000.0
tpot: 50.0
gemm_quant_mode: fp8
agg_num_gpu_candidates: [2, 4, 8]
agg_tp_candidates: [1, 2, 4, 8]"""

_EXAMPLE_DISAGG = """serving_mode: disagg
prefill_model_path: deepseek-ai/DeepSeek-V3
prefill_system_name: h200_sxm
decode_model_path: deepseek-ai/DeepSeek-V3
decode_system_name: h200_sxm
backend_name: trtllm
database_mode: HYBRID
total_gpus: 32
isl: 4000
osl: 1000
prefill_tp_candidates: [4, 8]
decode_tp_candidates: [4, 8]
num_gpu_per_replica: [8, 16, 24, 32]"""


def _quant_enum_values(enum_cls) -> list[str]:
    return [m.name for m in enum_cls]


def _build_system_prompt() -> str:
    """Describe the flat Task v2 schema from the dataclass + quant enums + examples."""
    field_names = sorted(f.name for f in dataclasses.fields(Task) if not f.name.startswith("_"))
    quant = {
        "gemm_quant_mode": _quant_enum_values(common.GEMMQuantMode),
        "moe_quant_mode": _quant_enum_values(common.MoEQuantMode),
        "kvcache_quant_mode": _quant_enum_values(common.KVCacheQuantMode),
        "fmha_quant_mode": _quant_enum_values(common.FMHAQuantMode),
        "comm_quant_mode": _quant_enum_values(common.CommQuantMode),
    }
    quant_lines = "\n".join(f"  - {k}: {', '.join(v)}" for k, v in quant.items())
    return f"""You generate a single Task definition (flat YAML) for the aiconfigurator \
performance sweep engine. Output ONLY one ```yaml code block — no prose.

The Task is a FLAT mapping: every key is a top-level field (no `config:` block, no \
nesting). `serving_mode` is required and is either `agg` or `disagg`.

For disagg, worker/search fields are prefixed `prefill_` and `decode_` (e.g. \
`prefill_tp_candidates`, `decode_tp_candidates`); do NOT use bare worker fields in \
disagg mode.

Search-space fields are lists of candidate integers (e.g. `agg_tp_candidates: [1,2,4,8]`).

Valid field names:
{", ".join(field_names)}

Quantization modes (use the exact string names):
{quant_lines}

Example (agg sweep):
```yaml
{_EXAMPLE_AGG}
```

Example (disagg sweep):
```yaml
{_EXAMPLE_DISAGG}
```

Pick sensible candidate lists for the search. Use only the field names listed above; \
unknown keys are rejected."""


def _extract_yaml(text: str) -> str:
    """Pull the first ```yaml ... ``` block, or fall back to the whole text."""
    m = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _validate(yaml_str: str) -> tuple[bool, str | None, dict | None]:
    try:
        data = yaml.safe_load(yaml_str)
    except Exception as e:
        return False, f"YAML parse error: {e}", None
    if not isinstance(data, dict):
        return False, "Top-level YAML must be a mapping of Task fields.", None
    # Unwrap a single named wrapper (e.g. `my_exp: {serving_mode: ...}`).
    if "serving_mode" not in data and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict):
            data = inner
    try:
        Task.from_yaml(data).validate()
    except Exception as e:
        return False, str(e), data
    return True, None, data


# --- provider adapters (official SDKs only) ---


def _call_anthropic(cfg: dict, system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg["api_key"])
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _call_openai_compatible(cfg: dict, system: str, user: str, base_url: str | None) -> str:
    """OpenAI SDK against the official API (base_url=None) or any OpenAI-compatible endpoint."""
    import openai

    # Many self-hosted servers (vLLM, SGLang, TGI, LiteLLM) accept any non-empty key.
    client = openai.OpenAI(api_key=cfg.get("api_key") or "EMPTY", base_url=base_url or None)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _call(cfg: dict, system: str, user: str) -> str:
    provider = cfg["provider"]
    if provider == "anthropic":
        return _call_anthropic(cfg, system, user)
    if provider == "openai":
        return _call_openai_compatible(cfg, system, user, base_url=None)
    if provider == "custom":
        return _call_openai_compatible(cfg, system, user, base_url=cfg.get("base_url"))
    raise RuntimeError(f"Unknown provider: {provider}")


def generate_task_yaml(prompt: str) -> dict:
    """Generate a validated Task YAML from a natural-language description."""
    cfg = settings.load()
    if cfg["provider"] == "custom":
        if not cfg.get("base_url"):
            raise RuntimeError("Custom provider needs a base URL. Set it on the Config page.")
    elif not cfg.get("api_key"):
        raise RuntimeError("No API key configured. Set provider/model/key on the Config page.")

    system = _build_system_prompt()
    user = f"Request: {prompt}\n\nReturn one YAML task definition in a ```yaml code block."

    last_yaml = None
    last_err = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        text = _call(cfg, system, user)
        yaml_str = _extract_yaml(text)
        last_yaml = yaml_str
        ok, err, _ = _validate(yaml_str)
        if ok:
            return {"yaml": yaml_str, "valid": True, "error": None, "attempts": attempt}
        last_err = err
        # Feed the validation error back for self-correction.
        user = (
            f"Your previous YAML failed validation with this error:\n\n{err}\n\n"
            f"Here is what you produced:\n```yaml\n{yaml_str}\n```\n\n"
            f"Fix it and return a corrected ```yaml block. Original request: {prompt}"
        )

    return {"yaml": last_yaml, "valid": False, "error": last_err, "attempts": _MAX_ATTEMPTS}
