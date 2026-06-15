# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Sweep job runner: Task.run() is synchronous and can take minutes, so run it on a
background thread and let the frontend poll. In-memory job store (v1; not durable).
"""

from __future__ import annotations

import logging
import math
import threading
import uuid

import numpy as np

from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk.task_v2 import Task

logger = logging.getLogger("webapp2.sweep")

# Classic aiconfigurator Pareto axes (both maximize): per-user speed vs per-GPU throughput.
PARETO_X = "tokens/s/user"
PARETO_Y = "tokens/s/gpu"

# Preferred table columns per mode (intersected with what the DataFrame actually has).
_AGG_COLS = [
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "bs",
    "concurrency",
    "ttft",
    "tpot",
    "tokens/s/user",
    "tokens/s/gpu",
    "seq/s",
    "memory",
    "num_total_gpus",
]
_DISAGG_COLS = [
    "(p)tp",
    "(p)workers",
    "(p)bs",
    "(d)tp",
    "(d)workers",
    "(d)bs",
    "ttft",
    "tpot",
    "tokens/s/user",
    "tokens/s/gpu",
    "seq/s",
    "num_total_gpus",
]

_LABEL_COLS = ["tp", "pp", "dp", "(p)tp", "(p)workers", "(d)tp", "(d)workers", "bs"]

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _safe(v):
    """Make a cell JSON-safe (NaN/inf -> None, numpy scalar -> python)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _label(row, cols) -> str:
    parts = [
        f"{c.replace('(p)', 'P:').replace('(d)', 'D:')}={int(row[c])}"
        for c in cols
        if c in row and _safe(row[c]) is not None
    ]
    return " ".join(parts)


def _run(job_id: str, data: dict) -> None:
    try:
        task = Task.from_yaml(data)
        df = task.run()
        if df is None or df.empty:
            with _LOCK:
                _JOBS[job_id].update(status="done", result={"empty": True})
            return

        pf = pareto_analysis.get_pareto_front(df, PARETO_X, PARETO_Y)
        pareto_idx = set(pf.index.tolist())

        preferred = _DISAGG_COLS if data.get("serving_mode") == "disagg" else _AGG_COLS
        table_cols = [c for c in preferred if c in df.columns]
        label_cols = [c for c in _LABEL_COLS if c in df.columns]

        points = [
            {
                "x": _safe(row.get(PARETO_X)),
                "y": _safe(row.get(PARETO_Y)),
                "on_pareto": idx in pareto_idx,
                "label": _label(row, label_cols),
            }
            for idx, row in df.iterrows()
        ]
        pareto_rows = [{c: _safe(row[c]) for c in table_cols} for _, row in pf.sort_values(PARETO_X).iterrows()]

        result = {
            "empty": False,
            "x_col": PARETO_X,
            "y_col": PARETO_Y,
            "n_feasible": len(df),
            "n_pareto": len(pf),
            "columns": table_cols,
            "points": [p for p in points if p["x"] is not None and p["y"] is not None],
            "pareto": pareto_rows,
        }
        with _LOCK:
            _JOBS[job_id].update(status="done", result=result)
    except Exception as e:
        logger.exception("sweep job %s failed", job_id)
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(e))


def start(data: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {"status": "running", "result": None, "error": None}
    threading.Thread(target=_run, args=(job_id, data), daemon=True).start()
    return job_id


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
