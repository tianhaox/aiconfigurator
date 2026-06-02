# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference implementation of run_static.

This is the *ground truth* the Rust Engine is compared against in the
parity test.  It mirrors what the production AIC backend does for one
phase:

    for op in ops:
        result = op.query(db, batch=..., seq=...)
        latency_dict[op._name] += float(result)

For PoC scope (only GEMM), the formula is:

    latency_ms = db[(m, n, k)] * (batch * seq) * scale_factor

Keep this implementation deliberately simple — the entire point of the
parity test is to catch deviations of the Rust path from this reference.
"""

from __future__ import annotations

from collections.abc import Iterable

from .mock_model import MockGemmOp

# Must match Rust's GpuSpec::default() in src/db.rs.
PEAK_TFLOPS_BF16 = 990.0
HBM_BW_GBPS = 3350.0


def query_gemm(db: dict[tuple[int, int, int], float], m: int, n: int, k: int) -> float:
    """Exact lookup; raises KeyError on a miss (matching Rust's behavior)."""
    return db[(m, n, k)]


def compute_dsa_latency_ms(op, batch_size: int, seq_len: int) -> float:
    """SoL roofline: max(compute_ms, mem_ms), scaled by scale_factor.

    Mirrors the Rust execute_op DSA arm exactly.
    """
    queries = batch_size * seq_len
    flops_per_query = op._num_heads * 2 * op._topk * (op._head_dim_qk + op._head_dim_v)
    bytes_per_query = op._num_heads * op._topk * (op._head_dim_qk + op._head_dim_v) * op._dtype_bytes
    total_flops = queries * flops_per_query
    total_bytes = queries * bytes_per_query
    compute_ms = total_flops / (PEAK_TFLOPS_BF16 * 1e12) * 1e3
    mem_ms = total_bytes / (HBM_BW_GBPS * 1e9) * 1e3
    sol_ms = max(compute_ms, mem_ms)
    return sol_ms * op._scale_factor


def _run_phase(
    ops: Iterable[MockGemmOp],
    db: dict[tuple[int, int, int], float],
    batch_size: int,
    seq_len: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for op in ops:
        if op.op_kind == "gemm":
            base = query_gemm(db, op._m, op._n, op._k)
            x = batch_size * seq_len
            latency = base * x * op._scale_factor
        elif op.op_kind == "dsa":
            latency = compute_dsa_latency_ms(op, batch_size, seq_len)
        else:
            raise ValueError(f"unknown op_kind {op.op_kind!r}")
        out[op._name] = out.get(op._name, 0.0) + latency
    return out


def run_static_reference(
    model,
    db: dict[tuple[int, int, int], float],
    batch_size: int,
    seq_len: int,
    mode: str,
) -> dict[str, float]:
    """Compute per-op latencies for one (or both) static phases.

    Returns a dict keyed by op._name.  Identical schema to what the
    Rust Engine.run_static returns.
    """
    if mode not in {"static_ctx", "static_gen", "static", "static_full"}:
        raise ValueError(f"unknown mode {mode!r}")

    out: dict[str, float] = {}
    if mode in {"static_ctx", "static", "static_full"}:
        out.update(_run_phase(model.context_ops, db, batch_size, seq_len))
    if mode in {"static_gen", "static", "static_full"}:
        for name, latency in _run_phase(model.generation_ops, db, batch_size, seq_len).items():
            out[name] = out.get(name, 0.0) + latency
    return out
