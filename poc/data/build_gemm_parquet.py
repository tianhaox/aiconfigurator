# SPDX-License-Identifier: Apache-2.0
"""Generate `data/gemm_perf.parquet` for the PoC.

Schema (must match ``poc/src/db.rs``):
    m, n, k       — UInt32     (GEMM shape)
    latency_ms    — Float64    (per (batch*seq) unit, see engine.rs:execute_op)

The values are arbitrary fake numbers — the PoC verifies the *path*
(Python build → Rust execute → matches Python reference), not the
numerical correctness of GEMM latency prediction.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# (m, n, k) → latency_ms (per unit batch*seq).
# Shapes match what MockLLMModel(hidden=4096) emits.
ROWS: list[tuple[int, int, int, float]] = [
    # qkv_proj / decode_qkv:   m=4096, n=12288, k=4096
    (4096, 12288, 4096, 1.0e-6),
    # out_proj:                m=4096, n=4096,  k=4096
    (4096, 4096, 4096, 0.3e-6),
    # Other shapes for the rayon demo with different hidden sizes:
    (8192, 24576, 8192, 4.0e-6),
    (8192, 8192, 8192, 1.2e-6),
]


def main(out_path: Path | None = None) -> Path:
    out_path = out_path or Path(__file__).parent / "gemm_perf.parquet"
    table = pa.table(
        {
            "m": pa.array([r[0] for r in ROWS], type=pa.uint32()),
            "n": pa.array([r[1] for r in ROWS], type=pa.uint32()),
            "k": pa.array([r[2] for r in ROWS], type=pa.uint32()),
            "latency_ms": pa.array([r[3] for r in ROWS], type=pa.float64()),
        }
    )
    pq.write_table(table, out_path)
    print(f"wrote {out_path} ({len(ROWS)} rows)")
    return out_path


if __name__ == "__main__":
    main()
