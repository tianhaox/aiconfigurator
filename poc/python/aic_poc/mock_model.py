# SPDX-License-Identifier: Apache-2.0
"""Mock model + ops for the PoC.

Production AIC has ~20 Operation subclasses across MLA, attention, MoE, etc.
The PoC supports exactly one (GEMM) to validate the end-to-end shape:

    Python model  -- compile -->  Rust Engine  -- run_static -->  dict
                                       ^
                                       │
                                  DbHandle (parquet)

Real models would expose `context_ops`, `generation_ops`, `encoder_ops`;
the PoC mock omits encoder_ops (Engine builder also reads only ctx/gen).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockGemmOp:
    """One GEMM operation.  Attributes mirror what the Rust binder reads."""

    _name: str
    _scale_factor: float  # multiplicity (typically = n_layers)
    _m: int
    _n: int
    _k: int
    op_kind: str = "gemm"


@dataclass
class MockDsaOp:
    """One DeepSeek Sparse Attention op.  Cost = max(compute, mem) roofline."""

    _name: str
    _scale_factor: float
    _num_heads: int
    _head_dim_qk: int
    _head_dim_v: int
    _topk: int
    _dtype_bytes: int
    op_kind: str = "dsa"


class MockLLMModel:
    """A toy LLM: 2 GEMMs + 1 DSA in context, 1 GEMM + 1 DSA in generation.

    The GEMM shapes stand in for a real transformer's qkv_proj /
    out_proj / decode_qkv.  All GEMM shapes must appear in the test
    perf parquet (see ``data/build_gemm_parquet.py``).  The DSA op is
    sized after DeepSeek-V3 (num_heads=128, qk=192, v=128, topk=2048,
    bf16) and uses an SoL roofline model that needs no DB lookup.
    """

    #: Identity of the DB this model expects.  Engine builder stamps it
    #: into engine.metadata["db_id"]; the parquet builder writes the same
    #: string into the parquet kv-metadata.  Mismatched values trigger a
    #: hard error in Engine.check_db_compat.
    DB_ID: str = "mock_h100"

    def __init__(self, hidden: int = 4096, n_layers: int = 32) -> None:
        dsa_ctx = MockDsaOp(
            _name="dsa_ctx",
            _scale_factor=float(n_layers),
            _num_heads=128,
            _head_dim_qk=192,
            _head_dim_v=128,
            _topk=2048,
            _dtype_bytes=2,
        )
        dsa_gen = MockDsaOp(
            _name="dsa_gen",
            _scale_factor=float(n_layers),
            _num_heads=128,
            _head_dim_qk=192,
            _head_dim_v=128,
            _topk=2048,
            _dtype_bytes=2,
        )
        self.context_ops = [
            MockGemmOp(
                _name="qkv_proj",
                _scale_factor=float(n_layers),
                _m=hidden,
                _n=3 * hidden,
                _k=hidden,
            ),
            MockGemmOp(
                _name="out_proj",
                _scale_factor=float(n_layers),
                _m=hidden,
                _n=hidden,
                _k=hidden,
            ),
            dsa_ctx,
        ]
        self.generation_ops = [
            MockGemmOp(
                _name="decode_qkv",
                _scale_factor=float(n_layers),
                _m=hidden,
                _n=3 * hidden,
                _k=hidden,
            ),
            dsa_gen,
        ]
        self.hidden = hidden
        self.n_layers = n_layers
