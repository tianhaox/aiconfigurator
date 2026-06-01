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


class MockLLMModel:
    """A toy LLM: 2 GEMMs in context phase, 1 GEMM in generation phase.

    The shapes are stand-ins for a real transformer's qkv_proj /
    out_proj / decode_qkv.  All GEMM shapes must appear in the test
    perf parquet (see ``data/build_gemm_parquet.py``).
    """

    def __init__(self, hidden: int = 4096, n_layers: int = 32) -> None:
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
        ]
        self.generation_ops = [
            MockGemmOp(
                _name="decode_qkv",
                _scale_factor=float(n_layers),
                _m=hidden,
                _n=3 * hidden,
                _k=hidden,
            ),
        ]
        self.hidden = hidden
        self.n_layers = n_layers
