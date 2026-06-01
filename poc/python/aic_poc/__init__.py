# SPDX-License-Identifier: Apache-2.0
"""Greenfield Python + Rust PoC for AIC perf engine.

Public surface:

- ``MockLLMModel`` / ``MockGemmOp`` — toy model the PoC exercises.
- ``run_static_reference`` — pure-Python ground truth (parity oracle).
- ``aic_step.Engine`` / ``aic_step.DbHandle`` / ``aic_step.build_engine`` —
  the Rust core, accessible directly via the ``aic_step`` package.
"""

from .mock_model import MockGemmOp, MockLLMModel
from .reference import query_gemm, run_static_reference

__all__ = [
    "MockGemmOp",
    "MockLLMModel",
    "query_gemm",
    "run_static_reference",
]
