# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""apply_moe_backend: wide-EP-only hardware choices must not reach tp1 renders.

Regression for the sm100 probe findings (results/findings.yaml
sm100_sglang_deepep_moe / sm100_trtllm_wideep_dp): the b200 hardware profile's
``moe_backend`` facts are wide-EP selections; applied to tp1/ep1 deployments
they are boot failures (trtllm asserts, sglang's parser rejects the flag).
"""

from __future__ import annotations

import pytest

from aiconfigurator.generator.facts.apply import apply_moe_backend

_B200_HW = {"moe_backend": {"trtllm": "WIDEEP", "sglang": "deepep_moe"}}


def _ctx(*, is_moe: bool = True, agg_params: dict | None = None) -> dict:
    return {"is_moe": is_moe, "agg_params": agg_params or {}}


@pytest.mark.parametrize("backend", ["trtllm", "sglang"])
def test_wide_ep_only_choice_skipped_on_tp1(backend: str) -> None:
    ctx = _ctx(agg_params={"moe_expert_parallel_size": 1})
    apply_moe_backend(ctx, _B200_HW, backend=backend)
    assert "moe_config" not in ctx and "moe_backend" not in ctx


def test_wideep_applied_when_attention_dp_enabled() -> None:
    ctx = _ctx(agg_params={"enable_attention_dp": True})
    apply_moe_backend(ctx, _B200_HW, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "WIDEEP"


def test_deepep_applied_when_ep_gt_1() -> None:
    ctx = _ctx(agg_params={"moe_expert_parallel_size": 8})
    apply_moe_backend(ctx, _B200_HW, backend="sglang")
    assert ctx["moe_backend"] == "deepep_moe"


def test_topology_neutral_choice_still_fills_on_tp1() -> None:
    ctx = _ctx()
    apply_moe_backend(ctx, {"moe_backend": {"trtllm": "CUTLASS"}}, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "CUTLASS"


def test_dense_model_untouched() -> None:
    ctx = _ctx(is_moe=False, agg_params={"enable_attention_dp": True})
    apply_moe_backend(ctx, _B200_HW, backend="trtllm")
    assert "moe_config" not in ctx


def test_user_value_wins_over_fact() -> None:
    ctx = _ctx(agg_params={"enable_attention_dp": True})
    ctx["moe_config"] = {"backend": "TRTLLM"}
    apply_moe_backend(ctx, _B200_HW, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "TRTLLM"


_B200_HW_QUANT = {
    "moe_backend": {"trtllm": "WIDEEP"},
    "moe_backend_quant": {"trtllm": {"fp8": "DEEPGEMM", "fp8_block": "DEEPGEMM"}},
}


def test_quant_conditional_choice_applies_on_tp1() -> None:
    ctx = _ctx(agg_params={"moe_expert_parallel_size": 1})
    ctx["ModelConfig"] = {"quant_algo": "fp8_block"}
    apply_moe_backend(ctx, _B200_HW_QUANT, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "DEEPGEMM"


def test_quant_conditional_wins_over_generic_on_wide_ep() -> None:
    ctx = _ctx(agg_params={"enable_attention_dp": True})
    ctx["ModelConfig"] = {"quant_algo": "fp8"}
    apply_moe_backend(ctx, _B200_HW_QUANT, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "DEEPGEMM"


def test_unmatched_quant_falls_back_to_generic_gating() -> None:
    ctx = _ctx(agg_params={"moe_expert_parallel_size": 1})
    ctx["ModelConfig"] = {"quant_algo": "nvfp4"}
    apply_moe_backend(ctx, _B200_HW_QUANT, backend="trtllm")
    assert "moe_config" not in ctx  # WIDEEP still gated off tp1


def test_quant_conditional_user_value_still_wins() -> None:
    ctx = _ctx(agg_params={"moe_expert_parallel_size": 1})
    ctx["ModelConfig"] = {"quant_algo": "fp8"}
    ctx["moe_config"] = {"backend": "CUTLASS"}
    apply_moe_backend(ctx, _B200_HW_QUANT, backend="trtllm")
    assert ctx["moe_config"]["backend"] == "CUTLASS"
