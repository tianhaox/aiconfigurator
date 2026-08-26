# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-default ``env:`` facts (worker environment variables).

Origin: findings.yaml sm100_sglang_dsv4_root_cause — the verified DSV4/sm100
workaround is an env var with no CLI spelling, so model defaults gained an
``env:`` surface rendered into run.sh as ``export K=V  # facts-env`` lines.
"""

from __future__ import annotations

from aiconfigurator.generator.facts.apply import apply_facts, collect_model_default_env


class _Facts:
    def __init__(self, model, hardware_key=None):
        self.model = model
        self.hardware = None
        self.hardware_key = hardware_key


_MODEL = {
    "defaults": [
        {"match": {"backend": "sglang", "system": "b200"},
         "env": {"SGLANG_OPT_FP8_WO_A_GEMM": "0"}},
        {"match": {"backend": "sglang"}, "roles": ["*"],
         "backend_args": {"trust-remote-code": True}},
    ]
}


def test_env_collected_for_matching_system() -> None:
    env = collect_model_default_env(_MODEL, backend="sglang", system="b200", variant=None)
    assert env == {"SGLANG_OPT_FP8_WO_A_GEMM": "0"}


def test_env_skipped_for_other_system_and_backend() -> None:
    assert collect_model_default_env(_MODEL, backend="sglang", system="h200_sxm", variant=None) == {}
    assert collect_model_default_env(_MODEL, backend="vllm", system="b200", variant=None) == {}


def test_apply_facts_emits_marked_export_lines() -> None:
    ctx = {"agg_cli_args_list": []}
    apply_facts(ctx, _Facts(_MODEL, hardware_key="b200"), "sglang")
    assert ctx["facts_env_exports"] == ["export SGLANG_OPT_FP8_WO_A_GEMM=0  # facts-env"]
    assert "--trust-remote-code" in ctx["agg_cli_args_list"]


def test_apply_facts_no_env_key_when_nothing_matches() -> None:
    ctx = {"agg_cli_args_list": []}
    apply_facts(ctx, _Facts(_MODEL, hardware_key="h200_sxm"), "sglang")
    assert "facts_env_exports" not in ctx


def test_later_entry_wins_on_conflict() -> None:
    model = {"defaults": [
        {"match": {"backend": "sglang"}, "env": {"K": "1"}},
        {"match": {"backend": "sglang", "system": "b200"}, "env": {"K": "2"}},
    ]}
    assert collect_model_default_env(model, backend="sglang", system="b200", variant=None) == {"K": "2"}
