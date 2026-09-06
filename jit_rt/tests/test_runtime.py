from __future__ import annotations

from pathlib import Path

import pytest

from jit_rt.config import ConfigError, RuntimeSpec, load_spec
from jit_rt.factory import load_runtime
from jit_rt.runtime import JitRuntime
from jit_rt.weights import make_synthetic_weights

MANIFEST = Path(__file__).resolve().parents[1] / "examples" / "tiny_static.yaml"
QWEN3_MANIFEST = Path(__file__).resolve().parents[1] / "examples" / "qwen3_0_6b.yaml"


def test_tiny_static_runtime_generates_tokens() -> None:
    runtime = JitRuntime.from_manifest(MANIFEST)
    result = runtime.generate([1, 2, 3], max_new_tokens=4)

    assert result.prompt_token_ids == [1, 2, 3]
    assert len(result.generated_token_ids) == 4
    assert result.token_ids == result.prompt_token_ids + result.generated_token_ids
    assert all(0 <= token_id < runtime.spec.model.vocab_size for token_id in result.generated_token_ids)


def test_tiny_static_runtime_is_deterministic() -> None:
    first = JitRuntime.from_manifest(MANIFEST).generate([1, 2, 3], max_new_tokens=4)
    second = JitRuntime.from_manifest(MANIFEST).generate([1, 2, 3], max_new_tokens=4)

    assert first.token_ids == second.token_ids


def test_manifest_guardrail_rejects_non_static_batch() -> None:
    spec = load_spec(MANIFEST)
    raw = {
        "name": spec.name,
        "target": {"gpu": spec.target.gpu, "sm": spec.target.sm},
        "model": {
            "family": spec.model.family,
            "vocab_size": spec.model.vocab_size,
            "hidden_size": spec.model.hidden_size,
            "num_layers": spec.model.num_layers,
            "num_heads": spec.model.num_heads,
            "intermediate_size": spec.model.intermediate_size,
        },
        "runtime": {
            "backend": spec.runtime.backend,
            "dtype": spec.runtime.dtype,
            "max_batch_size": 2,
            "max_seq_len": spec.runtime.max_seq_len,
            "kv_layout": spec.runtime.kv_layout,
        },
        "features": {
            "batching": spec.features.batching,
            "sampling": spec.features.sampling,
            "attention": spec.features.attention,
            "pd_separation": spec.features.pd_separation,
            "tensor_parallel": spec.features.tensor_parallel,
        },
        "weights": {"source": spec.weights.source, "seed": spec.weights.seed},
    }

    with pytest.raises(ConfigError, match="max_batch_size=1"):
        RuntimeSpec.from_dict(raw)


def test_request_guardrail_rejects_context_overflow() -> None:
    spec = load_spec(MANIFEST)
    runtime = JitRuntime(spec, make_synthetic_weights(spec.model, spec.weights.seed))

    with pytest.raises(ConfigError, match="max_seq_len"):
        runtime.generate([1] * spec.runtime.max_seq_len, max_new_tokens=1)


def test_qwen3_manifest_loads_without_downloading_weights() -> None:
    spec = load_spec(QWEN3_MANIFEST)

    assert spec.model.family == "qwen3_dense"
    assert spec.runtime.backend == "torch_reference"
    assert spec.model.resolved_head_dim == 128
    assert spec.model.resolved_num_key_value_heads == 8


def test_factory_keeps_toy_runtime_on_numpy_path() -> None:
    runtime = load_runtime(MANIFEST)

    assert isinstance(runtime, JitRuntime)
