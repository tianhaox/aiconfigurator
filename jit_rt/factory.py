from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jit_rt.config import ConfigError, load_spec
from jit_rt.runtime import GenerateResult, JitRuntime
from jit_rt.weights import make_synthetic_weights


class Runtime(Protocol):
    spec: object

    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> GenerateResult: ...


def load_runtime(path: str | Path) -> Runtime:
    spec = load_spec(path)
    if spec.model.family == "llama_like_dense":
        return JitRuntime(spec, make_synthetic_weights(spec.model, spec.weights.seed))
    if spec.model.family == "qwen3_dense":
        from jit_rt.qwen3_torch import Qwen3TorchRuntime

        return Qwen3TorchRuntime.from_spec(spec)
    raise ConfigError(f"unsupported model.family: {spec.model.family}")
