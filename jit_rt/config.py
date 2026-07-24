from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a manifest asks for an unsupported runtime combination."""


@dataclass(frozen=True)
class TargetSpec:
    gpu: str
    sm: str | None


@dataclass(frozen=True)
class ModelSpec:
    family: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    @property
    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_heads

    @property
    def resolved_num_key_value_heads(self) -> int:
        if self.num_key_value_heads is not None:
            return self.num_key_value_heads
        return self.num_heads


@dataclass(frozen=True)
class RuntimeLimits:
    backend: str
    dtype: str
    max_batch_size: int
    max_seq_len: int
    kv_layout: str
    device: str = "cpu"


@dataclass(frozen=True)
class FeatureSpec:
    batching: str
    sampling: str
    attention: str
    pd_separation: bool
    tensor_parallel: bool


@dataclass(frozen=True)
class WeightSpec:
    source: str
    seed: int = 0
    repo_id: str | None = None
    revision: str | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    target: TargetSpec
    model: ModelSpec
    runtime: RuntimeLimits
    features: FeatureSpec
    weights: WeightSpec

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeSpec:
        try:
            spec = cls(
                name=str(raw["name"]),
                target=TargetSpec(**raw["target"]),
                model=ModelSpec(**raw["model"]),
                runtime=RuntimeLimits(**raw["runtime"]),
                features=FeatureSpec(**raw["features"]),
                weights=WeightSpec(**raw["weights"]),
            )
        except KeyError as exc:
            raise ConfigError(f"missing manifest key: {exc.args[0]}") from exc
        except TypeError as exc:
            raise ConfigError(f"invalid manifest shape: {exc}") from exc

        spec.validate()
        return spec

    def validate(self) -> None:
        require(self.model.vocab_size > 1, "model.vocab_size must be greater than 1")
        require(self.model.hidden_size > 0, "model.hidden_size must be positive")
        require(self.model.num_layers > 0, "model.num_layers must be positive")
        require(self.model.num_heads > 0, "model.num_heads must be positive")
        require(self.model.resolved_head_dim > 0, "model.head_dim must be positive")
        require(self.model.intermediate_size > 0, "model.intermediate_size must be positive")
        require(self.runtime.max_batch_size == 1, "only runtime.max_batch_size=1 is supported")
        require(self.runtime.max_seq_len > 0, "runtime.max_seq_len must be positive")
        require(self.runtime.kv_layout == "static_dense", "only runtime.kv_layout=static_dense is supported")
        require(self.features.batching == "static", "only features.batching=static is supported")
        require(self.features.sampling == "greedy", "only features.sampling=greedy is supported")
        require(not self.features.pd_separation, "features.pd_separation must be false for this checkpoint")
        require(not self.features.tensor_parallel, "features.tensor_parallel must be false for this checkpoint")

        if self.model.family == "llama_like_dense":
            self._validate_toy_numpy()
            return

        if self.model.family == "qwen3_dense":
            self._validate_qwen3_torch()
            return

        raise ConfigError(f"unsupported model.family: {self.model.family}")

    def _validate_toy_numpy(self) -> None:
        require(self.target.gpu == "cpu_reference", "only target.gpu=cpu_reference is supported for toy numpy")
        require(self.target.sm is None, "target.sm must be null for cpu_reference")
        require(
            self.model.hidden_size % self.model.num_heads == 0,
            "model.hidden_size must be divisible by model.num_heads",
        )
        require(self.runtime.backend == "numpy_reference", "only runtime.backend=numpy_reference is supported")
        require(self.runtime.dtype == "float32", "only runtime.dtype=float32 is supported")
        require(self.runtime.device == "cpu", "only runtime.device=cpu is supported for numpy_reference")
        require(
            self.features.attention == "dense_causal_single_token",
            "only features.attention=dense_causal_single_token is supported",
        )
        require(self.weights.source == "synthetic", "only weights.source=synthetic is supported")

    def _validate_qwen3_torch(self) -> None:
        require(self.target.gpu in {"cuda", "cpu_reference"}, "qwen3_dense target.gpu must be cuda or cpu_reference")
        require(self.runtime.backend == "torch_reference", "qwen3_dense requires runtime.backend=torch_reference")
        require(self.runtime.dtype in {"bfloat16", "float32"}, "qwen3_dense dtype must be bfloat16 or float32")
        require(self.runtime.device in {"cuda", "cpu"}, "qwen3_dense device must be cuda or cpu")
        require(
            self.features.attention in {"qwen3_gqa_causal_single_token", "qwen3_gqa_flashinfer_single_decode"},
            "qwen3_dense attention must be qwen3_gqa_causal_single_token or qwen3_gqa_flashinfer_single_decode",
        )
        require(self.model.num_key_value_heads is not None, "qwen3_dense requires model.num_key_value_heads")
        require(self.model.head_dim is not None, "qwen3_dense requires model.head_dim")
        require(
            self.model.num_heads % self.model.resolved_num_key_value_heads == 0,
            "model.num_heads must be divisible by model.num_key_value_heads",
        )
        require(self.model.rope_theta > 0, "model.rope_theta must be positive")
        require(self.model.rms_norm_eps > 0, "model.rms_norm_eps must be positive")
        require(self.weights.source == "hf_safetensors", "qwen3_dense requires weights.source=hf_safetensors")
        require(
            bool(self.weights.repo_id) or bool(self.weights.local_path),
            "qwen3_dense requires weights.repo_id or weights.local_path",
        )


def load_spec(path: str | Path) -> RuntimeSpec:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("manifest root must be a mapping")
    return RuntimeSpec.from_dict(raw)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
