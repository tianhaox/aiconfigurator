from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jit_rt import kernels
from jit_rt.config import ConfigError, RuntimeSpec, load_spec
from jit_rt.kv_cache import StaticDenseKVCache
from jit_rt.weights import ModelWeights, make_synthetic_weights


@dataclass(frozen=True)
class GenerateResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    token_ids: list[int]


class JitRuntime:
    def __init__(self, spec: RuntimeSpec, weights: ModelWeights) -> None:
        spec.validate()
        self.spec = spec
        self.weights = weights
        self.kv_cache = StaticDenseKVCache(spec.model, spec.runtime)

    @classmethod
    def from_manifest(cls, path: str | Path) -> JitRuntime:
        spec = load_spec(path)
        weights = make_synthetic_weights(spec.model, spec.weights.seed)
        return cls(spec, weights)

    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> GenerateResult:
        self._validate_request(prompt_token_ids, max_new_tokens)
        self.kv_cache.reset()

        position = 0
        logits: np.ndarray | None = None
        for token_id in prompt_token_ids:
            logits = self._step(token_id, position)
            position += 1

        if logits is None:
            raise ConfigError("prompt must contain at least one token")

        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_token_id = int(np.argmax(logits))
            generated.append(next_token_id)
            logits = self._step(next_token_id, position)
            position += 1

        return GenerateResult(
            prompt_token_ids=list(prompt_token_ids),
            generated_token_ids=generated,
            token_ids=list(prompt_token_ids) + generated,
        )

    def _validate_request(self, prompt_token_ids: list[int], max_new_tokens: int) -> None:
        if not prompt_token_ids:
            raise ConfigError("prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ConfigError("max_new_tokens must be non-negative")
        total_tokens = len(prompt_token_ids) + max_new_tokens
        if total_tokens > self.spec.runtime.max_seq_len:
            raise ConfigError(
                f"request needs {total_tokens} tokens, max_seq_len is {self.spec.runtime.max_seq_len}",
            )
        vocab_size = self.spec.model.vocab_size
        for token_id in prompt_token_ids:
            if token_id < 0 or token_id >= vocab_size:
                raise ConfigError(f"token id {token_id} is outside vocab range [0, {vocab_size})")

    def _step(self, token_id: int, position: int) -> np.ndarray:
        x = self.weights.token_embedding[token_id].copy()
        for layer_idx, layer in enumerate(self.weights.layers):
            attn_input = kernels.rms_norm(x, layer.attn_norm)
            q = kernels.linear(attn_input, layer.wq).reshape(
                self.spec.model.num_heads,
                self.spec.model.resolved_head_dim,
            )
            k = kernels.linear(attn_input, layer.wk).reshape(
                self.spec.model.num_heads,
                self.spec.model.resolved_head_dim,
            )
            v = kernels.linear(attn_input, layer.wv).reshape(
                self.spec.model.num_heads,
                self.spec.model.resolved_head_dim,
            )

            self.kv_cache.write(layer_idx, position, k, v)
            keys, values = self.kv_cache.prefix(layer_idx, position)
            attn_out = kernels.attention_one_token(q, keys, values)
            x = x + kernels.linear(attn_out, layer.wo)

            ffn_input = kernels.rms_norm(x, layer.ffn_norm)
            gate = kernels.silu(kernels.linear(ffn_input, layer.w1))
            up = kernels.linear(ffn_input, layer.w3)
            x = x + kernels.linear(gate * up, layer.w2)

        x = kernels.rms_norm(x, self.weights.final_norm)
        return kernels.linear(x, self.weights.lm_head)
