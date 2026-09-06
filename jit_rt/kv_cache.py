from __future__ import annotations

import numpy as np

from jit_rt.config import ModelSpec, RuntimeLimits


class StaticDenseKVCache:
    """Fixed batch=1 KV cache with [layer, token, head, dim] layout."""

    def __init__(self, model: ModelSpec, runtime: RuntimeLimits) -> None:
        shape = (model.num_layers, runtime.max_seq_len, model.num_heads, model.resolved_head_dim)
        self.keys = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.max_seq_len = runtime.max_seq_len

    def reset(self) -> None:
        self.keys.fill(0.0)
        self.values.fill(0.0)

    def write(self, layer_idx: int, position: int, key: np.ndarray, value: np.ndarray) -> None:
        if position >= self.max_seq_len:
            raise IndexError(f"KV cache position {position} exceeds max_seq_len={self.max_seq_len}")
        self.keys[layer_idx, position] = key
        self.values[layer_idx, position] = value

    def prefix(self, layer_idx: int, position: int) -> tuple[np.ndarray, np.ndarray]:
        end = position + 1
        return self.keys[layer_idx, :end], self.values[layer_idx, :end]
