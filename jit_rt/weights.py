from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jit_rt.config import ModelSpec


@dataclass(frozen=True)
class LayerWeights:
    attn_norm: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    ffn_norm: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray


@dataclass(frozen=True)
class ModelWeights:
    token_embedding: np.ndarray
    layers: list[LayerWeights]
    final_norm: np.ndarray
    lm_head: np.ndarray


def make_synthetic_weights(model: ModelSpec, seed: int) -> ModelWeights:
    rng = np.random.default_rng(seed)
    hidden = model.hidden_size
    intermediate = model.intermediate_size

    token_embedding = normal(rng, (model.vocab_size, hidden), hidden)
    layers = [
        LayerWeights(
            attn_norm=np.ones((hidden,), dtype=np.float32),
            wq=normal(rng, (hidden, hidden), hidden),
            wk=normal(rng, (hidden, hidden), hidden),
            wv=normal(rng, (hidden, hidden), hidden),
            wo=normal(rng, (hidden, hidden), hidden),
            ffn_norm=np.ones((hidden,), dtype=np.float32),
            w1=normal(rng, (hidden, intermediate), hidden),
            w2=normal(rng, (intermediate, hidden), intermediate),
            w3=normal(rng, (hidden, intermediate), hidden),
        )
        for _ in range(model.num_layers)
    ]
    final_norm = np.ones((hidden,), dtype=np.float32)
    lm_head = normal(rng, (hidden, model.vocab_size), hidden)
    return ModelWeights(
        token_embedding=token_embedding,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def normal(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    scale = 1.0 / np.sqrt(float(fan_in))
    return (rng.standard_normal(shape) * scale).astype(np.float32)
