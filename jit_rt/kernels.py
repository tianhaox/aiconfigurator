from __future__ import annotations

import math

import numpy as np


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x @ weight


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(np.square(x), axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(variance + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def attention_one_token(q: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Dense causal attention for a single query token.

    q: [num_heads, head_dim]
    keys: [seq_len, num_heads, head_dim]
    values: [seq_len, num_heads, head_dim]
    returns: [hidden_size]
    """

    head_dim = q.shape[-1]
    scores = np.einsum("hd,thd->ht", q, keys) / math.sqrt(head_dim)
    probs = softmax(scores, axis=-1)
    context = np.einsum("ht,thd->hd", probs, values)
    return context.reshape(-1)
