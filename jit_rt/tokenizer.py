from __future__ import annotations

import re

from jit_rt.config import ConfigError


def parse_token_ids(raw: str, vocab_size: int) -> list[int]:
    parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if not parts:
        raise ConfigError("prompt must contain at least one token id")

    token_ids: list[int] = []
    for part in parts:
        try:
            token_id = int(part)
        except ValueError as exc:
            raise ConfigError(f"invalid token id: {part}") from exc
        if token_id < 0 or token_id >= vocab_size:
            raise ConfigError(f"token id {token_id} is outside vocab range [0, {vocab_size})")
        token_ids.append(token_id)
    return token_ids


def format_token_ids(token_ids: list[int]) -> str:
    return " ".join(str(token_id) for token_id in token_ids)
