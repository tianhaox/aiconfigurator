# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LLM configuration store for webapp2.

The provider / model / API key are set at runtime via the Config page and
persisted to a local gitignored JSON file. On first start, env vars seed the
config if no file exists. The API key is never returned to the frontend — only
a ``key_set`` boolean.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("webapp2.settings")

_CONFIG_PATH = Path(__file__).with_name(".llm_config.json")

# Suggested models per provider (the UI offers these; free-text is also allowed).
# "custom" is an OpenAI-compatible endpoint the user points at via base_url.
PROVIDER_MODELS = {
    "anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "custom": [],
}

_DEFAULTS = {"provider": "anthropic", "model": "claude-opus-4-8", "api_key": "", "base_url": ""}


def _seed_from_env() -> dict:
    cfg = dict(_DEFAULTS)
    provider = os.environ.get("AIC_LLM_PROVIDER")
    if provider in PROVIDER_MODELS:
        cfg["provider"] = provider
    if os.environ.get("AIC_LLM_MODEL"):
        cfg["model"] = os.environ["AIC_LLM_MODEL"]
    if os.environ.get("AIC_LLM_BASE_URL"):
        cfg["base_url"] = os.environ["AIC_LLM_BASE_URL"]
    env_key = "ANTHROPIC_API_KEY" if cfg["provider"] == "anthropic" else "OPENAI_API_KEY"
    if os.environ.get(env_key):
        cfg["api_key"] = os.environ[env_key]
    return cfg


def load() -> dict:
    """Return the full config dict (including api_key) for internal use."""
    if _CONFIG_PATH.exists():
        try:
            return {**_DEFAULTS, **json.loads(_CONFIG_PATH.read_text())}
        except Exception:
            logger.exception("failed to read %s; falling back to env", _CONFIG_PATH)
    return _seed_from_env()


def save(provider: str, model: str, api_key: str | None, base_url: str | None = None) -> dict:
    """Persist config. If api_key is None/empty, keep the existing key."""
    current = load()
    new = {
        "provider": provider if provider in PROVIDER_MODELS else current["provider"],
        "model": model or current["model"],
        "api_key": api_key if api_key else current.get("api_key", ""),
        # base_url is non-secret; empty string clears it (allow switching back to a hosted provider).
        "base_url": base_url if base_url is not None else current.get("base_url", ""),
    }
    try:
        _CONFIG_PATH.write_text(json.dumps(new, indent=2))
        _CONFIG_PATH.chmod(0o600)  # key is sensitive — restrict perms
    except Exception:
        logger.exception("failed to write %s", _CONFIG_PATH)
    return new


def public_status() -> dict:
    """Config view safe to send to the frontend (no api_key)."""
    cfg = load()
    return {
        "provider": cfg["provider"],
        "model": cfg["model"],
        "base_url": cfg.get("base_url", ""),
        "key_set": bool(cfg.get("api_key")),
        "provider_models": PROVIDER_MODELS,
    }
