"""Minimal static-batch inference runtime proof of concept."""

from jit_rt.config import ConfigError, RuntimeSpec, load_spec
from jit_rt.runtime import GenerateResult, JitRuntime

__all__ = [
    "ConfigError",
    "GenerateResult",
    "JitRuntime",
    "RuntimeSpec",
    "load_spec",
]
