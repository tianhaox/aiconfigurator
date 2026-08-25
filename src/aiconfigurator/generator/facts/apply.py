# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply model-profile ``defaults:`` to rendered cli token lists.

Model defaults sit at the *facts-default* layer of the precedence chain: they
only FILL flags that are not already present. A value supplied by the user, a
recipe, or a rule (and thus already in the token list) always wins.

The unit of mutation is the in-memory ``{role}_cli_args_list`` built in
``rendering/engine.py`` as ``shlex.split(cli)`` — a flat list of single tokens
where a value flag is two tokens (``--block-size`` then ``256``) and a bool flag
is one token (``--trust-remote-code``). Appended defaults match that convention
exactly so they render identically to user/recipe/rule flags.

This module also applies the hardware-derived ``moe_backend`` fact
(:func:`apply_moe_backend`). Unlike model ``defaults:`` (which key off
``ResolvedFacts.model``), ``moe_backend`` is a HARDWARE selection
(``ResolvedFacts.hardware``) and is applied for ANY MoE deployment on the
matched hardware — independent of whether the model has a profile entry.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycle
    from .resolve import ResolvedFacts


def _entry_matches(match: dict[str, Any], *, backend: str, system: str | None, variant: str | None) -> bool:
    """Return True when a ``defaults`` entry's ``match`` applies to this request.

    An absent match field is a wildcard. When present, ``backend`` / ``system``
    / ``model_variant`` must equal the request's value.
    """
    if match.get("backend") and match["backend"] != backend:
        return False
    if match.get("system") and match["system"] != (system or ""):
        return False
    return not (match.get("model_variant") and match["model_variant"] != (variant or ""))


def _flag(name: str) -> str:
    """Normalize a backend cli flag NAME to its ``--`` form (idempotent)."""
    return name if name.startswith("-") else f"--{name}"


def _stringify(value: Any) -> str:
    """Render a default value into a single cli token, matching the list convention.

    Scalars are ``str()``-ed (no quoting — ``shlex.split`` of the rendered string
    strips quotes, so the in-memory list holds bare tokens). dict/list values are
    compact JSON.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def apply_model_default_args(
    tokens: list[str],
    model: dict[str, Any] | None,
    *,
    backend: str,
    system: str | None,
    role: str,
    variant: str | None,
) -> None:
    """Append model-default flags onto ``tokens`` in place (skip-if-present).

    For each ``defaults`` entry whose ``match`` applies and whose ``roles``
    include ``role`` (or ``*``), for each ``{flag: value}``:

    * skip if the flag is already present in ``tokens`` (in any form);
    * bool ``True``  -> append the flag token only;
    * bool ``False`` / ``None`` -> omit entirely;
    * otherwise -> append the flag token then the stringified value token.
    """
    for entry in (model or {}).get("defaults", []) or []:
        if not _entry_matches(entry.get("match", {}) or {}, backend=backend, system=system, variant=variant):
            continue
        role_filter = entry.get("roles", ["*"]) or ["*"]
        if "*" not in role_filter and role not in role_filter:
            continue
        for name, value in (entry.get("backend_args", {}) or {}).items():
            flag = _flag(name)
            if flag in tokens:
                continue
            if value is False or value is None:
                continue
            if value is True:
                tokens.append(flag)
                continue
            tokens.append(flag)
            tokens.append(_stringify(value))


#: hardware ``moe_backend`` choices that are only valid on wide-EP deployments
#: (attention DP / EP>1). Probed on B200 2026-08-24 (collector/facts sm100 sweep):
#: trtllm 1.3.0rc23 asserts "Attention DP should be used with WideEP" at boot when
#: WIDEEP is selected on a tp1/ep1 render, and sglang 0.5.16's ServerArgs parser
#: rejects ``--moe-runner-backend deepep_moe`` outright (its deepep spelling is
#: ``--moe-a2a-backend deepep``, a wide-EP-only path). See
#: collector/facts/results/findings.yaml sm100_sglang_deepep_moe / sm100_trtllm_wideep_dp.
_WIDE_EP_ONLY_CHOICES = frozenset({"WIDEEP", "deepep_moe"})


def _is_wide_ep_deployment(context: dict[str, Any]) -> bool:
    """A deployment is wide-EP when any role enables attention DP or EP>1."""
    for role in ("prefill", "decode", "agg", "encode"):
        rp = context.get(f"{role}_params")
        if not isinstance(rp, dict):
            continue
        if rp.get("enable_attention_dp"):
            return True
        try:
            if int(rp.get("moe_expert_parallel_size") or 1) > 1:
                return True
        except (TypeError, ValueError):
            pass
    return False


def apply_moe_backend(context: dict[str, Any], hardware: dict[str, Any] | None, *, backend: str) -> None:
    """Apply the hardware-derived ``moe_backend`` selection (fill-if-absent, MoE-only).

    ``hardware`` is the resolved hardware-profile dict (``ResolvedFacts.hardware``).
    Its ``moe_backend`` entry maps backend -> kernel/runner choice, e.g.
    ``{"trtllm": "WIDEEP", "sglang": "deepep_moe"}``. These are the right choices
    for the wide-EP serving shape the Blackwell profiles were written for — and
    STARTUP CRASHES outside it (see ``_WIDE_EP_ONLY_CHOICES``), so wide-EP-only
    choices are applied only when the deployment actually is wide-EP; everything
    else falls through to the framework default (trtllm builder -> CUTLASS,
    sglang -> runner auto-selection).

    A sibling ``moe_backend_quant`` entry maps backend -> {quant_algo -> choice}
    and, when the checkpoint's inferred ``quant_algo`` (ModelConfig) matches,
    wins over the generic choice — e.g. sm100 trtllm FP8-MoE -> DEEPGEMM.

    Precedence: facts-default (fill-if-absent) — a user/recipe/rule value already
    in ``context`` always wins. Guarded to MoE deployments only via
    ``context["is_moe"]``; dense models are left untouched.

    Per-backend seam:

    * ``trtllm``: sets ``context["moe_config"]["backend"]`` (consumed by the typed
      engine-config builder). Only when unset — the builder defaults to CUTLASS.
    * ``sglang``: sets ``context["moe_backend"]`` (mapped to ``--moe-runner-backend``
      via ``backend_config_mapping.yaml``). Only when unset.
    * ``vllm``: no-op — vLLM's MoE backend comes from MODEL ``defaults:`` (handled
      by :func:`apply_model_default_args`), not this hardware fact.
    """
    choice = ((hardware or {}).get("moe_backend") or {}).get(backend)
    # quant-conditional fact wins over the generic one (more specific): keyed
    # on the checkpoint's inferred quant_algo (ModelConfig, set by naive.py).
    # sm100 origin: trtllm FP8-MoE needs DEEPGEMM (findings.yaml
    # sm100_trtllm_fp8moe_deepgemm); CUTLASS default aborts at first forward.
    qmap = ((hardware or {}).get("moe_backend_quant") or {}).get(backend) or {}
    model_cfg = context.get("ModelConfig")
    quant_algo = model_cfg.get("quant_algo") if isinstance(model_cfg, dict) else None
    qchoice = qmap.get(str(quant_algo)) if quant_algo else None
    if qchoice and not (qchoice in _WIDE_EP_ONLY_CHOICES and not _is_wide_ep_deployment(context)):
        choice = qchoice
    if not choice:
        return
    # GUARD: hardware moe_backend applies to MoE deployments only.
    if not context.get("is_moe"):
        return
    # GUARD: wide-EP-only choices apply only to wide-EP deployments (attention
    # DP / EP>1); on tp1-class renders they are boot failures, not defaults.
    if choice in _WIDE_EP_ONLY_CHOICES and not _is_wide_ep_deployment(context):
        return
    if backend == "trtllm":
        moe = context.setdefault("moe_config", {})
        if isinstance(moe, dict) and not moe.get("backend"):
            moe["backend"] = choice
    elif backend == "sglang":
        if not context.get("moe_backend"):
            context["moe_backend"] = choice
    # vllm: intentionally no-op (model-default driven).


def _system_key(facts: ResolvedFacts) -> str | None:
    """Derive the hardware-profile KEY used by ``defaults`` ``match.system``.

    ``ResolvedFacts.hardware`` is the profile *dict* (no bare key inside it), so
    the request resolver stashes the resolved key on ``facts.hardware_key``.
    Fall back to ``None`` when absent (then only system-agnostic entries match).
    """
    return getattr(facts, "hardware_key", None)


def apply_facts(context: dict[str, Any], facts: ResolvedFacts | None, backend: str) -> None:
    """Apply model-default cli args onto every role's token list in ``context``.

    No-op unless ``facts`` carries a matched model profile. Mutates the
    ``{role}_cli_args_list`` lists in place; callers re-sync the cli string
    artifacts from those lists so both the ``cli_args_*`` string and the typed
    k8s builder see identical appended flags.
    """
    if facts is None or facts.model is None:
        return
    system = _system_key(facts)
    # model_variant is not yet resolved per request.
    variant: str | None = None
    for role in ("prefill", "decode", "agg", "encode"):
        list_key = f"{role}_cli_args_list"
        tokens = context.get(list_key)
        if not isinstance(tokens, list):
            continue
        apply_model_default_args(
            tokens,
            facts.model,
            backend=backend,
            system=system,
            role=role,
            variant=variant,
        )
