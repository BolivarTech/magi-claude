#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 5.0.0
# Date: 2026-07-11
"""Preflight for the Ollama backend: reachability, lineage/capability/window guards.

Validates and MEASURES everything before a single agent is launched. Cheap
structural checks (lineage uniqueness) fail fast; only then do we pay for the
network (windows, capabilities, the exact token probe).
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fallback_policy import ModelCapability
from input_size import estimate_tokens
from model_context import compute_required_tokens, fetch_capabilities, probe_prompt_tokens
from ollama_config import ModelSpec, OllamaConfig
from redaction import redact_secrets
from validate import ValidationError

PREFLIGHT_TIMEOUT = 10

CONTEXT_GUARD_ENFORCED = "enforced"  # payload MEASURED; invariant #3 holds
CONTEXT_GUARD_ESTIMATED = "estimated"  # could not measure; invariant #3 does NOT hold

#: Tag prefix -> expected lineage. A TYPO DETECTOR, never an authority: the TOML
#: declaration always wins (decision #5). A stale table costs at worst a spurious
#: warning -- never a wrong decision. That asymmetry is why inference is acceptable
#: here for a WARNING and was rejected as a source of truth.
LINEAGE_PATTERNS: Mapping[str, str] = MappingProxyType(
    {
        "qwen": "alibaba",
        "kimi": "moonshot",
        "glm": "zhipu",
        "deepseek": "deepseek",
        "gpt-oss": "openai",
        "minimax": "minimax",
        "gemma": "google",
        "gemini": "google",
        "nemotron": "nvidia",
    }
)


class OllamaPreflightError(ValidationError):
    """Raised when the Ollama host is unreachable or a config guard fails."""


def _is_cloud_tag(tag: str) -> bool:
    """True for Ollama cloud tags, whose suffix is exactly ':cloud' or '-cloud'.

    Args:
        tag: A full Ollama model tag string (e.g. ``"gpt-oss:120b-cloud"``).

    Returns:
        ``True`` if *tag* ends with ``":cloud"`` or ``"-cloud"``, else ``False``.
    """
    return tag.endswith((":cloud", "-cloud"))


@dataclass(frozen=True)
class PreflightResult:
    """Everything the preflight MEASURED, handed to the orchestrator as one value.

    Attributes:
        capabilities: model id -> ModelCapability for the trio AND surviving
            fallbacks; the rotation path reads this cache and does no I/O (R20).
        min_window_tokens: RAW payload tokens -- the pre-filter threshold (C2-1).
        required_tokens: Padded worst case (payload + retry feedback + output
            headroom) -- the DEFINITIVE threshold.
        context_guard: CONTEXT_GUARD_ENFORCED or CONTEXT_GUARD_ESTIMATED. Reported,
            never hidden: a guard that did not run must not look like one that did.
        lineage_warnings: Declared lineages that disagree with LINEAGE_PATTERNS.
        fallback: The PRUNED fallback list -- entries whose tags are absent from
            the endpoint have already been dropped with a warning (R11.1).
        token_estimate_delta: Per trio model, the heuristic estimate, the measured
            count and the error -- so the margin can be validated with real data.
    """

    capabilities: dict[str, ModelCapability]
    min_window_tokens: int
    required_tokens: int
    context_guard: str
    lineage_warnings: list[str]
    fallback: tuple[ModelSpec, ...]
    token_estimate_delta: list[dict[str, Any]]


async def _list_models(config: OllamaConfig) -> set[str]:
    """Return the set of model tags the endpoint reports as available.

    Args:
        config: The resolved configuration (endpoint, auth).

    Returns:
        The available model tags. On 404/501 (listing unsupported) it warns and
        returns every configured tag, so the caller flags nothing missing.

    Raises:
        OllamaPreflightError: On 401/403 (auth), any other HTTP error, or an
            unreachable host. Endpoint text is redacted at the raising boundary.
    """
    url = f"{config.base_url}/models"
    headers = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")

    def _call() -> Any:
        with urllib.request.urlopen(req, timeout=PREFLIGHT_TIMEOUT) as resp:
            return json.loads(resp.read())

    try:
        payload = await asyncio.to_thread(_call)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise OllamaPreflightError(
                redact_secrets(
                    f"Auth failed ({exc.code}) for {config.base_url}; "
                    "check api_key / `ollama signin`.",
                    config.api_key,
                )
            ) from None
        if exc.code in (404, 501):
            print(
                f"WARNING: {config.base_url}/models unavailable ({exc.code}); "
                "skipping model-existence check.",
                file=sys.stderr,
            )
            return {s.model for s in config.models.values()} | {f.model for f in config.fallback}
        raise OllamaPreflightError(
            redact_secrets(f"Preflight HTTP {exc.code} at {url}.", config.api_key)
        ) from None
    except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
        raise OllamaPreflightError(
            f"Cannot reach Ollama at {config.base_url}: {exc}. "
            "Is it running? Try `ollama signin` for cloud."
        ) from None
    return {
        mid
        for m in payload.get("data", [])
        if isinstance(m, dict) and isinstance((mid := m.get("id")), str)
    }


def _check_trio_lineages_are_distinct(models: Mapping[str, ModelSpec]) -> None:
    """Abort if two mages declare the same lineage (R22).

    The PRIMARY path, unguarded for nine gate cycles while the rotation path was
    fortified: a trio with two mages of one lineage is born violating the invariant
    the whole feature exists to protect, and its consensus only LOOKS like three
    independent perspectives. That is not degraded MAGI -- it is fake MAGI.

    Args:
        models: agent -> ModelSpec, from [models].

    Raises:
        OllamaPreflightError: If any lineage is claimed by more than one mage.
    """
    by_lineage: dict[str, list[str]] = {}
    for agent, spec in models.items():
        by_lineage.setdefault(spec.lineage, []).append(agent)
    for lineage, agents in by_lineage.items():
        if len(agents) > 1:
            raise OllamaPreflightError(
                f"{' and '.join(agents)} both use lineage {lineage!r}. Each mage must have "
                "a unique lineage -- that independence is the entire premise of MAGI."
            )


def _check_fallback_lineages_are_unique(
    fallback: Sequence[ModelSpec],
    models: Mapping[str, ModelSpec],
) -> None:
    """Abort on duplicate fallback lineages (R11.3); warn on dead entries (R11.4).

    Args:
        fallback: The declared fallback list.
        models: The trio, to detect entries that can never be eligible.

    Raises:
        OllamaPreflightError: If two fallback entries share a lineage. Fail-closed:
            a duplicate is a config error that attacks the central invariant.
    """
    seen: dict[str, str] = {}
    for spec in fallback:
        if spec.lineage in seen:
            raise OllamaPreflightError(
                f"fallback entries {seen[spec.lineage]!r} and {spec.model!r} share lineage "
                f"{spec.lineage!r}; only one model per lineage is ever reachable."
            )
        seen[spec.lineage] = spec.model

    trio_lineages = {spec.lineage: agent for agent, spec in models.items()}
    for spec in fallback:
        if spec.lineage in trio_lineages:  # R11.4: dead weight, not dangerous
            print(
                f"WARNING: fallback {spec.model!r} has lineage {spec.lineage!r}, which trio mage "
                f"{trio_lineages[spec.lineage]!r} already holds -- it can never be eligible "
                "(dead entry).",
                file=sys.stderr,
            )


def _check_lineage_patterns(
    models: Mapping[str, ModelSpec],
    fallback: Sequence[ModelSpec],
) -> list[str]:
    """Flag declared lineages that disagree with the known tag prefixes (R21).

    A TYPO DETECTOR, never an authority: the declaration always wins (decision #5).
    A stale table costs at worst a spurious warning; it can never cause a wrong
    decision. That asymmetry is why inference is acceptable here and was rejected
    as a source of truth.

    Args:
        models: The trio.
        fallback: The fallback list.

    Returns:
        Warnings for every declared lineage that disagrees with its tag prefix.
    """
    warnings: list[str] = []
    for spec in list(models.values()) + list(fallback):
        for prefix, expected in LINEAGE_PATTERNS.items():
            if spec.model.startswith(prefix) and spec.lineage != expected:
                warnings.append(
                    f"{spec.model} declares lineage {spec.lineage!r} but its tag suggests "
                    f"{expected!r} -- if that is a typo, two mages may silently share a lab."
                )
    return warnings


async def _measure_payload(
    config: OllamaConfig, prompt: str
) -> tuple[dict[str, int], list[dict[str, Any]], int]:
    """Probe each trio model's OWN tokenizer count for *prompt* (R5c).

    Args:
        config: The resolved configuration.
        prompt: The exact payload the agents will receive.

    Returns:
        ``(measured, deltas, estimate)``: *measured* maps model id -> exact token
        count for every trio model that could be probed; *deltas* is the per-agent
        estimate/actual/error telemetry; *estimate* is the heuristic fallback count.
    """
    estimate = estimate_tokens(prompt)
    measured: dict[str, int] = {}
    deltas: list[dict[str, Any]] = []
    for agent, spec in config.models.items():
        exact = await probe_prompt_tokens(config, spec.model, prompt)
        if exact is None:
            continue
        measured[spec.model] = exact
        deltas.append(
            {
                "agent": agent,
                "estimated": estimate,
                "actual": exact,
                "error_pct": round((estimate - exact) / exact * 100, 1),
            }
        )
    return measured, deltas, estimate


async def preflight(config: OllamaConfig, prompt: str) -> PreflightResult:
    """Validate and MEASURE everything before a single agent is launched.

    Args:
        config: The resolved configuration (called exactly once, in setup).
        prompt: The exact payload the agents will receive -- what we measure.

    Returns:
        Everything the orchestrator needs, measured once and cached.

    Raises:
        OllamaPreflightError: Host unreachable; auth failure; a TRIO model missing;
            two trio mages sharing a lineage (R22); two fallbacks sharing a lineage
            (R11.3); a configured model without chat capability (R19); a trio model
            whose window cannot hold the payload (R5b); or an unmeasurable window
            under ``strict_context_guard`` (R18).
    """
    try:
        available = await _list_models(config)
    except OllamaPreflightError as exc:
        raise OllamaPreflightError(redact_secrets(str(exc), config.api_key)) from None

    # 1. Structural checks first -- they cost nothing and catch config errors.
    _check_trio_lineages_are_distinct(config.models)
    _check_fallback_lineages_are_unique(config.fallback, config.models)
    lineage_warnings = _check_lineage_patterns(config.models, config.fallback)

    # 2. The trio is a REQUIREMENT; the fallbacks are insurance (R11.1).
    missing = [spec.model for spec in config.models.values() if spec.model not in available]
    if missing:
        trio_tags = [s.model for s in config.models.values()]
        if all(_is_cloud_tag(t) for t in trio_tags) and not any(
            _is_cloud_tag(str(m)) for m in available
        ):
            raise OllamaPreflightError(
                f"No :cloud models available on {config.base_url} (the trio is all :cloud). "
                "Run `ollama signin` first (cloud models need a cloud session on the local "
                "daemon), or set api_key for the direct cloud API, or switch to local tags."
            )
        raise OllamaPreflightError(f"trio model(s) not available: {', '.join(missing)}.")

    fallback = tuple(f for f in config.fallback if f.model in available)
    for dropped in config.fallback:
        if dropped.model not in available:
            print(f"WARNING: fallback {dropped.model} is not available; dropped.", file=sys.stderr)

    # 3. Now pay for the network: windows + capabilities, concurrently, once (R20).
    models = [s.model for s in config.models.values()] + [f.model for f in fallback]
    caps = await fetch_capabilities(config, models)

    no_chat = [m for m in models if not caps[m].supports_completion]
    if no_chat:  # R19: embeddings models would fail 100% of the time and burn a rotation
        raise OllamaPreflightError(
            f"model(s) without chat/completion capability: {', '.join(no_chat)}."
        )

    # 4. MEASURE the payload with each trio model's OWN tokenizer (R5c).
    measured, deltas, estimate = await _measure_payload(config, prompt)

    def _required(payload_tokens: int, *, exact: bool) -> int:
        return compute_required_tokens(
            payload_tokens,
            output_headroom_tokens=config.output_headroom_tokens,
            input_margin_pct=config.input_margin_pct,
            exact=exact,
        )

    if len(measured) == len(config.models):  # every trio model measured
        guard = CONTEXT_GUARD_ENFORCED
        # R5c is PER MODEL: each trio model against ITS OWN exact count, never a
        # global max -- a model that tokenises efficiently must not be aborted over
        # a worse-tokenising sibling's larger count.
        needs = {
            spec.model: _required(measured[spec.model], exact=True)
            for spec in config.models.values()
        }
        payload, exact_flag = max(measured.values()), True
    else:
        if config.strict_context_guard:  # R18: strict is strict
            raise OllamaPreflightError(
                "could not measure the payload for every trio model and "
                "strict_context_guard is enabled."
            )
        print(
            "WARNING: the payload could not be measured for every trio model; falling back "
            "to the estimator. NOTE: on an endpoint without /api/show there is NO truncation "
            "protection at all.",
            file=sys.stderr,
        )
        guard = CONTEXT_GUARD_ESTIMATED
        est = _required(estimate, exact=False)
        needs = {spec.model: est for spec in config.models.values()}
        payload, exact_flag = estimate, False

    # 5. A trio model that cannot hold ITS OWN payload count does not run at all (R5b/R5c).
    too_small = [
        spec.model
        for spec in config.models.values()
        if (w := caps[spec.model].window) is not None and w < needs[spec.model]
    ]
    if too_small:
        raise OllamaPreflightError(
            "context window too small for this payload ("
            + ", ".join(f"{m} needs {needs[m]}" for m in too_small)
            + "). A model that truncates produces a verdict that looks legitimate and is not."
        )

    required = _required(payload, exact=exact_flag)
    return PreflightResult(
        capabilities=caps,
        min_window_tokens=payload,
        required_tokens=required,
        context_guard=guard,
        lineage_warnings=lineage_warnings,
        fallback=fallback,
        token_estimate_delta=deltas,
    )
