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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fallback_policy import ModelCapability
from input_size import estimate_tokens
from lineage_identity import LineageIdentityGuard
from model_context import compute_required_tokens, fetch_capabilities, probe_prompt_tokens
from ollama_config import ModelSpec, OllamaConfig
from redaction import redact_secrets
from validate import ValidationError

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


class DigestCollisionError(OllamaPreflightError):
    """Two magi resolve to the SAME model digest (R5/R5a) -- ensemble collapse."""


class FamilyContradictionError(OllamaPreflightError):
    """Declared lineage contradicts the model's real architecture family (R6, strict)."""


class ContextWindowUnmeasurableError(OllamaPreflightError):
    """No valid positive context window could be extracted (R2/R2b)."""


#: MS4: single source for the opt-out hint -- duplicating this literal across every
#: raise site would let one copy drift from the flag's real name/value.
_STRICT_CONTEXT_GUARD_OPTOUT_HINT = (
    "strict_context_guard is now true by default (MS4). Set strict_context_guard = "
    "false to proceed with an estimated guard."
)


def _context_window_unmeasurable_error(
    unmeasurable_models: Sequence[str],
) -> ContextWindowUnmeasurableError:
    """Build the fail-closed error naming the affected model(s) and the opt-out.

    Args:
        unmeasurable_models: Trio model tags whose payload or context window could
            not be measured, in the order they should be reported.

    Returns:
        The exception to raise. Building it does not raise it -- the caller decides
        whether ``strict_context_guard`` applies.

    Example:
        >>> "strict_context_guard = false" in str(_context_window_unmeasurable_error(["m1"]))
        True
    """
    return ContextWindowUnmeasurableError(
        f"context window unmeasurable for {', '.join(unmeasurable_models)}; "
        f"{_STRICT_CONTEXT_GUARD_OPTOUT_HINT}"
    )


class MissingDigestError(OllamaPreflightError):
    """/api/show omitted the digest -- uniqueness cannot be verified (R5b)."""


def _is_cloud_tag(tag: str) -> bool:
    """True for Ollama cloud tags, whose suffix is exactly ':cloud' or '-cloud'.

    Args:
        tag: A full Ollama model tag string (e.g. ``"gpt-oss:120b-cloud"``).

    Returns:
        ``True`` if *tag* ends with ``":cloud"`` or ``"-cloud"``, else ``False``.
    """
    return tag.endswith((":cloud", "-cloud"))


#: Task 0 spike verdict (2026-07-16): the ``:cloud`` trio's ``/api/show`` OMITS the
#: top-level ``digest`` entirely while STILL reporting architecture. For a cloud tag
#: an absent digest is therefore EXPECTED, never a probe failure -- MissingDigestError
#: (R5b) fires ONLY for a NON-cloud model, where an absent digest is a real gap that
#: leaves uniqueness unverifiable. This constant names that verdict at the one place
#: the R5b decision reads it, so the exemption cannot silently drift from the spike.
_CLOUD_HAS_DIGEST = False

#: Declared-lineage values that carry no real identity information -- an empty
#: string (rejected earlier by ollama_config, but a test/harness config can still
#: construct one) or a placeholder a user might type without knowing better. R6b's
#: "unmapped architecture" INFO note would otherwise fire for these, cross-checking
#: nothing against nothing.
_TRIVIAL_LINEAGES: frozenset[str] = frozenset({"", "unknown", "n/a", "none", "tbd", "placeholder"})


def _is_trivial_lineage(lineage: str) -> bool:
    """True if *lineage* carries no real identity information (R6b gate).

    Args:
        lineage: A declared lineage string.

    Returns:
        ``True`` for an empty string or a case-insensitive match in
        :data:`_TRIVIAL_LINEAGES`.
    """
    return lineage.strip().lower() in _TRIVIAL_LINEAGES


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
        digest_by_model: Trio model tag -> digest, seeded ONLY for models that
            HAVE one (today: empty for the ``:cloud`` trio -- ``_CLOUD_HAS_DIGEST``).
            Internal preflight data for a later rotation lookup to grow lazily
            (Task 5b); it never reaches the 7-key agent JSON or ``magi-report.json``.
    """

    capabilities: dict[str, ModelCapability]
    min_window_tokens: int
    required_tokens: int
    context_guard: str
    lineage_warnings: list[str]
    fallback: tuple[ModelSpec, ...]
    token_estimate_delta: list[dict[str, Any]]
    digest_by_model: dict[str, str] = field(default_factory=dict)


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
        # Honor the configured metadata timeout (MAGI gate, Balthasar): a hardcoded
        # value silently cut a slow-NAS operator's configured window on the /models call.
        with urllib.request.urlopen(req, timeout=config.preflight_timeout_seconds) as resp:
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


def check_config_offline(config: OllamaConfig) -> list[str]:
    """Run every config check that needs no network. The single source of truth.

    Both the preflight (before a run) and ``validate_magi_toml.py`` (before you even
    have a run) must answer the SAME question -- "is this config acceptable?" -- so
    they call this, not their own copy. The validator used to re-derive one of these
    checks by hand and consequently said ``OK`` to a duplicate-lineage fallback that
    the preflight fail-closes on: a pre-run tool that green-lights what the product
    refuses to run is worse than no tool.

    Args:
        config: The resolved configuration.

    Returns:
        Lineage warnings (R21). Non-empty is not a failure -- they are typo hints.

    Raises:
        OllamaPreflightError: Two trio mages sharing a lineage (R22), or two fallback
            entries sharing a lineage (R11.3). Both are fail-closed: they attack the
            one-lineage-one-mage invariant the whole system rests on.
    """
    _check_trio_lineages_are_distinct(config.models)
    _check_fallback_lineages_are_unique(config.fallback, config.models)
    return _check_lineage_patterns(config.models, config.fallback)


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


def _check_digest_collision(
    models: Mapping[str, ModelSpec],
    caps: Mapping[str, ModelCapability],
    guard: LineageIdentityGuard,
) -> None:
    """Abort if two trio mages resolve to the SAME model digest (R5/R5a).

    A digest collision is checked regardless of ``strict_lineage``: unlike a family
    contradiction (a plausible finetune/self-hosted mismatch, and the architecture
    map is non-exhaustive), two mages sharing one digest have no benign
    explanation -- they are, byte-for-byte, the same weights. Ensemble collapse.

    Args:
        models: agent -> ModelSpec, from [models].
        caps: model -> ModelCapability, from ``fetch_capabilities``.
        guard: The pure identity guard that compares the digests.

    Raises:
        DigestCollisionError: Naming the two colliding mages and their shared
            digest.
    """
    agents = list(models.items())
    digests = [caps[spec.model].digest for _, spec in agents]
    collision = guard.digest_collision(digests)
    if collision is None:
        return
    i, j = collision
    agent_i, spec_i = agents[i]
    agent_j, spec_j = agents[j]
    raise DigestCollisionError(
        f"{agent_i} ({spec_i.model}) and {agent_j} ({spec_j.model}) resolve to the "
        f"same model digest ({caps[spec_i.model].digest}) -- ensemble collapse: two "
        "mages would be running byte-identical weights."
    )


def _check_missing_digest(
    models: Mapping[str, ModelSpec],
    caps: Mapping[str, ModelCapability],
) -> None:
    """Abort if a NON-cloud trio model's /api/show omitted the digest (R5b).

    A ``:cloud`` tag's absent digest is EXPECTED and never raises here (see
    ``_CLOUD_HAS_DIGEST``): it simply does not participate in the digest-collision
    check above (its digest stays ``None``, non-comparable by
    :meth:`LineageIdentityGuard.digest_collision`). A NON-cloud model is expected
    to report one; its absence means uniqueness cannot be verified at all, so this
    fails closed rather than silently degrading the guarantee.

    Args:
        models: agent -> ModelSpec, from [models].
        caps: model -> ModelCapability, from ``fetch_capabilities``.

    Raises:
        MissingDigestError: If a non-cloud trio model has no digest.
    """
    for agent, spec in models.items():
        # `_CLOUD_HAS_DIGEST` is the SWITCH (Task 0 Step 3b): today the cloud trio
        # omits digests, so a cloud tag's absent digest is expected and skipped. If a
        # future spike shows cloud DOES report digests, flipping the flag to True makes
        # this line stop skipping -> R5b then guards cloud too, a one-line change.
        if _is_cloud_tag(spec.model) and not _CLOUD_HAS_DIGEST:
            continue
        if caps[spec.model].digest is None:
            raise MissingDigestError(
                f"{agent} ({spec.model}) is not a :cloud tag but /api/show gave no "
                "digest; model-identity uniqueness cannot be verified (R5b)."
            )


def _check_family_verdicts(
    models: Mapping[str, ModelSpec],
    caps: Mapping[str, ModelCapability],
    strict_lineage: bool,
    guard: LineageIdentityGuard,
    warnings: list[str],
) -> None:
    """Compare each trio mage's PROBED architecture against its declared lineage.

    A ``"contradiction"`` verdict either aborts (``strict_lineage=True``) or joins
    *warnings* -- the SAME ``lineage_warnings`` collection R21's tag-prefix typo
    detector feeds, never a second channel. An ``"unknown"`` verdict with a KNOWN
    architecture (probed, just unmapped) adds an R6b informational note UNLESS the
    declared lineage is trivial (nothing to cross-check). ``"ok"`` is silent.

    Args:
        models: agent -> ModelSpec, from [models].
        caps: model -> ModelCapability, from ``fetch_capabilities``.
        strict_lineage: If True, a contradiction raises instead of warning (R6).
        guard: The pure identity guard that compares architecture to lineage.
        warnings: Mutated in place with any WARNING/INFO produced.

    Raises:
        FamilyContradictionError: A contradiction found while strict_lineage=True.
    """
    for agent, spec in models.items():
        architecture = caps[spec.model].architecture
        verdict = guard.family_verdict(architecture, spec.lineage)
        if verdict == "contradiction":
            message = (
                f"{agent} ({spec.model}) declares lineage {spec.lineage!r} but its "
                f"probed architecture family {architecture!r} maps to a different "
                "vendor -- the declared lineage may be wrong, or two mages may "
                "silently share a lab."
            )
            if strict_lineage:
                raise FamilyContradictionError(message)
            warnings.append(f"WARNING: {message}")
        elif (
            verdict == "unknown"
            and architecture is not None
            and not _is_trivial_lineage(spec.lineage)
        ):
            warnings.append(
                f"INFO: {agent} ({spec.model}) architecture family {architecture!r} "
                f"is not in the known vendor map; declared lineage {spec.lineage!r} "
                "could not be cross-checked."
            )


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
    # Probe the trio CONCURRENTLY (NR6b: preflight I/O is O(M) concurrent calls, not
    # serialized round-trips -- MAGI gate, Balthasar). ``gather`` preserves input order,
    # so the deltas stay per-agent ordered. The three probes fit inside Ollama's 3-agent
    # cap and run before any agent launches, so there is no contention.
    items = list(config.models.items())
    exacts = await asyncio.gather(
        *(probe_prompt_tokens(config, spec.model, prompt) for _, spec in items)
    )
    for (agent, spec), exact in zip(items, exacts):
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
            (R11.3); a configured model without chat capability (R19); or a trio
            model whose window cannot hold the payload (R5b).
        ContextWindowUnmeasurableError: A trio model's payload or context window
            could not be measured (absent, unmeasurable, or invalid -- R2/R2b) and
            ``strict_context_guard`` is enabled, which is now the default (MS4). The
            message names both the unmeasurable model(s) and the opt-out.
        DigestCollisionError: Two trio mages resolve to the same model digest
            (Grieta 2) -- checked regardless of ``strict_lineage``.
        MissingDigestError: A non-``:cloud`` trio model's ``/api/show`` gave no
            digest (Grieta 2/R5b); a ``:cloud`` model's absent digest is expected
            and never raises (``_CLOUD_HAS_DIGEST``).
        FamilyContradictionError: A trio model's probed architecture contradicts
            its declared lineage AND ``strict_lineage`` is enabled (default False,
            in which case the contradiction only joins ``lineage_warnings``).
    """
    try:
        available = await _list_models(config)
    except OllamaPreflightError as exc:
        raise OllamaPreflightError(redact_secrets(str(exc), config.api_key)) from None

    # 1. Structural checks first -- they cost nothing and catch config errors. The
    #    tag-prefix typo warnings (R21) are deferred to step 3b, once the PROBED
    #    architecture is known and can supersede a mere tag-prefix guess for the
    #    trio -- see the invariant note at step 3b.
    _check_trio_lineages_are_distinct(config.models)
    _check_fallback_lineages_are_unique(config.fallback, config.models)

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

    # 3a. Model-identity guards (Grieta 2): a digest collision is ensemble collapse
    #     and is always fatal; a non-cloud trio model missing its digest cannot have
    #     its uniqueness verified at all and is always fatal (R5b). Neither depends
    #     on strict_lineage -- that flag governs ONLY the family (architecture vs.
    #     declared lineage) contradiction below.
    identity_guard = LineageIdentityGuard()
    _check_digest_collision(config.models, caps, identity_guard)
    _check_missing_digest(config.models, caps)

    # 3b. R21's tag-prefix typo detector and the family check are MUTUALLY
    #     EXCLUSIVE per trio model: where /api/show reported an architecture, the
    #     family check below is strictly more informative (it compares REAL probed
    #     identity, not a tag-prefix guess) and supersedes it; the tag check only
    #     fires where no architecture was probed at all (fallback models, always --
    #     Grieta 2 does not extend digest/family checks to them -- and any trio
    #     model whose /api/show omitted architecture).
    unknown_arch_trio = {
        agent: spec
        for agent, spec in config.models.items()
        if caps[spec.model].architecture is None
    }
    lineage_warnings = _check_lineage_patterns(unknown_arch_trio, fallback)
    _check_family_verdicts(
        config.models, caps, config.strict_lineage, identity_guard, lineage_warnings
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

    # ENFORCED requires BOTH: the payload was measured AND every window is known.
    # Payload-measurability alone is NOT enough -- invariant #3 ("no mage runs a model
    # whose payload would not fit its window") cannot be PROVEN when the window is
    # unknown, so labelling that run "enforced" lies about protection, and worse,
    # ``strict_context_guard`` would fail OPEN (the too_small check below skips unknown
    # windows). Context-guard review, 2026-07-12.
    payload_measured = len(measured) == len(config.models)
    windows_known = all(caps[spec.model].window is not None for spec in config.models.values())

    if payload_measured and windows_known:
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
        # Name the SPECIFIC model(s) that could not be measured -- a model is
        # unmeasurable if its payload was never probed OR its window is unknown
        # (R2/R2b; ``_read_window`` already collapses absent/zero/non-positive to
        # None, so "unknown" and "invalid" are the same case here).
        unmeasurable = sorted(
            spec.model
            for spec in config.models.values()
            if spec.model not in measured or caps[spec.model].window is None
        )
        if config.strict_context_guard:  # R18/MS4: strict is strict -- cannot prove the fit
            raise _context_window_unmeasurable_error(unmeasurable)
        reason = (
            "the payload could not be measured for every trio model"
            if not payload_measured
            else "the context window is unknown for some trio model (no /api/show data)"
        )
        print(
            f"WARNING: {reason}; falling back to the estimator. NOTE: without measured "
            "payloads AND known windows there is NO reliable truncation protection.",
            file=sys.stderr,
        )
        guard = CONTEXT_GUARD_ESTIMATED
        # Use each model's EXACT count where it was measured (decision #96: degrading
        # accuracy must not degrade prudence), the estimate only where it was not.
        needs = {
            spec.model: (
                _required(measured[spec.model], exact=True)
                if spec.model in measured
                else _required(estimate, exact=False)
            )
            for spec in config.models.values()
        }
        payload = max(measured.values()) if payload_measured else estimate
        exact_flag = payload_measured

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
    # Task 5b seed: trio model -> digest, ONLY where one was actually reported (today
    # empty for the :cloud trio -- _CLOUD_HAS_DIGEST). Internal preflight data; never
    # copied onto the 7-key agent JSON or magi-report.json.
    digest_by_model = {
        spec.model: digest
        for spec in config.models.values()
        if (digest := caps[spec.model].digest) is not None
    }
    return PreflightResult(
        capabilities=caps,
        min_window_tokens=payload,
        required_tokens=required,
        context_guard=guard,
        lineage_warnings=lineage_warnings,
        fallback=fallback,
        token_estimate_delta=deltas,
        digest_by_model=digest_by_model,
    )
