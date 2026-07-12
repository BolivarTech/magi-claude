#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 5.0.0
# Date: 2026-07-11
"""Pure decision logic for MAGI model fallback/rotation policy.

This module contains no I/O, no networking, and no concurrency: it only decides
which fallback model a mage should try next, given cached capability metadata
and per-mage bookkeeping. A later task owns ``LineageRegistry`` and the actual
agent slot/claim orchestration; this file only provides the policy state and
the rotation decision function.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Sequence

from ollama_config import ModelSpec

REJECT_TOO_SMALL = "too_small"  # measured; the payload provably does not fit
REJECT_UNMEASURABLE = "unmeasurable"  # the probe failed; window unknown

#: Distinct CONNECTION-level lineages that must refuse before we conclude the
#: endpoint itself is dead (R15). Two, not one: a single refusal can be one bad
#: model or one bad route; two distinct lineages refusing means nobody is listening.
ENDPOINT_DOWN_LINEAGE_THRESHOLD = 2


@dataclass(frozen=True)
class ModelCapability:
    """Capability metadata for a single model, pre-loaded by preflight.

    The preflight loads this data so that rotation can be tested as pure
    logic without performing any I/O.

    Attributes:
        window: Context window size in tokens, or ``None`` if the preflight
            probe could not measure it.
        supports_completion: Whether the model exposes chat/completions
            (embeddings-only rejection is enforced by preflight, not here).
    """

    window: int | None
    supports_completion: bool


@dataclass
class AgentRotationState:
    """Per-mage, per-run bookkeeping for fallback rotation decisions.

    This state is local to a single mage run and is never shared or locked.

    Attributes:
        model_configured: The model originally requested by the TOML
            ``[models]`` section.
        model_used: The model that actually produced the verdict. This differs
            from ``model_configured`` exactly when a rotation occurred, which
            is what the telemetry report must display.
        fallback_reason: Structured cause of a rotation (R13), or ``None`` when
            no rotation has happened.
        used: Model ids that this mage has already attempted.
        failed_lineages: Lineages that have SCHEMA-failed for this mage.
        window_rejected: Mapping of model id to the rejection reason
            (``REJECT_TOO_SMALL`` or ``REJECT_UNMEASURABLE``) after the exact
            probe has rejected it.
        rotations_done: Number of rotations already performed in this run.
        succeeded: ``True`` as soon as a valid verdict exists; this is the sole
            determinant of whether cleanup should run.
    """

    model_configured: ModelSpec | None = None
    model_used: ModelSpec | None = None
    fallback_reason: dict[str, Any] | None = None
    used: set[str] = field(default_factory=set)
    failed_lineages: set[str] = field(default_factory=set)
    window_rejected: dict[str, str] = field(default_factory=dict)
    rotations_done: int = 0
    succeeded: bool = False


class RotationPolicy:
    """Pure fallback decision policy for a mage run.

    Args:
        fallback: Ordered sequence of fallback ``ModelSpec`` candidates.
        max_rotations: Maximum rotations allowed for a single run; ``0``
            disables rotation entirely.
        min_window_tokens: Raw payload size threshold used as a pre-filter.
            This is the *raw* size, without retry-feedback or output-headroom
            padding, and it discards only certain misfits; the definitive probe
            makes the real window decision later.
        capabilities: Mapping from model id to pre-loaded ``ModelCapability``.
        strict_context_guard: If ``True``, models with an unknown window are
            rejected by the pre-filter; otherwise they remain eligible.
    """

    def __init__(
        self,
        fallback: Sequence[ModelSpec],
        max_rotations: int,
        min_window_tokens: int,
        capabilities: Mapping[str, ModelCapability],
        strict_context_guard: bool,
    ) -> None:
        """Initialize the policy with immutable fallback metadata."""
        self._fallback: tuple[ModelSpec, ...] = tuple(fallback)
        self._max_rotations: int = max_rotations
        self._min_window_tokens: int = min_window_tokens
        self._capabilities: Mapping[str, ModelCapability] = capabilities
        self._strict_context_guard: bool = strict_context_guard

    def _window_ok(self, spec: ModelSpec) -> bool:
        """Pre-filter a candidate based on its cached context window.

        Args:
            spec: Candidate model specification.

        Returns:
            ``True`` if the candidate passes the pre-filter, ``False`` otherwise.
            Unknown windows are treated as eligible unless
            ``strict_context_guard`` is enabled.
        """
        cap = self._capabilities.get(spec.model)
        if cap is None or cap.window is None:
            return not self._strict_context_guard
        return cap.window >= self._min_window_tokens

    def window_of(self, model: str) -> int | None:
        """Return the cached window for a model id, if known.

        Args:
            model: Model identifier.

        Returns:
            The cached window size, or ``None`` if the model is unknown or its
            window could not be measured. Unknown is never collapsed to zero.
        """
        cap = self._capabilities.get(model)
        if cap is None:
            return None
        return cap.window

    def next_model(
        self,
        agent: str,
        failed_lineages: set[str],
        run_failed_lineages: set[str],
        lineages_in_play: set[str],
        used: set[str],
        window_rejected: dict[str, str],
        rotations_done: int,
    ) -> ModelSpec | None:
        """Select the next fallback candidate for the given agent.

        Args:
            agent: Agent identifier; used for diagnostics only and does not
                affect the symmetric decision.
            failed_lineages: Lineages that have already schema-failed for this
                mage (the full accumulated set).
            run_failed_lineages: Lineages condemned run-wide by a transport
                failure.
            lineages_in_play: Lineages currently held by another live mage.
            used: Model ids this mage has already run.
            window_rejected: Model ids already rejected by the exact probe,
                mapped to their rejection reason.
            rotations_done: Number of rotations already performed.

        Returns:
            The first eligible ``ModelSpec`` from the fallback list, or
            ``None`` if the rotation cap is exhausted or no candidate qualifies.
        """
        if rotations_done >= self._max_rotations:
            return None

        for spec in self._fallback:
            if spec.lineage in lineages_in_play:
                continue
            if spec.lineage in failed_lineages:
                continue
            if spec.lineage in run_failed_lineages:
                continue
            if spec.model in used:
                continue
            if spec.model in window_rejected:
                continue
            if not self._window_ok(spec):
                continue
            return spec

        return None


class LineageRegistry:
    """The ONLY shared state between mages, behind the ONLY lock in the system.

    Holds the lineage each live mage is running, plus the lineages condemned
    run-wide by transport failures (R5a). Every read-decide-commit goes through
    ``claim_next`` so no caller can compose it wrongly (invariant #2/#2b).

    Exactly one :class:`asyncio.Lock` serializes all mutations and any read that
    needs a consistent snapshot. The lock is never held while acquiring another
    lock (the policy is pure; the probe runs outside), so deadlock is impossible
    by construction, not by discipline.
    """

    def __init__(self, initial: Mapping[str, ModelSpec]) -> None:
        """Reserve the trio's lineages.

        Args:
            initial: Mapping from agent name to the trio ``ModelSpec`` it holds.
        """
        self._lock = asyncio.Lock()
        self._active: dict[str, ModelSpec] = dict(initial)
        self._run_failed: set[str] = set()
        self._connection_failed: set[str] = set()
        self._endpoint_down_signalled: bool = False

    @property
    def run_failed_lineages(self) -> set[str]:
        """Return a snapshot of lineages condemned run-wide.

        Returns:
            A COPY of the internal run-failed set, so callers cannot mutate
            registry state outside the lock (invariant #2b).
        """
        return set(self._run_failed)

    async def lineages_in_play(self, exclude: str | None) -> set[str]:
        """Return the set of lineages currently held by live mages.

        Args:
            exclude: Agent name to omit from the result, if any.

        Returns:
            The distinct lineages reserved by every agent other than ``exclude``.
        """
        async with self._lock:
            return {spec.lineage for name, spec in self._active.items() if name != exclude}

    async def register_transport_failure(self, lineage: str, *, connection: bool) -> bool:
        """Condemn *lineage* run-wide (R5a) and decide the endpoint-down fast-fail.

        Schema failures are per-mage and must NOT be registered here. Registering
        the failure and deciding the fast-fail happen ATOMICALLY under the single
        lock: registering then reading the count in a separate call would be a
        TOCTOU (two mages failing at once could both read a crossed threshold, or
        neither).

        Args:
            lineage: The lineage whose model exhausted its attempts on transport
                errors.
            connection: True only for CONNECTION-level evidence the endpoint is
                dead (refused / host unreachable). A 5xx or a timeout is NOT such
                evidence: someone answered, or the model is merely slow.

        Returns:
            True for exactly the single caller that crosses
            ``ENDPOINT_DOWN_LINEAGE_THRESHOLD`` distinct connection-level lineages
            (it must abort the run); the latch guarantees exactly one True.
        """
        async with self._lock:
            self._run_failed.add(lineage)
            if not connection:
                return False
            self._connection_failed.add(lineage)
            if len(self._connection_failed) < ENDPOINT_DOWN_LINEAGE_THRESHOLD:
                return False
            if self._endpoint_down_signalled:
                return False
            self._endpoint_down_signalled = True
            return True

    async def release(self, agent: str) -> None:
        """Drop *agent*'s entry (it died), freeing its lineage. Idempotent.

        Args:
            agent: The mage whose lineage reservation should be freed.
        """
        async with self._lock:
            self._active.pop(agent, None)

    async def claim_next(
        self, agent: str, policy: RotationPolicy, state: AgentRotationState
    ) -> ModelSpec | None:
        """Decide and reserve *agent*'s next model, atomically.

        Under the lock: read the lineages in play, delegate the decision to the
        PURE ``policy.next_model`` (no I/O, no await), and -- only if there is a
        candidate -- replace *agent*'s entry. The mutation is the LAST statement,
        so a failure to decide cannot leave the registry half-updated.

        Args:
            agent: The rotating mage.
            policy: The pure eligibility policy.
            state: The mage's local rotation state.

        Returns:
            The reserved ``ModelSpec``, or None if no candidate qualifies.

        Postconditions:
            * ModelSpec returned -> *agent*'s entry HAS been replaced (old lineage
              freed, new one reserved).
            * None returned -> the registry is UNCHANGED; releasing the mage's
              current lineage is the caller's job, via ``agent_slot``'s finally.
        """
        async with self._lock:
            in_play = {spec.lineage for name, spec in self._active.items() if name != agent}
            chosen = policy.next_model(
                agent=agent,
                failed_lineages=state.failed_lineages,
                run_failed_lineages=self._run_failed,
                lineages_in_play=in_play,
                used=state.used,
                window_rejected=state.window_rejected,
                rotations_done=state.rotations_done,
            )
            if chosen is None:
                return None
            self._active[agent] = chosen
            return chosen

    @asynccontextmanager
    async def agent_slot(
        self, agent: str, state: AgentRotationState
    ) -> AsyncIterator[AgentRotationState]:
        """Own *agent*'s lineage for its run, and clean up correctly.

        ``state.succeeded`` is the SOLE determinant of the exit path -- NOT the
        presence of an exception. A late exception raised AFTER a valid verdict
        (telemetry, teardown) must not release a lineage that is still in play:
        that would break invariant #1 in the happy path.

            state.succeeded -> CONSERVE the entry (verdict counts; lineage stays)
            otherwise       -> release(agent) (the mage died; free its lineage)

        The exception, if any, always propagates.

        Args:
            agent: The mage.
            state: Its local rotation state (carries ``succeeded``).

        Yields:
            The same *state*, so the caller can set ``succeeded``.
        """
        try:
            yield state
        finally:
            if not state.succeeded:
                await self.release(agent)
