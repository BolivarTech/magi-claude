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
from typing import Any, AsyncIterator, Callable, Mapping, MutableMapping, Sequence

from lineage_identity import LineageIdentityGuard
from ollama_config import ModelSpec

REJECT_TOO_SMALL = "too_small"  # measured; the payload provably does not fit
REJECT_UNMEASURABLE = "unmeasurable"  # the probe failed; window unknown
#: A candidate resolves to the same digest as a mage already ACTIVE this run
#: (Task 5b, R5a-at-rotation) -- ensemble collapse, caught before commit.
REJECT_DIGEST_COLLISION = "digest_collision"
#: A model's digest could not be established (candidate: non-cloud with no
#: reported digest, R5b; or an active mage's digest is unrecoverable from the
#: lookup, R5c) -- uniqueness cannot be PROVEN, so the candidate is rejected
#: fail-closed rather than silently treated as non-comparable.
REJECT_DIGEST_UNVERIFIABLE = "digest_unverifiable"

#: Distinct CONNECTION-level lineages that must refuse before we conclude the
#: endpoint itself is dead (R15). Two, not one: a single refusal can be one bad
#: model or one bad route; two distinct lineages refusing means nobody is listening.
ENDPOINT_DOWN_LINEAGE_THRESHOLD = 2

#: The SOLE digest comparator (SRP): pairwise digest collision detection lives in
#: exactly one place, ``LineageIdentityGuard.digest_collision``. It is pure and
#: stateless (a frozen class constant is its only attribute), so ONE module-level
#: instance is shared by every ``claim_next`` call rather than re-constructed.
_IDENTITY_GUARD = LineageIdentityGuard()


def _assume_cloud(_model: str) -> bool:
    """Default ``is_cloud`` predicate for callers that do not track tag shape.

    Treats every model as a ``:cloud`` tag, i.e. a missing digest is always
    EXPECTED and never fails closed. Pre-Task-5b callers of :meth:`claim_next`
    (and any caller that omits ``is_cloud``) never reasoned about digests at
    all; assuming cloud reproduces that exact prior behaviour instead of
    silently introducing a new fail-closed path under them.

    Args:
        _model: Unused -- the permissive default ignores the model tag.

    Returns:
        Always ``True``.
    """
    return True


def _resolve_digest(
    model: str,
    digest_by_model: MutableMapping[str, str],
    policy: "RotationPolicy",
) -> str | None:
    """Look up *model*'s digest, growing the per-run lookup with ZERO I/O.

    The per-run ``digest_by_model`` lookup is checked first; on a miss, the
    digest is served from ``policy``'s already-fetched capability cache (the
    preflight measured it once, for the trio AND every surviving fallback --
    R20) and, if found, written into the lookup exactly once (append-only: an
    existing entry is never overwritten, matching that a model's digest is
    fixed for the life of the run). Neither branch performs I/O or an
    ``/api/show`` call (BDD-8/Task 5b).

    Args:
        model: The model tag to resolve.
        digest_by_model: The mutable per-run digest lookup (grown in place).
        policy: The pure rotation policy, whose cached capabilities are the
            zero-I/O source for a digest not yet in the lookup.

    Returns:
        The digest string, or ``None`` if no digest is on record for *model*
        (expected for a ``:cloud`` tag; a gap for anything else).
    """
    digest = digest_by_model.get(model)
    if digest is not None:
        return digest
    digest = policy.digest_of(model)
    if digest is not None:
        digest_by_model[model] = digest
    return digest


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
        digest: The model's identity fingerprint from ``/api/show``, or
            ``None``. Absent for the ``:cloud`` trio -- that is EXPECTED and
            CORRECT (Task 0 spike), not a probe failure.
        architecture: The model's architecture family (e.g. ``"qwen3.5"``),
            or ``None`` if ``/api/show`` did not report one.
    """

    window: int | None
    supports_completion: bool
    digest: str | None = None
    architecture: str | None = None


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
    #: True when the mage's COMMITTED (final) model was accepted on an estimated or
    #: unknown window rather than an exact probe. The run-level ``context_guard`` must
    #: read "estimated" if any surviving mage ran unmeasured (R16 honesty on the
    #: rotation path -- the highest-risk path per R5b).
    ran_unmeasured: bool = False


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

    def digest_of(self, model: str) -> str | None:
        """Return the cached digest for a model id, if known.

        Zero I/O: this reads the SAME capability cache the preflight fetched
        once for the trio and every surviving fallback (R20) -- it never
        performs a fresh ``/api/show`` call (BDD-8/Task 5b).

        Args:
            model: Model identifier.

        Returns:
            The cached digest, or ``None`` if the model is unknown or its
            ``/api/show`` payload omitted the digest (expected for a
            ``:cloud`` tag; see ``ollama_preflight._CLOUD_HAS_DIGEST``).
        """
        cap = self._capabilities.get(model)
        if cap is None:
            return None
        return cap.digest

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

    def _in_play_excluding(self, exclude: str | None) -> set[str]:
        """Lineages held by agents other than *exclude*. Caller MUST hold the lock.

        Deliberately NON-async and non-locking: it reads shared state, so it is
        only correct inside an ``async with self._lock`` block. Re-acquiring the
        (non-reentrant) lock here would deadlock -- hence a plain helper the two
        lock-holding callers share (DRY), not a public coroutine.

        Args:
            exclude: Agent name to omit, or None to include all.

        Returns:
            The distinct lineages currently reserved by every other agent.
        """
        return {spec.lineage for name, spec in self._active.items() if name != exclude}

    async def lineages_in_play(self, exclude: str | None) -> set[str]:
        """Return the set of lineages currently held by live mages.

        Args:
            exclude: Agent name to omit from the result, if any.

        Returns:
            The distinct lineages reserved by every agent other than ``exclude``.
        """
        async with self._lock:
            return self._in_play_excluding(exclude)

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
        self,
        agent: str,
        policy: RotationPolicy,
        state: AgentRotationState,
        digest_by_model: MutableMapping[str, str] | None = None,
        is_cloud: Callable[[str], bool] = _assume_cloud,
    ) -> ModelSpec | None:
        """Decide and reserve *agent*'s next model, atomically -- digest-unique too.

        Under the SAME lock that guards the lineage commit (no second lock is ever
        taken -- deadlock stays impossible by construction), this now also enforces
        model-DIGEST uniqueness across the currently-ACTIVE mages (Task 5b,
        R5a/R5b/R5c), delegating every pairwise comparison to
        :data:`_IDENTITY_GUARD` (SRP -- no comparison is re-implemented here).
        ``policy.next_model`` stays PURE (no I/O, no await); the digest lookup is
        equally I/O-free (:func:`_resolve_digest` only ever reads *digest_by_model*
        or ``policy``'s already-fetched capability cache).

        A candidate is retried (not just rejected once) IN THE SAME lock
        acquisition: a digest problem marks the model in ``state.window_rejected``
        (the same exclusion set the window/probe checks use) and the loop asks
        ``policy.next_model`` again, so a rotation that must skip several
        digest-unsafe candidates still resolves to one commit call. Three
        digest-shaped rejections, each fail-CLOSED (never a silent skip):

            * R5a (collision): the candidate's digest equals an ACTIVE mage's
              digest -- ensemble collapse, caught before it can ever run.
            * R5b (candidate unverifiable): the candidate is NOT a ``:cloud`` tag
              (per *is_cloud*) and has no digest -- uniqueness cannot be proven.
              A ``:cloud`` candidate with no digest is exempt (expected, per
              ``ollama_preflight._CLOUD_HAS_DIGEST``) and treated as
              non-comparable, exactly as :meth:`LineageIdentityGuard.digest_collision`
              already treats ``None``.
            * R5c (active unverifiable): an already-ACTIVE mage's model is NOT
              cloud and its digest cannot be resolved -- collision safety against
              THAT mage cannot be proven, so the candidate is rejected even though
              the fault is not the candidate's own.

        The candidate enters ``self._active`` ONLY on a successful commit, so a
        rejected candidate never becomes active and never compares against itself.

        Args:
            agent: The rotating mage.
            policy: The pure eligibility policy (also the zero-I/O digest source).
            state: The mage's local rotation state (mutated: ``window_rejected``
                gains an entry for every digest-rejected candidate).
            digest_by_model: The per-run digest lookup (model -> digest), seeded
                from the trio (``PreflightResult.digest_by_model``) and grown
                append-only as candidates are resolved. Lives OUTSIDE this
                registry's own state (never stored on ``self``); omit it (``None``)
                to get a throwaway dict scoped to this one call -- the permissive
                default for callers that do not track digests at all.
            is_cloud: Predicate for whether a model tag is a ``:cloud`` tag.
                Defaults to :func:`_assume_cloud` (always ``True``), which
                reproduces the exact pre-Task-5b behaviour for any caller that
                does not pass one.

        Returns:
            The reserved ``ModelSpec``, or None if no candidate qualifies (the
            eligibility policy is exhausted, or every remaining candidate is
            digest-unsafe).

        Postconditions:
            * ModelSpec returned -> *agent*'s entry HAS been replaced (old lineage
              freed, new one reserved) AND is digest-distinct from every other
              currently-active mage.
            * None returned -> the registry is UNCHANGED; releasing the mage's
              current lineage is the caller's job, via ``agent_slot``'s finally.
        """
        lookup: MutableMapping[str, str] = {} if digest_by_model is None else digest_by_model
        async with self._lock:
            in_play = self._in_play_excluding(agent)
            while True:
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

                candidate_digest = _resolve_digest(chosen.model, lookup, policy)
                if candidate_digest is None and not is_cloud(chosen.model):
                    # R5b: a non-cloud candidate with no digest leaves uniqueness
                    # unverifiable -- fail closed and try the next candidate.
                    state.window_rejected[chosen.model] = REJECT_DIGEST_UNVERIFIABLE
                    continue

                active_digests: list[str | None] = []
                active_unverifiable = False
                for other_agent, spec in self._active.items():
                    if other_agent == agent:
                        continue  # never compare the candidate against itself
                    other_digest = _resolve_digest(spec.model, lookup, policy)
                    if other_digest is None and not is_cloud(spec.model):
                        # R5c: an already-active mage's digest cannot be
                        # recovered -- collision safety cannot be PROVEN, so this
                        # candidate is rejected rather than silently treated as
                        # non-comparable.
                        active_unverifiable = True
                        break
                    active_digests.append(other_digest)
                if active_unverifiable:
                    state.window_rejected[chosen.model] = REJECT_DIGEST_UNVERIFIABLE
                    continue

                if (
                    _IDENTITY_GUARD.digest_collision([candidate_digest, *active_digests])
                    is not None
                ):
                    # R5a: the candidate would collapse the ensemble with a mage
                    # already active this run -- reject and re-propose.
                    state.window_rejected[chosen.model] = REJECT_DIGEST_COLLISION
                    continue

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
