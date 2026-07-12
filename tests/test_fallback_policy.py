# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-11
"""Tests for the pure fallback rotation policy (Task 3, R5/R5a/R6/R24)."""

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fallback_policy import (
    AgentRotationState,
    LineageRegistry,
    ModelCapability,
    RotationPolicy,
)
from ollama_config import ModelSpec

GLM = ModelSpec("glm-5.2:cloud", "zhipu")
OA = ModelSpec("gpt-oss:120b-cloud", "openai")
MM = ModelSpec("minimax-m3:cloud", "minimax")
FALLBACK = [GLM, OA, MM]
CAPS = {m.model: ModelCapability(window=200_000, supports_completion=True) for m in FALLBACK}
REQUIRED_RAW = 50_000  # RAW payload -- the pre-filter threshold (spec amendment C2-1)


def _policy(**kw):
    defaults = dict(
        fallback=FALLBACK,
        max_rotations=2,
        min_window_tokens=REQUIRED_RAW,
        capabilities=CAPS,
        strict_context_guard=False,
    )
    defaults.update(kw)
    return RotationPolicy(**defaults)


def _next(policy, **kw):
    args = dict(
        agent="caspar",
        failed_lineages=set(),
        run_failed_lineages=set(),
        lineages_in_play=set(),
        used=set(),
        window_rejected={},
        rotations_done=0,
    )
    args.update(kw)
    return policy.next_model(**args)


def test_returns_first_candidate_when_everything_is_free():
    assert _next(_policy()) == GLM


def test_skips_lineage_in_play_by_another_mage():
    assert _next(_policy(), lineages_in_play={"zhipu"}) == OA


def test_skips_lineage_that_already_failed_for_this_mage_accumulated():
    # BDD-22: A failed, rotated to B, B failed -> must NOT return to A.
    assert _next(_policy(), failed_lineages={"zhipu", "openai"}) == MM


def test_skips_lineage_condemned_by_transport_for_the_whole_run():
    # BDD-30: transport failure is global.
    assert _next(_policy(), run_failed_lineages={"zhipu"}) == OA


def test_schema_failure_of_another_mage_does_not_disqualify():
    # BDD-31: schema failures stay per-mage; they are NOT in run_failed_lineages.
    assert _next(_policy(), failed_lineages=set(), run_failed_lineages=set()) == GLM


def test_skips_model_already_used_by_this_mage():
    assert _next(_policy(), used={GLM.model}) == OA


def test_skips_candidate_rejected_by_the_exact_probe():
    # BDD-54: window_rejected feeds back into the next proposal.
    assert _next(_policy(), window_rejected={GLM.model: "too_small"}) == OA


def test_pre_filter_rejects_only_certain_misfits():
    caps = dict(CAPS, **{GLM.model: ModelCapability(window=1_000, supports_completion=True)})
    assert _next(_policy(capabilities=caps, min_window_tokens=REQUIRED_RAW)) == OA


def test_unknown_window_is_eligible_when_not_strict_and_rejected_when_strict():
    caps = dict(CAPS, **{GLM.model: ModelCapability(window=None, supports_completion=True)})
    assert _next(_policy(capabilities=caps)) == GLM
    assert _next(_policy(capabilities=caps, strict_context_guard=True)) == OA


def test_returns_none_when_rotation_cap_is_exhausted():
    # BDD-7
    assert _next(_policy(max_rotations=2), rotations_done=2) is None


def test_max_rotations_zero_disables_rotation_entirely():
    # BDD-8 (kill-switch)
    assert _next(_policy(max_rotations=0)) is None


def test_returns_none_when_no_candidate_qualifies():
    # BDD-9
    assert _next(_policy(), lineages_in_play={"zhipu", "openai", "minimax"}) is None


# ----------------------------------------------------------------------------
# Property-based invariants (hypothesis) over the eligibility state space (R5).
# LINEAGES here is a GENERIC 5-lineage fixture (synthetic "m-<lineage>" ids), not
# an assertion about DEFAULT_FALLBACK -- it only needs 5 distinct lineage strings.
# ----------------------------------------------------------------------------

LINEAGES = ["deepseek", "openai", "minimax", "nvidia", "google"]
SPECS = [ModelSpec(f"m-{lin}", lin) for lin in LINEAGES]
_sets = st.sets(st.sampled_from(LINEAGES))


@given(
    in_play=_sets,
    failed=_sets,
    condemned=_sets,
    used=st.sets(st.sampled_from([s.model for s in SPECS])),
    rotations=st.integers(min_value=0, max_value=4),
    cap=st.integers(min_value=0, max_value=4),
)
def test_property_result_never_violates_any_condition(
    in_play, failed, condemned, used, rotations, cap
):
    caps = {s.model: ModelCapability(window=200_000, supports_completion=True) for s in SPECS}
    policy = RotationPolicy(SPECS, cap, REQUIRED_RAW, caps, strict_context_guard=False)
    got = policy.next_model(
        agent="caspar",
        failed_lineages=failed,
        run_failed_lineages=condemned,
        lineages_in_play=in_play,
        used=used,
        window_rejected={},
        rotations_done=rotations,
    )
    if got is None:
        return
    assert rotations < cap  # never exceeds the cap
    assert got.lineage not in in_play  # never duplicates a live lineage
    assert got.lineage not in failed  # never revisits a failed lineage
    assert got.lineage not in condemned  # never uses a condemned lineage
    assert got.model not in used  # never repeats a model


@given(in_play=_sets, failed=_sets, condemned=_sets)
def test_property_none_iff_no_candidate_is_eligible(in_play, failed, condemned):
    caps = {s.model: ModelCapability(window=200_000, supports_completion=True) for s in SPECS}
    policy = RotationPolicy(SPECS, 5, REQUIRED_RAW, caps, strict_context_guard=False)
    got = policy.next_model(
        agent="caspar",
        failed_lineages=failed,
        run_failed_lineages=condemned,
        lineages_in_play=in_play,
        used=set(),
        window_rejected={},
        rotations_done=0,
    )
    eligible = [s for s in SPECS if s.lineage not in (in_play | failed | condemned)]
    assert (got is None) == (not eligible)  # None iff nothing qualifies


# ----------------------------------------------------------------------------
# Task 4: LineageRegistry -- the single lock and the state machine (BDD-23..55).
# ----------------------------------------------------------------------------

TRIO = {
    "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),
    "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
    "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
}


async def test_claim_next_reserves_the_new_lineage_and_frees_the_old():
    reg = LineageRegistry(TRIO)
    state = AgentRotationState(used={TRIO["caspar"].model})
    got = await reg.claim_next("caspar", _policy(), state)
    assert got == GLM
    assert "deepseek" not in await reg.lineages_in_play(exclude=None)  # old freed
    assert "zhipu" in await reg.lineages_in_play(exclude=None)  # new reserved


async def test_claim_next_returning_none_leaves_the_registry_unchanged():
    # BDD-34: postcondition of the None branch.
    reg = LineageRegistry(TRIO)
    before = await reg.lineages_in_play(exclude=None)
    state = AgentRotationState(rotations_done=99)
    assert await reg.claim_next("caspar", _policy(), state) is None
    assert await reg.lineages_in_play(exclude=None) == before


async def test_two_mages_failing_concurrently_never_claim_the_same_lineage():
    # BDD-23: the TOCTOU the gate found in cycle 1.
    reg = LineageRegistry(TRIO)
    policy = _policy()
    s1, s2 = AgentRotationState(), AgentRotationState()
    got = await asyncio.gather(
        reg.claim_next("melchior", policy, s1),
        reg.claim_next("balthasar", policy, s2),
    )
    assert got[0] is not None and got[1] is not None
    assert got[0].lineage != got[1].lineage


async def test_dead_mage_frees_its_lineage():
    # BDD-24
    reg = LineageRegistry(TRIO)
    await reg.release("caspar")
    assert "deepseek" not in await reg.lineages_in_play(exclude=None)


async def test_successful_mage_retains_its_lineage():
    # BDD-25 / BDD-50: success CONSERVES; only death releases.
    reg = LineageRegistry(TRIO)
    state = AgentRotationState()
    async with reg.agent_slot("caspar", state):
        state.succeeded = True
    assert "deepseek" in await reg.lineages_in_play(exclude=None)


async def test_late_exception_after_success_does_not_leak_the_lineage():
    # [CRITICAL] from cycle 17: succeeded is the SOLE determinant of the exit path.
    reg = LineageRegistry(TRIO)
    state = AgentRotationState()
    with pytest.raises(RuntimeError):
        async with reg.agent_slot("caspar", state):
            state.succeeded = True  # verdict emitted
            raise RuntimeError("telemetry blew up AFTER the verdict")
    assert "deepseek" in await reg.lineages_in_play(exclude=None)  # still in play


async def test_exception_before_success_releases_the_lineage():
    reg = LineageRegistry(TRIO)
    state = AgentRotationState()
    with pytest.raises(RuntimeError):
        async with reg.agent_slot("caspar", state):
            raise RuntimeError("died before any verdict")
    assert "deepseek" not in await reg.lineages_in_play(exclude=None)


async def test_transport_failure_condemns_the_lineage_run_wide():
    # BDD-38 / R5a. connection=False here: this checks run-wide condemnation, not
    # the endpoint-down fast-fail (which is driven by connection-level failures).
    reg = LineageRegistry(TRIO)
    await asyncio.gather(
        reg.register_transport_failure("deepseek", connection=False),
        reg.register_transport_failure("openai", connection=False),
    )
    assert reg.run_failed_lineages == {"deepseek", "openai"}  # no lost update
