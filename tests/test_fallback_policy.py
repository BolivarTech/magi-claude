# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-11
"""Tests for the pure fallback rotation policy (Task 3, R5/R5a/R6/R24)."""

from fallback_policy import ModelCapability, RotationPolicy
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
