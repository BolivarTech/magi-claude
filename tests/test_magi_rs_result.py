# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-18
"""Tests for reading a magi-rs consult envelope (MS3 Task 3c).

The fixtures under ``tests/fixtures/magi-rs-consult/`` derive from ONE real
captured run against the pinned binary (MS0c, 0.19.1); the variants were made
by editing that envelope, never by writing a shape from imagination.
"""

import json
import os

import pytest

from magi_rs_result import ConsultResult
from magi_rs_trio import MagiRsContractError

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "magi-rs-consult")


def _fixture(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as handle:
        return handle.read()


def test_the_three_verdicts_are_read_from_inside_the_consult_object():
    # MS0 corrected this: the JSON is NESTED. agents lives at
    # data["consult"]["agents"], never at the top level.
    result = ConsultResult.from_json(_fixture("three-agents.json"))
    assert len(result.agents) == 3
    assert {a["agent"] for a in result.agents} == {"melchior", "balthasar", "caspar"}


def test_each_verdict_carries_the_seven_contract_keys_with_no_translation():
    # The whole point of the milestone: magi-rs emits exactly this project's
    # contract, field for field. An adapter here would be a SECOND place the
    # contract lives, which is how two definitions diverge.
    for verdict in ConsultResult.from_json(_fixture("three-agents.json")).agents:
        assert set(verdict) >= {
            "agent",
            "verdict",
            "confidence",
            "summary",
            "reasoning",
            "findings",
            "recommendation",
        }


def test_the_captured_verdicts_are_accepted_by_the_existing_validator(tmp_path):
    # The strongest test in the milestone, and the one that would catch a
    # contract drift: it does not check field NAMES, it runs the real
    # load_agent_output over a real captured envelope. The validator reads a
    # FILE (measured signature), exactly as the Claude path feeds it.
    from synthesize import load_agent_output

    for verdict in ConsultResult.from_json(_fixture("three-agents.json")).agents:
        path = tmp_path / f"{verdict['agent']}.json"
        path.write_text(json.dumps(verdict), encoding="utf-8")
        load_agent_output(str(path))  # raises if the contract broke


def test_a_two_agent_run_is_read_and_flagged_degraded():
    result = ConsultResult.from_json(_fixture("two-agents-degraded.json"))
    assert len(result.agents) == 2
    assert result.degraded is True


def test_a_null_consult_raises_relaying_stop_reason_and_error():
    # MS0 corrected R17: with fewer than 2 mages, `consult` comes back NULL --
    # no agents, no consensus, no telemetry. There is nothing to report but the
    # failure, so the plugin must not pretend to build a degraded report.
    with pytest.raises(MagiRsContractError) as exc:
        ConsultResult.from_json(_fixture("consult-null.json"))
    message = str(exc.value)
    assert "stop_reason" in message or "error" in message


def test_json_without_the_agents_key_names_the_forgotten_flag():
    # The real failure mode of this path is forgetting --structured-verdicts.
    # "a key is missing" would be true and useless.
    with pytest.raises(MagiRsContractError) as exc:
        ConsultResult.from_json(_fixture("no-agents.json"))
    assert "--structured-verdicts" in str(exc.value)


def test_output_that_is_not_json_says_so_and_never_scrapes_the_report():
    with pytest.raises(MagiRsContractError) as exc:
        ConsultResult.from_json("## Report\n\nApproved by all three.")
    assert "json" in str(exc.value).lower()


def test_the_eleven_telemetry_fields_are_carried_verbatim():
    # Persisted, never recomputed: `rotations` replaces the rotation telemetry
    # of the v5 plan's MS1 -- a different milestone from this project's MS1 --
    # and extraction_failures is the upstream equivalent of the v5 plan's
    # marker-adherence counter.
    telemetry = ConsultResult.from_json(_fixture("three-agents.json")).telemetry
    for field in (
        "degraded",
        "mode",
        "mode_source",
        "extraction_failures",
        "input_size",
        "report_truncated",
        "endpoint_divergence",
        "timeout_below_formula",
        "failed_agents",
        "rotations",
        "ran_unmeasured",
    ):
        assert field in telemetry


def test_empty_telemetry_values_are_preserved_and_not_dropped():
    # "Always present" is an explicit contract guarantee: an empty
    # extraction_failures list is the POSITIVE certificate that nothing failed
    # to extract, not silence about whether anything did. Dropping falsy values
    # would erase exactly that. The captured run rotated once, so `rotations`
    # is asserted verbatim rather than empty -- the fixture is real, not made.
    raw = json.loads(_fixture("three-agents.json"))
    telemetry = ConsultResult.from_json(_fixture("three-agents.json")).telemetry
    assert telemetry["rotations"] == raw["consult"]["rotations"]
    assert telemetry["failed_agents"] == {}
    assert telemetry["extraction_failures"]["melchior"] == []


def test_the_completions_records_are_carried_with_their_finish_reason():
    # `completions` is a mapping mage -> list of attempts (measured in MS0c).
    completions = ConsultResult.from_json(_fixture("three-agents.json")).telemetry["completions"]
    assert set(completions) == {"melchior", "balthasar", "caspar"}
    for attempts in completions.values():
        assert all({"model", "cap", "finish"} <= set(attempt) for attempt in attempts)


def test_reading_the_result_never_mutates_the_parsed_payload():
    raw = _fixture("three-agents.json")
    result = ConsultResult.from_json(raw)
    _ = result.agents, result.telemetry, result.degraded
    assert ConsultResult.from_json(raw).agents == result.agents


def test_the_captured_envelope_shape_is_exactly_the_pinned_contract():
    # R2 of the spec-base: "at every pin move the WHOLE shape of the JSON is
    # re-verified, and it stays as a TEST, not a procedure". Exact key sets,
    # never `>=`: an added key is as much a contract change as a missing one
    # (MS0b caught `+run_id` and `majority_summary` -> `dissent` this way).
    #
    # IF THIS GOES RED AFTER A PIN MOVE: re-verify the contract with a REAL run
    # (the MS0b procedure in planning/magi-rs-pin-bump-v0.19.0.md section 5),
    # understand what changed and why, update the spec -- and only THEN
    # regenerate the fixture. Regenerating first blesses the drift this test
    # exists to catch.
    raw = json.loads(_fixture("three-agents.json"))
    consult = raw["consult"]
    assert set(raw) == {
        "schema_version",
        "provider",
        "model",
        "response",
        "consult",
        "error",
        "stop_reason",
        "usage",
        "timings",
        "tool_calls",
        "transcript",
        "applied_caps",
        "run_id",
    }, "top-level shape drifted: re-verify with a real run, do not regenerate"
    assert set(consult) == {
        "agents",
        "consensus",
        "report",
        "degraded",
        "mode",
        "mode_source",
        "extraction_failures",
        "input_size",
        "report_truncated",
        "endpoint_divergence",
        "timeout_below_formula",
        "failed_agents",
        "rotations",
        "ran_unmeasured",
        "completions",
        "pool_eligibility",
    }, "consult shape drifted: re-verify with a real run, do not regenerate"
    assert set(consult["consensus"]) == {
        "consensus",
        "consensus_verdict",
        "confidence",
        "score",
        "agent_count",
        "votes",
        "dissent",
    }, "consensus shape drifted: re-verify with a real run, do not regenerate"
    for verdict in consult["agents"]:
        assert set(verdict) == {
            "agent",
            "verdict",
            "confidence",
            "summary",
            "reasoning",
            "findings",
            "recommendation",
        }
        for finding in verdict["findings"]:
            assert set(finding) == {"severity", "title", "detail", "file", "line", "category"}
    for attempts in consult["completions"].values():
        for attempt in attempts:
            assert set(attempt) == {"model", "cap", "finish", "completion_tokens", "prompt_tokens"}
    for failures in consult["extraction_failures"].values():
        for failure in failures:
            assert set(failure) == {"attempt", "cause", "model"}


def test_the_run_id_is_read_from_the_top_level_and_absent_means_none():
    # 0.18.0 added `run_id` at the TOP level (not inside `consult`): it is the
    # pointer into magi-rs's own daily log, where the per-mage ceiling now
    # lives instead of stderr. Telemetry: fail-soft, never a contract error.
    result = ConsultResult.from_json(_fixture("three-agents.json"))
    assert result.run_id == json.loads(_fixture("three-agents.json"))["run_id"]
    without = json.loads(_fixture("three-agents.json"))
    del without["run_id"]
    assert ConsultResult.from_json(json.dumps(without)).run_id is None


def test_a_contract_failure_is_deterministic_and_not_retryable():
    # D6: a forgotten --structured-verdicts fails identically every time, so
    # the orchestrator's `except (ValidationError, ...)` must not consume it.
    from synthesize import ValidationError

    assert not issubclass(MagiRsContractError, ValidationError)
