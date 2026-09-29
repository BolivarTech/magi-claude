# Author: Julian Bolivar
# Version: 1.3.0
# Date: 2026-09-29
"""Tests for reading a magi-rs consult envelope (MS3 Task 3c).

The fixtures under ``tests/fixtures/magi-rs-consult/`` derive from ONE real
captured run against the pinned binary (v6.2.0 M1, 0.21.0); the variants were
made by editing that envelope, never by writing a shape from imagination. Their
origin and edits are listed in that directory's README.md.
"""

import json
import os

import pytest

from magi_rs_result import ConsultResult, describe_reasoning
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
    # (the MS0b procedure in
    # planning/_archive/magi-rs-pin-bump-0.19-v6.0.0/magi-rs-pin-bump-v0.19.0.md section 5),
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
            assert set(attempt) == {
                "model",
                "cap",
                "finish",
                "completion_tokens",
                "prompt_tokens",
                "reasoning",
                "control",
            }, "completion record shape drifted: re-verify with a real run, do not regenerate"
            assert attempt["control"] in {"default", "disabled", "enabled"}
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


def test_a_null_report_reads_as_empty_not_the_string_None():
    # Gate, cycle 4: ``str(None)`` would have turned an absent report into the
    # literal text "None" -- a value that LOOKS like content.
    envelope = json.loads(_fixture("three-agents.json"))
    envelope["consult"]["report"] = None
    assert ConsultResult.from_json(json.dumps(envelope)).report == ""


# --------------------------------------------------------------------------
# MS4 Task 2 -- capped_agents (D2): a `finish: length` on the LAST attempt counts, and is shouted
# --------------------------------------------------------------------------


def _result(**telemetry):
    """A ConsultResult carrying only the telemetry a test cares about.

    Every default is the shape a clean run REALLY produces -- read off the
    MS0b fixture, not imagined. Mappings keyed by mage, never flat lists with
    an "agent" field: a helper that invents a shape makes every test on top
    of it pass against the wrong contract.
    """
    fields = dict(
        degraded=False,
        mode="code-review",
        mode_source="explicit",
        extraction_failures={"balthasar": [], "caspar": [], "melchior": []},
        input_size={"estimated_tokens": 0, "exceeded": False, "warn_threshold": 98304},
        report_truncated="none",
        endpoint_divergence=False,
        timeout_below_formula=False,
        failed_agents={},
        rotations=[],
        ran_unmeasured=[],
        completions={},
        pool_eligibility={},
    )
    fields.update(telemetry)
    return ConsultResult(
        agents=[], telemetry=fields, report="", degraded=fields["degraded"], run_id=None
    )


def _result_with_completions(records):
    """A ConsultResult whose only interesting telemetry is `completions`.

    `records` is the measured shape: a mapping of mage -> list of attempts,
    each `{model, cap, finish, completion_tokens, prompt_tokens}`. `None`
    models an older magi-rs that emits no completions at all -- which must
    yield no capped agents rather than raising.
    """
    return _result(completions=records if records is not None else None)


def _attempt(finish, model="m", cap=16384):
    """One completion record in the measured shape (no `agent` key)."""
    return {
        "model": model,
        "cap": cap,
        "finish": finish,
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }


def test_an_agent_whose_last_attempt_stopped_cleanly_is_not_capped():
    assert _result_with_completions({"caspar": [_attempt("stop")]}).capped_agents == []


def test_an_agent_whose_last_attempt_hit_the_cap_is_capped():
    assert _result_with_completions({"caspar": [_attempt("length")]}).capped_agents == ["caspar"]


def test_an_agent_that_recovered_on_retry_is_NOT_capped():
    # The measured case from §2.4: attempt 1 `length`, attempt 2 `stop`. The
    # LAST attempt is what the verdict came from.
    result = _result_with_completions({"caspar": [_attempt("length"), _attempt("stop")]})
    assert result.capped_agents == []


def test_an_agent_that_hit_the_cap_on_its_LAST_attempt_is_capped_despite_an_earlier_stop():
    result = _result_with_completions({"caspar": [_attempt("stop"), _attempt("length")]})
    assert result.capped_agents == ["caspar"]


def test_a_finish_that_is_neither_stop_nor_length_still_counts_as_capped():
    # D2 says `finish != "stop"`, not `== "length"`: magi-core's enum is
    # non-exhaustive, and an unknown reason is exactly where silence is worst.
    result = _result_with_completions({"melchior": [_attempt("content_filter", cap=1)]})
    assert result.capped_agents == ["melchior"]


def test_several_capped_agents_come_back_sorted_by_name():
    result = _result_with_completions(
        {
            "melchior": [_attempt("length", cap=1)],
            "balthasar": [_attempt("stop")],
            "caspar": [_attempt("length", cap=1)],
        }
    )
    assert result.capped_agents == ["caspar", "melchior"]


def test_an_absent_completions_key_yields_no_capped_agents_and_does_not_raise():
    assert _result_with_completions(None).capped_agents == []


def test_a_malformed_completions_record_is_skipped_not_fatal():
    # An empty attempt list, an attempt without `finish`, a null `finish`
    # (MEASURED in MS0c: a timed-out attempt), and a value that is not a list:
    # none may raise, none may count. A null finish is "no output at all".
    result = _result_with_completions(
        {"caspar": [], "melchior": [{"model": "m"}], "balthasar": "?", "x": [_attempt(None)]}
    )
    assert result.capped_agents == []


# --------------------------------------------------------------------------
# MS4 Task 3 -- the warnings: what degrades a gate is shouted (stderr, verbatim)
# --------------------------------------------------------------------------


def _announce(result):
    from run_magi import announce_magi_rs_telemetry

    announce_magi_rs_telemetry(result)


def test_the_capped_warning_names_the_mage_the_model_and_the_cap(capsys):
    _announce(
        _result(
            completions={"caspar": [_attempt("length", model="deepseek-v4-pro:cloud", cap=16384)]}
        )
    )
    err = capsys.readouterr().err
    assert "caspar" in err and "deepseek-v4-pro:cloud" in err and "16384" in err


def test_the_unmeasured_warning_names_every_mage_that_ran_unmeasured(capsys):
    # Measured in §2.4: ran_unmeasured hit ALL THREE main models.
    _announce(_result(ran_unmeasured=["balthasar", "caspar", "melchior"]))
    err = capsys.readouterr().err
    assert all(name in err for name in ("balthasar", "caspar", "melchior"))


def test_the_unmeasured_warning_names_the_remedy_without_opening_the_toml(capsys):
    # D3: half a warning is no warning -- it says what to set and where, and the
    # plugin never reads the file it points at (R7).
    _announce(_result(ran_unmeasured=["caspar"]))
    err = capsys.readouterr().err
    assert "strict_context_guard = true" in err and ".magi/magi.toml" in err


def test_all_three_unmeasured_raises_the_tone_over_one_or_two(capsys):
    # D11: with one or two seats unmeasured the guard still bounded the others;
    # with all three nobody measured anything, and the message says so.
    _announce(_result(ran_unmeasured=["caspar"]))
    partial = capsys.readouterr().err
    _announce(_result(ran_unmeasured=["balthasar", "caspar", "melchior"]))
    total = capsys.readouterr().err
    assert "nobody measured" in total.lower() and "nobody measured" not in partial.lower()


def test_the_extraction_failure_warning_names_the_count(capsys):
    _announce(
        _result(
            extraction_failures={
                "balthasar": [
                    {"attempt": 1, "cause": "invalid-json", "model": "m"},
                    {"attempt": 2, "cause": "unterminated", "model": "m"},
                ],
                "caspar": [],
                "melchior": [],
            }
        )
    )
    assert "2" in capsys.readouterr().err


def test_an_extraction_failure_cause_nobody_has_seen_is_relayed_verbatim(capsys):
    # `cause` is an OPEN set on magi-core's side. Never matched against a list.
    _announce(
        _result(
            extraction_failures={
                "caspar": [{"attempt": 1, "cause": "some-cause-added-upstream", "model": "m"}],
                "balthasar": [],
                "melchior": [],
            }
        )
    )
    assert "some-cause-added-upstream" in capsys.readouterr().err


def test_the_degraded_warning_fires_and_says_the_gate_is_invalid(capsys):
    _announce(_result(degraded=True))
    err = capsys.readouterr().err.lower()
    assert "degraded" in err
    assert "invalid" in err


def test_a_clean_run_emits_no_warning_at_all(capsys):
    _announce(_result())
    assert capsys.readouterr().err == ""


def test_a_malformed_telemetry_value_never_takes_the_announcer_down(capsys):
    # Fail-soft by contract: it runs right before json.dump.
    _announce(_result(ran_unmeasured=None, extraction_failures="?", completions=42))
    capsys.readouterr()


# --------------------------------------------------------------------------
# MS4 Task 4 -- artefacts and the `magi_rs` block of magi-report.json
# --------------------------------------------------------------------------

_CLEAN_REPORT = os.path.join(_FIXTURES, "clean-magi-report.json")
"""A magi-report.json from a healthy Claude-path run (the MS1 SMOKE #2 opus
certification), captured not authored: the test that uses it proves the REAL
shape stays valid without the magi_rs block."""


def _report_without(key):
    """A magi-report payload with one block absent, to prove it is optional."""
    report = json.loads(open(_CLEAN_REPORT, encoding="utf-8").read())
    report.pop(key, None)
    return report


def test_the_raw_output_is_persisted_verbatim_under_its_own_name(tmp_path):
    # R16: two producers, two names.
    from magi_rs_trio import write_raw_artifact

    raw = _fixture("three-agents.json")
    write_raw_artifact(str(tmp_path), raw)
    assert (tmp_path / "magi-rs-raw.json").read_text(encoding="utf-8") == raw
    assert not (tmp_path / "magi-report.json").exists()


def test_a_raw_artifact_that_cannot_be_written_raises_not_swallows(tmp_path):
    from magi_rs_trio import write_raw_artifact

    with pytest.raises(OSError):
        write_raw_artifact(str(tmp_path / "missing" / "dir"), "{}")


def test_the_report_carries_a_magi_rs_block_with_the_telemetry_verbatim():
    from run_magi import build_report_block

    report = build_report_block(_result(rotations=[]))
    assert report["magi_rs"]["rotations"] == []  # the POSITIVE certificate
    assert "capped_agents" in report["magi_rs"]
    assert "run_id" in report["magi_rs"] and report["magi_rs"]["run_id"] is None
    assert set(report["magi_rs"]) >= {
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
    }


def test_the_block_carries_the_tag_each_seat_actually_used_as_model_used():
    # D8: the tag magi-rs USED is telemetry under a name that says so -- never
    # the `served` of the `model` block, which means server-confirmed.
    from run_magi import build_report_block

    result = _result(
        completions={
            "caspar": [
                _attempt("length", model="deepseek-v4-pro:cloud"),
                _attempt("stop", model="glm-5.2:cloud"),
            ],
            "melchior": [_attempt("stop", model="qwen3.5:397b-cloud")],
        }
    )
    assert build_report_block(result)["magi_rs"]["model_used"] == {
        "caspar": "glm-5.2:cloud",
        "melchior": "qwen3.5:397b-cloud",
    }


def test_three_unmeasured_seats_leave_the_marker_in_the_report_and_two_do_not():
    # D11 / BDD-34: who consumes the gate reads the report, not the terminal.
    from run_magi import build_report_block

    every = build_report_block(_result(ran_unmeasured=["balthasar", "caspar", "melchior"]))
    some = build_report_block(_result(ran_unmeasured=["caspar", "melchior"]))
    assert every["magi_rs"]["gate_ran_on_unverified_context"] is True
    assert some["magi_rs"]["gate_ran_on_unverified_context"] is False


def test_a_report_without_the_magi_rs_block_is_still_valid():
    # Additive and fail-soft, exactly like v5.4.0's `model` block.
    from synthesize import format_report

    report = _report_without("magi_rs")
    format_report(report["agents"], report["consensus"])  # raises if it were required


def test_the_open_weight_cost_is_unpriced_never_zero():
    # R20: magi-rs exposes no usage per mage. `null` fails a consumer that sums
    # it LOUDLY; `0.0` sums in silence and lies; an absent key breaks each
    # consumer differently.
    from run_magi import unpriced_cost

    assert unpriced_cost() == {"total_usd": None, "per_agent": {}, "cost_source": "unpriced"}


# Loop 1 (MS4) findings.


def test_a_run_degraded_locally_by_a_broken_verdict_is_shouted_invalid_too(capsys):
    # magi-rs delivered three verdicts (its own degraded: false) but one failed
    # OUR seven-key contract in _validate: the run is degraded on this side,
    # and the "gate is INVALID" warning must fire for that path as well.
    from run_magi import announce_magi_rs_telemetry

    announce_magi_rs_telemetry(_result(degraded=False), degraded_locally=True)
    err = capsys.readouterr().err.lower()
    assert "degraded" in err and "the gate is invalid" in err


def test_the_unmeasured_marker_and_the_tone_share_one_definition(capsys):
    # A duplicated seat name must not make stderr say "nobody measured" while
    # the report's marker stays False: one predicate, both sides.
    from run_magi import announce_magi_rs_telemetry, build_report_block

    dup = _result(ran_unmeasured=["caspar", "caspar", "caspar"])
    announce_magi_rs_telemetry(dup)
    assert "nobody measured" not in capsys.readouterr().err.lower()
    assert build_report_block(dup)["magi_rs"]["gate_ran_on_unverified_context"] is False


def test_the_announcer_survives_telemetry_that_raises_on_access(capsys):
    # The wrapper's own catch branch, executed rather than asserted.
    from run_magi import announce_magi_rs_telemetry

    class _Hostile(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("telemetry exploded")

    hostile = ConsultResult(agents=[], telemetry=_Hostile(), report="", degraded=False, run_id=None)
    announce_magi_rs_telemetry(hostile)
    assert "could not render" in capsys.readouterr().err


def test_the_extraction_failure_count_is_the_number_of_records_not_an_attempt_index(capsys):
    _announce(
        _result(
            extraction_failures={
                "balthasar": [{"attempt": 7, "cause": "invalid-json", "model": "m"}],
                "caspar": [{"attempt": 9, "cause": "unterminated", "model": "m"}],
                "melchior": [],
            }
        )
    )
    assert "2 verdict extraction failure(s)" in capsys.readouterr().err


# --------------------------------------------------------------------------
# v6.2.0 Task 10 -- the reasoning state of a capped seat (R5, R5a; DB-5)
# --------------------------------------------------------------------------


def _attempt_r(finish, reasoning=..., model="m", cap=65536, control="default"):
    """A 0.21.0 completion record; reasoning=... omits the key (pre-0.20 shape)."""
    record = {
        "model": model,
        "cap": cap,
        "finish": finish,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "control": control,
    }
    if reasoning is not ...:
        record["reasoning"] = reasoning
    return record


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        ({"Measured": {"chars": 211124, "text": None}}, "Measured 211124 chars"),
        ("NotMeasured", "NotMeasured"),
        (
            {"Unsupported": {"backend": "openai-compat", "chars": 0, "text": None}},
            "Unsupported (openai-compat, 0 chars)",
        ),
        ({"Unsupported": {}}, "Unsupported"),
        ({"Future": 1}, "unknown (Future)"),
        (42, "unknown (int)"),
        ("Future", "unknown (str)"),
        (None, "unknown (NoneType)"),
    ],
)
def test_the_reasoning_state_is_described_without_values(reasoning, expected):
    # Scenarios 6, 8, 21, 23 and Review Focus 1.
    assert describe_reasoning(_attempt_r("length", reasoning)) == expected


def test_an_absent_reasoning_key_says_not_reported():
    # Scenario 18.
    assert describe_reasoning(_attempt_r("length")) == "not reported"


@pytest.mark.parametrize(
    "reasoning",
    [
        {"Measured": {"chars": 10, "text": "SECRET"}},
        {"Future": {"text": "SECRET", "chars": 3}},
        {"Unsupported": {"backend": "openai-compat", "chars": 0, "text": "SECRET"}},
    ],
)
def test_no_model_text_or_unknown_value_is_ever_printed(reasoning):
    # Scenarios 19, 20, 21 (the JSON is untrusted input).
    described = describe_reasoning(_attempt_r("length", reasoning))
    assert "SECRET" not in described
    if "Future" in reasoning:
        assert "3" not in described


@pytest.mark.parametrize(
    "reasoning",
    [
        {"Fu\u001b[31mture": 1},
        {"Future\u001b[31m": 1},
        {"Unsupported": {"backend": "x\u001b]0;pwn\u0007", "chars": 1}},
        {"Unsupported": {"backend": "openai-compat\u0007", "chars": 1}},
    ],
)
def test_a_name_that_is_not_a_printable_identifier_is_replaced(reasoning):
    # Scenario 22: the JSON is untrusted; escape sequences never reach stderr.
    described = describe_reasoning(_attempt_r("length", reasoning))
    assert "unprintable" in described
    assert "\u001b" not in described and "\u0007" not in described


def test_the_capped_warning_carries_the_reasoning_state(capsys):
    # Scenario 6: the diagnosis E-E asked for.
    _announce(
        _result(
            completions={
                "caspar": [
                    _attempt_r(
                        "length",
                        {"Measured": {"chars": 211124, "text": None}},
                        model="glm-5.3:cloud",
                    )
                ]
            }
        )
    )
    err = capsys.readouterr().err
    assert "caspar" in err and "glm-5.3:cloud" in err and "Measured 211124 chars" in err


def test_a_timed_out_last_attempt_is_not_capped_with_the_real_shape():
    # Scenario 7: finish null + NotMeasured is the clock, not the cap.
    # Guard test: green in Red on purpose.
    result = _result(completions={"melchior": [_attempt_r(None, "NotMeasured")]})
    assert result.capped_agents == []


def test_the_reasoning_fields_travel_verbatim_into_the_report():
    # Scenario 9 / R5a. Guard test: green in Red on purpose.
    from run_magi import build_report_block

    record = _attempt_r("stop", {"Measured": {"chars": 5, "text": None}}, control="enabled")
    block = build_report_block(_result(completions={"caspar": [record]}))
    assert block["magi_rs"]["completions"]["caspar"][0] == record


def test_the_capped_warning_prints_on_a_cp1252_console(monkeypatch):
    # Review Focus 4. Guard test: green in Red on purpose.
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    from run_magi import announce_magi_rs_telemetry

    announce_magi_rs_telemetry(
        _result(completions={"caspar": [_attempt_r("length", {"Fu\u001b[31mture": 1})]}),
        stream,
    )
    stream.flush()  # a UnicodeEncodeError above would already have failed the test
    assert b"caspar" in stream.buffer.getvalue()
