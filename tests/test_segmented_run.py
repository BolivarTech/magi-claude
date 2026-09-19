# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-19
"""Tests for the segmented evaluation of MS4b: the map, the warning, the gate."""

import json
from types import SimpleNamespace

import pytest

from bundle_segmenter import CONSULT_INPUT_CEILING_BYTES, Segment, SegmentPlan

_CEILING = CONSULT_INPUT_CEILING_BYTES


def _plan_with(count, references=()):
    """A SegmentPlan of `count` segments, for the tests that only care how many."""
    return SegmentPlan(
        segments=[
            Segment(
                order=i + 1,
                section_names=[f"s{i}"],
                size_bytes=1000,
                defines=[f"Name{i}"],
                references=list(references) if i else [],
            )
            for i in range(count)
        ],
        is_segmented=count > 1,
        ceiling_bytes=_CEILING,
        headroom=0.65,
    )


def _sections(count):
    return [{"name": f"s{i}", "body": f"body {i}", "defines": [f"Name{i}"]} for i in range(count)]


def _agents():
    """Three minimal seven-key verdicts, enough for format_report to render."""
    return [
        {
            "agent": name,
            "verdict": "approve",
            "confidence": 0.9,
            "summary": "s",
            "reasoning": "r",
            "findings": [],
            "recommendation": "rec",
        }
        for name in ("caspar", "melchior", "balthasar")
    ]


def _report():
    """A real report shape: the existing consensus over the three verdicts."""
    from synthesize import determine_consensus

    agents = _agents()
    return {"agents": agents, "consensus": determine_consensus(agents)}


def _args(**overrides):
    """A parsed-args namespace shaped like run_magi's (fields read by run_segmented)."""
    fields = dict(ollama=True, model=None, mode="design", timeout=None, show_status=False)
    fields.update(overrides)
    return SimpleNamespace(**fields)


# --------------------------------------------------------------------------
# Task 2 -- the map and the warning
# --------------------------------------------------------------------------


def test_the_map_is_written_before_the_first_segment_is_evaluated(monkeypatch, tmp_path):
    # The map is the DECISION, not a record of what happened.
    import run_magi

    order = []
    monkeypatch.setattr(run_magi, "_write_segmentation_map", lambda *a, **k: order.append("map"))

    def _fake_segment(*a, **k):
        order.append("segment")
        return _report()

    monkeypatch.setattr(run_magi, "_evaluate_segment", _fake_segment)
    run_magi.run_segmented(_plan_with(3), _sections(3), _args(), str(tmp_path), "agents", 1800)
    assert order[0] == "map", f"the map must come first, got {order}"
    assert order.count("segment") == 3


def test_the_warning_says_how_many_segments_and_why(capsys):
    from run_magi import announce_segmentation

    announce_segmentation(_plan_with(3), payload_bytes=520_000, ceiling=_CEILING)
    err = capsys.readouterr().err
    assert "3" in err  # how many
    assert "520" in err and "262" in err  # and why: the two numbers compared


# --------------------------------------------------------------------------
# Task 3 -- the gate: every segment must pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdicts", "approved"),
    [
        (["GO (3-0)", "GO (2-1)", "STRONG GO"], True),
        (["GO WITH CAVEATS (3-0)", "GO (2-1)"], True),
        (["GO (3-0)", "GO (3-0)", "HOLD (1-2)"], False),  # one NO CUMPLE sinks the gate
        (["STRONG GO", "HOLD -- TIE"], False),
        (["STRONG GO", "STRONG NO-GO"], False),
        (["GO (3-0)", "INVALID (degraded)"], False),  # a degraded segment approves nothing
        ([], False),  # nothing evaluated approves nothing
    ],
)
def test_the_gate_passes_only_when_EVERY_segment_passes(verdicts, approved):
    from run_magi import segmentation_verdict

    assert segmentation_verdict(verdicts).approved is approved


def test_the_integration_pass_runs_only_when_the_split_cut_through_a_contract():
    from run_magi import needs_integration_pass

    assert needs_integration_pass(_plan_with(3, references=["Name0"])) is True
    assert needs_integration_pass(_plan_with(3)) is False
    assert needs_integration_pass(_plan_with(1, references=["Name0"])) is False  # nothing split


def test_the_input_is_split_into_sections_at_top_level_headings():
    from run_magi import split_sections

    text = (
        "preamble line\n\n# Alpha\n<!-- defines: TrioExecutor, Pin -->\nbody a\n\n"
        "## Beta\nbody b\n### not a section\nmore b\n"
    )
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["(preamble)", "Alpha", "Beta"]
    assert sections[1]["defines"] == ["TrioExecutor", "Pin"]
    assert sections[2]["defines"] == []
    assert "### not a section" in sections[2]["body"]
    assert "".join(s["body"] for s in sections).count("body") == 2


def test_a_segmented_run_evaluates_in_series_and_writes_the_table(monkeypatch, tmp_path, capsys):
    # End to end below the trio: the segments run ONE AT A TIME in plan order,
    # each into its own directory, and the top-level report carries the map,
    # every verdict verbatim and the gate's decision.
    import run_magi

    calls = []

    async def _orch(agents_dir, prompt, output_dir, timeout, **kwargs):
        calls.append((output_dir, prompt))
        return _report()

    monkeypatch.setattr(run_magi, "run_orchestrator", _orch)
    plan = _plan_with(2)
    report = run_magi.run_segmented(plan, _sections(2), _args(), str(tmp_path), "agents", 1800)
    assert [c[0] for c in calls] == [str(tmp_path / "segment-1"), str(tmp_path / "segment-2")]
    assert "body 0" in calls[0][1] and "body 1" not in calls[0][1]
    assert report["segmentation"]["segments"][0]["order"] == 1
    assert [r["consensus"] for r in report["segmentation"]["results"]] == ["STRONG GO", "STRONG GO"]
    assert report["segmentation"]["approved"] is True
    assert json.loads((tmp_path / "magi-report.json").read_text(encoding="utf-8"))["segmentation"]
    out = capsys.readouterr().out
    assert "segment-1" in out and "STRONG GO" in out and "APPROVED" in out
