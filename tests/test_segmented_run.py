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
        "## Beta\nbody b\n#### not a section\nmore b\n"
    )
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["(preamble)", "Alpha", "Beta"]
    assert sections[1]["defines"] == ["TrioExecutor", "Pin"]
    assert sections[2]["defines"] == []
    assert "#### not a section" in sections[2]["body"]  # depth 4 stays inside (1-3 open)
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


def test_a_heading_inside_a_code_fence_does_not_open_a_section():
    # SMOKE (Task 4) found it: every module starts with `# Author:` at column
    # zero, and a bundle of fenced sources split into 257 "sections", cutting
    # fences in half. A fence is opaque to the section scanner.
    from run_magi import split_sections

    text = "## file.py\n\n```python\n# Author: someone\n# Version: 1\ncode\n```\n\n## other.py\n\nmore\n"
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["file.py", "other.py"]
    assert "# Author: someone" in sections[0]["body"]


# Loop 1 (MS4b) findings.


def test_a_third_level_heading_opens_a_section_because_the_enrichment_uses_it():
    # review_context.py nests every touched file at `### path` under two `##`
    # wrappers: with `##` only, a real code-review bundle was at most three
    # sections, each far larger than the headroom -> refused, never split.
    from run_magi import split_sections

    text = "## Touched files (full content)\n\n### a.py\n```\ncode a\n```\n### b.py\n```\ncode b\n```\n"
    names = [s["name"] for s in split_sections(text)]
    assert names == ["Touched files (full content)", "a.py", "b.py"]


def test_a_raw_diff_is_split_at_file_boundaries():
    # The code-review payload before enrichment is a git diff: no headings at
    # all. Each `diff --git` line opens a section named after the file.
    from run_magi import split_sections

    text = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-# old\n+# new\n"
        "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n+line\n"
    )
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["diff: x.py", "diff: y.py"]
    assert "".join(s["body"] for s in sections) == text


def test_a_diff_inside_a_diff_fence_is_split_per_file_with_the_fence_reopened():
    # The project's own gate bundles embed `git diff` in a ```diff fence. A
    # fence is opaque to headings, but a per-file split INSIDE it is safe when
    # each piece closes and reopens the fence, so every segment still parses.
    from run_magi import split_sections

    text = (
        "## The diff\n\n```diff\ndiff --git a/x.py b/x.py\n+x\ndiff --git a/y.py b/y.py\n+y\n```\n"
        "## After\n\ntext\n"
    )
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["The diff", "diff: x.py", "diff: y.py", "After"]
    assert sections[1]["body"].startswith("```diff\n")
    assert sections[1]["body"].rstrip().endswith("```")
    assert sections[2]["body"].startswith("```diff\ndiff --git a/y.py")
    assert sections[0]["body"] == "## The diff\n\n"


def test_duplicate_section_names_are_made_unique_instead_of_collapsing():
    from run_magi import split_sections

    text = "## same\n\na\n\n## same\n\nb\n"
    sections = split_sections(text)
    assert [s["name"] for s in sections] == ["same", "same (2)"]
    assert "b" in sections[1]["body"]


def test_the_defines_line_is_honoured_only_right_under_the_heading():
    from run_magi import split_sections

    text = "## s\nbody\n<!-- defines: Late -->\n\n## t\n<!-- defines: Early -->\nbody\n"
    sections = split_sections(text)
    assert sections[0]["defines"] == []
    assert sections[1]["defines"] == ["Early"]


def test_a_reference_matches_whole_words_only():
    # `Pin` must not match `Pinout` or `opinion`: a false reference buys a
    # real, paid integration run for nothing.
    from bundle_segmenter import BundleSegmenter

    sections = [
        {"name": "owner", "body": "x" * 100_000 + " class Pin", "defines": ["Pin"]},
        {"name": "near", "body": "x" * 100_000 + " opinion Pinout", "defines": []},
        {"name": "user", "body": "x" * 100_000 + " uses Pin here", "defines": []},
    ]
    plan = BundleSegmenter.plan(sections, _CEILING)
    by_name = {name: seg for seg in plan.segments for name in seg.section_names}
    assert by_name["near"].references == []
    assert by_name["user"].references == ["Pin"]


def test_a_segment_whose_run_raises_is_tabled_and_the_rest_still_run(monkeypatch, tmp_path):
    # §6.2.5: one failure does not abort the run -- it is tabled as it came
    # back and the next segment runs, so the report is complete, not partial.
    import run_magi
    from magi_rs_trio import MagiRsProcessError

    seen = []

    def _segment(order, *a, **k):
        seen.append(order)
        if order == 2:
            raise MagiRsProcessError("magi-rs exited 2: boom")
        return _report()

    monkeypatch.setattr(run_magi, "_evaluate_segment", _segment)
    report = run_magi.run_segmented(_plan_with(3), _sections(3), _args(), str(tmp_path), "a", 1)
    assert seen == [1, 2, 3]
    results = report["segmentation"]["results"]
    assert results[1]["consensus"].startswith("FAILED") and "boom" in results[1]["consensus"]
    assert report["segmentation"]["approved"] is False
    assert results[1]["consensus"] in report["segmentation"]["failing"]


def test_a_fail_closed_error_in_a_segment_aborts_the_whole_run(monkeypatch, tmp_path):
    # A digest that does not verify fails identically for every later segment:
    # this one is a genuine whole-run abort, not a segment's verdict.
    import run_magi
    from magi_rs_bootstrap import DigestMismatchError

    def _segment(order, *a, **k):
        raise DigestMismatchError("digest differs")

    monkeypatch.setattr(run_magi, "_evaluate_segment", _segment)
    with pytest.raises(DigestMismatchError):
        run_magi.run_segmented(_plan_with(2), _sections(2), _args(), str(tmp_path), "a", 1)


def test_the_integration_segment_is_planned_against_the_ceiling_too(monkeypatch, tmp_path):
    # The extra segment gets the same size guarantee as every other one: too
    # large to fit, it is tabled as not run and sinks the gate -- never sent
    # to be rejected by the transport.
    import run_magi

    big_line = "uses Name0 " + "y" * 200_000
    sections = [
        {"name": "s0", "body": "class Name0", "defines": ["Name0"]},
        {"name": "s1", "body": big_line, "defines": []},
    ]
    plan = _plan_with(2, references=["Name0"])
    monkeypatch.setattr(run_magi, "_evaluate_segment", lambda *a, **k: _report())
    report = run_magi.run_segmented(plan, sections, _args(), str(tmp_path), "a", 1)
    results = report["segmentation"]["results"]
    assert results[-1]["section_names"] == ["(integration)"]
    assert results[-1]["consensus"].startswith("NOT RUN")
    assert report["segmentation"]["approved"] is False


def test_the_map_records_both_ceilings_and_the_wrapper(monkeypatch, tmp_path):
    # Two numbers both called "the ceiling" confused a reader: the map names
    # the transport ceiling, the wrapper charged against it, and what is left.
    import run_magi

    monkeypatch.setattr(run_magi, "_evaluate_segment", lambda *a, **k: _report())
    report = run_magi.run_segmented(
        _plan_with(2), _sections(2), _args(), str(tmp_path), "a", 1, wrapper_bytes=127
    )
    seg = report["segmentation"]
    assert seg["transport_ceiling_bytes"] == _CEILING and seg["wrapper_bytes"] == 127
    assert seg["ceiling_bytes"] == _CEILING  # what the plan itself was computed against


def test_a_segment_prompt_over_the_transport_ceiling_is_never_sent(monkeypatch, tmp_path):
    # Gate (cycle 1): the arithmetic says a planned segment plus the wrapper
    # always fits; the guarantee is EXECUTED anyway -- the real prompt is
    # measured right before the launch and an oversized one is refused, tabled
    # as NOT RUN, and the next segment still runs.
    import run_magi

    launched = []

    async def _orch(agents_dir, prompt, output_dir, timeout, **kwargs):
        launched.append(len(prompt))
        return _report()

    monkeypatch.setattr(run_magi, "run_orchestrator", _orch)
    monkeypatch.setattr(run_magi, "_select_executor", lambda args, timeout: None)
    huge = "x" * (run_magi.CONSULT_INPUT_CEILING_BYTES + 1)
    sections = [
        {"name": "s0", "body": huge, "defines": []},
        {"name": "s1", "body": "ok", "defines": []},
    ]
    plan = _plan_with(2)
    report = run_magi.run_segmented(plan, sections, _args(), str(tmp_path), "a", 1)
    results = report["segmentation"]["results"]
    assert results[0]["consensus"].startswith("NOT RUN") and "ceiling" in results[0]["consensus"]
    assert results[1]["consensus"] == "STRONG GO"
    assert len(launched) == 1  # the oversized one never reached the transport
    assert report["segmentation"]["approved"] is False
