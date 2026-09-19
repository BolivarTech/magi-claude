# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-09-19
"""Tests for BundleSegmenter, the pure planner of MS4b (CLAUDE.local.md §6.2)."""

import pytest

from bundle_segmenter import BundleSegmenter, SegmentTooLargeError

_CEILING = 262_144  # measured in MS0, inclusive
_HEADROOM = 0.65  # §6.2: never fill to 100%


def _section(name, size, defines=()):
    return {"name": name, "body": "x" * size, "defines": list(defines)}


def test_a_bundle_that_fits_is_NOT_segmented():
    # §6.2: a fallback for a context limit, never the default mode.
    plan = BundleSegmenter.plan([_section("all", 1000)], _CEILING)
    assert plan.is_segmented is False
    assert len(plan.segments) == 1


def test_a_bundle_that_fits_the_CEILING_but_not_the_headroom_IS_segmented():
    # 0.80 of the ceiling is under the ceiling and far ABOVE the 0.65 headroom:
    # the system prompt and the model's own reasoning live in the same window.
    over_headroom = int(_CEILING * 0.80)
    plan = BundleSegmenter.plan(
        [_section("a", over_headroom // 2), _section("b", over_headroom // 2)], _CEILING
    )
    assert plan.is_segmented is True


def test_the_ceiling_is_measured_in_BYTES_not_characters():
    # A UTF-8 accent is TWO bytes. per section 45 000 chars = 90 000 bytes (fits);
    # two of them 180 000 bytes > the 170 393 headroom. Counting characters
    # would say "it fits" and truncate silently.
    accented = {"name": "a", "body": "á" * 45_000, "defines": []}
    other = {"name": "b", "body": "á" * 45_000, "defines": []}
    plan = BundleSegmenter.plan([accented, other], _CEILING)
    assert plan.is_segmented is True, "sizes must be counted in bytes, not characters"
    assert len(plan.segments) == 2


def test_a_pure_ascii_bundle_of_the_same_CHARACTER_count_is_NOT_segmented():
    # The control: same characters, half the bytes, fits.
    plan = BundleSegmenter.plan([_section("a", 45_000), _section("b", 45_000)], _CEILING)
    assert plan.is_segmented is False


def test_no_segment_exceeds_the_headroom_fraction():
    sections = [_section(f"s{i}", 40_000) for i in range(12)]
    for segment in BundleSegmenter.plan(sections, _CEILING).segments:
        assert segment.size_bytes <= _CEILING * _HEADROOM


def test_segments_are_cut_at_section_boundaries_never_mid_section():
    sections = [_section(f"s{i}", 40_000) for i in range(12)]
    plan = BundleSegmenter.plan(sections, _CEILING)
    seen = [name for segment in plan.segments for name in segment.section_names]
    assert seen == [f"s{i}" for i in range(12)]  # every section whole, in order


def test_a_shared_interface_lands_in_the_segment_that_DEFINES_it():
    # §6.2.3. Splitting a definition from its consumers is how a segment gets
    # reviewed against an interface the reviewer cannot see.
    sections = [
        _section("consumer", 100_000),
        _section("owner", 100_000, defines=["TrioExecutor"]),
        _section("other_consumer", 100_000),
    ]
    plan = BundleSegmenter.plan(sections, _CEILING)
    owning = [s for s in plan.segments if "owner" in s.section_names][0]
    assert "TrioExecutor" in owning.defines


def test_the_segment_order_is_deterministic_across_runs():
    sections = [_section(f"s{i}", 40_000) for i in range(12)]
    first = BundleSegmenter.plan(sections, _CEILING)
    second = BundleSegmenter.plan(sections, _CEILING)
    assert [s.section_names for s in first.segments] == [s.section_names for s in second.segments]


def test_one_section_larger_than_the_headroom_raises_instead_of_truncating():
    # The case with no good answer; the only honest response is to refuse.
    huge = _section("monolith", int(_CEILING * 0.9))
    with pytest.raises(SegmentTooLargeError) as exc:
        BundleSegmenter.plan([huge], _CEILING)
    assert "monolith" in str(exc.value)


def test_the_error_names_the_section_its_size_and_the_ceiling():
    size = int(_CEILING * 0.9)
    with pytest.raises(SegmentTooLargeError) as exc:
        BundleSegmenter.plan([_section("monolith", size)], _CEILING)
    message = str(exc.value)
    assert "monolith" in message  # which one
    assert str(size) in message  # what it measures
    assert str(int(_CEILING * _HEADROOM)) in message  # what it must fit under


def test_a_bundle_exactly_AT_the_headroom_is_not_segmented():
    # The boundary, inclusive, so the rule has no undefined point.
    exact = int(_CEILING * _HEADROOM)
    plan = BundleSegmenter.plan([_section("a", exact)], _CEILING)
    assert plan.is_segmented is False


def test_an_empty_bundle_is_not_segmented_and_does_not_raise():
    plan = BundleSegmenter.plan([], _CEILING)
    assert plan.is_segmented is False
    assert plan.segments == []


def test_every_section_appears_in_exactly_one_segment():
    # Nothing lost, nothing reviewed twice.
    sections = [_section(f"s{i}", 30_000) for i in range(20)]
    plan = BundleSegmenter.plan(sections, _CEILING)
    seen = [name for segment in plan.segments for name in segment.section_names]
    assert sorted(seen) == sorted(s["name"] for s in sections)
    assert len(seen) == len(set(seen))


def test_the_plan_records_the_map_before_anything_is_evaluated():
    # §6.2.3: the map and the order are registered BEFORE evaluating.
    plan = BundleSegmenter.plan([_section(f"s{i}", 40_000) for i in range(12)], _CEILING)
    assert plan.as_map()["segments"][0]["order"] == 1
    assert all("section_names" in entry for entry in plan.as_map()["segments"])
    assert plan.as_map()["ceiling_bytes"] == _CEILING
    assert plan.as_map()["headroom"] == _HEADROOM


def test_the_overhead_per_section_counts_toward_the_size():
    # The renderer's heading and separator travel too; the caller measures them
    # and passes them in, so the planner stays pure.
    plan = BundleSegmenter.plan([_section("a", 100)], _CEILING, overhead_bytes=50)
    assert plan.segments[0].size_bytes == 150


def test_a_segment_records_the_names_it_references_but_does_not_define():
    # The datum needs_integration_pass reads: a split that cut through a
    # contract has a definer on one side and a reference on the other. The
    # three sections are 100 kB each so each lands in its own segment.
    sections = [
        {
            "name": "owner",
            "body": "x" * 100_000 + " class TrioExecutor",
            "defines": ["TrioExecutor"],
        },
        {"name": "user", "body": "x" * 100_000 + " uses TrioExecutor here", "defines": []},
        {"name": "other", "body": "x" * 100_000, "defines": []},
    ]
    plan = BundleSegmenter.plan(sections, _CEILING)
    by_name = {name: seg for seg in plan.segments for name in seg.section_names}
    assert "TrioExecutor" in by_name["user"].references
    assert "TrioExecutor" not in by_name["owner"].references  # defining is not referencing
