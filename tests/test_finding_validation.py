# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for finding_validation.py — diff-grounded finding guard."""

from __future__ import annotations

_DIFF = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -10,3 +10,4 @@ def f():
 ctx
+added1
+added2
 ctx2
"""


class TestParseDiffRanges:
    def test_changed_lines_per_file(self):
        from finding_validation import parse_diff_ranges, valid_files

        ranges = parse_diff_ranges(_DIFF)
        assert valid_files(_DIFF) == {"src/a.py"}
        # added1 at post-image line 11, added2 at 12
        assert 11 in ranges["src/a.py"] and 12 in ranges["src/a.py"]


class TestValidateFindings:
    def _ranges(self):
        from finding_validation import parse_diff_ranges, valid_files

        return valid_files(_DIFF), parse_diff_ranges(_DIFF)

    def test_hard_drop_file_not_in_diff(self):
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "ghost.py", "line": 1}],
            vf,
            rg,
        )
        assert kept == [] and dropped == 1 and annotated == 0

    def test_soft_annotate_line_out_of_range(self):
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "src/a.py", "line": 999}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        assert "outside changed range" in kept[0]["detail"]

    def test_keep_finding_without_file(self):
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        f = {"severity": "info", "title": "design point", "detail": "d", "file": None, "line": None}
        kept, dropped, annotated = validate_findings([f], vf, rg)
        assert kept == [f] and dropped == 0 and annotated == 0

    def test_line_in_range_passes_clean(self):
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "src/a.py", "line": 11}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 0 and kept[0]["detail"] == "d"

    # A3 tests: unique-basename fallback and ambiguous-basename hard-drop

    def test_a3_unique_basename_soft_annotates_not_drops(self):
        """A3: file="a.py" vs diff touching src/a.py (unique basename) -> annotated,
        not dropped. The agent under-qualified the path but the finding is real."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        # _DIFF touches src/a.py; basename "a.py" is unique in the diff.
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "a.py", "line": 11}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        assert "[path unverified]" in kept[0]["detail"]

    def test_a3_no_basename_match_is_hard_dropped(self):
        """A3: file="ghost.py" has no basename match in the diff -> hard-dropped."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "ghost.py", "line": 5}],
            vf,
            rg,
        )
        assert kept == [] and dropped == 1 and annotated == 0

    def test_parse_diff_ranges_ignores_no_newline_marker(self):
        """FIX 1: a backslash-space 'no newline at end of file' marker must not
        advance the post-image line counter — subsequent line numbers must be exact."""
        from finding_validation import parse_diff_ranges

        diff = (
            "diff --git a/z.py b/z.py\n"
            "--- a/z.py\n"
            "+++ b/z.py\n"
            "@@ -1,2 +1,3 @@\n"
            " ctx\n"
            "+added_line\n"
            "\\ No newline at end of file\n"
            "+second_added\n"
        )
        ranges = parse_diff_ranges(diff)
        # added_line is at post-image line 2, second_added at 3.
        # If the marker is mistakenly counted as a context line, second_added
        # would be recorded as line 4 (off-by-one).
        assert ranges == {"z.py": {2, 3}}, (
            f"no-newline marker must not advance the line counter; got {ranges}"
        )

    def test_a3_ambiguous_basename_is_hard_dropped(self):
        """A3 (iter-3): diff with TWO files sharing basename a.py (src/a.py + lib/a.py)
        and a finding file="x/a.py" -> hard-dropped because the basename is not unique
        (too weak a signal to distinguish a real finding from a fabrication)."""
        from finding_validation import parse_diff_ranges, valid_files, validate_findings

        # Build a diff that touches BOTH src/a.py and lib/a.py.
        ambiguous_diff = (
            "diff --git a/src/a.py b/src/a.py\n"
            "--- a/src/a.py\n"
            "+++ b/src/a.py\n"
            "@@ -1,2 +1,3 @@\n"
            " ctx\n"
            "+added\n"
            " ctx2\n"
            "diff --git a/lib/a.py b/lib/a.py\n"
            "--- a/lib/a.py\n"
            "+++ b/lib/a.py\n"
            "@@ -5,2 +5,3 @@\n"
            " x\n"
            "+change\n"
            " y\n"
        )
        vf = valid_files(ambiguous_diff)
        rg = parse_diff_ranges(ambiguous_diff)
        assert "src/a.py" in vf and "lib/a.py" in vf  # sanity

        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "x/a.py", "line": 2}],
            vf,
            rg,
        )
        # Ambiguous basename -> hard-drop (too weak a signal)
        assert kept == [] and dropped == 1 and annotated == 0
