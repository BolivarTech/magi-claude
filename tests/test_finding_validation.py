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


class TestF2NonGitDiffParity:
    """F2: the guard parser and the enrichment parser must recognize the same
    touched files, including non-git unified diffs (no ``b/`` prefix) and
    ``diff -u`` headers that append a tab+timestamp. A divergence false-drops a
    finding that cites a real file."""

    # ``diff -u`` style: no ``b/`` prefix, tab + timestamp after the path.
    _NONGIT_DIFF = (
        "--- src/app.py\t2026-05-24 10:00:00.000000000 +0000\n"
        "+++ src/app.py\t2026-05-24 10:05:00.000000000 +0000\n"
        "@@ -1,2 +1,3 @@\n"
        " ctx\n"
        "+added\n"
        " ctx2\n"
    )

    def test_parse_diff_ranges_recognizes_non_git_plus_header(self):
        """A '+++ <path>' header without git's 'b/' prefix must be recognized,
        and a trailing tab+timestamp must be stripped from the path."""
        from finding_validation import parse_diff_ranges, valid_files

        assert valid_files(self._NONGIT_DIFF) == {"src/app.py"}
        # 'added' is the post-image line 2.
        assert 2 in parse_diff_ranges(self._NONGIT_DIFF)["src/app.py"]

    def test_non_git_real_file_finding_is_not_false_dropped(self):
        """Core F2 bug: a finding citing a REAL file from a non-git diff was
        hard-dropped because the guard required 'b/'. It must be kept."""
        from finding_validation import parse_diff_ranges, valid_files, validate_findings

        vf = valid_files(self._NONGIT_DIFF)
        rg = parse_diff_ranges(self._NONGIT_DIFF)
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "src/app.py", "line": 2}],
            vf,
            rg,
        )
        assert dropped == 0 and len(kept) == 1, "real file in a non-git diff must not be dropped"

    def test_guard_and_enrichment_parsers_agree(self):
        """F2 consistency guarantee: review_context._extract_touched_files and
        finding_validation.valid_files must derive the SAME touched-file set, so
        the guard can never hard-drop a file the enrichment grounded on."""
        from finding_id import normalize_path
        from finding_validation import valid_files
        from review_context import _extract_touched_files

        for diff in (_DIFF, self._NONGIT_DIFF):
            enr = {normalize_path(p) for p in _extract_touched_files(diff)}
            assert enr == valid_files(diff), f"parsers disagree on touched files for: {diff!r}"

    def test_added_content_line_resembling_plus_header_not_treated_as_file(self):
        """Robustness: an added line whose content begins with '++ ' renders as a
        raw '+++ ...' diff line. Because it does NOT follow a '--- ' old-file
        header, it must never be misparsed as a new-file header (no phantom file).
        This pins the '--- '/'+++ ' pairing against a naive optional-'b/' regex."""
        from finding_validation import parse_diff_ranges, valid_files

        diff = (
            "--- a/real.py\n"
            "+++ b/real.py\n"
            "@@ -1,2 +1,3 @@\n"
            " ctx\n"
            "+++ phantom\n"  # added line; content is '++ phantom'
            " ctx2\n"
        )
        # Only the real file is recognized; 'phantom' must NOT become a file.
        assert valid_files(diff) == {"real.py"}
        assert "real.py" in parse_diff_ranges(diff)

    def test_deleted_comment_adjacent_to_addition_is_not_phantom_file(self):
        """Adversarial (Caspar): a DELETED '-- ' comment line (rendered '--- ')
        immediately followed by an ADDED '++ ' line (rendered '+++ ') must not be
        misparsed as a phantom file header. Hunk-body line counting keeps both
        inside the open hunk, so only the real file is recognized (no phantom) and
        the '++ new note' line is counted as a normal addition under it."""
        from finding_validation import parse_diff_ranges, valid_files

        diff = (
            "diff --git a/db.sql b/db.sql\n"
            "--- a/db.sql\n"
            "+++ b/db.sql\n"
            "@@ -1,3 +1,3 @@\n"
            " ctx\n"
            "--- old comment\n"  # a deleted SQL comment line: '-- old comment'
            "+++ new note\n"  # an added line whose content is '++ new note'
            " ctx2\n"
        )
        assert valid_files(diff) == {"db.sql"}, "deleted '--'/added '++' must not create a phantom"
        # No phantom key in the ranges either; the '++ new note' line is counted
        # under db.sql (a legitimate addition), never as a separate file.
        assert set(parse_diff_ranges(diff).keys()) == {"db.sql"}

    def test_new_file_with_content_is_recognized(self):
        """A new file WITH content carries '--- /dev/null' / '+++ b/new.py' / '@@'
        (verified against git output). Its header is outside any open hunk, so the
        walker recognizes it — no recall loss for real new files."""
        from finding_validation import parse_diff_ranges, valid_files

        diff = (
            "diff --git a/new.py b/new.py\n"
            "new file mode 100644\n"
            "index 0000000..422c2b7\n"
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+a\n"
            "+b\n"
        )
        assert valid_files(diff) == {"new.py"}
        assert parse_diff_ranges(diff) == {"new.py": {1, 2}}

    def test_empty_new_file_emits_no_header_and_is_not_touched(self):
        """An EMPTY new file emits only 'diff --git' + 'new file mode' (verified:
        git emits NO '---'/'+++' for a 0-byte file), so there is no header to
        recognize — non-recognition is git's doing, and an empty file has no
        citable lines anyway."""
        from finding_validation import valid_files

        diff = "diff --git a/empty.py b/empty.py\nnew file mode 100644\nindex 0000000..e69de29\n"
        assert valid_files(diff) == set()

    def test_deleted_comment_added_line_before_next_hunk_not_phantom(self):
        """W4 (Mel+Caspar): the deleted-'-- '/added-'++ ' adjacency, when it falls
        immediately before the NEXT hunk header, must still not create a phantom
        file. Hunk-body line counting (not a trailing-'@@' peek) immunizes the
        walker: the content pair is inside an open hunk, so it is never a header.
        The '++ new note' line is a legitimate addition and is counted."""
        from finding_validation import parse_diff_ranges, valid_files

        diff = (
            "diff --git a/db.sql b/db.sql\n"
            "--- a/db.sql\n"
            "+++ b/db.sql\n"
            "@@ -1,2 +1,2 @@\n"
            " ctx1\n"
            "--- old comment\n"  # deleted '-- old comment'
            "+++ new note\n"  # added '++ new note' ...
            "@@ -10,2 +10,2 @@\n"  # ... immediately before the next hunk header
            " ctx2\n"
            "+real\n"
        )
        assert valid_files(diff) == {"db.sql"}, "content adjacency before a hunk must not phantom"
        assert set(parse_diff_ranges(diff).keys()) == {"db.sql"}


class TestF3BasenameLineRangeCheck:
    """F3 (Caspar): a unique-basename match identifies the exact diff file, so
    the line-range check must STILL run (instead of being skipped). A fabricated
    line far outside the resolved file's changed range is then surfaced via the
    ``[outside changed range]`` marker, narrowing the fabrication surface while
    keeping the finding (recall preserved). _DIFF touches src/a.py at lines 11,12."""

    def _ranges(self):
        from finding_validation import parse_diff_ranges, valid_files

        return valid_files(_DIFF), parse_diff_ranges(_DIFF)

    def test_basename_match_with_line_outside_range_gets_both_markers(self):
        """file='a.py' (unique basename for src/a.py), line=999 outside {11,12}
        -> kept, annotated, BOTH '[path unverified]' and 'outside changed range'."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "a.py", "line": 999}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        detail = kept[0]["detail"]
        assert "[path unverified]" in detail
        assert "outside changed range" in detail, (
            "unique-basename match must still run the line-range check (F3)"
        )

    def test_basename_match_with_line_inside_range_only_path_unverified(self):
        """file='a.py', line=11 inside {11,12} -> '[path unverified]' only, no
        outside-range marker (the line is valid for the resolved file)."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "a.py", "line": 11}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        detail = kept[0]["detail"]
        assert "[path unverified]" in detail
        assert "outside changed range" not in detail

    def test_basename_match_without_line_only_path_unverified(self):
        """file='a.py', no line -> '[path unverified]' only; nothing to range-check."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [{"severity": "warning", "title": "x", "detail": "d", "file": "a.py"}],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        detail = kept[0]["detail"]
        assert "[path unverified]" in detail
        assert "outside changed range" not in detail

    def test_basename_match_when_finding_path_is_nested(self):
        """A finding with a DIFFERENT directory prefix (file='tests/a.py') whose
        basename uniquely matches the diff file (src/a.py) is still resolved via
        basename on BOTH sides -> '[path unverified]', not hard-dropped."""
        from finding_validation import validate_findings

        vf, rg = self._ranges()
        kept, dropped, annotated = validate_findings(
            [
                {
                    "severity": "warning",
                    "title": "x",
                    "detail": "d",
                    "file": "tests/a.py",
                    "line": 11,
                }
            ],
            vf,
            rg,
        )
        assert dropped == 0 and annotated == 1 and len(kept) == 1
        assert "[path unverified]" in kept[0]["detail"]
        # line 11 is in src/a.py's changed range -> no outside-range marker.
        assert "outside changed range" not in kept[0]["detail"]
