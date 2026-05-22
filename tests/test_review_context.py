# tests/test_review_context.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-22
"""Tests for review_context.py — deterministic code-review enrichment."""

import os
import subprocess
import tempfile

from review_context import enrich_code_review_context, _git_toplevel, _tree_is_clean
from review_context import _contains_diff, _extract_touched_files, _read_file_safe
from review_context import _git_diff
from review_context import _candidate_identifiers, _grep_defs, _MAX_CANDIDATES
from review_context import _assemble


def _init_repo(repo: str) -> None:
    def run(*a):
        subprocess.run(
            ["git", "-C", repo, *a],
            check=True,  # noqa: E704
            capture_output=True,
            text=True,
        )

    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    run("checkout", "-q", "-b", "main")
    with open(os.path.join(repo, "pkg.py"), "w", encoding="utf-8") as f:
        f.write("def base():\n    return 0\n")
    run("add", "-A")
    run("commit", "-q", "-m", "base")
    run("checkout", "-q", "-b", "feat")
    with open(os.path.join(repo, "pkg.py"), "w", encoding="utf-8") as f:
        f.write("def base():\n    return 0\n\n\ndef added():\n    return base() + helper()\n")
    with open(os.path.join(repo, "helpers.py"), "w", encoding="utf-8") as f:
        f.write("def helper():\n    return 1\n")
    run("add", "-A")
    run("commit", "-q", "-m", "feat")


_SAMPLE_DIFF = (
    "diff --git a/pkg.py b/pkg.py\n"
    "--- a/pkg.py\n"
    "+++ b/pkg.py\n"
    "@@ -1,1 +1,5 @@\n"
    " def base():\n"
    "+    return base() + helper()\n"
    "diff --git a/old.py b/old.py\n"
    "--- a/old.py\n"
    "+++ /dev/null\n"
)


class TestDiffAndRead:
    def test_contains_diff(self):
        assert _contains_diff(_SAMPLE_DIFF) is True
        assert _contains_diff("prose only") is False

    def test_extract_touched_skips_devnull_strips_prefix(self):
        assert _extract_touched_files(_SAMPLE_DIFF) == ["pkg.py"]

    def test_read_file_safe_reads_worktree_and_memoizes(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            cache: dict = {}
            content = _read_file_safe(repo, "pkg.py", cache)
            assert content is not None and "def added():" in content
            assert "pkg.py" in cache

    def test_read_file_safe_missing_and_binary(self):
        with tempfile.TemporaryDirectory() as repo:
            assert _read_file_safe(repo, "nope.py", {}) is None
            with open(os.path.join(repo, "b.bin"), "wb") as f:
                f.write(b"\x00\x01")
            assert _read_file_safe(repo, "b.bin", {}) is None

    def test_read_file_safe_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            assert _read_file_safe(repo, "../../etc/passwd", {}) is None
            assert _read_file_safe(repo, "../outside.py", {}) is None

    def test_read_file_safe_skips_oversized(self):
        from review_context import _MAX_FILE_BYTES

        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            with open(os.path.join(repo, "big.py"), "w", encoding="utf-8") as f:
                f.write("x" * (_MAX_FILE_BYTES + 1))
            assert _read_file_safe(repo, "big.py", {}) is None


class TestScaffold:
    def test_non_repo_is_noop(self):
        with tempfile.TemporaryDirectory() as not_repo:
            content, note = enrich_code_review_context("Review this.", repo_root=not_repo)
        assert content == "Review this." and "skip" in note.lower()

    def test_git_toplevel_none_outside_repo(self):
        with tempfile.TemporaryDirectory() as not_repo:
            assert _git_toplevel(not_repo) is None

    def test_tree_is_clean_true_on_committed_repo(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            assert _tree_is_clean(repo) is True

    def test_untracked_file_still_counts_as_clean(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            with open(os.path.join(repo, "untracked.md"), "w", encoding="utf-8") as f:
                f.write("a review bundle\n")
            assert _tree_is_clean(repo) is True  # untracked ignored

    def test_modified_tracked_file_is_noop(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            with open(os.path.join(repo, "pkg.py"), "a", encoding="utf-8") as f:
                f.write("\n# uncommitted edit\n")
            assert _tree_is_clean(repo) is False
            content, note = enrich_code_review_context("Review.", repo_root=repo, base_ref="main")
            assert content == "Review." and "clean" in note.lower()


class TestTouchedFiles:
    def test_input_diff_includes_worktree_content(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            content, note = enrich_code_review_context(_SAMPLE_DIFF, repo_root=repo)
            assert "## Touched files (full content)" in content
            assert "def added():" in content
            assert content.startswith(_SAMPLE_DIFF)
            assert "1 file" in note

    def test_missing_touched_file_skipped(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            content, _ = enrich_code_review_context(
                _SAMPLE_DIFF.replace("pkg.py", "ghost.py"), repo_root=repo
            )
            assert "## Touched files" not in content

    def test_diff_head_mismatch_file_skipped(self):
        # An added line that does NOT exist in the working-tree file → mismatch → skip.
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            bad = (
                "diff --git a/pkg.py b/pkg.py\n"
                "--- a/pkg.py\n"
                "+++ b/pkg.py\n"
                "@@ -1,1 +1,2 @@\n"
                " def base():\n"
                "+    return NONEXISTENT_TOKEN_XYZ()\n"
            )
            content, note = enrich_code_review_context(bad, repo_root=repo)
            assert "## Touched files" not in content  # pkg.py skipped (mismatch)
            assert "mismatch" in note.lower()

    def test_duplicate_touched_paths_deduped(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            dup = _SAMPLE_DIFF + (
                "diff --git a/pkg.py b/pkg.py\n"
                "--- a/pkg.py\n"
                "+++ b/pkg.py\n"
                "@@ -1,1 +1,1 @@\n"
                " def base():\n"
            )
            content, note = enrich_code_review_context(dup, repo_root=repo)
            assert content.count("### pkg.py\n") == 1  # not duplicated


class TestAutoCompute:
    def test_no_input_diff_autocomputes(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            content, _ = enrich_code_review_context(
                "Review the branch.", repo_root=repo, base_ref="main"
            )
            assert "## Touched files (full content)" in content
            assert "def added():" in content

    def test_bad_base_returns_none(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            assert _git_diff(repo, "no-such-ref") is None


class TestSymbols:
    def test_candidates_strip_keywords_defined_noise_comments_strings(self):
        diff = (
            "+def added():\n"
            "+    return helper(x)  # call something noise\n"
            '+    msg = "ignore zzz inside string"\n'
            "+# pure comment qqq\n"
        )
        ids = _candidate_identifiers(diff, defined={"added"})
        assert "helper" in ids and "x" in ids and "msg" in ids
        assert "added" not in ids and "return" not in ids
        assert "zzz" not in ids  # inside a string literal
        assert "qqq" not in ids  # pure-comment line
        assert "self" not in ids  # _EXTRA_EXCLUDE

    def test_candidate_cap(self):
        diff = "\n".join(f"+    v{i} = f{i}()" for i in range(200))
        assert len(_candidate_identifiers(diff, set())) <= _MAX_CANDIDATES

    def test_grep_single_batched_and_cross_file(self, monkeypatch):
        calls = {"n": 0}
        import review_context

        real = review_context._git

        def counting(repo, *args):
            if args and args[0] == "grep":
                calls["n"] += 1
            return real(repo, *args)

        monkeypatch.setattr(review_context, "_git", counting)
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            defs = _grep_defs(repo, ["helper", "base"], {})
            assert calls["n"] == 1  # ONE batched grep, not per-name
            assert any(p == "helpers.py" and "def helper():" in ex for p, _l, ex in defs)

    def test_enrich_appends_symbol_defs_section(self):
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            content, note = enrich_code_review_context(_SAMPLE_DIFF, repo_root=repo)
            assert "## Referenced symbol definitions" in content
            assert (
                "def helper():" in content
            )  # helper() referenced in the added line, defined cross-file
            assert "def(s)" in note


class TestBudgetAndFailSafe:
    def test_hard_cap_holds_and_defs_drop_first(self):
        base = "DIFF"
        touched = [("a.py", "x" * 60), ("big.py", "y" * 400)]
        defs = [("d.py", 1, "z" * 90)]
        content, note = _assemble(base, touched, defs, max_chars=len(base) + 160)
        assert len(content) <= len(base) + 160  # HARD cap holds
        assert content.startswith(base)  # input kept
        assert "a.py" in content  # smallest file kept
        assert "big.py" not in content  # largest dropped
        assert "z" * 90 not in content  # defs dropped first
        assert "omitted" in note.lower()

    def test_input_over_cap_kept_with_note(self):
        big = "Q" * 500
        content, note = _assemble(big, [("a.py", "aaa")], [], max_chars=100)
        assert content == big  # input always kept (documented carve-out)
        assert "skip" in note.lower() or "omitted" in note.lower()

    def test_within_budget_keeps_all(self):
        content, _ = _assemble("D", [("a.py", "aaa")], [("d.py", 1, "ddd")], 10_000)
        assert "a.py" in content and "ddd" in content

    def test_outer_failsafe_trips(self, monkeypatch):
        import review_context

        monkeypatch.setattr(
            review_context, "_assemble", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with tempfile.TemporaryDirectory() as repo:
            _init_repo(repo)
            content, note = enrich_code_review_context(_SAMPLE_DIFF, repo_root=repo)
        assert content == _SAMPLE_DIFF and "error" in note.lower()
