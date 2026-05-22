# tests/test_review_context.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-22
"""Tests for review_context.py — deterministic code-review enrichment."""

import os
import subprocess
import tempfile

from review_context import enrich_code_review_context, _git_toplevel, _tree_is_clean


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
