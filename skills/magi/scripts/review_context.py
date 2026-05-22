# skills/magi/scripts/review_context.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-22
"""Deterministic, bounded, fail-safe review-context enrichment for MAGI
code-review mode. Runs only when the working tree is clean (== HEAD), so all
reads come from one coherent source. Never raises into the orchestrator (R7)."""

from __future__ import annotations

import keyword  # noqa: F401
import os
import re
import subprocess

_ENRICH_MAX_CHARS = 512_000
_DEF_WINDOW_LINES = 40
_MAX_CANDIDATES = 60
_MAX_DEFS = 40
_GIT_TIMEOUT = 30
_MAX_FILE_BYTES = 262_144
_DIFF_MARKERS = ("diff --git ", "--- a/", "+++ b/")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEF_RE = re.compile(r"^[\t ]*(?:def|class)[\t ]+([A-Za-z_][A-Za-z0-9_]*)")
_STRING_RE = re.compile(r"""(['"]).*?\1""")
_EXTRA_EXCLUDE = frozenset(
    {
        "self",
        "cls",
        "True",
        "False",
        "None",
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "set",
        "tuple",
    }
)


def _contains_diff(text: str) -> bool:
    """Return True if text looks like a unified diff.

    Args:
        text: The text to inspect.

    Returns:
        True if any of the canonical diff markers are present.
    """
    return any(marker in text for marker in _DIFF_MARKERS)


def _extract_touched_files(diff_text: str) -> list[str]:
    """Return the list of paths modified by diff_text (new-file side only).

    Skips /dev/null targets (deleted files) and strips the ``b/`` prefix
    that git unified diffs add.

    Args:
        diff_text: A unified diff string (git format).

    Returns:
        Ordered list of relative file paths that were added or modified.
    """
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                files.append(path)
    return files


def _read_file_safe(repo_root: str, rel_path: str, cache: "dict[str, str | None]") -> "str | None":
    """Read a working-tree file (UTF-8 with replace). Return None if the file
    is missing, binary (contains NUL), oversized, or outside the repo root.
    Results are memoized in *cache*.

    Path-traversal containment guard: resolves ``os.path.realpath`` and
    requires the result is inside *repo_root*. Skips files larger than
    ``_MAX_FILE_BYTES`` without reading them into memory.

    Args:
        repo_root: Absolute path to the git repository root.
        rel_path: Relative path (as it appears in the diff) to read.
        cache: Mutable dict used for memoization; key is *rel_path*.

    Returns:
        File text or None on any skip condition.
    """
    if rel_path in cache:
        return cache[rel_path]
    content: "str | None" = None
    root_real = os.path.realpath(repo_root)
    full = os.path.realpath(os.path.join(repo_root, rel_path))
    try:
        inside = os.path.commonpath([root_real, full]) == root_real
    except ValueError:  # e.g. different drives on Windows
        inside = False
    if inside and os.path.isfile(full) and os.path.getsize(full) <= _MAX_FILE_BYTES:
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            content = None if "\x00" in text else text
        except OSError:
            content = None
    cache[rel_path] = content
    return content


def _git(repo_root: str, *args: str) -> tuple[int, str]:
    """Run git; return (returncode, stdout). errors='replace' so non-UTF-8
    (binary) output degrades instead of collapsing the run."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return result.returncode, result.stdout


def _git_toplevel(start: str) -> str | None:
    """Return the absolute path to the git repo root, or None if not in a repo.

    Args:
        start: Directory path to start searching from.

    Returns:
        Absolute path string to the toplevel repo directory, or None.
    """
    rc, out = _git(start, "rev-parse", "--show-toplevel")
    if rc != 0:
        return None
    return out.strip() or None


def _tree_is_clean(repo_root: str) -> bool:
    """Return True iff no uncommitted changes to TRACKED files (untracked ignored).

    Uses --untracked-files=no so the self-review workflow can leave an untracked
    bundle file in the repo without triggering a no-op.

    Args:
        repo_root: Absolute path to the git repository root.

    Returns:
        True if tracked files are all clean, False otherwise.
    """
    rc, out = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return rc == 0 and out.strip() == ""


def enrich_code_review_context(
    input_content: str,
    *,
    repo_root: str | None = None,
    base_ref: str = "main",
    max_chars: int = _ENRICH_MAX_CHARS,
) -> tuple[str, str]:
    """Return (content, note); content unchanged on no-op. Never raises (R7).

    Args:
        input_content: The original review content to potentially enrich.
        repo_root: Optional path to the git repository root. Defaults to cwd.
        base_ref: The base git ref to diff against. Defaults to "main".
        max_chars: Maximum characters for the enriched output. Defaults to
            _ENRICH_MAX_CHARS.

    Returns:
        A tuple of (content, note) where content is either the enriched
        content or the original input_content on no-op, and note describes
        what happened.
    """
    try:
        return _enrich(input_content, repo_root, base_ref, max_chars)
    except Exception as exc:  # noqa: BLE001 — fail-safe contract
        return input_content, f"enrichment skipped (error: {exc!r})"


def _enrich(
    input_content: str, repo_root: str | None, base_ref: str, max_chars: int
) -> tuple[str, str]:
    """Internal enrichment logic; may raise (caller wraps in try/except).

    Args:
        input_content: The original review content to potentially enrich.
        repo_root: Optional path to the git repository root.
        base_ref: The base git ref to diff against.
        max_chars: Maximum characters for the enriched output.

    Returns:
        A tuple of (content, note).
    """
    root = _git_toplevel(repo_root or os.getcwd())
    if root is None:
        return input_content, "enrichment skipped (not a git repo)"
    if not _tree_is_clean(root):
        return input_content, "enrichment skipped (working tree not clean / not at HEAD)"
    return input_content, "enrichment skipped (no diff context)"
