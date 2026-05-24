# skills/magi/scripts/review_context.py
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-22
"""Deterministic, bounded, fail-safe review-context enrichment for MAGI
code-review mode. Runs only when the working tree is clean (== HEAD), so all
reads come from one coherent source. Never raises into the orchestrator (R7)."""

from __future__ import annotations

import keyword
import os
import re
import subprocess

_ENRICH_MAX_CHARS = 512_000
_DEF_WINDOW_LINES = 40
_MAX_CANDIDATES = 60
_MAX_DEFS = 40
_MAX_DEFS_PER_NAME = 5
_GIT_TIMEOUT = 30
_MAX_FILE_BYTES = 262_144
_MAX_TOUCHED_FILES = 50
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
# keyword.softkwlist was added in CPython 3.12; pyproject pins >=3.9.
_SOFT_KWLIST: frozenset[str] = frozenset(getattr(keyword, "softkwlist", []))


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
    if inside:
        try:
            if os.path.isfile(full) and os.path.getsize(full) <= _MAX_FILE_BYTES:
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


def _added_lines_by_file(diff_text: str) -> dict[str, list[str]]:
    """Map each post-image path to its added (``+``) lines from the diff.

    Args:
        diff_text: A unified diff string (git format).

    Returns:
        Dict mapping relative file path to list of added line bodies (the
        leading ``+`` character is stripped).
    """
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if (not path or path == "/dev/null") else path
        elif current and line.startswith("+") and not line.startswith("+++"):
            result.setdefault(current, []).append(line[1:])
    return result


def _coheres(content: str, added: list[str]) -> bool:
    """Return True iff every non-blank added line appears in *content*.

    This is a cheap HEAD-coherence check: with a clean working tree
    (working tree == HEAD), any added line from the diff must be present
    in the file. A mismatch means the diff doesn't correspond to HEAD.

    Args:
        content: Full text of the working-tree file.
        added: List of added line bodies from the diff for this file.

    Returns:
        True if all non-blank added lines are found in content.
    """
    return all(a.strip() == "" or a.strip() in content for a in added)


def _collect_touched(
    repo_root: str, diff_text: str, cache: "dict[str, str | None]"
) -> "tuple[list[tuple[str, str]], list[str]]":
    """Return (touched, mismatched_paths).

    *touched* holds ``(path, content)`` for files that exist, are readable,
    and whose added lines cohere with the working tree. *mismatched_paths*
    holds paths where the coherence check failed.

    Args:
        repo_root: Absolute path to the git repository root.
        diff_text: A unified diff string (git format).
        cache: Mutable dict used for memoization by _read_file_safe.

    Returns:
        Tuple of (touched list of (path, content), mismatched path list).
    """
    added_by_file = _added_lines_by_file(diff_text)
    touched: list[tuple[str, str]] = []
    mismatched: list[str] = []
    for path in list(dict.fromkeys(_extract_touched_files(diff_text)))[
        :_MAX_TOUCHED_FILES
    ]:  # dedup, preserve order, cap
        content = _read_file_safe(repo_root, path, cache)
        if content is None:
            continue
        if not _coheres(content, added_by_file.get(path, [])):
            mismatched.append(path)
            continue
        touched.append((path, content))
    return touched, mismatched


def _code_part(line: str) -> str:
    """Drop pure-comment lines, string-literal contents, and inline comments
    (single-line heuristic; multi-line strings/docstrings may leak — documented).

    Args:
        line: A single source line (the leading ``+`` already stripped).

    Returns:
        The portion of the line that counts as code, with string contents and
        inline ``  #`` comments removed; empty string for pure-comment lines.
    """
    if line.lstrip().startswith("#"):
        return ""
    line = _STRING_RE.sub("", line)
    idx = line.find("  #")
    return line[:idx] if idx != -1 else line


def _defined_names(texts: list[str]) -> set[str]:
    """Return all ``def``/``class`` names declared in the given source texts.

    Args:
        texts: List of full file contents to scan.

    Returns:
        Set of identifier strings that appear as ``def`` or ``class`` names.
    """
    names: set[str] = set()
    for text in texts:
        for line in text.splitlines():
            m = _DEF_RE.match(line)
            if m:
                names.add(m.group(1))
    return names


def _candidate_identifiers(diff_text: str, defined: set[str]) -> list[str]:
    """Extract candidate identifiers from added lines, bounded by ``_MAX_CANDIDATES``.

    Strips keywords, soft-keywords, ``_EXTRA_EXCLUDE`` tokens, and names
    already defined in touched files (passed via *defined*). Comment-only lines
    and string-literal contents are excluded via ``_code_part``.

    Args:
        diff_text: A unified diff string (git format).
        defined: Set of names already declared in the touched files.

    Returns:
        Ordered list of unique candidate identifier strings (insertion order,
        capped at ``_MAX_CANDIDATES``).
    """
    ordered: dict[str, None] = {}
    for raw in diff_text.splitlines():
        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        for tok in _IDENT_RE.findall(_code_part(raw[1:])):
            if (
                tok in keyword.kwlist
                or tok in _SOFT_KWLIST
                or tok in _EXTRA_EXCLUDE
                or tok in defined
            ):
                continue
            ordered.setdefault(tok, None)
            if len(ordered) >= _MAX_CANDIDATES:
                return list(ordered)
    return list(ordered)


def _read_excerpt(
    repo_root: str, rel_path: str, line_no: int, cache: "dict[str, str | None]"
) -> "str | None":
    """Return up to ``_DEF_WINDOW_LINES`` lines starting at *line_no* (1-based).

    Args:
        repo_root: Absolute path to the git repository root.
        rel_path: Relative path to the file.
        line_no: 1-based line number where the definition starts.
        cache: Mutable dict used for memoization by ``_read_file_safe``.

    Returns:
        Multi-line string excerpt, or None if the file cannot be read.
    """
    content = _read_file_safe(repo_root, rel_path, cache)
    if content is None:
        return None
    lines = content.splitlines()
    start = max(0, line_no - 1)
    return "\n".join(lines[start : start + _DEF_WINDOW_LINES])


def _grep_defs(
    repo_root: str, names: list[str], cache: "dict[str, str | None]"
) -> "list[tuple[str, int, str]]":
    """Single batched ``git grep`` for ``def``/``class`` of any candidate name.

    Portable word boundary ``([^A-Za-z0-9_]|$)`` is used (not ``\\b``,
    which is unsupported in git ERE on all platforms). Bounded by
    ``_MAX_DEFS`` total and ``_MAX_DEFS_PER_NAME`` per symbol (collision-
    flooding guard from iter-3).

    Args:
        repo_root: Absolute path to the git repository root.
        names: List of candidate identifier strings to look up.
        cache: Mutable dict used for memoization by ``_read_excerpt``.

    Returns:
        List of ``(rel_path, line_no, excerpt)`` tuples, at most ``_MAX_DEFS``
        entries and at most ``_MAX_DEFS_PER_NAME`` per distinct name.
    """
    if not names:
        return []
    alt = "|".join(re.escape(n) for n in names)
    pattern = rf"^[\t ]*(def|class)[\t ]+({alt})([^A-Za-z0-9_]|$)"
    rc, out = _git(repo_root, "grep", "-nE", pattern)
    if rc != 0:
        return []
    defs: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    per_name: dict[str, int] = {}
    for hit in out.splitlines():
        parts = hit.split(":", 2)
        if len(parts) < 3:
            continue
        path, line_s, body = parts
        try:
            line_no = int(line_s)
        except ValueError:
            continue
        m = _DEF_RE.match(body)
        name = m.group(1) if m else None
        if name is not None and per_name.get(name, 0) >= _MAX_DEFS_PER_NAME:
            continue
        key = (path, line_no)
        if key in seen:
            continue
        seen.add(key)
        excerpt = _read_excerpt(repo_root, path, line_no, cache)
        if excerpt is not None:
            defs.append((path, line_no, excerpt))
            if name is not None:
                per_name[name] = per_name.get(name, 0) + 1
        if len(defs) >= _MAX_DEFS:
            break
    return defs


def _git_diff(repo_root: str, base_ref: str) -> "str | None":
    """Return the output of ``git diff <base_ref>...HEAD``, or None on failure.

    Args:
        repo_root: Absolute path to the git repository root.
        base_ref: The base git ref to compare against HEAD.

    Returns:
        The diff text if the command succeeds and produces output, otherwise
        None (non-zero exit code, git unavailable, bad ref, or empty diff).
    """
    rc, out = _git(repo_root, "diff", f"{base_ref}...HEAD")
    if rc != 0:
        return None
    return out or None


def resolve_diff(input_content: str, repo_root: str, base_ref: str) -> str:
    """Resolve the review diff: input-embedded diff, else ``git diff <base>...HEAD``.

    This is the single shared diff-resolution seam used by BOTH enrichment and
    the finding guard, so the two never diverge (decision A2). Resolution rules
    mirror :func:`_enrich`:

    * If *input_content* already contains a unified diff, it is returned verbatim.
    * Otherwise, under a clean working tree (== HEAD, decision F),
      ``git diff <base_ref>...HEAD`` is returned.
    * Any no-op condition — not a git repo, dirty tree, empty diff — yields ``""``.

    TOTAL — returns ``""`` on ANY failure. It now runs in ``main()`` outside the
    ``_maybe_enrich`` boundary, so it must never raise into the orchestrator.

    Args:
        input_content: The raw review content (may itself embed a diff).
        repo_root: Path to start git resolution from (e.g. ``os.getcwd()``).
        base_ref: The base git ref to diff against HEAD.

    Returns:
        The resolved unified diff text, or ``""`` when no diff is available.
    """
    try:
        if _contains_diff(input_content):
            return input_content
        root = _git_toplevel(repo_root or os.getcwd())
        if root is None or not _tree_is_clean(root):
            return ""
        return _git_diff(root, base_ref) or ""
    except Exception:  # noqa: BLE001 — TOTAL: runs in main() outside _maybe_enrich
        return ""


def _assemble(
    input_content: str,
    touched: list[tuple[str, str]],
    defs: list[tuple[str, int, str]],
    max_chars: int,
) -> tuple[str, str]:
    """Assemble enriched content within a HARD char budget.

    Input is always kept (may itself exceed max_chars — documented carve-out).
    Touched files (smallest first) are added before defs; defs are dropped first
    when over budget. Header and join bytes are accounted up front so
    ``len(result) <= max_chars`` whenever ``len(input_content) <= max_chars``.

    Args:
        input_content: The original diff/review text; always preserved in output.
        touched: List of ``(path, content)`` pairs for files to include.
        defs: List of ``(path, line_no, excerpt)`` tuples for symbol definitions.
        max_chars: Hard character budget for the assembled result.

    Returns:
        A tuple of ``(content, note)`` where note summarises what was kept or
        omitted.
    """
    _TF_HDR = "## Touched files (full content)"
    _SD_HDR = "## Referenced symbol definitions"
    _JOIN = 2  # len("\n\n")

    parts: list[str] = [input_content]
    used = len(input_content)
    omitted: list[str] = []

    # Sort touched files smallest-first so cheapest blocks are kept preferentially.
    file_blocks = sorted(
        ((p, f"### {p}\n```\n{c}\n```") for p, c in touched), key=lambda pb: len(pb[1])
    )
    kept_files: list[str] = []
    for path, block in file_blocks:
        # Account for join + optional section header on the first file.
        extra = len(block) + _JOIN + (len(_TF_HDR) + _JOIN if not kept_files else 0)
        if used + extra <= max_chars:
            kept_files.append(block)
            used += extra
        else:
            omitted.append(f"file {path}")
    if kept_files:
        parts.append(_TF_HDR + "\n\n" + "\n\n".join(kept_files))

    kept_defs: list[str] = []
    for path, line_no, excerpt in defs:
        block = f"### {path}:{line_no}\n```\n{excerpt}\n```"
        extra = len(block) + _JOIN + (len(_SD_HDR) + _JOIN if not kept_defs else 0)
        if used + extra <= max_chars:
            kept_defs.append(block)
            used += extra
        else:
            omitted.append(f"def {path}:{line_no}")
    if kept_defs:
        parts.append(_SD_HDR + "\n\n" + "\n\n".join(kept_defs))

    if len(parts) == 1:
        return input_content, "enrichment skipped (nothing within budget)"
    note = f"enriched: {len(kept_files)} file(s), {len(kept_defs)} def(s)"
    if omitted:
        note += f"; omitted {len(omitted)} unit(s) over budget"
    return "\n\n".join(parts), note


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
        return input_content, "enrichment skipped (working tree not clean: uncommitted changes)"
    # Resolve the diff via the shared seam so enrichment and the finding guard
    # see the same diff (A2). The repo/clean gates above are kept for their
    # distinct skip notes; past them, ``resolve_diff`` is byte-for-byte
    # equivalent to the prior ``input_content if _contains_diff(...) else
    # _git_diff(...)`` expression (it returns "" rather than None for an empty
    # diff, which the ``not diff_text`` guard below treats identically).
    diff_text = resolve_diff(input_content, root, base_ref)
    if not diff_text:
        return input_content, "enrichment skipped (no diff context)"
    cache: dict[str, str | None] = {}
    touched, mismatched = _collect_touched(root, diff_text, cache)
    defined = _defined_names([c for _p, c in touched])
    defs = _grep_defs(root, _candidate_identifiers(diff_text, defined), cache)
    if not touched and not defs:
        note = "enrichment skipped (no readable context)"
        if mismatched:
            note = f"enrichment skipped (diff/HEAD mismatch: {len(mismatched)} file(s))"
        return input_content, note
    content, note = _assemble(input_content, touched, defs, max_chars)
    if mismatched and content != input_content:
        note += f"; {len(mismatched)} file(s) skipped (diff/HEAD mismatch)"
    return content, note
