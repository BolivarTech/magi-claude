# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Diff-grounded validation of MAGI findings (code-review only).

Ports panóptico's hallucination guard and adds the line-range check it only
planned. Pure stdlib and **total** — never raises into the orchestrator.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from finding_id import normalize_path

#: ``@@ -a,b +c,d @@`` — old start/count (g1/g2) and new start/count (g3/g4).
#: The counts drive hunk-body line tracking in :func:`_iter_diff_events`; an
#: absent count means 1 (git omits ``,1``).
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
#: New-file (post-image) header path. The ``b/`` prefix is git-specific; plain
#: ``diff -u`` output omits it, so it is optional. A literal ``+++`` line is
#: promoted to a header only by :func:`_iter_diff_events` (outside any open hunk
#: body), never by this pattern alone.
_NEWFILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
#: Old-file header prefix; it opens a candidate ``--- ``/``+++ `` header pair.
_OLDFILE_PREFIX = "--- "
#: A finding's ``line`` may be off by a few from the diff's exact post-image
#: numbering (LLM counting fuzz); accept within this margin of a changed line.
LINE_RANGE_MARGIN = 3


def _clean_newfile_path(captured: str) -> str | None:
    """Normalize a captured ``+++`` header path to a file path, or ``None``.

    Strips a trailing ``\\t<timestamp>`` that ``diff -u`` appends, trims
    surrounding whitespace, and rejects empty paths and the ``/dev/null``
    deletion target.
    """
    path = captured.split("\t", 1)[0].strip()
    if not path or path == "/dev/null":
        return None
    return path


def _iter_diff_events(diff: str) -> Iterator[tuple[Any, ...]]:
    """Walk a unified diff once, yielding its structural events.

    The single source of truth for the three diff consumers
    (:func:`extract_touched_files`, :func:`added_lines_by_file`,
    :func:`parse_diff_ranges`) so they can never disagree on which files a diff
    touches or where its added lines fall. Yields:

    * ``("file", path)`` — a new-file header (raw post-image path).
    * ``("add", lineno, body)`` — an added post-image line (``+`` stripped).

    Header vs content is disambiguated by **hunk-body line counting**: an
    ``@@ -a,b +c,d @@`` opens a hunk of ``b`` old and ``d`` new lines, counted
    down as body lines are consumed (context decrements both, ``-`` the old,
    ``+`` the new). While a hunk is open, every line — including one that renders
    as ``--- `` or ``+++ `` (a deleted ``-- `` comment, an added ``++ `` line) —
    is content, never a file header. A ``--- ``/``+++ `` pair is promoted to a
    header only **outside** any open hunk, which is why a ``-- ``/``++ ``
    adjacency cannot inject a phantom file even when it sits right before the next
    ``@@``. A ``diff --git`` line force-closes the current hunk (a hard file
    boundary), keeping git diffs robust even when a hunk's ``@@`` counts
    under/over-state its body. An added line whose body begins with ``++ `` is
    counted correctly (it is consumed by the ``+`` branch inside the hunk).

    The git ``b/`` prefix is optional; ``/dev/null`` targets (deleted files) and
    ``diff -u`` tab-timestamps are stripped (see :func:`_clean_newfile_path`).
    Paths are yielded raw for callers that read them from disk;
    :func:`parse_diff_ranges` applies :func:`normalize_path` itself, so the
    consumers share recognition but differ in path normalization (a caller
    comparing across them must normalize, as the guard does).

    Limitations (both low-likelihood, accepted trade-offs):

    * **Non-git overstated-count recall loss.** A non-git diff (no ``diff --git``
      boundary) whose hunk ``@@`` count *overstates* its body keeps the hunk
      "open", so a later file's ``--- ``/``+++ `` header is read as content and
      that file is left unrecognized — a finding on it is then hard-dropped. This
      is the deliberate cost of using hunk-counting to immunize against the
      ``-- ``/``++ `` phantom edge; the two are structurally indistinguishable
      without counts. Git diffs are immune: ``diff --git`` force-closes each hunk
      and git's counts are exact.
    * **Non-git understated-count misparse (the dual).** Symmetrically, a non-git
      diff whose ``@@`` count *understates* its body closes the hunk early, so
      trailing body lines are read as structural — a ``-- ``/``++ `` content
      adjacency there can register a phantom file and trailing added lines are
      missed. Same root cause and same immunity (git counts are exact).
    * **C-quoted paths.** Git C-quoted paths (octal-escaped unicode/control
      chars) are not unquoted.
    """
    lines = diff.splitlines()
    n = len(lines)
    current: str | None = None
    new_line = 0
    old_rem = 0  # old-side lines left in the open hunk (0 and new_rem 0 => no hunk)
    new_rem = 0  # new-side lines left in the open hunk
    i = 0
    while i < n:
        raw = lines[i]
        if raw.startswith("diff --git "):
            old_rem = new_rem = 0  # new file section: force-close any open hunk
            i += 1
            continue
        if old_rem > 0 or new_rem > 0:
            # Inside a hunk body: every line is content. A line rendering as
            # '--- '/'+++ ' here is a deleted/added line, never a file header.
            if raw.startswith("\\ "):
                i += 1  # "\ No newline at end of file" — not a real line
                continue
            if raw.startswith("+"):
                if current is not None:
                    yield ("add", new_line, raw[1:])
                new_line += 1
                new_rem -= 1
                i += 1
                continue
            if raw.startswith("-"):
                old_rem -= 1  # deletion: no post-image line
                i += 1
                continue
            new_line += 1  # context line advances the post-image counter
            old_rem -= 1
            new_rem -= 1
            i += 1
            continue
        # Outside any hunk: only structural lines (headers, '@@', index, modes).
        if raw.startswith(_OLDFILE_PREFIX):
            m = _NEWFILE_RE.match(lines[i + 1]) if i + 1 < n else None
            if m:
                current = _clean_newfile_path(m.group(1))
                if current is not None:
                    yield ("file", current)
                i += 2  # consume '--- ' and its paired '+++ '
                continue
            i += 1  # lone '--- ' without a '+++ ' pair — not a header
            continue
        h = _HUNK_RE.match(raw)
        if h:
            new_line = int(h.group(3))
            old_rem = int(h.group(2)) if h.group(2) else 1
            new_rem = int(h.group(4)) if h.group(4) else 1
            i += 1
            continue
        i += 1  # 'index', mode, prose, etc. — ignored outside a hunk


def extract_touched_files(diff: str) -> list[str]:
    """Return the ordered (raw) post-image paths a unified diff touches.

    Thin consumer of :func:`_iter_diff_events`; paths are raw (not normalized)
    for callers that read them from disk.
    """
    return [ev[1] for ev in _iter_diff_events(diff) if ev[0] == "file"]


def added_lines_by_file(diff: str) -> dict[str, list[str]]:
    """Map each (raw) post-image path to its added (``+``) line bodies.

    Thin consumer of :func:`_iter_diff_events` so the enrichment coherence check
    keys added lines under the SAME paths the touched-file set uses (F2).
    """
    result: dict[str, list[str]] = {}
    current: str | None = None
    for ev in _iter_diff_events(diff):
        if ev[0] == "file":
            current = ev[1]
        elif current is not None:  # ("add", lineno, body)
            result.setdefault(current, []).append(ev[2])
    return result


def parse_diff_ranges(diff: str) -> dict[str, set[int]]:
    """Map each touched file (normalized) to its changed post-image line numbers.

    Thin consumer of :func:`_iter_diff_events`; applies :func:`normalize_path`
    so the guard's file/line keys match normalized finding paths.
    """
    ranges: dict[str, set[int]] = {}
    current: str | None = None
    for ev in _iter_diff_events(diff):
        if ev[0] == "file":
            current = normalize_path(ev[1])
            ranges.setdefault(current, set())
        elif current is not None:  # ("add", lineno, body)
            ranges[current].add(ev[1])
    return ranges


def valid_files(diff: str) -> set[str]:
    """Return the set of normalized file paths present in *diff*."""
    return set(parse_diff_ranges(diff).keys())


def _line_outside_range(line: Any, rng: set[int], margin: int) -> bool:
    """True iff *line* is an integer outside non-empty *rng* (within *margin*).

    A non-int/bool ``line`` or an empty *rng* yields ``False`` (nothing to flag).
    Shared by the exact-file and unique-basename branches of
    :func:`validate_findings`.
    """
    if not isinstance(line, int) or isinstance(line, bool):
        return False
    return bool(rng) and not any(abs(line - r) <= margin for r in rng)


def validate_findings(
    findings: list[dict[str, Any]],
    files: set[str],
    ranges: dict[str, set[int]],
    margin: int = LINE_RANGE_MARGIN,
) -> tuple[list[dict[str, Any]], int, int]:
    """Filter *findings* against the diff. Returns ``(kept, dropped, annotated)``.

    * Finding without ``file`` -> kept untouched (not validatable).
    * ``file`` (normalized) in *files* -> in-diff; if ``line`` is outside its
      changed range (+/- *margin*) -> soft-annotate ``"[outside changed range] "``.
    * ``file`` not exact but its **basename** uniquely matches a diff file (A3)
      -> the agent under-qualified the path: soft-annotate ``"[path unverified] "``.
      Because a unique basename identifies the exact file, the line-range check
      (F3) STILL runs against that file; a ``line`` outside its changed range
      additionally gets ``"[outside changed range] "``. The finding is kept
      regardless (recall preserved) — these are observability markers, not drops.
    * No exact and no unique-basename match -> **hard-drop** (hallucinated file).
    Never raises.
    """
    # A3 (iter-3): only a UNIQUE basename is a strong enough signal for the
    # soft-annotate fallback; an ambiguous basename (shared by 2+ diff files) is
    # hard-dropped — too weak to tell a real finding from a fabrication.
    base_counts: dict[str, int] = {}
    base_to_file: dict[str, str] = {}
    for vf in files:
        b = vf.rsplit("/", 1)[-1]
        base_counts[b] = base_counts.get(b, 0) + 1
        base_to_file[b] = vf  # only consulted when the basename is unique
    kept: list[dict[str, Any]] = []
    dropped = 0
    annotated = 0
    for f in findings:
        file = f.get("file")
        if not file or not isinstance(file, str):
            kept.append(f)
            continue
        nf = normalize_path(file)
        if nf in files:
            if _line_outside_range(f.get("line"), ranges.get(nf, set()), margin):
                f = {**f, "detail": "[outside changed range] " + str(f.get("detail", ""))}
                annotated += 1
            kept.append(f)
        elif base_counts.get(nf.rsplit("/", 1)[-1], 0) == 1:
            # F3: a unique basename resolves to exactly one diff file, so run the
            # line-range check against it instead of skipping it. Both markers may
            # apply; the finding is kept either way (observability, not a drop).
            resolved = base_to_file[nf.rsplit("/", 1)[-1]]
            detail = str(f.get("detail", ""))
            if _line_outside_range(f.get("line"), ranges.get(resolved, set()), margin):
                detail = "[outside changed range] " + detail
            f = {**f, "detail": "[path unverified] " + detail}
            annotated += 1
            kept.append(f)
        else:
            dropped += 1
    return kept, dropped, annotated
