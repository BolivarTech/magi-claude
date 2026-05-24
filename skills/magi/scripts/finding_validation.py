# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Diff-grounded validation of MAGI findings (code-review only).

Ports panóptico's hallucination guard and adds the line-range check it only
planned. Pure stdlib and **total** — never raises into the orchestrator.
"""

from __future__ import annotations

import re
from typing import Any

from finding_id import normalize_path

#: ``@@ -a,b +c,d @@`` — capture the new-file (post-image) start + count.
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
#: New-file (post-image) header. The ``b/`` prefix is git-specific; plain
#: ``diff -u`` output omits it, so it is optional. A literal ``+++`` line is
#: only honored as a header when it directly follows a ``--- `` old-file header
#: (see :func:`_iter_newfile_paths`); that pairing prevents an added content
#: line that renders as ``+++ ...`` from being misread as a file header.
_NEWFILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
#: Old-file header prefix; its presence gates whether the next ``+++`` is a header.
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


def extract_touched_files(diff: str) -> list[str]:
    """Return the ordered post-image paths a unified diff touches.

    Single source of truth for new-file recognition, shared with
    :func:`parse_diff_ranges` so the finding guard and the enrichment layer
    can never disagree on which files a diff touches. Honors a ``+++ `` header
    only when it follows a ``--- `` header (git ``b/`` prefix optional), skips
    ``/dev/null`` targets, and strips ``diff -u`` tab timestamps. Paths are
    returned raw (not normalized) for callers that read them from disk.
    """
    files: list[str] = []
    prev_minus = False
    for raw in diff.splitlines():
        if raw.startswith(_OLDFILE_PREFIX):
            prev_minus = True
            continue
        if prev_minus:
            prev_minus = False
            m = _NEWFILE_RE.match(raw)
            if m:
                path = _clean_newfile_path(m.group(1))
                if path is not None:
                    files.append(path)
    return files


def added_lines_by_file(diff: str) -> dict[str, list[str]]:
    """Map each post-image path to its added (``+``) line bodies.

    Uses the same new-file recognition as :func:`extract_touched_files` (the
    ``--- ``/``+++ `` pairing, optional ``b/``, tab-timestamp + ``/dev/null``
    handling) so the enrichment coherence check keys added lines under the SAME
    paths the touched-file set uses — the two can never disagree (F2). The
    leading ``+`` is stripped from each returned body.
    """
    result: dict[str, list[str]] = {}
    current: str | None = None
    prev_minus = False
    for raw in diff.splitlines():
        if raw.startswith(_OLDFILE_PREFIX):
            prev_minus = True
            continue
        if prev_minus:
            prev_minus = False
            m = _NEWFILE_RE.match(raw)
            if m:
                current = _clean_newfile_path(m.group(1))
                continue
        if current and raw.startswith("+") and not raw.startswith("+++"):
            result.setdefault(current, []).append(raw[1:])
    return result


def parse_diff_ranges(diff: str) -> dict[str, set[int]]:
    """Map each touched file to the set of changed post-image line numbers.

    Walks the unified diff: a ``--- `` header followed by a ``+++ <file>``
    header opens a file (git ``b/`` prefix optional), ``@@`` resets the
    post-image counter, added/context lines advance it (deletions do not). The
    ``--- ``/``+++ `` pairing (same primitive as :func:`extract_touched_files`)
    distinguishes a real header from an added content line rendered as ``+++``.
    """
    ranges: dict[str, set[int]] = {}
    current: str | None = None
    new_line = 0
    prev_minus = False
    for raw in diff.splitlines():
        if raw.startswith(_OLDFILE_PREFIX):
            prev_minus = True
            continue
        if prev_minus:
            prev_minus = False
            m = _NEWFILE_RE.match(raw)
            if m:
                path = _clean_newfile_path(m.group(1))
                current = normalize_path(path) if path is not None else None
                if current is not None:
                    ranges.setdefault(current, set())
                continue
            # '--- ' not followed by a '+++ ' header: fall through to handle raw.
        h = _HUNK_RE.match(raw)
        if h:
            new_line = int(h.group(1))
            continue
        if current is None:
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("\\ "):
            continue  # "\ No newline at end of file" marker — not a real line
        if raw.startswith("+"):
            ranges[current].add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            continue  # deletion: no post-image line
        else:
            new_line += 1  # context line advances the post-image counter
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
