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

#: ``@@ -a,b +c,d @@`` — capture the new-file (post-image) start + count.
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
#: New-file (post-image) header path. The ``b/`` prefix is git-specific; plain
#: ``diff -u`` output omits it, so it is optional. A literal ``+++`` line is
#: promoted to a header only by :func:`_iter_diff_events` (which also requires
#: the preceding ``--- `` and a following ``@@``), never by this pattern alone.
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

    Header recognition: a ``+++ `` line is a new-file header only when it
    directly follows a ``--- `` line and is itself followed by an ``@@`` hunk
    header. The trailing-``@@`` requirement prevents a deleted ``-- ``-comment
    line (rendered ``--- ``) adjacent to an added ``++ `` line (rendered
    ``+++ ``) from being misparsed as a phantom file header. The git ``b/``
    prefix is optional; ``/dev/null`` targets and ``diff -u`` tab-timestamps are
    stripped (see :func:`_clean_newfile_path`). The ``@@`` requirement loses no
    real file: a content-bearing file's header is always followed by a hunk, and
    an empty new file emits no ``--- ``/``+++ `` header at all (git verified), so
    there was never a header to recognize — and an empty file has no citable line.

    Paths are yielded raw for callers that read them from disk;
    :func:`parse_diff_ranges` applies :func:`normalize_path` itself, so the
    consumers share recognition but differ in path normalization (a caller
    comparing across them must normalize, as the guard does). Limitation: git
    C-quoted paths (octal-escaped unicode/control chars) are not unquoted —
    pre-existing and low-likelihood.
    """
    lines = diff.splitlines()
    n = len(lines)
    current: str | None = None
    new_line = 0
    i = 0
    while i < n:
        raw = lines[i]
        if raw.startswith(_OLDFILE_PREFIX):
            # A real file header is '--- ' then '+++ ' then '@@'. Requiring the
            # trailing '@@' (count-free, so it tolerates imprecise hunk counts)
            # disambiguates a real header from a deleted '-- '-comment line
            # adjacent to an added '++ ' line — both render as '--- '/'+++ '.
            if i + 2 < n and _HUNK_RE.match(lines[i + 2]):
                m = _NEWFILE_RE.match(lines[i + 1])
                if m:
                    current = _clean_newfile_path(m.group(1))
                    if current is not None:
                        yield ("file", current)
                    i += 2  # consume '--- ' and '+++ '; '@@' handled next loop
                    continue
            i += 1  # '--- ' is a deletion/content line, not a header
            continue
        h = _HUNK_RE.match(raw)
        if h:
            new_line = int(h.group(1))
            i += 1
            continue
        if current is None:
            i += 1
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            # A '+++ '/'--- ' line that reached here is content, not a header
            # (real headers are consumed above). It is skipped rather than
            # counted: an added line whose body itself begins with '++ ' is left
            # uncounted (a documented off-by-one), the conservative choice — it
            # avoids miscounting a malformed/truncated '+++ ' header (e.g. a
            # trailing '+++ /dev/null' deletion with no hunk) as a real line.
            i += 1
            continue
        if raw.startswith("\\ "):
            i += 1  # "\ No newline at end of file" marker — not a real line
            continue
        if raw.startswith("+"):
            yield ("add", new_line, raw[1:])
            new_line += 1
            i += 1
            continue
        if raw.startswith("-"):
            i += 1  # deletion: no post-image line
            continue
        new_line += 1  # context line advances the post-image counter
        i += 1


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
