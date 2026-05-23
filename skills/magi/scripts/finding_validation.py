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
_NEWFILE_RE = re.compile(r"^\+\+\+ b/(.+)$")
#: A finding's ``line`` may be off by a few from the diff's exact post-image
#: numbering (LLM counting fuzz); accept within this margin of a changed line.
LINE_RANGE_MARGIN = 3


def parse_diff_ranges(diff: str) -> dict[str, set[int]]:
    """Map each touched file to the set of changed post-image line numbers.

    Walks the unified diff: ``+++ b/<file>`` opens a file, ``@@`` resets the
    post-image counter, added/context lines advance it (deletions do not).
    """
    ranges: dict[str, set[int]] = {}
    current: str | None = None
    new_line = 0
    for raw in diff.splitlines():
        m = _NEWFILE_RE.match(raw)
        if m:
            current = normalize_path(m.group(1))
            ranges.setdefault(current, set())
            continue
        h = _HUNK_RE.match(raw)
        if h:
            new_line = int(h.group(1))
            continue
        if current is None:
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            continue
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
    * ``file`` not exact but its **basename** matches a diff file (A3) -> the
      agent under-qualified the path: soft-annotate ``"[path unverified] "`` and
      **skip the line-range check** (the path->line mapping is untrusted). This
      deliberately trades a possible false-keep on a common basename for not
      false-dropping a real finding; hard-drop is reserved for clearly fabricated
      files (no exact and no basename match).
    * No exact and no basename match -> **hard-drop** (hallucinated file).
    Never raises.
    """
    # A3 (iter-3): only a UNIQUE basename is a strong enough signal for the
    # soft-annotate fallback; an ambiguous basename (shared by 2+ diff files) is
    # hard-dropped — too weak to tell a real finding from a fabrication.
    base_counts: dict[str, int] = {}
    for vf in files:
        b = vf.rsplit("/", 1)[-1]
        base_counts[b] = base_counts.get(b, 0) + 1
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
            line = f.get("line")
            if isinstance(line, int) and not isinstance(line, bool):
                rng = ranges.get(nf, set())
                if rng and not any(abs(line - r) <= margin for r in rng):
                    f = {**f, "detail": "[outside changed range] " + str(f.get("detail", ""))}
                    annotated += 1
            kept.append(f)
        elif base_counts.get(nf.rsplit("/", 1)[-1], 0) == 1:
            f = {**f, "detail": "[path unverified] " + str(f.get("detail", ""))}
            annotated += 1
            kept.append(f)
        else:
            dropped += 1
    return kept, dropped, annotated
