#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-05-21
"""Parse and validate agent JSON output from Claude CLI.

Extracts structured JSON from various Claude CLI output formats,
strips markdown code fences, recovers the JSON verdict even when an
agent wraps it in natural-language prose (2.4.2), validates the result,
and writes clean JSON to the specified output file.

Usage:
    python3 parse_agent_output.py <input_file> <output_file>

Exit codes:
    0 - Success: valid JSON extracted and written to output file.
    1 - Failure: input could not be parsed or did not contain valid JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Bootstrap: see CLAUDE.md "Open technical debt / synthesize import gap [LOCKED]".
# Direct invocation and pytest already cover this; ``python -m
# skills.magi.scripts.parse_agent_output`` does not.
_SCRIPT_DIR = str(Path(__file__).parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from validate import MAX_INPUT_FILE_SIZE  # noqa: E402


# Regex to strip leading ```json (case-insensitive, optional whitespace) or bare ```
_FENCE_START = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_END = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping the text.

    Handles variants such as ```json, ```JSON, ``` json, and bare ```.

    Args:
        text: Raw text potentially wrapped in code fences.

    Returns:
        Text with leading/trailing fences removed and whitespace trimmed.
    """
    text = text.strip()
    text = _FENCE_START.sub("", text)
    text = _FENCE_END.sub("", text)
    return text.strip()


def _extract_text(data: object) -> str:
    """Extract the meaningful text payload from Claude CLI JSON output.

    Supports multiple output shapes:
        - ``{"result": "..."}``
        - ``{"content": [{"type": "text", "text": "..."}]}``
        - Plain string

    Args:
        data: Deserialised JSON value from Claude CLI output.

    Returns:
        The extracted text content as a string.

    Raises:
        ValueError: If the data format is not recognised (no ``result``
            or ``content`` key in a dict, or unexpected type).
    """
    if isinstance(data, dict) and "result" in data:
        return str(data["result"])

    if isinstance(data, dict) and "content" in data:
        content = data["content"]
        if not isinstance(content, list):
            # A malformed ``content`` (e.g. a bare string or a dict) would
            # otherwise iterate character-by-character or by dict key and
            # quietly miss every text block. Reject the shape up front so
            # the caller gets a clear signal instead of a silent "No text
            # block found".
            raise ValueError(f"'content' must be a list, got {type(content).__name__}.")
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block["text"])
        raise ValueError("No text block found in 'content' array")

    if isinstance(data, str):
        return data

    raise ValueError(
        f"Unexpected Claude CLI output type: {type(data).__name__}. "
        f"Expected dict with 'result' or 'content' key, or plain string."
    )


# Minimal keys that identify an agent verdict among other JSON objects an
# agent might echo (config files, schema examples, quoted payloads). Kept to
# the two discriminating keys rather than the full 7-key schema so a verdict
# merely missing a key (e.g. ``recommendation``) is still recovered and then
# rejected by ``load_agent_output``'s full check — preserving the retry path.
_VERDICT_KEYS = ("agent", "verdict")

# Lenient recovery is a fallback for prose-wrapped output, which is a few KB
# in practice. Above this budget the scan is skipped: a multi-MB blob is
# almost certainly echoed tool-use content, not a clean verdict, and scanning
# it risks the O(n^2) ``raw_decode`` worst case. The agent is dropped and
# retried instead.
_LENIENT_RECOVERY_MAX_CHARS = 1_000_000

# Hard cap on candidate ``{`` positions probed, bounding the scan within the
# size budget against adversarial deeply-nested-unterminated input. The real
# verdict is found within the first few probes; a legitimate output never
# approaches this.
_MAX_BRACE_PROBES = 2_000


def _embedded_verdict_object(text: str) -> dict[str, Any] | None:
    """Return the *sole* embedded JSON object that looks like an agent verdict.

    Scans for ``{`` and attempts ``json.JSONDecoder().raw_decode`` from each
    position — which parses one complete JSON value and reports where it
    ended, so nested braces and braces inside strings are handled without
    hand-rolled counting.

    Selection is **schema-aware, not span-based**: only objects carrying the
    verdict discriminator keys (:data:`_VERDICT_KEYS`) qualify, so a large
    JSON document an agent echoes from tool use — ``package.json``, an API
    payload — cannot be mistaken for the verdict even when it out-spans it.

    Recovery succeeds **only when exactly one** qualifying object decodes
    *within the probe budget* (:data:`_MAX_BRACE_PROBES`). If two or more do
    (the agent quoted the schema example — which is a complete valid verdict,
    see ``agents/*.md`` — beside its real verdict, or content under review
    embedded one), the choice is ambiguous: picking either risks a fabricated
    ``approve`` entering consensus, which ``load_agent_output`` cannot catch
    because both are well-formed. We return ``None`` so the caller fails
    closed and the orchestrator retries rather than guessing. (2.4.2 pass-2
    review — consensus integrity.) Note the budget bound: a second qualifying
    object beyond the probe cap would not be seen, so a verdict followed by
    >2000 brace positions then a second verdict is the one ambiguity shape the
    guard cannot observe — acceptable, as that input is already pathological.

    The scan is bounded by :data:`_MAX_BRACE_PROBES` so adversarial
    deeply-nested-unterminated input cannot degrade to O(n^2). A
    :class:`RecursionError` from a deeply nested candidate is treated like a
    decode failure (skip the candidate) rather than aborting the scan.

    Known residual (single-match fabrication): when exactly one verdict-shaped
    object decodes but it is NOT the real verdict — a quoted example beside a
    truncated real verdict, an early echo with the real verdict beyond the
    probe cap, or a lone echoed example — it is recovered and a fabricated
    ``approve`` can reach consensus. The durable fix is a verdict
    sentinel/delimiter (or Option C). See CLAUDE.md "Durable verdict-recovery
    fix". Do not add more heuristic tuning here; the next change should be the
    sentinel.

    Args:
        text: Text that may contain a verdict object embedded in prose.

    Returns:
        The single qualifying verdict ``dict``, or ``None`` if zero qualify,
        more than one qualify (ambiguous), or the probe budget is exhausted.
    """
    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    index = 0
    length = len(text)
    probes = 0
    while index < length and probes < _MAX_BRACE_PROBES:
        brace = text.find("{", index)
        if brace == -1:
            break
        probes += 1
        try:
            candidate, end = decoder.raw_decode(text, brace)
        except (json.JSONDecodeError, RecursionError):
            index = brace + 1
            continue
        if isinstance(candidate, dict) and all(key in candidate for key in _VERDICT_KEYS):
            matches.append(candidate)
            if len(matches) > 1:
                return None  # ambiguous — fail closed rather than guess
        # Advance past the decoded value so the next iteration looks for a
        # later object; guard against a zero-width decode pinning the scan.
        index = end if end > brace else brace + 1
    return matches[0] if len(matches) == 1 else None


def _loads_lenient(text: str) -> Any:
    """Parse JSON from *text*, tolerating natural-language prose around it.

    The fast path is a strict :func:`json.loads`: in the common case the
    text *is* the JSON object (optionally after fence stripping) and the
    behaviour is byte-for-byte identical to before 2.4.2. When that raises —
    which happens when an agent doing multi-turn tool use prepends a
    transitional sentence before the JSON verdict (the 2.4.2 exit-1 root
    cause) — the embedded verdict object is recovered via
    :func:`_embedded_verdict_object`.

    Recovery is skipped for input larger than
    :data:`_LENIENT_RECOVERY_MAX_CHARS` (likely echoed tool-use content, and
    a scan hazard). If nothing qualifies, the original
    :class:`json.JSONDecodeError` is re-raised so output with no JSON object,
    a truncated verdict (whose stray complete sub-objects lack the verdict
    keys), an ambiguous multi-verdict output, or only echoed non-verdict
    objects still fails closed at this layer. The orchestrator relies on that
    exception to drive its single retry and degraded-mode handling; the full
    7-key schema is still enforced downstream by ``load_agent_output``.

    A :class:`RecursionError` (CPython's ``json`` raises it, not
    ``JSONDecodeError``, on deeply nested input) is mapped to a
    ``JSONDecodeError`` so deeply-nested echoed/adversarial output stays on
    the same fail-closed/retry path instead of escaping as an uncaught error.

    Args:
        text: Candidate JSON text, possibly wrapped in prose.

    Returns:
        The parsed JSON value.

    Raises:
        json.JSONDecodeError: If *text* yields no qualifying verdict object,
            including the deeply-nested ``RecursionError`` case.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        if len(text) <= _LENIENT_RECOVERY_MAX_CHARS:
            verdict = _embedded_verdict_object(text)
            if verdict is not None:
                return verdict
        if isinstance(exc, RecursionError):
            raise json.JSONDecodeError(
                "Input nesting exceeds the JSON decoder limit", text, 0
            ) from exc
        raise


def parse_agent_output(input_path: str, output_path: str) -> None:
    """Read raw Claude CLI output, extract and validate JSON, write result.

    Args:
        input_path:  Path to the raw Claude CLI JSON output file.
        output_path: Destination path for the cleaned JSON.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        json.JSONDecodeError: If the extracted text contains no decodable
            JSON object (after both a strict parse and embedded-object
            recovery).
        ValueError: If content extraction fails or file exceeds size limit.
    """
    file_size = os.path.getsize(input_path)
    if file_size > MAX_INPUT_FILE_SIZE:
        raise ValueError(
            f"Input file {input_path} is {file_size} bytes, "
            f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes."
        )

    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    text = _extract_text(data)
    text = _strip_code_fences(text)

    # Validate that the cleaned text is valid JSON. Agents that do
    # multi-turn tool use sometimes wrap the verdict in prose, so a strict
    # parse falls back to recovering the embedded object; output with no
    # JSON object at all still raises (fail closed). See ``_loads_lenient``.
    parsed = _loads_lenient(text)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(parsed, fh, indent=2)
        fh.write("\n")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) != 3:
        print(
            "Usage: parse_agent_output.py <input_file> <output_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        parse_agent_output(input_path, output_path)
    except (json.JSONDecodeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
