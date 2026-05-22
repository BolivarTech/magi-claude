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


def _largest_json_object(text: str) -> dict[str, Any] | None:
    """Return the widest-spanning embedded JSON object in *text*, or ``None``.

    Scans for every ``{`` and attempts ``json.JSONDecoder().raw_decode``
    from that position, which parses one complete JSON value and reports
    where it ended — so nested braces, braces inside strings, and arrays
    are handled correctly without hand-rolled brace counting.

    Only ``dict`` values qualify: the agent schema is a JSON object, so a
    bare array or scalar appearing in the surrounding prose must not be
    mistaken for the verdict. The widest span wins so an incidental small
    object in the preamble (e.g. a schema example ``{"agent": "name"}``)
    cannot shadow the real multi-key verdict that follows it.

    Args:
        text: Text that may contain a JSON object embedded in prose.

    Returns:
        The widest-spanning embedded ``dict``, or ``None`` if no JSON
        object decodes anywhere in *text*.
    """
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_span = -1
    index = 0
    length = len(text)
    while index < length:
        brace = text.find("{", index)
        if brace == -1:
            break
        try:
            candidate, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(candidate, dict) and end - brace > best_span:
            best, best_span = candidate, end - brace
        # Advance past the decoded value so the next iteration looks for a
        # later object; guard against a zero-width decode pinning the scan.
        index = end if end > brace else brace + 1
    return best


def _loads_lenient(text: str) -> Any:
    """Parse JSON from *text*, tolerating natural-language prose around it.

    The fast path is a strict :func:`json.loads`: in the common case the
    text *is* the JSON object (optionally after fence stripping) and the
    behaviour is byte-for-byte identical to before 2.4.2. When that raises
    — which happens when an agent doing multi-turn tool use prepends a
    transitional sentence before the JSON verdict (the 2.4.2 exit-1 root
    cause) — the largest embedded JSON object is recovered instead.

    If no embedded object decodes, the original
    :class:`json.JSONDecodeError` is re-raised so genuinely malformed or
    truncated output still fails closed. The orchestrator relies on that
    exception to drive its single retry and, failing that, degraded-mode
    handling; silently salvaging a truncated verdict would defeat both.

    Args:
        text: Candidate JSON text, possibly wrapped in prose.

    Returns:
        The parsed JSON value.

    Raises:
        json.JSONDecodeError: If *text* contains no decodable JSON object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        embedded = _largest_json_object(text)
        if embedded is not None:
            return embedded
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
