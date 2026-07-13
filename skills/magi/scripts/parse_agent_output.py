#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.2.0
# Date: 2026-07-11
"""Parse and validate an agent's JSON verdict from any supported backend.

Unwraps the TRANSPORT envelope a backend may add (the Claude CLI's shapes; the Ollama
backend's content arrives unwrapped), then EXTRACTS the verdict from between the
``<MAGI_VERDICT>`` / ``</MAGI_VERDICT>`` marker lines and reads **nothing else** (MS2,
v5.1.0 — see ``verdict_markers.VerdictSentinel``).

It does not SEARCH for the verdict, and there is deliberately no fallback that does. The
heuristic that used to scan the whole response for whatever object "looked like" a verdict
is DELETED: it could return the worked example baked into the agent's own system prompt —
a fabricated ``approve`` in the adversarial seat. An output with no markers has no verdict,
however clean its JSON looks; it fails closed and the mage is retried with corrective
feedback. Any change that reintroduces a search outside the markers reverts that fix.

It does NOT validate the schema. The 7-key contract and the verdict enum are
enforced downstream by ``load_agent_output`` -- deliberately, because an object
that decodes but violates the schema must reach that check: its error message is
the corrective feedback the agent's retry depends on.

Usage:
    python3 parse_agent_output.py <input_file> <output_file>

Exit codes:
    0 - Success: valid JSON extracted and written to output file.
    1 - Failure: input could not be parsed or did not contain valid JSON.
"""

from __future__ import annotations

import json
import os
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
from verdict_markers import VerdictSentinel  # noqa: E402

#: The sentinel is **stateless** (it only carries the marker pair): one instance is enough.
_SENTINEL = VerdictSentinel()


#: The keys that make a decoded object a TRANSPORT envelope (the CLI's wrapper) rather than
#: the model's own words. The distinction decides which retry instruction the mage gets: a
#: malformed envelope is the transport's fault -- telling the model "you left out the markers"
#: would be a false instruction about text it never wrote -- while anything else IS the model's
#: text, and there the missing markers are exactly the defect it can fix.
#:
#: This is a CONTRACT with the ``claude -p`` CLI, and it is pinned by the fixtures in
#: ``tests/fixtures/claude-cli-outputs/`` -- captured from the real CLI, one per shape. If a
#: future CLI adds an envelope key, those fixtures are where it must be re-captured, and the
#: failure mode until then is a mage retried with the wrong instruction (fail-closed, never a
#: fabricated verdict). Guessing at unknown keys instead -- "if it looks like a wrapper" -- is
#: the same shape-guessing this milestone deleted, so the tuple stays explicit (MAGI gate,
#: Caspar).
_ENVELOPE_KEYS = ("result", "content")


def _is_transport_envelope(data: dict[str, Any]) -> bool:
    """Return True if *data* is a backend's transport wrapper, not the model's own output.

    Args:
        data: A decoded top-level JSON object.

    Returns:
        True if it carries an envelope key.
    """
    return any(key in data for key in _ENVELOPE_KEYS)


def _extract_text(data: object) -> str:
    """Extract the meaningful text payload from a backend's raw output.

    Supports every shape a backend can produce:
        - ``{"result": "..."}``                              (Claude CLI envelope)
        - ``{"content": [{"type": "text", "text": "..."}]}`` (Claude CLI envelope)
        - Plain string                                       (incl. fenced or
          prose-wrapped content, which reaches here as raw text when the file is
          not JSON at the top level — the Ollama path, 4.0.6)
        - Bare 7-key verdict dict                            (Ollama, unwrapped)

    Args:
        data: A decoded Claude CLI envelope, or — since 4.0.6 — the raw model text
            itself, when the file was never an envelope (the Ollama backend path).

    Returns:
        The extracted text content as a string.

    Raises:
        ValueError: If the data format is not recognised — no ``result`` or ``content``
            key in a dict, an unexpected type, or a ``content`` array with no usable
            text block (including a block that declares ``type: text`` but omits
            ``text``). This is the ONLY exception this function raises for a bad shape,
            and the caller maps it to ``JSONDecodeError`` so the mage keeps its retry;
            anything else escaping here (a ``KeyError`` did, until 4.0.6) costs the mage
            its second attempt.
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
            # ``"text" in block`` matters: a block that declares the type and omits the
            # key would otherwise raise ``KeyError`` — which ``run_magi`` does not retry,
            # so the mage would be dropped without a second attempt. Falling through to
            # the ValueError below (mapped to JSONDecodeError by the caller) retries it.
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                return str(block["text"])
        raise ValueError("No text block found in 'content' array")

    if isinstance(data, str):
        return data

    # There used to be a bare-verdict-dict branch here (``agent`` + ``verdict`` -> re-serialise
    # -> the sentinel says "you forgot the markers"). It is gone because the caller now routes
    # EVERY non-envelope object to the raw text, which reaches the same instruction by a rule
    # that does not depend on guessing which keys make an object "verdict-shaped" -- and
    # guessing that was the whole disease MS2 cured. What is left here is the transport
    # contract, and nothing else.
    raise ValueError(
        f"Unexpected agent output type: {type(data).__name__}. "
        f"Expected dict with 'result' or 'content' key, or plain string."
    )


def _extract_verdict(text: str) -> Any:
    """Extract the verdict from between the markers and decode it. **Extracts, never searches.**

    Replaces the heuristic recovery MS2 deleted -- and it was called ``_loads_lenient`` until
    the MAGI gate pointed out that the name now lies: **there is nothing lenient left
    underneath**. The only thing tolerated is NORMALIZATION inside an **already-delimited**
    region (stripping a fence). **Outside the markers nothing is ever looked at, ever.**

    What was deleted, and why it does not come back: the scanner decoded every JSON object it
    found in the output and kept whichever one "looked like" a verdict. Like every heuristic
    it had false positives (recovering something that was not the verdict -- the system
    prompt's own example, producing a **fabricated** ``approve`` in the adversarial seat) and
    false negatives (discarding a real verdict because it found two candidates). A fallback
    "just in case" **reintroduces the entire residual**.

    Args:
        text: The raw content of the agent's file (already unwrapped from the transport).

    Returns:
        The JSON object from between the markers.

    Raises:
        MissingVerdictMarkers: There are no markers. Inherits from ``ValidationError``, so
            the orchestrator retries with corrective feedback.
        UnterminatedVerdictBlock: The closing marker is missing (truncated output).
        AmbiguousVerdictMarkers: There is more than one delimited block.
        json.JSONDecodeError: The content **between** the markers is not decodable JSON.
            Every way the decoder can reject the payload arrives as this exception --
            syntax error, deep nesting (``RecursionError``), or an integer longer than
            ``int_max_str_digits``. This is the contract the orchestrator depends on: it
            retries on ``(ValidationError, JSONDecodeError)``, so anything else escaping
            here costs the mage its second attempt.
    """
    block = _SENTINEL.extract(text)
    try:
        return json.loads(block)
    except RecursionError as exc:
        # CPython raises RecursionError (not JSONDecodeError) on deep nesting. Mapping it
        # keeps the adversarial output on the fail-closed/retry path instead of escaping as
        # an uncaught error that would cost the mage its retry.
        raise json.JSONDecodeError(
            "Input nesting exceeds the JSON decoder limit", block, 0
        ) from exc
    except ValueError as exc:
        if isinstance(exc, json.JSONDecodeError):
            raise
        # A ValueError from the decoder that is NOT a JSONDecodeError: today that means an
        # integer literal above ``int_max_str_digits`` (4300). It must come out as a
        # JSONDecodeError or the orchestrator's retry guard does not catch it and the mage
        # is dropped without a second attempt.
        raise json.JSONDecodeError(f"Agent output is not decodable JSON: {exc}", block, 0) from exc


def parse_agent_output(input_path: str, output_path: str) -> None:
    """Read a raw agent response, extract its verdict object, and write it out.

    Backend-agnostic by contract: the raw file may be a Claude CLI transport
    envelope, or the agent's verdict itself (the Ollama backend writes the
    unwrapped content), optionally inside a markdown fence or surrounded by
    prose. All four shapes converge on the same decodable JSON object.

    It does **not** validate the schema, despite what an earlier version of this line
    claimed: the 7-key contract and the verdict enum are enforced downstream by
    ``load_agent_output``. That is deliberate — an object that decodes but violates the
    schema must reach that check, because its error message ("Invalid verdict 'Reject'")
    is the corrective feedback the agent's retry depends on.

    Args:
        input_path:  Path to the raw agent output file (envelope OR bare content).
        output_path: Destination path for the cleaned JSON.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        MissingVerdictMarkers: The output carries no marker line at all -- there is no
            verdict, however clean its JSON looks.
        UnterminatedVerdictBlock: Exactly one of the two markers is missing -- the
            signature of a TRUNCATED response.
        AmbiguousVerdictMarkers: The marker count is not exactly one open and one close,
            or the close precedes the open. There is no tie-break rule, by design.
        json.JSONDecodeError: The text BETWEEN the markers does not decode as JSON.
        ValueError: If content extraction fails or file exceeds size limit.

    Note:
        The three extraction errors are ``ValidationError`` subclasses, and the
        orchestrator picks the retry's corrective instruction from the exception TYPE
        (``retry_feedback``). An incomplete ``Raises:`` here is therefore not a
        documentation nit: it is how a future caller ends up spending a retry on the
        wrong instruction.
    """
    file_size = os.path.getsize(input_path)
    if file_size > MAX_INPUT_FILE_SIZE:
        raise ValueError(
            f"Input file {input_path} is {file_size} bytes, "
            f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes."
        )

    # errors="replace": these are the backend's bytes verbatim, and a strict decode
    # raises UnicodeDecodeError — a ValueError, but NOT a JSONDecodeError, so run_magi's
    # retry guard would miss it and the mage would be dropped without a second attempt.
    #
    # A bad byte can land ANYWHERE — an earlier version of this comment said it could
    # only land inside a string, and that was measured and false. What is true is the
    # part that matters: a bad byte that SURVIVES to a successful parse can only be
    # inside a string value, so the worst case is one replacement character in a summary.
    # In the structure it breaks the parse (JSONDecodeError → retry); in a key it mangles
    # the key (ValidationError → retry). Every outcome stays on the retry path, and none
    # of them fabricates: what is recovered is still the model's own object. Same
    # convention the codebase already uses for untrusted output (cost.py,
    # review_context.py, and the cp1252 hardening of 2.2.6).
    with open(input_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Read as TEXT, then decide whether it is an envelope (4.0.6).
    #
    # The Claude backend writes a transport envelope, so the raw file is JSON at
    # the top level. The Ollama backend writes ``choices[0].message.content``
    # already unwrapped, so the raw file is the agent's verdict ITSELF — and many
    # models emit that verdict inside a markdown fence, which is not JSON at all.
    #
    # Parsing before stripping the fence (the pre-4.0.6 order) made the fence
    # handling unreachable on exactly the path that needed it: ``json.load`` blew
    # up at character 0, the mage was retried, the retry produced the same
    # perfectly valid fenced verdict, and the mage was dropped. A schema-correct
    # verdict was discarded because of the ORDER of two operations.
    try:
        data: object = json.loads(raw)
    except (ValueError, RecursionError):
        # RecursionError, not just JSONDecodeError: CPython raises it on deeply
        # nested input, and on the Ollama path this text is MODEL-AUTHORED, so a
        # pathological response reaches this call directly.
        #
        # What letting it escape actually costs — checked, not assumed, because this
        # file has a history of comments asserting more than they can: the orchestrator
        # gathers with ``return_exceptions=True`` and re-raises only non-Exception
        # BaseExceptions other than CancelledError, so a stray RecursionError does NOT kill
        # the run — the mage is excluded and the run degrades. But the retry at
        # ``run_magi.py`` only catches ``(ValidationError, JSONDecodeError)``, so the
        # mage would be dropped **without its second attempt**. Mapping it here buys
        # back the retry, exactly as ``_extract_verdict`` does on the decode side.
        #
        # Scope of the behaviour change: a WELL-FORMED envelope is untouched (it
        # parses here exactly as before). A *malformed* Claude envelope, which used
        # to raise immediately, now goes through prose-recovery and could succeed.
        # That path is practically unreachable — a non-zero ``claude -p`` exit is
        # caught before the parse, and ``--output-format json`` always emits an
        # envelope — but the honest claim is "no change for well-formed envelopes",
        # not "no change on the Claude path".
        data = raw  # not an envelope: fenced or prose-wrapped content

    if isinstance(data, dict) and not _is_transport_envelope(data):
        # Decoded, but NOT a transport envelope: it is the model's own text, so the RAW TEXT is
        # the payload (BDD-8b) -- and with no markers in it there is no verdict, which is the
        # instruction the model can actually act on ("you left out the markers"). Routing it to
        # "unrecognised output shape" instead teaches it nothing and spends the retry on a
        # message about a defect it cannot locate.
        #
        # This cannot fabricate: JSON escapes newlines inside strings, so a marker quoted
        # anywhere in this object cannot appear alone on a line, and the sentinel needs a
        # line-anchored pair. No markers, no verdict.
        data = raw

    try:
        text = _extract_text(data)
    except ValueError as exc:
        # An unrecognised SHAPE must still reach the retry. ``_extract_text``
        # discriminates the bare-verdict-dict branch on exactly ``agent`` + ``verdict``,
        # so a model that omits one of those two keys falls through to its "unexpected
        # shape" ``ValueError`` — which ``run_magi``'s ``(ValidationError,
        # JSONDecodeError)`` guard does not catch, dropping the mage **without a second
        # attempt**. The identical content inside a markdown fence was retried with
        # corrective feedback. Same defect, opposite treatment, decided by a fence — and
        # reading-as-text-first (4.0.6) makes the bare route the PRIMARY one for Ollama.
        #
        # Mapping it here makes the two routes agree and keeps the promise this module's
        # docstring makes: an output that decodes but violates the schema reaches the
        # check whose error message the retry is built from. The size-cap ``ValueError``
        # is raised before this block, so it is not swallowed.
        raise json.JSONDecodeError(f"Unrecognised agent output shape: {exc}", raw, 0) from exc

    # Extract the verdict from between the marker lines and decode ONLY that. Prose,
    # <think> blocks, tool-use JSON and the prompt's own worked example all live OUTSIDE
    # the markers, so none of them is ever a candidate -- there is nothing left to
    # disambiguate. No markers -> no verdict (fail closed). See ``_extract_verdict``.
    parsed = _extract_verdict(text)

    try:
        # COMPACT, not ``indent=2``. Reading-as-text-first routes a deeply-nested valid
        # container here for the first time, and per-level indentation adds ``2 × depth``
        # spaces per element — a 16 KB fenced payload of nested arrays re-encoded to
        # ~128 MB (measured), an untrusted input below ``MAX_INPUT_FILE_SIZE`` defeating
        # the cap that exists to bound resource use, and a comb payload within the cap
        # projecting to a ``MemoryError`` the orchestrator does not retry. The output is
        # never read for its formatting — ``load_agent_output`` re-parses it — so the
        # indentation bought nothing. Compact makes the output O(input).
        payload = json.dumps(parsed)
    except RecursionError as exc:
        # The final encode still recurses per level (C encoder: ~15.5k deep on 3.14,
        # ~3.0k on 3.12 — see the table above), so an object that decoded cleanly can
        # still fail to re-encode. An escaping RecursionError is not caught by the
        # orchestrator's ``(ValidationError, JSONDecodeError)`` retry, so the mage would
        # be dropped without a second attempt. Map it, as ``_extract_verdict`` maps the
        # decode side.
        raise json.JSONDecodeError(
            "Recovered verdict is nested too deeply to re-encode", text, 0
        ) from exc

    # Encode BEFORE opening the file. Streaming straight into ``open(..., "w")`` left a
    # truncated artifact behind whenever the encoder raised mid-write — for a mage that
    # was then dropped, in the very run dir a reviewer is told to read.
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
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
