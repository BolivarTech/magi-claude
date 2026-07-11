#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.2.0
# Date: 2026-07-11
"""Parse and validate an agent's JSON verdict from any supported backend.

Extracts the structured verdict from every shape a backend can produce —
the Claude CLI's transport envelopes, and the Ollama backend's *unwrapped*
content, whether bare or wrapped in a markdown fence (4.0.6) — strips those
fences, and recovers the verdict even when an agent buries it in prose (2.4.2).

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

    # Bare-verdict dict: Ollama backend returns ``choices[0].message.content``
    # already decoded as the 7-key agent JSON object.  Serialize it back so the
    # rest of the pipeline (fence-strip → lenient parse → validate) is unchanged.
    if isinstance(data, dict) and "agent" in data and "verdict" in data:
        return json.dumps(data)

    raise ValueError(
        f"Unexpected agent output type: {type(data).__name__}. "
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


def _is_verdict_shaped(candidate: object) -> bool:
    """Whether *candidate* has the shape of an agent verdict: the two keys, no further conditions.

    Deliberately **key-only**. Every attempt to make it smarter has been a fail-open,
    and 4.0.6 made three of them in three consecutive review rounds before the
    evidence settled it. This docstring is the record, because the next person to
    "harden" this function will feel exactly as certain as I did.

    **The trap.** This predicate does not only *select* the verdict — it also feeds the
    **ambiguity guard** (``_embedded_verdict_object``: two candidates ⇒ fail closed).
    So every exclusion added here narrows the guard: fewer candidates ⇒ fewer ambiguity
    trips ⇒ **more single-match fabrications**. *Widening the exclusion looks like
    tightening the check*, because the guard you are disarming is invisible in the line
    you are editing.

    **The three attempts, all rejected:**

    ================  ===============================  ==============================
    Attempt           Exclusion added                  What stopped being a rival
    ================  ===============================  ==============================
    enum member       ``verdict in VALID_VERDICTS``    ``"Reject"`` (case drift)
    pipe-union regex  any ``a | b`` union              ``"approve | conditional"``
    type guard        ``isinstance(verdict, str)``     ``null``, ``["reject"]``, ``0``
    ================  ===============================  ==============================

    In each one the excluded object was **the mage's real verdict**; the echoed
    system-prompt example (which literally carries ``"verdict": "approve"``) became the
    sole match; and consensus received a schema-perfect fabricated ``approve`` **from
    the adversarial seat** — where the previous behaviour was a clean fail-closed drop
    and a retry.

    **Why no exclusion at all, not even a "correct" one.** The three attempts were all
    trying to disqualify a *schema restatement* — the object a thinking model emits when
    it quotes its own schema (``"verdict": "approve | reject | conditional"``), which
    counts as a rival candidate and drops the mage. A restatement exclusion derived from
    ``VALID_VERDICTS`` was written, and it was correct as far as it went. It was removed
    anyway, on evidence:

    * Across **171 captured agent outputs** from the real default trio, it changed the
      outcome of **zero**. Not one payload contained a restatement rival.
    * Reverting it left the **entire suite green** — it was unpinned by any positive test.
    * It matched only **one of six** plausible spellings of a restatement (the agent
      prompts actually teach the comma form, ``"approve", "reject", or "conditional"``).
    * And it had a **real, verified cost**: it removed an *accidental* rival, so a
      payload with a restatement beside an echoed example — which used to fail closed —
      fabricated an ``approve`` instead.

    A guard-narrowing whose benefit no test and no artifact can demonstrate, paid for in
    the one currency this system cannot afford, is not a hardening. The mage drop it
    aimed at **fails closed** (degraded run ⇒ by the Integrity rule, approves nothing);
    the fabrication it introduced fails **open**, and silently. Given the choice, take
    the loud failure every time.

    The durable fix for both is the verdict **sentinel** (MS2): stop *searching* for the
    verdict and *extract* it from between markers. Then a restatement, an echo, and a
    tool-use blob are all simply outside the markers, and none of this is a judgement
    call. See :func:`_embedded_verdict_object` and ``CLAUDE.techdebt.md``.

    Args:
        candidate: A JSON value decoded from the agent's output.

    Returns:
        True if it has the shape of a genuine verdict, False otherwise.
    """
    if not isinstance(candidate, dict):
        return False
    # ONE condition. Do not add a second — see the table above; each of the three
    # attempts read as a harmless tightening and each one fabricated an ``approve``.
    return all(key in candidate for key in _VERDICT_KEYS)


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
    ``approve`` can reach consensus. The durable fix is the verdict
    **sentinel** — scheduled as **MS2** (``sbtdd/spec-behavior-base-MS2.md``;
    debt in ``CLAUDE.techdebt.md``). Do not add more heuristic tuning here, and
    do not try to make :func:`_is_verdict_shaped` smarter either: 4.0.6 attempted
    that three times and produced a fail-open every time. The next change here is
    the sentinel.

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
        if _is_verdict_shaped(candidate):
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
    except (json.JSONDecodeError, RecursionError):
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
        # back the retry, exactly as ``_loads_lenient`` does on the decode side.
        #
        # Scope of the behaviour change: a WELL-FORMED envelope is untouched (it
        # parses here exactly as before). A *malformed* Claude envelope, which used
        # to raise immediately, now goes through prose-recovery and could succeed.
        # That path is practically unreachable — a non-zero ``claude -p`` exit is
        # caught before the parse, and ``--output-format json`` always emits an
        # envelope — but the honest claim is "no change for well-formed envelopes",
        # not "no change on the Claude path".
        data = raw  # not an envelope: fenced or prose-wrapped content

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
    except RecursionError as exc:
        # A SECOND encode site, on a DIFFERENT route. ``_extract_text``'s bare-verdict
        # branch re-serialises with ``json.dumps``, so a BARE (unfenced) verdict blows up
        # here, while a FENCED one reaches the encode further down instead. Mapping only
        # the fenced route left the plainest Ollama payload there is escaping, and the
        # mage losing the retry that ``run_magi``'s ``(ValidationError, JSONDecodeError)``
        # guard would otherwise have given it.
        #
        # MEASURED, because two earlier versions of this comment reasoned instead and were
        # both wrong (max nesting depth before RecursionError):
        #
        #                    decoder   dumps()   dumps(indent=2)
        #     CPython 3.14     16909     15500     15500   <- C encoder handles indent
        #     CPython 3.12      2997      2997       993   <- pure-Python for indent
        #
        # So there is no clean "encoders are weaker than the decoder" rule to lean on: on
        # 3.14 both encoders are, on 3.12 only the indent one is (and this catch never
        # fires there — anything that decodes also dumps). Which catch fires depends on
        # the interpreter AND the route, which is exactly why both exist and why neither
        # may be deleted as redundant.
        raise json.JSONDecodeError(
            "Agent output is nested too deeply to re-serialise", raw, 0
        ) from exc

    text = _strip_code_fences(text)

    # Validate that the cleaned text is valid JSON. Agents that do
    # multi-turn tool use sometimes wrap the verdict in prose, so a strict
    # parse falls back to recovering the embedded object; output with no
    # JSON object at all still raises (fail closed). See ``_loads_lenient``.
    parsed = _loads_lenient(text)

    try:
        payload = json.dumps(parsed, indent=2)
    except RecursionError as exc:
        # The final encode, reached by fenced and prose-wrapped payloads. ``indent=2``
        # is the weakest of the JSON calls on BOTH supported interpreters (see the
        # measured table above: 15500 on 3.14, 993 on 3.12), so an object that decoded
        # cleanly can still fail to re-encode here. An escaping RecursionError is not
        # caught by the orchestrator's ``(ValidationError, JSONDecodeError)`` retry, so
        # the mage would be dropped without a second attempt. Map it, exactly as
        # ``_loads_lenient`` maps the decode side.
        raise json.JSONDecodeError(
            "Recovered verdict is nested too deeply to re-encode", text, 0
        ) from exc

    # Encode BEFORE opening the file. Streaming straight into ``open(..., "w")`` left a
    # truncated ~1 MB artifact behind whenever the encoder raised mid-write — for a mage
    # that was then dropped, in the very run dir a reviewer is told to read.
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
