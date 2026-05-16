# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-16
"""Defense-in-depth user-prompt construction for MAGI orchestrator.

Sanitizes consumer-supplied content before embedding it in the LLM
user prompt. Four ordered layers: newline normalization, invisible
stripping, header neutralization, nonce-wrapped delimiters with a
fail-closed collision check.

Threat model and rationale: ``docs/python-prompt-hardening-port.md`` and
``docs/sbtdd/sanitize-spec-behavior.md``. The layer order is
load-bearing; see spec section 3.5 for the bypass analysis.
"""

from __future__ import annotations

import re
import secrets
from typing import Protocol

from validate import ValidationError


class _RngLike(Protocol):
    """Structural type for the injectable RNG.

    Only ``getrandbits(int) -> int`` is required. ``random.Random``,
    test doubles, and any object exposing the same single method
    satisfy the contract.
    """

    def getrandbits(self, k: int, /) -> int: ...


class InvalidInputError(ValidationError):
    """Raised when ``content`` cannot be safely embedded in a user prompt.

    The only current trigger is a nonce collision in
    :func:`build_user_prompt` (probability ~2^-128 per call with the
    default ``secrets`` RNG). The error message deliberately omits the
    nonce value: disclosing it would hand the attacker the very token
    they need to spoof the delimiters.
    """


# --- Layer 1: newline normalization ----------------------------------------

# CRLF listed before lone CR so the pair is consumed as a single unit.
# Codepoints expressed via ``\uXXXX`` escapes inside the raw-string
# regex: Python re engine interprets them even under ``r"..."`` strings,
# and the source file stays free of literal non-ASCII bytes so it
# survives any editor / shell / pipeline that mishandles UTF-8.
#     U+000B  VT  vertical tab
#     U+000C  FF  form feed
#     U+0085  NEL next line
#     U+2028  LS  line separator
#     U+2029  PS  paragraph separator
_NEWLINE_RE = re.compile(r"\r\n|\r|[\u000B\u000C\u0085\u2028\u2029]")


def normalize_newlines(s: str) -> str:
    r"""Convert every Unicode line separator in ``s`` to ``\n``.

    Recognised separators: ``\r\n``, ``\r``, U+000B (VT), U+000C (FF),
    U+0085 (NEL), U+2028 (LS), U+2029 (PS). ASCII tabs and spaces are
    not separators and pass through unchanged. Idempotent.
    """
    return _NEWLINE_RE.sub("\n", s)


# --- Layer 2: invisible-character stripping --------------------------------

# Parity with ``validate.py:_ZERO_WIDTH_RE``: same codepoint set kept in
# lockstep across the two modules. Bumping one without the other splits
# the contract.
#     U+200B..U+200F  zero-width spaces, ZWNJ, ZWJ, LRM, RLM
#     U+2028..U+202F  line/paragraph seps, bidi embedding, NNBSP
#     U+2060..U+206F  word joiner, invisible math operators, deprecated
#                     formatting and language-tag controls
#     U+FEFF          BOM / zero-width no-break space
#     U+00AD          soft hyphen
_INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u2028-\u202F\u2060-\u206F\uFEFF\u00AD]")


def strip_invisibles(s: str) -> str:
    """Remove zero-width, bidi, soft-hyphen, and Unicode separator codepoints.

    Operates on the post-:func:`normalize_newlines` string in the
    canonical pipeline; safe to call directly. Idempotent.
    """
    return _INVISIBLE_RE.sub("", s)


# --- Layer 3: header neutralization ----------------------------------------

# (?m) -- multiline; ^ matches start of string AND positions after \n.
# ([\t ]*) -- group 1: absorbs ASCII tabs/spaces so injections like
#             "   MODE: x" cannot bypass via leading whitespace.
# (MODE|CONTEXT|---BEGIN|---END) -- group 2: the four reserved keywords.
# (\s|:|$) -- group 3: separator after keyword. Without this, MODESTY,
#              CONTEXTUAL, ---BEGINNING would also match. The $ branch
#              lets a keyword sit at end-of-string.
# Substitution \1  \2\3 preserves original whitespace, injects "  ",
# preserves keyword and separator.
_HEADER_RE = re.compile(r"(?m)^([\t ]*)(MODE|CONTEXT|---BEGIN|---END)(\s|:|$)")


def neutralize_headers(s: str) -> str:
    r"""Insert a two-space prefix before lines starting with reserved keywords.

    Case-sensitive by design; see spec section 4 IS-NOT. Non-ASCII
    leading whitespace (NBSP, ideographic space) is NOT absorbed;
    documented gap, parity with the Rust implementation.
    """
    return _HEADER_RE.sub(r"\1  \2\3", s)


# --- Layer 4: nonce + delimiters + fail-closed -----------------------------


def build_user_prompt(
    mode: str,
    content: str,
    rng: _RngLike | None = None,
) -> str:
    """Build the canonical MAGI user prompt with defense-in-depth sanitization.

    Args:
        mode: One of ``"code-review"``, ``"design"``, ``"analysis"``.
            Not validated here -- the caller (argparse in
            ``run_magi.py``) owns that contract.
        content: Raw consumer-supplied content. May be adversarial.
        rng: Optional injectable RNG. Must expose
            ``getrandbits(int) -> int``. When ``None``, uses
            :func:`secrets.randbits` for cryptographic unpredictability.
            Production callers pass ``None``; tests inject a
            ``random.Random`` or a deterministic stub.

    Raises:
        InvalidInputError: If the generated nonce appears as a literal
            substring of the sanitized content. Probability ~2^-128 per
            call with the default RNG.

    Returns:
        The user prompt string ready to send to the LLM. Exactly four
        logical lines, no trailing newline.
    """
    step1 = normalize_newlines(content)
    step2 = strip_invisibles(step1)
    sanitized = neutralize_headers(step2)

    if rng is None:
        nonce_val = secrets.randbits(128)
    else:
        nonce_val = rng.getrandbits(128)
    nonce = f"{nonce_val:032x}"

    if nonce in sanitized:
        # Message must not include the nonce -- information disclosure.
        raise InvalidInputError("content contains generated nonce; refuse and retry")

    return (
        f"MODE: {mode}\n"
        f"---BEGIN USER CONTEXT {nonce}---\n"
        f"{sanitized}\n"
        f"---END USER CONTEXT {nonce}---"
    )
