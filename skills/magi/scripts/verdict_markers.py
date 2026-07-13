# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""The verdict sentinel: delimits the agent's verdict (MS2).

The rule that carries ALL of this feature's security:

    **NORMALIZING inside an already-delimited region is allowed.
    SEARCHING outside the markers is FORBIDDEN, always.**

Any PR that adds a search outside the markers **reverts MS2**, however useful it may look:
the parser would go back to GUESSING which of the objects was the verdict, which is exactly
the defect this module exists to erase.

This module **delimits and nothing else**: it does not validate the 7-key schema (that is
``validate.py``), it does not speak HTTP and it does not launch agents. It is **pure** -- it
is tested with no network, no event loop and no disk.
"""

from __future__ import annotations

import re
import unicodedata

from validate import ValidationError

VERDICT_OPEN = "<MAGI_VERDICT>"
VERDICT_CLOSE = "</MAGI_VERDICT>"

#: Fingerprint of the example shipped in ``agents/*.md`` (R6). The three prompts share
#: these two values; they differ only in ``reasoning`` and in the finding's ``detail``.
#: Anchored by ``tests/test_agent_prompt_contract.py``: if someone edits a ``.md``'s
#: example without updating this, the canary would be left comparing against a text that
#: nobody emits any more -- a SILENT FAIL-OPEN.
ECHO_CANARY: dict[str, str] = {
    "summary": "One-line verdict",
    "recommendation": "What you recommend",
}

#: Unicode categories stripped before comparing a line against a marker.
#: ``Cf`` (format) covers ALL the invisibles -- including ``U+00AD`` (SOFT HYPHEN) and
#: ``U+180E`` (MONGOLIAN VOWEL SEPARATOR), which a hardcoded list left out.
#: ``Mn`` covers the variation selectors. **By CATEGORY, not by list: the category is
#: exhaustive and does not age.**
_STRIPPED_CATEGORIES = frozenset({"Cf", "Mn"})

#: The ONLY separators that count as a line ending when looking for a marker: the ones JSON
#: **escapes inside a string**. ``str.splitlines()`` also splits on ``\v``, ``\f``,
#: ``\x1c-\x1e``, ``U+0085``, ``U+2028`` and ``U+2029``; the last three are **legal raw
#: inside a JSON string**, so with ``splitlines`` a marker QUOTED in the payload could end
#: up alone on a line and cut the block short -- exactly the case that the line anchoring
#: promises is impossible, and exactly the one MAGI produces **when reviewing itself** (a
#: finding about the sentinel quotes the marker). Narrowing the separator set cannot open a
#: fail-open: a line containing ``U+2028`` does not normalize to the marker, so it can only
#: fail closed.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")

#: Opening fence: ``` or ~~~, with or without an info string. The info string is accepted
#: PERMISSIVELY -- any text after the marker, spaces included (``json``, ``json schema``,
#: ``json title="x"``). A whitelist would enumerate what is allowed, so every language with an
#: odd character (``c#``, ``asp.net``) and every model that writes two words would be a future
#: failure -- and each one costs a retry on a verdict that was never wrong (MAGI gate, Caspar,
#: twice). Permissiveness is FREE here: the fence is stripped only when the FIRST and the LAST
#: line are both fences, and what is inside is still decided by ``json.loads``.
#: **Permissive where it does not matter, strict where it does (the markers).**
_FENCE_OPEN_RE = re.compile(r"^\s*(```|~~~)[^`~]*$")

#: The CLOSING fence tolerates an info string too, because models echo the opening one on the
#: close (```json ... ```json). Demanding a bare fence there left both fence lines inside the
#: block, ``json.loads`` choked on them, and a verdict that was never wrong came back for a
#: retry (MAGI gate, Caspar). Being permissive costs nothing: a fence is only stripped when the
#: FIRST and the LAST line are both fences, and what is inside is still decided by
#: ``json.loads``. Permissive where it does not matter, strict where it does (the markers).
_FENCE_CLOSE_RE = _FENCE_OPEN_RE


class VerdictExtractionError(ValidationError):
    """Base of the verdict-extraction failures.

    **It inherits from ``ValidationError`` ON PURPOSE, and that is load-bearing.** The
    orchestrator's retry guard catches ``(ValidationError, json.JSONDecodeError)``, and
    here **the retry IS the fix**: the model can correct itself from the feedback.

    The fail-closed derogation of ``CLAUDE.local.md`` §0.2 (exceptions that are **siblings**
    of ``ValidationError``, not children) exists for the **opposite** case -- events the
    retry must NOT swallow -- and does **not** apply here. Compare ``PromptContractError``,
    which IS a sibling: a stale prompt **is not fixed by retrying**.

    **The rule, in one line: inherit from ``ValidationError`` if the retry fixes it;
    inherit from ``Exception`` if it does not.**
    """


class MissingVerdictMarkers(VerdictExtractionError):
    """No marker at all: the model did not emit the delimited block."""


class UnterminatedVerdictBlock(VerdictExtractionError):
    """An open with no close (or the reverse): the signature of a TRUNCATED output."""


class AmbiguousVerdictMarkers(VerdictExtractionError):
    """More than one block, or a close before an open.

    Fail-closed **with no tie-break**: choosing between two blocks would be a heuristic,
    and heuristics are exactly what MS2 erases.
    """


class EchoedExampleRejected(VerdictExtractionError):
    """The "verdict" is the system prompt's example, copied (R6)."""


class AgentIdentityError(VerdictExtractionError):
    """The verdict claims to be from a mage other than the one launched (R10)."""


class VerdictSentinel:
    """Delimits the verdict between two line-anchored markers.

    The TWO predicates live here together **on purpose**: their **asymmetry of trust** is
    the invariant, and splitting them into loose functions is exactly how it gets lost.

    ======================== ==================== =========================================
    Predicate                On what              Criterion
    ======================== ==================== =========================================
    :meth:`is_marker_line`   The MODEL's output   **PERMISSIVE**: normalizes invisibles,
                                                  trims spaces, ignores case. The model is
                                                  **untrusted** and we do not control its
                                                  output; killing it over a zero-width
                                                  space is a retry given away for free.
    :meth:`is_exact_marker_  OUR OWN ``.md``      **STRICT**: the line **is** the ASCII
    line`                                         marker. These are files **we ship**; an
                                                  invisible in there is **corruption**, and
                                                  it has to be seen.
    ======================== ==================== =========================================

    **A single predicate has already failed TWICE in this design:** shared and permissive,
    it let a corrupted ``.md`` through; shared and strict, it aborted the run with a false
    FATAL on a BOM. The fix was neither predicate: it was putting the BOM in the **encoding
    layer** (``utf-8-sig``), where it is resolved **before** anything is compared.

    Args:
        open_marker: Opening marker. Defaults to :data:`VERDICT_OPEN`.
        close_marker: Closing marker. Defaults to :data:`VERDICT_CLOSE`.
    """

    def __init__(self, open_marker: str = VERDICT_OPEN, close_marker: str = VERDICT_CLOSE) -> None:
        """Build a sentinel with the given marker pair.

        Args:
            open_marker: Opening marker.
            close_marker: Closing marker.
        """
        self.open = open_marker
        self.close = close_marker

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Canonical form of a line of the MODEL's output. **O(c)**, one pass.

        **This is the ONLY definition of "this line is a marker"** (DRY): it is consumed by
        :meth:`is_marker_line` and, further down, by the extraction. Duplicating it would
        plant the seed of the two diverging one day -- and *"two pieces deciding the same
        thing by different criteria"* is the bug this module has already suffered twice.

        Strips the characters in Unicode categories ``Cf`` and ``Mn`` (see
        :data:`_STRIPPED_CATEGORIES`), trims spaces and casefolds. Removing characters
        **can never fabricate** a marker where there was none, so the operation is safe in
        the direction that matters.

        **HOMOGLYPHS are left alone** (e.g. fullwidth ``U+FF1C``, category ``Sm``): they are
        not invisibles, they are **a different character**. Normalizing them would mean
        accepting as a marker something that **is not** the marker -- the class of laxity
        MS2 eliminates.

        Args:
            line: Candidate line (**untrusted** model output).

        Returns:
            The line with no invisibles, no surrounding spaces, and casefolded.
        """
        stripped = "".join(
            char for char in line if unicodedata.category(char) not in _STRIPPED_CATEGORIES
        )
        return stripped.strip().casefold()

    def is_marker_line(self, line: str, marker: str) -> bool:
        """PERMISSIVE -- for the MODEL's output (untrusted, not under our control).

        Args:
            line: One line of the agent's output.
            marker: :data:`VERDICT_OPEN` or :data:`VERDICT_CLOSE`.

        Returns:
            ``True`` if the line is that marker, tolerating invisibles, surrounding spaces
            and case drift.
        """
        return self._normalize_line(line) == marker.casefold()

    def is_exact_marker_line(self, line: str, marker: str) -> bool:
        """STRICT -- for OUR OWN ``.md`` (installation-time guard, R9).

        Args:
            line: One line of an ``agents/*.md`` that we ship.
            marker: :data:`VERDICT_OPEN` or :data:`VERDICT_CLOSE`.

        Returns:
            ``True`` only if the line, trimmed, **is** the ASCII marker. An invisible in
            there is file corruption, not tolerance that is owed.
        """
        return line.strip() == marker

    def extract(self, text: str) -> str:
        """Return the ONE delimited block. **Does NOT scan outside the markers.**

        **The ORDER of the first three checks is LOAD-BEARING.** The orchestrator picks the
        retry's feedback **by the exception TYPE**, so telling a model that **emitted no
        marker at all** *"you emitted more than one block"* spends the retry on a **false**
        instruction, and the mage dies from a bug in the very algorithm that exists to save
        it. (Found as a ``[CRITICAL]`` in review.)

        **Every** open and **every** close is counted, in **a single pass** and with **a
        single normalization per line**. **There is no "first close" nor "last close"**:
        choosing between them would be a tie-break rule, and tie-break rules are heuristics
        -- exactly what this module erases.

        Args:
            text: The agent's raw output (**untrusted** input).

        Returns:
            The content between the markers, with the markdown fence stripped if it wrapped
            the **whole** block.

        Raises:
            MissingVerdictMarkers: There is no marker at all (neither open nor close).
            UnterminatedVerdictBlock: Exactly one of the two is missing -- the signature of
                a **truncated** output.
            AmbiguousVerdictMarkers: The count is not exactly 1 and 1, or the close
                precedes the open.
        """
        # A "line" is what JSON **escapes inside a string**: ``\n``, ``\r\n``, ``\r``. NOT
        # ``str.splitlines()``, which also splits on ``\v``, ``\f``, ``\x1c-\x1e``,
        # ``U+0085``, ``U+2028`` and ``U+2029`` -- and the last three are **legal raw inside
        # a JSON string**. With splitlines, a finding ABOUT the sentinel that quoted the
        # marker behind a ``U+2028`` left it **alone on its own line** -> 2 closes -> the
        # mage died from an invisible separator. It failed closed, yes, but on a case the
        # guarantee above says cannot happen: the guarantee was wider than the code. Not
        # splitting on ``U+2028`` **cannot** open a fail-open (a line containing it does not
        # normalize to the marker).
        lines = _LINE_BREAK.split(text)

        opens: list[int] = []
        closes: list[int] = []
        want_open = self.open.casefold()
        want_close = self.close.casefold()
        for index, line in enumerate(lines):
            normalized = self._normalize_line(line)
            if normalized == want_open:
                opens.append(index)
            elif normalized == want_close:
                closes.append(index)

        if not opens and not closes:
            raise MissingVerdictMarkers(
                f"no verdict markers found: expected both {self.open!r} and {self.close!r}"
            )
        if not opens or not closes:
            raise UnterminatedVerdictBlock(
                f"unterminated verdict block: {len(opens)} open marker(s) and "
                f"{len(closes)} close marker(s) (likely a truncated response)"
            )
        if len(opens) != 1 or len(closes) != 1:
            raise AmbiguousVerdictMarkers(
                f"expected exactly one verdict block, found {len(opens)} open and "
                f"{len(closes)} close markers"
            )
        if closes[0] < opens[0]:
            raise AmbiguousVerdictMarkers(
                # +1: the message says "line N" to a person who is about to open the file at
                # line N, and every editor, grep and human counts from 1 (MAGI gate, Balthasar).
                f"close marker precedes the open marker (open at line {opens[0] + 1}, "
                f"close at line {closes[0] + 1})"
            )

        return self._strip_fence(lines[opens[0] + 1 : closes[0]])

    @staticmethod
    def _strip_fence(block_lines: list[str]) -> str:
        """Strip a markdown fence that wraps the **whole** block.

        **NORMALIZING INSIDE an already-delimited region is allowed; SEARCHING outside the
        markers, never.** If the fence does not wrap the complete block, the content is
        left **INTACT** and ``json.loads`` decides: trimming *"whatever is in the way"*
        until something decodes **would be searching again**.

        The close must be **of the SAME type** as the open (` ``` ` with ` ``` `, ``~~~``
        with ``~~~``). A **mismatched** pair is not a fence: it is text -- no markdown
        parser accepts it and no model emits it.

        Args:
            block_lines: The lines between the markers.

        Returns:
            The block as text, without the two fence lines if they were a valid pair.
        """
        body = list(block_lines)
        if len(body) < 2:
            return "\n".join(body).strip()

        opened = _FENCE_OPEN_RE.match(body[0])
        closed = _FENCE_CLOSE_RE.match(body[-1])
        if opened and closed and opened.group(1) == closed.group(1):
            body = body[1:-1]
        return "\n".join(body).strip()
