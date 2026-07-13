# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Verdict sentinel suite (MS2).

**Pure**: no network, no ``asyncio``, no disk. The day an HTTP mock is needed here, the
module has lost its low coupling.
"""

import ast
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from validate import ValidationError
from verdict_markers import (
    _LINE_BREAK,
    VERDICT_CLOSE,
    VERDICT_OPEN,
    AgentIdentityError,
    AmbiguousVerdictMarkers,
    EchoedExampleRejected,
    MissingVerdictMarkers,
    UnterminatedVerdictBlock,
    VerdictExtractionError,
    VerdictSentinel,
)


class TestErrorHierarchy:
    def test_every_extraction_error_is_a_validation_error(self):
        """The orchestrator's retry catches ``(ValidationError, JSONDecodeError)``.

        If an extraction error did NOT inherit from ``ValidationError``, the guard would
        not catch it and the mage would **die instead of retrying** -- turning a
        RECOVERABLE failure into the death of the mage. The fail-closed derogation of
        ``CLAUDE.local.md`` section 0.2 does NOT apply here: **here the retry IS the fix**.
        """
        for cls in (
            MissingVerdictMarkers,
            UnterminatedVerdictBlock,
            AmbiguousVerdictMarkers,
            EchoedExampleRejected,
            AgentIdentityError,
        ):
            assert issubclass(cls, VerdictExtractionError)
            assert issubclass(cls, ValidationError)


class TestQualityContract:
    """The Quality section of ``~/.claude/CLAUDE.md``, made VERIFIABLE, not aspirational."""

    @staticmethod
    def _source() -> str:
        import verdict_markers

        return Path(verdict_markers.__file__).read_text(encoding="utf-8")

    def test_the_sentinel_imports_no_io_and_no_orchestrator(self):
        """Low coupling: the module can be tested -- and broken -- ALONE.

        The day someone imports ``urllib`` or ``run_magi`` here, the module stops being
        pure and stops being testable without a network. This test prevents it **today**,
        not once it already hurts.
        """
        source = self._source()
        for forbidden in (
            "urllib",
            "asyncio",
            "requests",
            "run_magi",
            "parse_agent_output",
        ):
            assert f"import {forbidden}" not in source

    def test_no_magic_numbers_in_function_bodies(self):
        """Named constants: zero **semantic** numeric literals in the body.

        **Indexing** literals are allowed (``body[0]``, ``body[-1]``, ``len(body) < 2``):
        they are not magic numbers, they are **the arithmetic of a list**. A magic number
        is one that encodes a **decision** -- 2000 probes, 400 chars, 4 tokens/char -- and
        not one of those is left here.
        """
        indexing = {-1, 0, 1, 2}
        tree = ast.parse(self._source())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            magic = [
                n.value
                for n in ast.walk(fn)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)
                and n.value not in indexing
            ]
            assert not magic, f"{fn.name} tiene numeros magicos: {magic}"


class TestMarkerLinePredicates:
    """PERMISSIVE -- for the MODEL's output (untrusted, and not ours to control)."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    def test_a_plain_marker_line_is_recognized(self):
        assert self.sentinel.is_marker_line(VERDICT_OPEN, VERDICT_OPEN)

    def test_surrounding_whitespace_is_tolerated(self):
        assert self.sentinel.is_marker_line("   <MAGI_VERDICT>  ", VERDICT_OPEN)

    def test_crlf_line_endings_are_tolerated(self):
        """Windows exists, and the model may emit \\r\\n."""
        assert self.sentinel.is_marker_line("<MAGI_VERDICT>\r\n", VERDICT_OPEN)

    # Invisibles go in as ESCAPES, NEVER as the literal character: a literal ZWSP is
    # **invisible to the reviewer**, and an editor or a copy-paste can delete it in
    # silence -- leaving a test that compares "<MAGI_VERDICT>" against "<MAGI_VERDICT>" and
    # **passes without testing anything**. The escape is visible, reviewable, and cannot
    # get lost.
    @pytest.mark.parametrize(
        "invisible",
        [
            "​",  # ZERO WIDTH SPACE
            "‍",  # ZERO WIDTH JOINER
            "﻿",  # BOM
            "⁠",  # WORD JOINER
            "­",  # SOFT HYPHEN               <- a hardcoded list left this one out
            "᠎",  # MONGOLIAN VOWEL SEPARATOR <- same
            "\ufe0f",  # VARIATION SELECTOR-16 (Mn) -- ESCAPED, per the rule above
        ],
    )
    def test_an_invisible_inside_the_marker_does_not_kill_the_mage(self, invisible):
        """An invisible in the MODEL's output is a retry given away for free."""
        assert self.sentinel.is_marker_line(f"<MAGI{invisible}_VERDICT>", VERDICT_OPEN)

    def test_case_drift_is_tolerated_from_the_model(self):
        """A model that writes ``<magi_verdict>`` DID EMIT the marker."""
        assert self.sentinel.is_marker_line("<magi_verdict>", VERDICT_OPEN)

    def test_a_fullwidth_homoglyph_is_NOT_a_marker(self):
        """A homoglyph is not an invisible: it is **another character**. It fails closed."""
        assert not self.sentinel.is_marker_line("＜MAGI_VERDICT>", VERDICT_OPEN)

    def test_prose_mentioning_the_marker_is_not_a_marker_line(self):
        assert not self.sentinel.is_marker_line("emit <MAGI_VERDICT> at the end", VERDICT_OPEN)

    def test_the_close_marker_is_not_the_open_marker(self):
        assert not self.sentinel.is_marker_line(VERDICT_CLOSE, VERDICT_OPEN)
        assert self.sentinel.is_marker_line(VERDICT_CLOSE, VERDICT_CLOSE)


class TestExactMarkerLinePredicate:
    """STRICT -- for OUR OWN .md files: an invisible there is CORRUPTION, not tolerance.

    It is the exact reverse of the permissive predicate, and the asymmetry **is** the
    invariant: the model's output is not ours to control; our own files are.
    """

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    def test_our_own_file_must_carry_the_exact_ascii_marker(self):
        assert self.sentinel.is_exact_marker_line(VERDICT_OPEN, VERDICT_OPEN)
        assert self.sentinel.is_exact_marker_line("  <MAGI_VERDICT>  ", VERDICT_OPEN)

    def test_an_invisible_in_OUR_file_is_corruption_and_is_rejected(self):
        assert not self.sentinel.is_exact_marker_line("<MAGI​_VERDICT>", VERDICT_OPEN)

    def test_case_drift_in_OUR_file_is_rejected(self):
        assert not self.sentinel.is_exact_marker_line("<magi_verdict>", VERDICT_OPEN)


VERDICT = '{"agent": "caspar", "verdict": "reject"}'


def _block(body: str) -> str:
    """Wrap *body* in the delimited block that MS2 demands."""
    return f"{VERDICT_OPEN}\n{body}\n{VERDICT_CLOSE}"


#: A **fabricated** verdict: exactly what the pre-MS2 heuristic scanner would have picked
#: up from OUTSIDE the markers (the system prompt's own example, an ``approve`` in the
#: adversarial seat). The property's noise carries it on purpose: innocent noise has nothing
#: to scan, and a regression back to the scanner would **survive** the test.
_FABRICATED_APPROVE = json.dumps(
    {"agent": "caspar", "verdict": "approve", "confidence": 0.85, "summary": "One-line verdict"}
)

#: HOSTILE noise for the safety property: arbitrary text, a loose fabricated verdict, an
#: orphan marker, or the combination. It is what makes the property **kill** a tie-break
#: rule and a return to the scanner, instead of waving them through.
_NOISE = st.one_of(
    st.text(),
    st.just(_FABRICATED_APPROVE),
    st.just(VERDICT_OPEN),
    st.just(VERDICT_CLOSE),
    st.just(f"prosa\n{_FABRICATED_APPROVE}\nmas prosa"),
    st.just(_block(_FABRICATED_APPROVE)),
)


def _has_marker_line(text: str) -> bool:
    """Is any line of *text* a marker (using the parser's permissive predicate)?"""
    sentinel = VerdictSentinel()
    return any(
        sentinel.is_marker_line(line, VERDICT_OPEN) or sentinel.is_marker_line(line, VERDICT_CLOSE)
        for line in _LINE_BREAK.split(text)
    )


#: Noise with NO marker at all -- but with a fabricated verdict inside, which is what a
#: scanner would pick up. It is the input that kills the pre-MS2 residual.
_MARKERLESS_NOISE = st.one_of(
    st.text(),
    st.just(_FABRICATED_APPROVE),
    st.just(f"prosa\n{_FABRICATED_APPROVE}\nmas prosa"),
    st.just(f"<think>razono</think>\n```json\n{_FABRICATED_APPROVE}\n```"),
).filter(lambda text: not _has_marker_line(text))


class TestExtract:
    """The ORDER of the checks is load-bearing: it selects the retry's feedback."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    def test_extracts_the_block_and_ignores_everything_outside(self):
        raw = f"<think>reasoning for a while</think>\nprosa\n{_block(VERDICT)}\nmas prosa"
        assert json.loads(self.sentinel.extract(raw))["agent"] == "caspar"

    def test_zero_markers_raises_MISSING_not_ambiguous(self):
        """The ``[CRITICAL]`` from cycle 6: the TYPE of exception selects the feedback.

        Telling *"you emitted more than one block"* to a model that emitted NO marker at
        all burns the retry on a **false** instruction, and the mage dies from a bug in the
        very algorithm that exists to save it.
        """
        with pytest.raises(MissingVerdictMarkers):
            self.sentinel.extract(VERDICT)  # perfect JSON, but with NO markers

    def test_an_open_without_a_close_is_a_truncated_output(self):
        with pytest.raises(UnterminatedVerdictBlock):
            self.sentinel.extract(f"{VERDICT_OPEN}\n{VERDICT}")

    def test_a_close_without_an_open_is_a_truncated_output(self):
        with pytest.raises(UnterminatedVerdictBlock):
            self.sentinel.extract(f"{VERDICT}\n{VERDICT_CLOSE}")

    def test_two_blocks_fail_closed_without_a_tie_break(self):
        """The echoed example + the real verdict. One is **NEVER** picked."""
        with pytest.raises(AmbiguousVerdictMarkers):
            self.sentinel.extract(f"{_block(VERDICT)}\n{_block(VERDICT)}")

    def test_nested_open_markers_fail_closed(self):
        raw = f"{VERDICT_OPEN}\n{VERDICT_OPEN}\n{VERDICT}\n{VERDICT_CLOSE}"
        with pytest.raises(AmbiguousVerdictMarkers):
            self.sentinel.extract(raw)

    def test_a_close_before_the_open_fails_closed(self):
        with pytest.raises(AmbiguousVerdictMarkers):
            self.sentinel.extract(f"{VERDICT_CLOSE}\n{VERDICT}\n{VERDICT_OPEN}")

    def test_a_marker_quoted_inside_the_json_does_not_truncate_the_block(self):
        """MAGI reviews itself: a finding ABOUT the sentinel quotes the marker.

        JSON **escapes newlines**, so a marker quoted inside the payload **cannot** appear
        alone on a line of its own -> the line anchoring makes it harmless.
        """
        payload = json.dumps(
            {"agent": "caspar", "detail": "the close </MAGI_VERDICT> is line-anchored"}
        )
        assert json.loads(self.sentinel.extract(_block(payload)))["agent"] == "caspar"

    def test_a_raw_newline_inside_a_string_yields_two_closes_and_fails_closed(self):
        """A model emitting invalid JSON with a RAW newline -> 1 open, 2 closes."""
        body = '{"detail": "roto\n</MAGI_VERDICT>\nsigue"}'
        with pytest.raises(AmbiguousVerdictMarkers):
            self.sentinel.extract(_block(body))

    @pytest.mark.parametrize("separator", [" ", " ", ""])
    def test_a_JSON_LEGAL_line_separator_does_not_cut_the_block(self, separator):
        """A VALID verdict that quotes the marker behind a separator **legal in JSON**.

        ``str.splitlines()`` also splits on ``\\v``, ``\\f``, ``\\x1c-\\x1e``, ``U+0085``,
        ``U+2028`` and ``U+2029`` -- and the last three **are legal raw inside a JSON
        string** (``json.loads`` accepts them). A finding about the sentinel that quotes
        the marker behind one of them leaves `</MAGI_VERDICT>` **alone on its own line** ->
        2 closes -> `AmbiguousVerdictMarkers` -> the mage dies from an invisible separator.

        It fails closed (it never fabricates), but the guarantee the docstring promises
        --*"JSON escapes newlines, so a quoted marker cannot end up alone on a line"*-- is
        **wider than the code**: it holds for ``\\n``, not for the whole separator set of
        ``splitlines()``. And the scenario is the one MAGI produces **when reviewing
        itself**.
        """
        detail = (
            f"the finding quotes the marker:{separator}</MAGI_VERDICT>{separator}and it goes on"
        )
        payload = json.dumps({"agent": "caspar", "detail": detail}, ensure_ascii=False)
        json.loads(payload)  # premise: the RAW separator is valid JSON

        extracted = self.sentinel.extract(_block(payload))

        assert json.loads(extracted)["detail"] == detail

    def test_extract_documents_every_cause_it_raises(self):
        """The ``Raises:`` is LOAD-BEARING: an incomplete one is a blind retry."""
        doc = VerdictSentinel.extract.__doc__ or ""
        assert "Raises:" in doc
        for cause in (
            "MissingVerdictMarkers",
            "UnterminatedVerdictBlock",
            "AmbiguousVerdictMarkers",
        ):
            assert cause in doc


class TestFenceNormalization:
    """Normalizing INSIDE an already-delimited region: allowed. Searching outside: never."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    @pytest.mark.parametrize(
        "opener", ["```json", "```", "~~~json", "``` json", "```json  ", "```json5"]
    )
    def test_a_fence_around_the_json_is_stripped(self, opener):
        """glm-5.2 fences out of habit. Failing over a SPACE would be fragile, not strict."""
        closer = opener[:3]
        raw = _block(f"{opener}\n{VERDICT}\n{closer}")
        assert json.loads(self.sentinel.extract(raw))["agent"] == "caspar"

    def test_text_between_the_fence_and_the_json_is_left_INTACT(self):
        """Trimming "what gets in the way" until something decodes would be SEARCHING AGAIN."""
        raw = _block(f"```json\naqui va mi veredicto:\n{VERDICT}\n```")
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.sentinel.extract(raw))

    def test_a_MISMATCHED_fence_pair_is_not_a_fence(self):
        """``` opened and ~~~ closed is not a fence: it is text. It is left intact."""
        raw = _block(f"```json\n{VERDICT}\n~~~")
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.sentinel.extract(raw))

    def test_each_line_is_normalized_ONCE_not_twice(self, monkeypatch):
        """The plan's O(N) bound, made EXECUTABLE instead of merely promised.

        The obvious version of ``extract`` (two comprehensions, one per marker) normalizes
        **each line TWICE**: same O(N), twice the work, and for nothing. This test makes
        that regression **break the build** instead of slipping by unnoticed.
        """
        calls = 0
        original = VerdictSentinel._normalize_line

        def counting(line: str) -> str:
            nonlocal calls
            calls += 1
            return original(line)

        monkeypatch.setattr(VerdictSentinel, "_normalize_line", staticmethod(counting))
        raw = _block(VERDICT)
        VerdictSentinel().extract(raw)
        # Counted with the SAME notion of a line that ``extract`` uses (``_LINE_BREAK``), not
        # with ``splitlines()``: two different criteria for "what is a line" is how a test and
        # its implementation drift apart without anyone noticing.
        assert calls == len(_LINE_BREAK.split(raw))  # ONE per line, not one more


class TestExtractProperties:
    """hypothesis: properties over GENERATED inputs, not examples (section 0.3)."""

    @given(_NOISE, _NOISE, st.dictionaries(st.text(min_size=1), st.text(), min_size=1))
    def test_nothing_from_OUTSIDE_the_markers_can_reach_the_output(self, prefix, suffix, obj):
        """THE safety property of MS2 -- and now it is actually EXERCISED.

        Two earlier versions of this test **proved nothing**, each in its own way, and both
        passed green:

        1. It generated plain ``st.text()`` and only asserted **if** the text happened to
           carry a well-formed block. Measured: **0 of 2000 examples** reached the assertion
           -- hypothesis does not produce ``<MAGI_VERDICT>`` alone on its line by chance. It
           was ``assert True`` wearing the shape of a property, over the milestone's nuclear
           invariant.
        2. It did build the block, but the **noise outside was innocent text**: never a
           marker, never a decodable JSON. Measured by mutation: two critical regressions
           **survived** -- a tie-break rule ("the last close wins") and **the scanner over
           the whole text**, which is *literally the pre-MS2 residual*. The noise never gave
           them anything to scan, nor any tie to break.

        Now the noise is **hostile by construction** (:data:`_NOISE`): a FABRICATED
        ``approve`` verdict, loose markers, or both. And failing closed is no longer a free
        pass: if ``extract`` raises, the noise is required to **actually have carried** a
        marker line -- otherwise it is a mage killed for nothing, and the test says so.
        """
        sentinel = VerdictSentinel()
        raw = f"{prefix}\n{_block(json.dumps(obj))}\n{suffix}"
        noise_carries_a_marker = _has_marker_line(prefix) or _has_marker_line(suffix)

        try:
            block = sentinel.extract(raw)
        except VerdictExtractionError:
            # Failing closed is NOT a free pass: it only counts if the noise really carried
            # a marker. If it did not, it is a mage killed for no cause.
            assert noise_carries_a_marker, "fail-closed on noise carrying NO marker at all"
            return

        # The other half of R2, and the one a weakened counting guard skips: one marker too
        # many **fails closed**, it is never "resolved" by picking a block. Without this
        # assertion, an ``extract`` that accepted 2 closes and kept the first one passed the
        # test -- measured by mutation. **Any** tie-break rule is a heuristic in disguise.
        assert not noise_carries_a_marker, (
            "a marker too many MUST fail closed -- resolving it is a tie-break rule"
        )
        # The oracle reconstructs nothing: we know **by construction** that ``obj`` is what
        # sits between the markers. If anything from outside leaked in -the fabricated
        # approve, a neighbouring line-, the block would not decode to ``obj``.
        assert json.loads(block) == obj

    @given(_MARKERLESS_NOISE)
    def test_a_text_with_NO_markers_never_yields_a_verdict(self, noise):
        """The OTHER half of the invariant, and the one that kills the pre-MS2 residual.

        The property above **always** builds a block, so the *"there are no markers"* branch
        -the one the heuristic scanner came in through- **was never exercised**. Measured by
        mutation: an ``extract`` that, seeing no markers, scans the whole text and returns
        the first JSON object that decodes -*exactly* the residual MS2 deletes, the one that
        fabricated an ``approve`` in the adversarial seat- **survived** the test. Only an
        example test killed it; now the property does too.

        With no markers there **is no verdict**, however decodable the text may come (R15).
        """
        with pytest.raises(VerdictExtractionError):
            VerdictSentinel().extract(noise)

    @given(st.text())
    def test_never_raises_anything_but_a_VerdictExtractionError(self, noise):
        """Arbitrary input NEVER triggers an uncontrolled exception."""
        try:
            VerdictSentinel().extract(noise)
        except VerdictExtractionError:
            pass

    @given(st.dictionaries(st.text(min_size=1), st.text(), min_size=1))
    def test_a_verdict_wrapped_in_markers_round_trips_EXACTLY(self, obj):
        """KILLS the cheating implementation the previous property left alive.

        An ``extract`` that **always returned the empty string** would pass ``block in
        intra`` -- ``"" in anything`` is **True**. The no-leak property is necessary but
        **not sufficient**.
        """
        raw = f"prosa\n<think>noise</think>\n{_block(json.dumps(obj))}\nmas prosa"
        assert json.loads(VerdictSentinel().extract(raw)) == obj
