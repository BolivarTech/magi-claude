# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""Tests for parse_agent_output.py — Claude CLI JSON extraction."""

import json
import os
import tempfile

import pytest

from parse_agent_output import _strip_code_fences, _extract_text, parse_agent_output


class TestStripCodeFences:
    """Verify markdown code fence removal."""

    def test_no_fences_unchanged(self):
        assert _strip_code_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fences_stripped(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_bare_fences_stripped(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_uppercase_json_fences_stripped(self):
        text = '```JSON\n{"key": "value"}\n```'
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_fences_with_surrounding_whitespace(self):
        text = '  ```json\n{"key": "value"}\n```  '
        assert _strip_code_fences(text) == '{"key": "value"}'

    def test_nested_backticks_in_content_preserved(self):
        text = '```json\n{"code": "use `var`"}\n```'
        result = _strip_code_fences(text)
        assert "`var`" in result


class TestExtractText:
    """Verify text extraction from various Claude CLI output formats."""

    def test_result_format(self):
        data = {"result": '{"agent": "melchior", "verdict": "approve"}'}
        assert _extract_text(data) == '{"agent": "melchior", "verdict": "approve"}'

    def test_content_block_format(self):
        data = {
            "content": [{"type": "text", "text": '{"agent": "balthasar", "verdict": "reject"}'}]
        }
        assert _extract_text(data) == '{"agent": "balthasar", "verdict": "reject"}'

    def test_content_block_skips_non_text(self):
        data = {
            "content": [
                {"type": "image", "url": "http://example.com"},
                {"type": "text", "text": "extracted"},
            ]
        }
        assert _extract_text(data) == "extracted"

    def test_content_block_no_text_raises(self):
        data = {"content": [{"type": "image", "url": "http://example.com"}]}
        with pytest.raises(ValueError, match="No text block"):
            _extract_text(data)

    def test_content_must_be_a_list(self):
        """A non-list ``content`` value must be rejected, not silently
        iterated character-by-character."""
        data = {"content": "not-a-list"}
        with pytest.raises(ValueError, match="'content' must be a list"):
            _extract_text(data)

    def test_content_dict_not_accepted(self):
        """A dict under ``content`` would silently iterate its keys; reject it."""
        data = {"content": {"type": "text", "text": "inline"}}
        with pytest.raises(ValueError, match="'content' must be a list"):
            _extract_text(data)

    def test_plain_string(self):
        assert _extract_text("hello world") == "hello world"

    def test_bare_verdict_dict_serialized_to_json(self):
        """_extract_text must serialize a bare verdict dict to a valid JSON string."""
        verdict = {
            "agent": "melchior",
            "verdict": "approve",
            "confidence": 0.8,
            "summary": "s",
            "reasoning": "r",
            "findings": [],
            "recommendation": "go",
        }
        result = _extract_text(verdict)
        parsed = json.loads(result)
        assert parsed["agent"] == "melchior"
        assert parsed["verdict"] == "approve"

    def test_fallback_dict_raises_value_error(self):
        data = {"unknown_key": "some_value"}
        with pytest.raises(ValueError, match="Unexpected agent output type"):
            _extract_text(data)

    def test_result_key_takes_precedence_over_content(self):
        data = {
            "result": "from_result",
            "content": [{"type": "text", "text": "from_content"}],
        }
        assert _extract_text(data) == "from_result"


def _write_temp(content: str, *, suffix: str = ".json") -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _sample_agent_payload() -> dict:
    """A minimal but schema-complete agent verdict for prose-wrapping tests.

    ``findings`` is left empty on purpose so the only JSON object in the
    serialised payload is the verdict itself — that lets the truncation
    test assert "no recoverable sub-object survives" without a nested
    finding accidentally decoding.
    """
    return {
        "agent": "melchior",
        "verdict": "conditional",
        "confidence": 0.82,
        "summary": "API-correct plan, ready after minor fixes.",
        "reasoning": "Cross-checked every load-bearing call against the source.",
        "findings": [],
        "recommendation": "Ship after fixing the dependency-graph error.",
    }


class TestParseAgentOutput:
    """Integration tests for the full parse pipeline."""

    def test_result_format_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.9,
                "summary": "OK",
                "reasoning": "Fine",
                "findings": [],
                "recommendation": "Merge",
            }
        )
        raw = json.dumps({"result": agent_json})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "melchior"
            assert result["verdict"] == "approve"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_content_block_format_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "caspar",
                "verdict": "reject",
                "confidence": 0.7,
                "summary": "Bad",
                "reasoning": "Risky",
                "findings": [],
                "recommendation": "Rework",
            }
        )
        raw = json.dumps({"content": [{"type": "text", "text": agent_json}]})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "caspar"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_code_fenced_result_end_to_end(self):
        agent_json = json.dumps(
            {
                "agent": "balthasar",
                "verdict": "conditional",
                "confidence": 0.8,
                "summary": "Maybe",
                "reasoning": "Depends",
                "findings": [],
                "recommendation": "Add tests",
            }
        )
        fenced = f"```json\n{agent_json}\n```"
        raw = json.dumps({"result": fenced})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert result["agent"] == "balthasar"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_invalid_json_raises(self):
        raw = json.dumps({"result": "not valid json at all"})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(input_path, output_path)
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_missing_input_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_agent_output("/nonexistent/input.json", "/tmp/out.json")

    def test_output_has_trailing_newline(self):
        agent_json = json.dumps(
            {
                "agent": "melchior",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "Good",
                "reasoning": "Clean",
                "findings": [],
                "recommendation": "Ship",
            }
        )
        raw = json.dumps({"result": agent_json})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path) as f:
                content = f.read()
            assert content.endswith("\n")
        finally:
            os.unlink(input_path)
            os.unlink(output_path)


class TestOllamaFencedContent:
    """4.0.6: an Ollama model that fences its verdict must not kill its mage.

    The Ollama backend returns ``choices[0].message.content`` **already unwrapped**
    — the raw file therefore holds the agent's verdict *itself*, not an envelope
    around it. Many models emit that verdict inside a markdown fence, so the raw
    file starts with ``` and is not JSON at the top level.

    Before 4.0.6, ``parse_agent_output`` called ``json.load(fh)`` **before**
    stripping fences. On the Claude path that is fine (the raw *is* an envelope).
    On the Ollama path it blew up at character 0, the mage was retried, the retry
    produced the same (perfectly valid) fenced verdict, and the mage was dropped —
    a degraded run whose verdict, by the Integrity rule, approves nothing.

    The fence-stripping code existed all along. It was simply unreachable on the
    one path that needed it: it ran *after* a parse that could never succeed.

    Observed 2026-07-11: glm-5.2 emitted a schema-perfect 7-key verdict with 7
    findings, fenced; MAGI discarded it twice and reported a degraded run.
    """

    def test_bare_fenced_verdict_is_parsed(self):
        """The exact shape that killed Caspar: a fenced verdict, no envelope."""
        payload = _sample_agent_payload()
        raw = f"```json\n{json.dumps(payload)}\n```"
        in_path = _write_temp(raw)
        out_path = _write_temp("", suffix=".out.json")
        try:
            parse_agent_output(in_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                result = json.load(f)
            assert result == payload
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_bare_fenced_verdict_without_language_tag(self):
        """A bare ``` fence is as common as ```json — both must work."""
        payload = _sample_agent_payload()
        raw = f"```\n{json.dumps(payload)}\n```"
        in_path = _write_temp(raw)
        out_path = _write_temp("", suffix=".out.json")
        try:
            parse_agent_output(in_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                assert json.load(f) == payload
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_bare_unfenced_verdict_still_works(self):
        """Regression: models that emit bare JSON kept working (the 4.0.0 path)."""
        payload = _sample_agent_payload()
        in_path = _write_temp(json.dumps(payload))
        out_path = _write_temp("", suffix=".out.json")
        try:
            parse_agent_output(in_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                assert json.load(f) == payload
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_unparseable_content_still_fails_closed(self):
        """The fix must not turn "unparseable" into "silently accepted".

        Widening the input a parser tolerates is exactly how a fail-closed guard
        becomes fail-open by accident. A response with no JSON at all must still
        raise, so the agent is retried and — if it persists — dropped, rather than
        contributing an empty verdict to a consensus that would look complete.
        """
        in_path = _write_temp("I refuse to answer, and there is no JSON here at all.")
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_two_verdicts_at_top_level_still_fail_closed(self):
        """Ambiguity must fail closed on the BARE path too, not just the envelope.

        A model that echoes the example verdict from its own system prompt beside its
        real one produces two schema-shaped objects. Picking either would be
        fabrication. The guard already existed, but was pinned only through the
        envelope path — and this fix is what makes the bare path reachable, so the
        guard needs its own pin here.
        """
        first = json.dumps(_sample_agent_payload())
        second = json.dumps({**_sample_agent_payload(), "verdict": "approve"})
        in_path = _write_temp(f"Two of them:\n```json\n{first}\n```\n```json\n{second}\n```")
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_truncated_verdict_at_top_level_still_fails_closed(self):
        """A verdict cut mid-object is not a verdict — on the bare path either.

        This is the LOUD half of the truncation asymmetry: a cut-off *output* breaks
        the JSON and dies here, noisily, which is exactly why only a truncated
        *input* (which yields perfectly valid JSON) needs a guard of its own.
        """
        cut = json.dumps(_sample_agent_payload())[:60]
        in_path = _write_temp(f"```json\n{cut}")
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_lone_echoed_example_is_a_known_fabrication_residual(self):
        """CHARACTERIZATION: this asserts a BUG, so it fails when the bug is fixed.

        The LOCKED single-match fabrication residual, now reachable on the Ollama
        path. Every agent system prompt carries a complete example verdict whose
        value is literally ``"verdict": "approve"`` (see skills/magi/agents/*.md).
        A model that echoes it — and emits nothing else decodable — has that example
        recovered as its verdict: a fabricated ``approve``, in the adversarial seat.

        The ambiguity guard does NOT save us here, and that is the subtle part: it
        only fires when TWO objects decode. One echo alone is a single match. So is
        an echo beside a *truncated* real verdict. The two guards interact, and the
        interaction is the hole.

        This test exists so the residual is visible and measured rather than merely
        described in a docstring. The durable fix is the verdict sentinel
        (CLAUDE.techdebt.md), NOT more heuristics in ``_embedded_verdict_object``.
        When the sentinel lands, this test MUST fail — and its failure is the signal
        to delete it, not to weaken it.
        """
        echoed_example = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "One-line verdict",
                "reasoning": "Your risk-focused analysis",
                "findings": [],
                "recommendation": "What you recommend",
            }
        )
        in_path = _write_temp(f"Let me recall the required shape:\n{echoed_example}")
        out_path = _write_temp("", suffix=".out.json")
        try:
            parse_agent_output(in_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                recovered = json.load(f)
            assert recovered["verdict"] == "approve", (
                "fabrication residual has changed shape — re-read the docstring "
                "before touching this test"
            )
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_echoed_example_beside_a_case_drifted_verdict_still_fails_closed(self):
        """A rival verdict with a DRIFTED enum value must still trip the guard.

        The predicate is **key-only**, so this object — a genuine verdict whose enum
        value merely drifted in case — is still verdict-shaped, still a rival, and the
        ambiguity guard still trips. That is the whole point of this test.

        It exists because 4.0.6 tried to make the predicate smarter and broke exactly
        this. Requiring ``verdict`` to be a valid enum member excluded ``"Reject"``
        from candidacy, so the echoed system-prompt example became the *sole* match and
        the parser handed consensus a schema-perfect fabricated ``approve`` — in the
        adversarial seat, silently — on a payload that had failed closed before.

        The reason is structural, and it is why no exclusion belongs in that predicate:
        ``_is_verdict_shaped`` does not only *select* the verdict, **it feeds the
        ambiguity guard**, which is the fail-closed mechanism. Narrowing the predicate
        narrows the guard: fewer candidates ⇒ fewer ambiguity trips ⇒ MORE single-match
        fabrications. Enum drift is common — it is why ``_build_retry_prompt`` exists —
        so a drifted verdict must stay a rival.
        """
        echoed_example = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "One-line verdict",
                "reasoning": "Your risk-focused analysis",
                "findings": [],
                "recommendation": "What you recommend",
            }
        )
        drifted_real_verdict = json.dumps(
            {
                "agent": "caspar",
                "verdict": "Reject",  # a real verdict, wrong case
                "confidence": 0.93,
                "summary": "Six concrete defects.",
                "reasoning": "Traced every claim against the code.",
                "findings": [{"severity": "critical", "title": "Race", "detail": "TOCTOU."}],
                "recommendation": "Do not merge.",
            }
        )
        raw = (
            f"<think>The shape I must follow:\n{echoed_example}\n"
            f"Now my actual verdict.</think>\n"
            f"```json\n{drifted_real_verdict}\n```"
        )
        in_path = _write_temp(raw)
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_a_pipe_union_verdict_stays_a_rival(self):
        """A ``verdict`` value of ``"approve | conditional"`` is still a rival.

        Second attempt at the same reasoning error, one notch finer. That version
        excluded any candidate whose ``verdict`` matched a regex for "word-token pipe
        union" — broader than the rule it claimed to enforce, and broader in the
        fail-open direction: a real verdict that drifted into ``"approve |
        conditional"`` stopped being a rival, so the echoed system-prompt example
        became the sole match and consensus received a fabricated ``approve``.

        With the key-only predicate this object is verdict-shaped, the guard sees two
        candidates, and the parse fails closed. No exclusion is correct here — see
        ``_is_verdict_shaped``; the durable fix is the sentinel (MS2).
        """
        echoed_example = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "One-line verdict",
                "reasoning": "Your risk-focused analysis",
                "findings": [],
                "recommendation": "What you recommend",
            }
        )
        partial_union = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve | conditional",  # a subset, NOT the definition
                "confidence": 0.93,
                "summary": "Six concrete defects.",
                "reasoning": "Traced every claim against the code.",
                "findings": [{"severity": "critical", "title": "Race", "detail": "TOCTOU."}],
                "recommendation": "Do not merge.",
            }
        )
        in_path = _write_temp(f"{echoed_example}\n```json\n{partial_union}\n```")
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    @pytest.mark.parametrize(
        "type_drifted_verdict", [None, ["reject"], {"value": "reject"}, 0, False]
    )
    def test_a_type_drifted_verdict_stays_a_rival(self, type_drifted_verdict):
        """A ``verdict`` of the wrong TYPE is still a rival candidate.

        Third and last instance of the one bug this function kept growing, and the
        reason it now carries **no** exclusion at all: every condition added there is
        one fewer thing the ambiguity guard can see. An ``isinstance(verdict, str)``
        early-return reads like harmless type hygiene — it is an exclusion, and it
        removed the mage's real verdict from the rival set whenever the model
        type-drifted that field (``null``, a list, a number). The echoed system-prompt
        example was then the sole match: a fabricated ``approve``, in Caspar's seat.

        Type drift is a real failure mode — ``validate`` has a dedicated error for it
        (``Invalid verdict 'None'``), which is exactly the feedback the retry needs.
        With the key-only predicate the object stays a rival and the guard trips, which
        is what this test pins. **Nothing** may be disqualified in that predicate: the
        exclusion it was chasing was reverted after three fail-opens. See
        ``_is_verdict_shaped``.
        """
        echoed_example = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "One-line verdict",
                "reasoning": "Your risk-focused analysis",
                "findings": [],
                "recommendation": "What you recommend",
            }
        )
        real_verdict = json.dumps(
            {
                "agent": "caspar",
                "verdict": type_drifted_verdict,
                "confidence": 0.9,
                "summary": "Blocking defects.",
                "reasoning": "Traced every claim against the code.",
                "findings": [{"severity": "critical", "title": "Race", "detail": "TOCTOU."}],
                "recommendation": "Do not merge.",
            }
        )
        in_path = _write_temp(f"<think>{echoed_example}</think>\nVerdict:\n{real_verdict}")
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_a_schema_restatement_drops_the_mage_and_that_is_the_SAFE_failure(self):
        """CHARACTERIZATION: asserts a limitation, so it fails when MS2 fixes it.

        A thinking model that quotes its own schema (``"verdict": "approve | reject |
        conditional"``) emits a decodable object carrying both discriminating keys. It
        counts as a rival candidate, the ambiguity guard fires, and the mage is
        **dropped** → degraded run → by the Integrity rule, approves nothing.

        4.0.6 tried three times to exclude that object from candidacy, and **every
        attempt fabricated an ``approve``** instead (see ``_is_verdict_shaped``). The
        exclusion was then removed on evidence: over 171 captured outputs from the real
        trio it changed **zero** results, while its cost — a narrowed ambiguity guard —
        was reproducible.

        So this drop is deliberate, and this test says so. **It is the safe failure:**
        loud, fail-closed, and it blocks the gate rather than approving it. The cure is
        not a cleverer predicate — it is the verdict **sentinel** (MS2), which stops
        *searching* for the verdict and *extracts* it from between markers, at which
        point a restatement is simply outside them.

        When the sentinel lands this test MUST fail. That failure is the signal to
        delete it — never to weaken the guard so it passes.
        """
        restatement = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve | reject | conditional",  # the schema, quoted
                "confidence": 0.0,
                "summary": "The shape I must emit",
                "reasoning": "Recalling the contract",
                "findings": [],
                "recommendation": "n/a",
            }
        )
        real_verdict = json.dumps(
            {
                "agent": "caspar",
                "verdict": "reject",
                "confidence": 0.93,
                "summary": "Six concrete defects.",
                "reasoning": "Traced every claim against the code.",
                "findings": [{"severity": "critical", "title": "Race", "detail": "TOCTOU."}],
                "recommendation": "Do not merge.",
            }
        )
        raw = f"<think>{restatement}</think>\n```json\n{real_verdict}\n```"
        in_path = _write_temp(raw)
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_deeply_nested_json_degrades_the_mage_instead_of_crashing_the_run(self):
        """CPython raises RecursionError, not JSONDecodeError, on deep nesting.

        ``_loads_lenient`` goes to deliberate trouble to map that, precisely so the
        orchestrator's ``except (ValidationError, json.JSONDecodeError)`` retry can
        catch it. The top-level parse must do the same — and on the Ollama path this
        input is MODEL-AUTHORED content, so a pathological response reaches the call
        directly. An escaping ``RecursionError`` is not caught by that retry, so the
        mage is lost **without a second attempt**; mapped, it is simply retried.
        """
        in_path = _write_temp("[" * 100_000)
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    @pytest.mark.parametrize("depth", [5_000, 16_200, 30_000])
    @pytest.mark.parametrize("fenced", [False, True], ids=["bare", "fenced"])
    def test_a_deeply_nested_verdict_never_escapes_as_RecursionError(self, depth, fenced):
        """No stack overflow may escape this module — from ANY of its four JSON calls.

        Four places can blow the stack on a pathological payload, and they do **not**
        share a depth window — which is why pinning one depth at one site is not a
        guarantee, and why the first attempt at this guard missed:

        * ``json.loads`` on the raw file,
        * ``raw_decode`` during prose recovery (``_loads_lenient``),
        * ``_extract_text``'s ``json.dumps`` — reached only by a **bare** verdict,
        * the final ``json.dumps(..., indent=2)`` — reached by a **fenced** one.

        Measured on the shipped interpreter (3.14): the decoder survives ~16.9k levels,
        both encoders only ~15.5k — so an object that decodes can still fail to encode,
        and *which* encode site raises depends on the route the payload took. That is
        why one depth at one site proves nothing, and why the first attempt at this
        guard passed while the sibling shape was still broken.

        That first attempt mapped the fenced route only, so the sibling shape — a bare,
        unfenced verdict, the plainest Ollama payload there is — still escaped. So this
        test asserts the **property**, not a site: whatever blows, the mage sees a
        ``JSONDecodeError`` and gets its retry. A ``RecursionError`` escapes the
        orchestrator's ``(ValidationError, JSONDecodeError)`` catch and costs the mage
        its second attempt.

        Surviving a depth is allowed — that is not a failure. Only ``RecursionError`` is.
        """
        verdict = (
            '{"agent":"caspar","verdict":"reject","confidence":0.9,"summary":"s",'
            '"reasoning":"r","recommendation":"x","findings":' + "[" * depth + "]" * depth + "}"
        )
        raw = f"```json\n{verdict}\n```" if fenced else verdict
        in_path = _write_temp(raw)
        out_path = _write_temp("", suffix=".out.json")
        try:
            try:
                parse_agent_output(in_path, out_path)
            except json.JSONDecodeError:
                pass  # mapped: the mage is retried
            except RecursionError as exc:  # pragma: no cover - the bug this pins
                raise AssertionError(
                    f"RecursionError escaped at depth={depth} fenced={fenced}. The mage "
                    "loses its retry. Map it to JSONDecodeError at the site that raised."
                ) from exc
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    @pytest.mark.parametrize("missing", ["verdict", "agent"])
    def test_a_bare_verdict_missing_a_key_is_retried_not_dropped(self, missing):
        """A bare payload missing a key must reach the schema check, like a fenced one.

        ``_extract_text`` discriminates the bare-verdict-dict branch on exactly two keys
        — ``agent`` and ``verdict`` — so a model that omits one of *those* falls through
        to its "unexpected shape" ``ValueError``. The orchestrator retries only on
        ``(ValidationError, JSONDecodeError)``, so that ``ValueError`` **dropped the mage
        without a second attempt** — while the identical content inside a fence was
        retried with corrective feedback. Same defect, opposite treatment, decided by a
        markdown fence.

        It matters more since 4.0.6, because reading-as-text-first makes the **bare**
        route the primary one for Ollama. And a missing key is precisely what the retry
        exists to correct: ``load_agent_output`` names it (*"Missing required key"*), and
        that message is the feedback the second attempt is built from.
        """
        payload = {
            "agent": "caspar",
            "verdict": "reject",
            "confidence": 0.9,
            "summary": "s",
            "reasoning": "r",
            "findings": [],
            "recommendation": "x",
        }
        del payload[missing]
        in_path = _write_temp(json.dumps(payload))  # bare: no fence, no prose
        out_path = _write_temp("", suffix=".out.json")
        try:
            # JSONDecodeError -- not ValueError -- so the orchestrator's retry catches it.
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_a_failed_encode_leaves_no_truncated_artifact(self):
        """When the encoder blows up, the output file must not be written at all.

        The verdict is encoded to a string BEFORE the output file is opened. Streaming
        straight into ``open(..., "w")`` used to leave ~1 MB of half-written JSON behind
        whenever the encoder raised mid-write — for a mage that was then dropped, in the
        very run directory ``CLAUDE.md`` tells a reviewer to read. A truncated artifact
        that looks like a verdict is exactly the kind of thing someone reads at 3am.

        **The payload must be FENCED**, and that detail is the whole test. A *bare*
        payload raises earlier, at ``_extract_text``'s own encode, so the final encode —
        the one this guarantee is about — is never reached and the test passes with or
        without the code it claims to pin. The first version of this test did exactly
        that: it was vacuous, and a review caught it. Only a fenced payload takes the
        route that opens the output file.
        """
        depth = 16_200  # decodes, does not re-encode
        verdict = (
            '{"agent":"caspar","verdict":"reject","confidence":0.9,"summary":"s",'
            '"reasoning":"r","recommendation":"x","findings":' + "[" * depth + "]" * depth + "}"
        )
        sentinel = "UNTOUCHED"
        in_path = _write_temp(f"```json\n{verdict}\n```")
        out_path = _write_temp(sentinel, suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                assert f.read() == sentinel, (
                    "the output file was opened and truncated despite the failure — "
                    "encode the payload before opening it"
                )
        finally:
            os.unlink(in_path)
            os.unlink(out_path)

    def test_a_bare_too_nested_verdict_is_also_retryable(self):
        """The BARE Ollama shape must be mapped too — it has its own encoder.

        Sibling of the fenced case, and the reason that one was not enough: a bare
        verdict (no fence) decodes at the top level, so it reaches ``_extract_text``'s
        bare-dict branch and is re-serialised there — a *different* encode site from the
        one a fenced payload reaches. Same limit, different route.

        Two sites, one guarantee. Mapping only the fenced route left the plainest Ollama
        payload there is losing its retry.
        """
        depth = 16_200
        verdict = (
            '{"agent":"caspar","verdict":"reject","confidence":0.9,"summary":"s",'
            '"reasoning":"r","recommendation":"x","findings":' + "[" * depth + "]" * depth + "}"
        )
        in_path = _write_temp(verdict)  # bare: no fence, no prose
        out_path = _write_temp("", suffix=".out.json")
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(in_path, out_path)
        finally:
            os.unlink(in_path)
            os.unlink(out_path)


class TestProseWrappedJson:
    """Recover the JSON verdict when an agent wraps it in natural language.

    Agents doing multi-turn tool use (e.g. verifying a plan against the
    real source) emit a transitional sentence before — and occasionally
    after — the JSON object, such as ``"Now I have enough to render my
    verdict.\\n\\n{...}"``. The strict ``json.loads`` then raised
    ``JSONDecodeError`` and, after one failed retry, the orchestrator
    dropped the agent; with all three dropped it exited 1. The parser
    must recover the embedded object while still failing closed on
    output that contains no JSON object at all. (v2.4.2 root cause.)

    Selection is schema-aware (objects carrying the verdict keys), not by
    character span, and the scan is bounded against oversized/adversarial
    input — hardening added after the 2.4.2 MAGI self-review.
    """

    def _round_trip(self, result_text: str) -> dict:
        """Run *result_text* through the full parser and return the parsed dict."""
        raw = json.dumps({"result": result_text})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(input_path, output_path)
            with open(output_path, encoding="utf-8") as f:
                return json.load(f)
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def _expect_raises(self, result_text: str) -> None:
        """Assert *result_text* still fails closed with ``JSONDecodeError``."""
        raw = json.dumps({"result": result_text})
        input_path = _write_temp(raw)
        fd, output_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with pytest.raises(json.JSONDecodeError):
                parse_agent_output(input_path, output_path)
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_prose_preamble_before_json_is_recovered(self):
        payload = _sample_agent_payload()
        result = (
            "I have verified the plan's code against the real source. "
            "Now I have enough to render my technical verdict.\n\n" + json.dumps(payload)
        )
        assert self._round_trip(result) == payload

    def test_trailing_prose_after_json_is_recovered(self):
        payload = _sample_agent_payload()
        result = json.dumps(payload) + "\n\nThat concludes my analysis."
        assert self._round_trip(result) == payload

    def test_partial_key_object_is_ignored(self):
        """An object with only some verdict keys must not shadow the real verdict."""
        payload = _sample_agent_payload()
        result = (
            'The required schema looks like {"agent": "name"}. '
            "Here is my verdict:\n\n" + json.dumps(payload)
        )
        assert self._round_trip(result) == payload

    def test_echoed_larger_object_without_verdict_keys_is_ignored(self):
        """A large JSON doc echoed from tool use must not shadow the verdict.

        In code-review mode agents Read source/config and quote it; an
        echoed object can out-span the verdict. Selection is by verdict
        keys, not character span, so the echoed object (no ``agent`` /
        ``verdict``) is ignored even though it is larger.
        """
        payload = _sample_agent_payload()
        echoed = {f"config_key_{i}": f"value_{i}" for i in range(40)}
        result = (
            "I read the project config:\n\n"
            + json.dumps(echoed)
            + "\n\nBased on that, here is my verdict:\n\n"
            + json.dumps(payload)
        )
        assert self._round_trip(result) == payload

    def test_prose_with_no_json_object_still_raises(self):
        """No JSON object anywhere → fail closed so the orchestrator can react."""
        self._expect_raises("I am unable to complete this analysis.")

    def test_preamble_with_truncated_json_still_raises(self):
        """A truncated verdict with no complete sub-object re-raises."""
        truncated = json.dumps(_sample_agent_payload())[:-12]
        self._expect_raises("Here is my verdict:\n\n" + truncated)

    def test_truncated_verdict_with_intact_findings_still_raises(self):
        """Truncation after a complete findings element must still fail closed.

        The stray complete finding object lacks the verdict keys, so it is
        not mistaken for the verdict; with no verdict object the parser
        re-raises rather than returning a partial dict.
        """
        payload = _sample_agent_payload()
        payload["findings"] = [
            {"severity": "info", "title": "A finding", "detail": "Complete object."}
        ]
        full = json.dumps(payload)
        truncated = full[: full.rindex('"recommendation"')]
        self._expect_raises("Here is my verdict:\n\n" + truncated)

    def test_oversized_output_skips_recovery_and_raises(self):
        """Output beyond the recovery size budget is not scanned; it re-raises.

        A multi-MB blob is almost certainly echoed tool-use content, not a
        clean verdict, and scanning it risks the O(n^2) raw_decode worst case.
        """
        import parse_agent_output as pao

        payload = _sample_agent_payload()
        filler = "x" * (pao._LENIENT_RECOVERY_MAX_CHARS + 1)
        self._expect_raises(filler + "\n\n" + json.dumps(payload))

    def test_brace_scan_is_bounded(self):
        """The brace scan stops after a bounded number of probes.

        Guards against adversarial deeply-nested-unterminated input
        degrading to O(n^2): a verdict placed beyond the probe cap is not
        recovered.
        """
        import parse_agent_output as pao

        payload = _sample_agent_payload()
        lone_braces = "{" * (pao._MAX_BRACE_PROBES + 5)
        self._expect_raises(lone_braces + json.dumps(payload))

    def test_multiple_verdict_objects_fail_closed(self):
        """Two complete verdict-shaped objects are ambiguous → fail closed.

        If an agent quotes the schema example (a full valid verdict) beside
        its real verdict, or content under review embeds one, picking either
        risks a fabricated verdict entering consensus. Recover only when a
        single verdict object is present; otherwise re-raise so the
        orchestrator retries. (2.4.2 pass-2 review, consensus integrity.)
        """
        real = _sample_agent_payload()
        echoed = _sample_agent_payload()
        echoed["verdict"] = "approve"
        echoed["summary"] = "Quoted schema example."
        result = (
            "For reference the schema is:\n\n"
            + json.dumps(echoed)
            + "\n\nMy actual verdict:\n\n"
            + json.dumps(real)
        )
        self._expect_raises(result)

    def test_deeply_nested_input_raises_json_error_not_recursion(self):
        """Deeply nested input must surface as JSONDecodeError, not RecursionError.

        CPython's json raises RecursionError on deeply nested input; the
        orchestrator's retry catches JSONDecodeError, so the parser maps it
        to keep deeply-nested (echoed or adversarial) output on the
        fail-closed/retry path rather than letting it escape. (2.4.2 pass-2.)
        """
        self._expect_raises('{"a":' * 100_000)


class TestPython39Compatibility:
    """Pin the Python 3.9 compatibility invariant flagged across MAGI reviews."""

    def test_module_annotations_stay_lazy(self):
        """`from __future__ import annotations` must remain in effect.

        ``parse_agent_output`` uses PEP 604 ``X | None`` annotations, which are
        runtime-valid only on CPython 3.10+. ``pyproject`` pins ``>=3.9``, so
        the module relies on ``from __future__ import annotations`` (PEP 563)
        keeping annotations as non-evaluated strings. This guard fails if a
        refactor drops that import: on 3.10+ the annotation becomes an
        evaluated ``types.UnionType`` (caught here); on 3.9 the import itself
        would break. Pins the recurring review concern as a tested invariant.
        """
        import parse_agent_output as pao

        annotation = pao._embedded_verdict_object.__annotations__["return"]
        assert isinstance(annotation, str), (
            "annotations must stay lazy strings (from __future__ import "
            f"annotations); got an evaluated {type(annotation)!r} — PEP 604 "
            "unions break module import on Python 3.9"
        )


# ---------------------------------------------------------------------------
# TestClaudeCliFixtureContract — pinned contract with the Claude CLI output.
# ---------------------------------------------------------------------------

_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "claude-cli-outputs",
)


def _discovered_fixtures() -> list[str]:
    """Return every ``*.json`` file under the fixtures directory.

    Auto-discovery keeps the contract test cheap to extend: drop a captured
    backend output into the fixtures dir and the suite validates it on the
    next run, with no test edit. See
    ``tests/fixtures/claude-cli-outputs/README.md`` for the capture procedure.

    The directory is named for the backend that came first, but the contract it
    pins is **every** backend: since 4.0.6 it also holds Ollama shapes, whose raw
    output is the verdict itself rather than a CLI envelope. The name is kept for
    continuity; the scope is not limited by it.
    """
    if not os.path.isdir(_FIXTURE_DIR):
        return []
    return sorted(
        os.path.join(_FIXTURE_DIR, name)
        for name in os.listdir(_FIXTURE_DIR)
        if name.endswith(".json")
    )


class TestClaudeCliFixtureContract:
    """Pin the contract between ``parse_agent_output`` and ``claude -p``.

    ``parse_agent_output._extract_text`` documents three accepted output
    shapes (``{"result": ...}``, ``{"content": [...]}``, plain string)
    but nothing else in the suite actually exercises them end-to-end
    because ``claude -p`` needs the CLI and a paid API key. Without a
    pinned fixture set, a silent CLI wrapper change would surface only
    as a parse failure in production.

    Each fixture below is a captured sample of what the CLI produces.
    The parametrized test auto-discovers every ``.json`` file in the
    fixtures directory and asserts that it round-trips through the
    parser to a valid agent payload. Adding a new shape is a fixture
    drop; no test edit required.
    """

    def test_fixture_directory_is_populated(self):
        """Regression guard: the directory must exist and be non-empty.

        Without this, a rename that silently empties the fixtures
        directory would turn the parametrized contract below into a
        zero-case test that passes vacuously.
        """
        fixtures = _discovered_fixtures()
        assert fixtures, (
            f"Fixtures directory {_FIXTURE_DIR!r} is empty or missing — "
            "the Claude CLI contract test has no cases to run."
        )

    @pytest.mark.parametrize(
        "fixture_path",
        _discovered_fixtures(),
        ids=lambda p: os.path.basename(p),
    )
    def test_fixture_round_trips_to_valid_agent_output(self, fixture_path):
        """Each captured ``claude -p`` output must parse to valid agent JSON.

        Parses the fixture with ``parse_agent_output``, then re-loads
        the cleaned output and verifies every top-level key required
        by the agent schema is present. A schema mismatch here means
        either the fixture was captured wrong (fix the fixture) or
        ``parse_agent_output`` no longer understands a previously-
        working CLI shape (fix the parser).
        """
        fd, out_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            parse_agent_output(fixture_path, out_path)
            with open(out_path, encoding="utf-8") as f:
                parsed = json.load(f)
            required_keys = {
                "agent",
                "verdict",
                "confidence",
                "summary",
                "reasoning",
                "findings",
                "recommendation",
            }
            missing = required_keys - set(parsed.keys())
            assert not missing, (
                f"Fixture {os.path.basename(fixture_path)!r} did not round-trip "
                f"to a valid agent payload — missing keys: {sorted(missing)}"
            )
        finally:
            os.unlink(out_path)
