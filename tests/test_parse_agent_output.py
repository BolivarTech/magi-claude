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

    def test_fallback_dict_raises_value_error(self):
        data = {"unknown_key": "some_value"}
        with pytest.raises(ValueError, match="Unexpected Claude CLI output type"):
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

    Auto-discovery keeps the contract test cheap to extend: drop a new
    captured ``claude -p`` output into the fixtures dir and the suite
    will validate it on the next run without a test edit. See
    ``tests/fixtures/claude-cli-outputs/README.md`` for the capture
    procedure.
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
