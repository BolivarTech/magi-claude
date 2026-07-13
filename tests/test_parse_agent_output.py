# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-07-13
"""Tests de ``parse_agent_output`` — extraccion del veredicto (MS2).

**El parser ya no BUSCA: EXTRAE.** La recuperacion heuristica se borro (no se dejo como
fallback: un fallback reintroduce el residual entero). Lo que sobrevive de la suite previa
es lo que prueba el **TRANSPORTE** (las formas de envelope), que MS2 no toca; lo que
probaba **la heuristica** murio con ella.
"""

import json
import os
import tempfile

import pytest

from parse_agent_output import _extract_text, parse_agent_output
from verdict_markers import (
    VERDICT_CLOSE,
    VERDICT_OPEN,
    AmbiguousVerdictMarkers,
    MissingVerdictMarkers,
    UnterminatedVerdictBlock,
)

VERDICT = json.dumps(
    {
        "agent": "melchior",
        "verdict": "approve",
        "confidence": 0.9,
        "summary": "s",
        "reasoning": "r",
        "findings": [],
        "recommendation": "x",
    }
)


def marked(payload: str) -> str:
    """Envuelve *payload* en el bloque delimitado que MS2 exige."""
    return "\n".join((VERDICT_OPEN, payload, VERDICT_CLOSE))


def _parse(raw: str) -> dict:
    """Corre el parser real sobre *raw* y devuelve el veredicto escrito."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "raw.json")
        dst = os.path.join(tmp, "out.json")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(raw)
        parse_agent_output(src, dst)
        with open(dst, encoding="utf-8") as fh:
            return json.load(fh)


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


class TestTheHeuristicIsGone:
    """R1/R16: se BORRA, no se deja como fallback. Un fallback reintroduce el residual."""

    def test_the_scanner_symbols_no_longer_exist(self):
        import parse_agent_output as pao

        for dead in (
            "_embedded_verdict_object",
            "_is_verdict_shaped",
            "_MAX_BRACE_PROBES",
            "_LENIENT_RECOVERY_MAX_CHARS",
            "_strip_code_fences",
        ):
            assert not hasattr(pao, dead), f"{dead} sigue vivo: MS2 seria TEATRO"


class TestSentinelExtraction:
    """El unico camino valido: el bloque delimitado."""

    def test_a_marked_verdict_is_extracted(self):
        assert _parse(marked(VERDICT))["agent"] == "melchior"

    def test_prose_and_think_around_the_block_are_IGNORED(self):
        """El incidente 2.4.2 (que pario la heuristica) ahora es **inofensivo**."""
        raw = "\n".join(
            ("He verificado el plan.", "<think>razono</think>", marked(VERDICT), "Fin.")
        )
        assert _parse(raw)["agent"] == "melchior"

    def test_a_fence_INSIDE_the_block_is_normalized(self):
        """glm-5.2 fencea por costumbre, incluso con json_schema activo."""
        fenced = "\n".join(("```json", VERDICT, "```"))
        assert _parse(marked(fenced))["agent"] == "melchior"

    def test_a_BARE_verdict_without_markers_is_REJECTED(self):
        """R15 -- el requisito mas doloroso y el mas importante.

        Antes de MS2 esto FUNCIONABA (es como llegaban las 3/3 salidas de Claude medidas),
        y es EXACTAMENTE la variante 1 del residual en su forma pura: el eco solitario.
        **Si se acepta un veredicto sin marcas, MS2 es teatro.**
        """
        with pytest.raises(MissingVerdictMarkers):
            _parse(VERDICT)

    def test_a_claude_envelope_is_unwrapped_THEN_extracted(self):
        assert _parse(json.dumps({"result": marked(VERDICT)}))["agent"] == "melchior"

    def test_json_that_decodes_but_is_NOT_an_envelope_fails_closed(self):
        with pytest.raises((MissingVerdictMarkers, json.JSONDecodeError)):
            _parse(json.dumps({"foo": "bar"}))

    def test_two_blocks_fail_closed(self):
        with pytest.raises(AmbiguousVerdictMarkers):
            _parse("\n".join((marked(VERDICT), marked(VERDICT))))

    def test_a_truncated_block_fails_closed(self):
        with pytest.raises(UnterminatedVerdictBlock):
            _parse("\n".join((VERDICT_OPEN, VERDICT)))

    def test_invalid_json_between_the_markers_still_raises_JSONDecodeError(self):
        """El orquestador reintenta ante ``(ValidationError, JSONDecodeError)``: si el
        contenido entre marcas no decodifica, el mago debe conservar su reintento."""
        with pytest.raises(json.JSONDecodeError):
            _parse(marked("{roto,"))


class TestClaudeCliFixtureContract:
    """Los fixtures pinnean el TRANSPORTE (las formas de envelope), que MS2 no cambia.

    Lo que cambia es su **contenido interno**, que ahora lleva marcas -- porque a partir de
    MS2 **eso es lo que ``claude -p`` devuelve de verdad**.
    """

    FIXTURES = [
        "result-shape.json",
        "content-block-shape.json",
        "content-block-not-first.json",
        "plain-string-shape.json",
        "result-with-markdown-fences.json",
        "result-with-prose-preamble.json",
        "ollama-fenced-content.json",
    ]

    @pytest.mark.parametrize("name", FIXTURES)
    def test_every_pinned_fixture_yields_a_valid_verdict(self, name):
        from pathlib import Path

        raw = (Path(__file__).parent / "fixtures" / "claude-cli-outputs" / name).read_text(
            encoding="utf-8"
        )
        verdict = _parse(raw)
        assert verdict["agent"] in {"melchior", "balthasar", "caspar"}
        assert verdict["verdict"] in {"approve", "reject", "conditional"}
