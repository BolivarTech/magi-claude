# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Suite del verdict sentinel (MS2).

**Pura**: sin red, sin ``asyncio``, sin disco. Si algun dia hace falta un mock de HTTP
aqui, es que el modulo perdio su bajo acoplamiento.
"""

import ast
from pathlib import Path

import pytest

from validate import ValidationError
from verdict_markers import (
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
        """El retry del orquestador captura ``(ValidationError, JSONDecodeError)``.

        Si un error de extraccion NO heredara de ``ValidationError``, el guard no lo
        capturaria y el mago **moriria en vez de reintentar** -- convirtiendo un fallo
        RECUPERABLE en la muerte del mago. La derogacion fail-closed de
        ``CLAUDE.local.md`` §0.2 NO aplica aqui: **aqui el retry ES el arreglo**.
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
    """§Quality de ``~/.claude/CLAUDE.md``, hecho VERIFICABLE en vez de aspiracional."""

    @staticmethod
    def _source() -> str:
        import verdict_markers

        return Path(verdict_markers.__file__).read_text(encoding="utf-8")

    def test_the_sentinel_imports_no_io_and_no_orchestrator(self):
        """Bajo acoplamiento: el modulo se puede testear -- y romper -- SOLO.

        Si un dia alguien importa ``urllib`` o ``run_magi`` aqui, el modulo deja de ser
        puro y deja de ser testeable sin red. Este test lo impide **hoy**, no cuando ya
        duela.
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
        """Constantes con nombre: cero literales numericos **semanticos** en el cuerpo.

        Se permiten los literales de **indexacion** (``body[0]``, ``body[-1]``,
        ``len(body) < 2``): no son numeros magicos, son **la aritmetica de una lista**.
        Un numero magico es el que codifica una **decision** -- 2000 probes, 400 chars,
        4 tokens/char -- y de esos aqui no queda ninguno.
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
    """PERMISIVO -- para la salida del MODELO (no confiable, no la controlamos)."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    def test_a_plain_marker_line_is_recognized(self):
        assert self.sentinel.is_marker_line(VERDICT_OPEN, VERDICT_OPEN)

    def test_surrounding_whitespace_is_tolerated(self):
        assert self.sentinel.is_marker_line("   <MAGI_VERDICT>  ", VERDICT_OPEN)

    def test_crlf_line_endings_are_tolerated(self):
        """Windows existe, y el modelo puede emitir \\r\\n."""
        assert self.sentinel.is_marker_line("<MAGI_VERDICT>\r\n", VERDICT_OPEN)

    # Los invisibles van como ESCAPES, NUNCA como el caracter literal: un ZWSP literal
    # es **invisible para el revisor**, y un editor o un copy-paste puede borrarlo en
    # silencio -- dejando un test que compara "<MAGI_VERDICT>" contra "<MAGI_VERDICT>" y
    # **pasa sin probar nada**. El escape es visible, revisable y no se puede perder.
    @pytest.mark.parametrize(
        "invisible",
        [
            "​",  # ZERO WIDTH SPACE
            "‍",  # ZERO WIDTH JOINER
            "﻿",  # BOM
            "⁠",  # WORD JOINER
            "­",  # SOFT HYPHEN               <- una lista hardcodeada lo dejaba fuera
            "᠎",  # MONGOLIAN VOWEL SEPARATOR <- idem
            "️",  # VARIATION SELECTOR-16 (categoria Mn)
        ],
    )
    def test_an_invisible_inside_the_marker_does_not_kill_the_mage(self, invisible):
        """Un invisible en la salida del MODELO es un reintento regalado."""
        assert self.sentinel.is_marker_line(f"<MAGI{invisible}_VERDICT>", VERDICT_OPEN)

    def test_case_drift_is_tolerated_from_the_model(self):
        """Un modelo que escribe ``<magi_verdict>`` EMITIO la marca."""
        assert self.sentinel.is_marker_line("<magi_verdict>", VERDICT_OPEN)

    def test_a_fullwidth_homoglyph_is_NOT_a_marker(self):
        """Un homoglifo no es un invisible: es **otro caracter**. Falla cerrado."""
        assert not self.sentinel.is_marker_line("＜MAGI_VERDICT>", VERDICT_OPEN)

    def test_prose_mentioning_the_marker_is_not_a_marker_line(self):
        assert not self.sentinel.is_marker_line("emite <MAGI_VERDICT> al final", VERDICT_OPEN)

    def test_the_close_marker_is_not_the_open_marker(self):
        assert not self.sentinel.is_marker_line(VERDICT_CLOSE, VERDICT_OPEN)
        assert self.sentinel.is_marker_line(VERDICT_CLOSE, VERDICT_CLOSE)


class TestExactMarkerLinePredicate:
    """ESTRICTO -- para NUESTROS .md: un invisible ahi es CORRUPCION, no tolerancia.

    Es el reverso exacto del predicado permisivo, y la asimetria **es** el invariante:
    la salida del modelo no la controlamos; nuestros archivos, si.
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
