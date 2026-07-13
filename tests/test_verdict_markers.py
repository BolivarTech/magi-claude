# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Suite del verdict sentinel (MS2).

**Pura**: sin red, sin ``asyncio``, sin disco. Si algun dia hace falta un mock de HTTP
aqui, es que el modulo perdio su bajo acoplamiento.
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


VERDICT = '{"agent": "caspar", "verdict": "reject"}'


def _block(body: str) -> str:
    """Envuelve *body* en el bloque delimitado que MS2 exige."""
    return f"{VERDICT_OPEN}\n{body}\n{VERDICT_CLOSE}"


class TestExtract:
    """El ORDEN de los chequeos es load-bearing: selecciona el feedback del reintento."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    def test_extracts_the_block_and_ignores_everything_outside(self):
        raw = f"<think>razono un rato</think>\nprosa\n{_block(VERDICT)}\nmas prosa"
        assert json.loads(self.sentinel.extract(raw))["agent"] == "caspar"

    def test_zero_markers_raises_MISSING_not_ambiguous(self):
        """El ``[CRITICAL]`` del ciclo 6: el TIPO de excepcion selecciona el feedback.

        Decirle *"emitiste mas de un bloque"* a quien no emitio NINGUNA marca gasta el
        reintento en una instruccion **falsa**, y el mago muere por un bug del algoritmo
        que existe para salvarlo.
        """
        with pytest.raises(MissingVerdictMarkers):
            self.sentinel.extract(VERDICT)  # JSON perfecto, pero SIN marcas

    def test_an_open_without_a_close_is_a_truncated_output(self):
        with pytest.raises(UnterminatedVerdictBlock):
            self.sentinel.extract(f"{VERDICT_OPEN}\n{VERDICT}")

    def test_a_close_without_an_open_is_a_truncated_output(self):
        with pytest.raises(UnterminatedVerdictBlock):
            self.sentinel.extract(f"{VERDICT}\n{VERDICT_CLOSE}")

    def test_two_blocks_fail_closed_without_a_tie_break(self):
        """El eco del ejemplo + el veredicto real. **NUNCA** se elige uno."""
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
        """MAGI se revisa a si mismo: un finding SOBRE el sentinel cita la marca.

        JSON **escapa los saltos de linea**, asi que una marca citada dentro del payload
        **no puede** aparecer sola en un renglon -> el anclaje a linea la hace inofensiva.
        """
        payload = json.dumps(
            {"agent": "caspar", "detail": "el cierre </MAGI_VERDICT> se ancla a linea"}
        )
        assert json.loads(self.sentinel.extract(_block(payload)))["agent"] == "caspar"

    def test_a_raw_newline_inside_a_string_yields_two_closes_and_fails_closed(self):
        """Un modelo que emite JSON invalido con un salto CRUDO -> 1 apertura, 2 cierres."""
        body = '{"detail": "roto\n</MAGI_VERDICT>\nsigue"}'
        with pytest.raises(AmbiguousVerdictMarkers):
            self.sentinel.extract(_block(body))

    @pytest.mark.parametrize("separator", [" ", " ", ""])
    def test_a_JSON_LEGAL_line_separator_does_not_cut_the_block(self, separator):
        """Un veredicto VALIDO que cita la marca tras un separador **legal en JSON**.

        ``str.splitlines()`` corta ademas por ``\\v``, ``\\f``, ``\\x1c-\\x1e``, ``U+0085``,
        ``U+2028`` y ``U+2029`` -- y los tres ultimos **son legales crudos dentro de un
        string JSON** (``json.loads`` los acepta). Un finding sobre el sentinel que cite la
        marca detras de uno de ellos deja `</MAGI_VERDICT>` **solo en su renglon** -> 2
        cierres -> `AmbiguousVerdictMarkers` -> el mago muere por un separador invisible.

        Falla cerrado (nunca fabrica), pero la garantia que la docstring promete --*"JSON
        escapa los saltos de linea, asi que una marca citada no puede quedar sola en un
        renglon"*-- es **mas ancha que el codigo**: vale para ``\\n``, no para el juego de
        separadores de ``splitlines()``. Y el escenario es el que MAGI produce **al
        revisarse a si mismo**.
        """
        detail = f"el finding cita la marca:{separator}</MAGI_VERDICT>{separator}y sigue"
        payload = json.dumps({"agent": "caspar", "detail": detail}, ensure_ascii=False)
        json.loads(payload)  # premisa: el separador CRUDO es JSON valido

        extracted = self.sentinel.extract(_block(payload))

        assert json.loads(extracted)["detail"] == detail

    def test_extract_documents_every_cause_it_raises(self):
        """El ``Raises:`` es LOAD-BEARING: uno incompleto es un reintento a ciegas."""
        doc = VerdictSentinel.extract.__doc__ or ""
        assert "Raises:" in doc
        for cause in (
            "MissingVerdictMarkers",
            "UnterminatedVerdictBlock",
            "AmbiguousVerdictMarkers",
        ):
            assert cause in doc


class TestFenceNormalization:
    """Normalizar DENTRO de una region ya delimitada: permitido. Buscar fuera: jamas."""

    def setup_method(self):
        self.sentinel = VerdictSentinel()

    @pytest.mark.parametrize(
        "opener", ["```json", "```", "~~~json", "``` json", "```json  ", "```json5"]
    )
    def test_a_fence_around_the_json_is_stripped(self, opener):
        """glm-5.2 fencea por costumbre. Fallar por un ESPACIO seria fragil, no estricto."""
        closer = opener[:3]
        raw = _block(f"{opener}\n{VERDICT}\n{closer}")
        assert json.loads(self.sentinel.extract(raw))["agent"] == "caspar"

    def test_text_between_the_fence_and_the_json_is_left_INTACT(self):
        """Quitar "lo que estorba" hasta que algo decodifique seria VOLVER A BUSCAR."""
        raw = _block(f"```json\naqui va mi veredicto:\n{VERDICT}\n```")
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.sentinel.extract(raw))

    def test_a_MISMATCHED_fence_pair_is_not_a_fence(self):
        """``` abierto y ~~~ cerrado no es un fence: es texto. Se deja intacto."""
        raw = _block(f"```json\n{VERDICT}\n~~~")
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.sentinel.extract(raw))

    def test_each_line_is_normalized_ONCE_not_twice(self, monkeypatch):
        """La cota O(N) del plan, hecha EJECUTABLE en vez de prometida.

        La version obvia de ``extract`` (dos comprensiones, una por marca) normaliza
        **cada linea DOS veces**: mismo O(N), el doble de trabajo, y por nada. Este test
        hace que esa regresion **rompa el build** en vez de pasar desapercibida.
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
        # Se cuenta con el MISMO criterio de linea que usa ``extract`` (``_LINE_BREAK``), no
        # con ``splitlines()``: dos criterios distintos para "que es una linea" es como se
        # desincronizan un test y su implementacion sin que nadie lo note.
        assert calls == len(_LINE_BREAK.split(raw))  # UNA por linea, ni una mas


class TestExtractProperties:
    """hypothesis: propiedades sobre entradas GENERADAS, no ejemplos (§0.3)."""

    @given(
        st.text(),
        st.text(),
        st.dictionaries(st.text(min_size=1), st.text(), min_size=1),
    )
    def test_nothing_from_OUTSIDE_the_markers_can_reach_the_output(self, prefix, suffix, obj):
        """LA propiedad de seguridad de MS2 -- y ahora se EJERCITA de verdad.

        La version previa generaba ``st.text()`` a secas y comprobaba la propiedad **solo
        si** el texto traia un bloque bien formado. Medido: **0 de 2000 ejemplos** llegaban
        a la asercion -- hypothesis no produce ``<MAGI_VERDICT>`` solo en su renglon por
        azar. Era ``assert True`` con forma de propiedad, justo sobre el invariante nuclear
        del milestone. (Su oraculo, ademas, re-partia con ``splitlines()``, que ya no es lo
        que hace ``extract``: dos piezas decidiendo lo mismo con criterios distintos.)

        Ahora el bloque se **construye** y el ruido va **fuera** (y puede ser cualquier
        cosa, incluidas marcas: entonces se exige fallo cerrado). El oraculo no reconstruye
        nada -- sabemos por construccion que ENTRE las marcas va ``obj``, asi que si algo
        de fuera se colara, el bloque no decodificaria a ``obj``.
        """
        sentinel = VerdictSentinel()
        raw = f"{prefix}\n{_block(json.dumps(obj))}\n{suffix}"
        try:
            block = sentinel.extract(raw)
        except VerdictExtractionError:
            return  # el ruido generado traia una marca -> fallar cerrado es correcto
        assert json.loads(block) == obj

    @given(st.text())
    def test_never_raises_anything_but_a_VerdictExtractionError(self, noise):
        """Entrada arbitraria NUNCA provoca una excepcion no controlada."""
        try:
            VerdictSentinel().extract(noise)
        except VerdictExtractionError:
            pass

    @given(st.dictionaries(st.text(min_size=1), st.text(), min_size=1))
    def test_a_verdict_wrapped_in_markers_round_trips_EXACTLY(self, obj):
        """MATA la implementacion tramposa que la propiedad anterior dejaba viva.

        Un ``extract`` que devolviera **siempre la cadena vacia** pasaria ``block in
        intra`` -- ``"" in cualquier_cosa`` es **True**. La propiedad de no-fuga es
        necesaria pero **no suficiente**.
        """
        raw = f"prosa\n<think>ruido</think>\n{_block(json.dumps(obj))}\nmas prosa"
        assert json.loads(VerdictSentinel().extract(raw)) == obj
