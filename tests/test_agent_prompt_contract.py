# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Ancla el contrato cross-file entre el codigo y los tres system prompts.

Las marcas y la huella del canario viven en ``verdict_markers.py``, pero **el modelo solo
ve los ``.md``**. Si divergen, pasan dos cosas y ninguna es buena:

* si se rompe una **marca**, el mago **muere** (ruidoso);
* si se rompe la **huella del canario**, el guard **se desarma en SILENCIO** -- sigue ahi,
  sigue ejecutandose, y ya no protege nada. Esa es la peor forma de fallo posible.

Estos tests hacen que la divergencia **rompa el build**, no la produccion.
"""

import json
from pathlib import Path

import pytest

from verdict_markers import ECHO_CANARY, VERDICT_CLOSE, VERDICT_OPEN, VerdictSentinel

AGENTS_DIR = Path(__file__).parent.parent / "skills" / "magi" / "agents"
PROMPTS = ["melchior.md", "balthasar.md", "caspar.md"]


@pytest.mark.parametrize("name", PROMPTS)
class TestPromptContract:
    @staticmethod
    def _text(name: str) -> str:
        # utf-8-sig, no utf-8: se come el BOM ANTES de comparar nada.
        return (AGENTS_DIR / name).read_text(encoding="utf-8-sig")

    def _example_line(self, name: str) -> str:
        return next(
            line for line in self._text(name).splitlines() if line.lstrip().startswith('{"agent"')
        )

    def test_declares_exactly_one_marker_pair(self, name):
        """Exactamente UN par: la misma regla de R3a, aplicada a NUESTROS archivos.

        Si el prompt ensena el formato **dos veces**, el modelo ve **dos ejemplos** y el
        guard ni siquiera sabe **cual** par inspeccionar.
        """
        sentinel = VerdictSentinel()
        lines = self._text(name).splitlines()
        assert sum(sentinel.is_exact_marker_line(ln, VERDICT_OPEN) for ln in lines) == 1
        assert sum(sentinel.is_exact_marker_line(ln, VERDICT_CLOSE) for ln in lines) == 1

    def test_nothing_between_the_markers_is_a_valid_json_object(self, name):
        """El eco SOLITARIO produce UN bloque -- el guard de ambiguedad **no lo ve**.

        Si entre las marcas hubiera un veredicto valido, un modelo que copiara ese bloque
        fabricaria un ``approve`` **con sentinel y todo**. Entre las marcas va un HUECO.
        """
        with pytest.raises(json.JSONDecodeError):
            json.loads(VerdictSentinel().extract(self._text(name)))

    def test_the_canary_fingerprint_is_still_in_the_example_line(self, name):
        """Si alguien edita el ejemplo sin actualizar ``ECHO_CANARY``, el canario compara
        contra un texto que **ya nadie emite**: un test verde sobre un guard desarmado."""
        for value in ECHO_CANARY.values():
            assert value in self._example_line(name)

    def test_the_example_names_ITS_OWN_agent(self, name):
        """Un ejemplo con el nombre del mago equivocado **mata a ese mago en CADA run**.

        Si ``melchior.md`` shippea un ejemplo que dice ``"agent": "caspar"``, el modelo lo
        imita y la verificacion de identidad (R10) lo descarta **siempre**.
        """
        agent = name.removesuffix(".md")
        assert f'"agent": "{agent}"' in self._example_line(name)

    def test_the_worked_example_is_NOT_an_approve(self, name):
        """Cinturon sobre el canario: si algo se copiara pese a todo, que **no** sea un
        aprobado fabricado en el asiento adversarial."""
        assert '"verdict": "approve"' not in self._example_line(name)
