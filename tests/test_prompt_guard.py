# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Suite del guard de arranque del contrato de prompts (R9, MS2).

Cubre lo que el test de anclaje **no puede ver**: la **instalacion del usuario**. El test
de anclaje corre en el repo del desarrollador; el bug de la **copia rancia** (``mklink /D``
degradando a copia en Windows) produce prompts viejos con parser nuevo **en la maquina del
usuario**, donde ningun test llega.
"""

import json

import pytest

from prompt_guard import AgentPromptGuard, PromptContractError
from validate import ValidationError
from verdict_markers import VERDICT_CLOSE, VERDICT_OPEN, VerdictSentinel

GOOD = f"prosa\n{VERDICT_OPEN}\n{{ ...tu objeto JSON de 7 claves... }}\n{VERDICT_CLOSE}\n"


def _agents(tmp_path, **overrides):
    """Crea un directorio de prompts sano, con los overrides que se pidan."""
    directory = tmp_path / "agents"
    directory.mkdir()
    for name in ("melchior", "balthasar", "caspar"):
        (directory / f"{name}.md").write_text(overrides.get(name, GOOD), encoding="utf-8")
    return directory


class TestErrorIsNotRetryable:
    def test_PromptContractError_is_a_SIBLING_of_ValidationError_not_a_child(self):
        """``[CRITICAL]`` del Checkpoint 2: si heredara, **el retry se la tragaria**.

        El guard de reintento del orquestador captura ``(ValidationError,
        JSONDecodeError)``. Un prompt rancio **no se arregla reintentando** -- el archivo
        no cambia por volver a llamar al modelo. Es un evento **fail-closed**: aborta.

        Es exactamente el caso de la derogacion locked de ``CLAUDE.local.md`` §0.2
        (precedente: ``InvalidInputError``). **Regla: hereda de ValidationError si el
        reintento lo arregla; de Exception si no.**
        """
        assert issubclass(PromptContractError, Exception)
        assert not issubclass(PromptContractError, ValidationError)


class TestAgentPromptGuard:
    def test_a_healthy_prompt_set_passes(self, tmp_path):
        AgentPromptGuard(_agents(tmp_path), VerdictSentinel()).check()

    def test_a_stale_prompt_without_markers_is_FATAL(self, tmp_path):
        directory = _agents(tmp_path, caspar="Respond with ONLY a JSON object.\n")
        with pytest.raises(PromptContractError, match="caspar.md"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_two_marker_pairs_are_FATAL(self, tmp_path):
        """El usuario documenta el formato dos veces -> el modelo ve **dos ejemplos**."""
        directory = _agents(tmp_path, caspar=GOOD + GOOD)
        with pytest.raises(PromptContractError, match="2 open"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_VALID_verdict_between_the_markers_is_FATAL(self, tmp_path):
        """El ULTIMO camino de fabricacion, y esta donde ningun test nuestro llega.

        Un usuario "mejora" el prompt metiendo un ejemplo completo **entre las marcas** y
        **reinstala la variante 1 en SU maquina**: el modelo copia ese bloque, produce UN
        solo bloque delimitado, valida... y fabrica. Ni el canario (no es el ejemplo
        shippeado) ni el test de anclaje (corre en NUESTRO repo) lo ven.
        """
        verdict = json.dumps(
            {
                "agent": "caspar",
                "verdict": "approve",
                "confidence": 0.85,
                "summary": "s",
                "reasoning": "r",
                "findings": [],
                "recommendation": "x",
            }
        )
        directory = _agents(tmp_path, caspar=f"{VERDICT_OPEN}\n{verdict}\n{VERDICT_CLOSE}\n")
        with pytest.raises(PromptContractError, match="valid verdict"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_harmless_placeholder_that_happens_to_be_json_PASSES(self, tmp_path):
        """``{}`` es JSON valido y **no puede fabricar nada** (no tiene las 7 claves).

        Abortar por el seria castigar al usuario por algo inofensivo. La pregunta correcta
        no es *"¿es JSON valido?"* sino *"¿esto, copiado, se aceptaria como veredicto?"*.
        """
        directory = _agents(tmp_path, caspar=f"{VERDICT_OPEN}\n{{}}\n{VERDICT_CLOSE}\n")
        AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_file_with_BOM_does_NOT_trigger_a_false_FATAL(self, tmp_path):
        """El BOM se resuelve en la **capa de codificacion** (``utf-8-sig``), no relajando
        el predicado. Por eso el guard puede ser ESTRICTO sin falsos FATAL."""
        directory = _agents(tmp_path)
        (directory / "caspar.md").write_text(GOOD, encoding="utf-8-sig")
        AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_an_invisible_inside_OUR_marker_is_corruption_and_is_FATAL(self, tmp_path):
        """El reverso del permisivo: la salida del MODELO con ese invisible SI se acepta.

        *Dominios de confianza distintos, estrictez distinta.*
        """
        corrupted = GOOD.replace(VERDICT_OPEN, "<MAGI​_VERDICT>")
        directory = _agents(tmp_path, caspar=corrupted)
        with pytest.raises(PromptContractError):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_a_missing_file_is_FATAL(self, tmp_path):
        directory = _agents(tmp_path)
        (directory / "caspar.md").unlink()
        with pytest.raises(PromptContractError, match="caspar.md"):
            AgentPromptGuard(directory, VerdictSentinel()).check()

    def test_the_SHIPPED_prompts_pass_the_guard(self):
        """El guard corre contra los prompts **de verdad**, no solo contra fixtures."""
        from pathlib import Path

        agents = Path(__file__).parent.parent / "skills" / "magi" / "agents"
        AgentPromptGuard(agents, VerdictSentinel()).check()
