# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""El verdict sentinel: delimita el veredicto del agente (MS2).

La regla que sostiene TODA la seguridad de este feature:

    **NORMALIZAR dentro de una region ya delimitada esta permitido.
    BUSCAR fuera de las marcas esta PROHIBIDO, siempre.**

Cualquier PR que anada una busqueda fuera de las marcas **revierte MS2**, por util que
parezca: el parser volveria a ADIVINAR cual de los objetos era el veredicto, que es
exactamente el defecto que este modulo existe para borrar.

Este modulo **delimita y nada mas**: no valida el schema de 7 claves (eso es
``validate.py``), no habla HTTP y no lanza agentes. Es **puro** — se testea sin red, sin
event loop y sin disco.
"""

from __future__ import annotations

import unicodedata

from validate import ValidationError

VERDICT_OPEN = "<MAGI_VERDICT>"
VERDICT_CLOSE = "</MAGI_VERDICT>"

#: Huella dactilar del ejemplo shippeado en ``agents/*.md`` (R6). Los tres prompts
#: comparten estos dos valores; solo difieren en ``reasoning`` y en el ``detail`` del
#: finding. Anclada por ``tests/test_agent_prompt_contract.py``: si alguien edita el
#: ejemplo de un ``.md`` sin actualizar esto, el canario quedaria comparando contra un
#: texto que ya nadie emite -- un FAIL-OPEN SILENCIOSO.
ECHO_CANARY: dict[str, str] = {
    "summary": "One-line verdict",
    "recommendation": "What you recommend",
}

#: Categorias Unicode que se eliminan antes de comparar una linea con una marca.
#: ``Cf`` (format) cubre TODOS los invisibles -- incluidos ``U+00AD`` (SOFT HYPHEN) y
#: ``U+180E`` (MONGOLIAN VOWEL SEPARATOR), que una lista hardcodeada se dejaba fuera.
#: ``Mn`` cubre los selectores de variacion. **Por CATEGORIA, no por lista: la categoria
#: es exhaustiva y no envejece.**
_STRIPPED_CATEGORIES = frozenset({"Cf", "Mn"})


class VerdictExtractionError(ValidationError):
    """Base de los fallos de extraccion del veredicto.

    **Hereda de ``ValidationError`` A PROPOSITO, y es load-bearing.** El guard de
    reintento del orquestador captura ``(ValidationError, json.JSONDecodeError)``, y
    aqui **el reintento ES el arreglo**: el modelo puede corregirse con el feedback.

    La derogacion fail-closed de ``CLAUDE.local.md`` §0.2 (excepciones **hermanas** de
    ``ValidationError``, no hijas) existe para lo **contrario** -- eventos que el retry
    NO debe consumir-- y **no aplica** aqui. Compara con ``PromptContractError``, que si
    es hermana: un prompt rancio **no se arregla reintentando**.

    **La regla, en una linea: hereda de ``ValidationError`` si el reintento lo arregla;
    hereda de ``Exception`` si no.**
    """


class MissingVerdictMarkers(VerdictExtractionError):
    """No hay ninguna marca: el modelo no emitio el bloque delimitado."""


class UnterminatedVerdictBlock(VerdictExtractionError):
    """Hay apertura sin cierre (o al reves): firma de una salida TRUNCADA."""


class AmbiguousVerdictMarkers(VerdictExtractionError):
    """Mas de un bloque, o cierre antes de apertura.

    Fail-closed **sin desempate**: elegir entre dos bloques seria una heuristica, y las
    heuristicas son justo lo que MS2 borra.
    """


class EchoedExampleRejected(VerdictExtractionError):
    """El "veredicto" es el ejemplo del system prompt, copiado (R6)."""


class AgentIdentityError(VerdictExtractionError):
    """El veredicto dice ser de otro mago distinto al que se lanzo (R10)."""


class VerdictSentinel:
    """Delimita el veredicto entre dos marcas ancladas a linea.

    Los DOS predicados viven aqui juntos **a proposito**: su **asimetria de confianza**
    es el invariante, y separarlos en funciones sueltas es exactamente como se pierde.

    ======================== ==================== =========================================
    Predicado                Sobre que            Criterio
    ======================== ==================== =========================================
    :meth:`is_marker_line`   La salida del MODELO **PERMISIVO**: normaliza invisibles,
                                                  recorta espacios, ignora mayusculas. El
                                                  modelo es **no confiable** y su salida no
                                                  la controlamos; matarlo por un zero-width
                                                  space es un reintento regalado.
    :meth:`is_exact_marker_  NUESTROS ``.md``     **ESTRICTO**: la linea **es** la marca
    line`                                         ASCII. Son archivos que **shipeamos**; un
                                                  invisible ahi es **corrupcion**, y hay
                                                  que verla.
    ======================== ==================== =========================================

    **Un predicado unico ya fallo DOS veces en este diseno:** compartido y permisivo, dejo
    pasar un ``.md`` corrupto; compartido y estricto, aborto el run con un FATAL falso ante
    un BOM. El arreglo no era ninguno de los dos predicados: era poner el BOM en la **capa
    de codificacion** (``utf-8-sig``), donde se resuelve **antes** de comparar nada.

    Args:
        open_marker: Marca de apertura. Default :data:`VERDICT_OPEN`.
        close_marker: Marca de cierre. Default :data:`VERDICT_CLOSE`.
    """

    def __init__(self, open_marker: str = VERDICT_OPEN, close_marker: str = VERDICT_CLOSE) -> None:
        """Construye un sentinel con el par de marcas dado.

        Args:
            open_marker: Marca de apertura.
            close_marker: Marca de cierre.
        """
        self.open = open_marker
        self.close = close_marker

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Forma canonica de una linea de la salida del MODELO. **O(c)**, una pasada.

        **Es la UNICA definicion de "esta linea es una marca"** (DRY): la consumen
        :meth:`is_marker_line` y, mas adelante, la extraccion. Duplicarla seria plantar
        la semilla de que un dia diverjan -- y *"dos piezas decidiendo lo mismo con
        criterios distintos"* es el bug que este modulo ya sufrio dos veces.

        Elimina los caracteres de categoria Unicode ``Cf`` y ``Mn`` (ver
        :data:`_STRIPPED_CATEGORIES`), recorta espacios y pasa a minusculas. Quitar
        caracteres **nunca puede fabricar** una marca donde no la habia, asi que la
        operacion es segura en la direccion que importa.

        **Los HOMOGLIFOS no se tocan** (p.ej. ``U+FF1C`` fullwidth, categoria ``Sm``): no
        son invisibles, son **otro caracter**. Normalizarlos significaria aceptar como
        marca algo que **no es** la marca -- la clase de laxitud que MS2 elimina.

        Args:
            line: Linea candidata (salida **no confiable** del modelo).

        Returns:
            La linea sin invisibles, sin espacios alrededor y en minusculas.
        """
        stripped = "".join(
            char for char in line if unicodedata.category(char) not in _STRIPPED_CATEGORIES
        )
        return stripped.strip().casefold()

    def is_marker_line(self, line: str, marker: str) -> bool:
        """PERMISIVO -- para la salida del MODELO (no confiable, no la controlamos).

        Args:
            line: Una linea de la salida del agente.
            marker: :data:`VERDICT_OPEN` o :data:`VERDICT_CLOSE`.

        Returns:
            ``True`` si la linea es esa marca, tolerando invisibles, espacios alrededor
            y deriva de mayusculas.
        """
        return self._normalize_line(line) == marker.casefold()

    def is_exact_marker_line(self, line: str, marker: str) -> bool:
        """ESTRICTO -- para NUESTROS ``.md`` (guard de arranque, R9).

        Args:
            line: Una linea de un ``agents/*.md`` que shipeamos.
            marker: :data:`VERDICT_OPEN` o :data:`VERDICT_CLOSE`.

        Returns:
            ``True`` solo si la linea, recortada, **es** la marca ASCII. Un invisible ahi
            es corrupcion del archivo, no tolerancia debida.
        """
        return line.strip() == marker
