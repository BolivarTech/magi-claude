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

import re
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

#: Fence de apertura: ``` o ~~~, con o sin info-string, con espacios alrededor.
#: El info-string se acepta **permisivo** (cualquier token sin espacios) y NO como lista
#: blanca: una lista **enumera** lo permitido, asi que cada lenguaje con un punto o una
#: almohadilla (``json5``, ``c#``, ``asp.net``) seria un fallo futuro. Aqui ser permisivo
#: es **gratis**: el fence solo se quita si la primera **Y** la ultima linea lo son, y lo
#: de dentro **lo decide ``json.loads``**. Permisivo en lo que no importa, estricto en lo
#: que si (las marcas).
_FENCE_OPEN_RE = re.compile(r"^\s*(```|~~~)\s*[^\s`~]*\s*$")
_FENCE_CLOSE_RE = re.compile(r"^\s*(```|~~~)\s*$")


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

    def extract(self, text: str) -> str:
        """Devuelve el UNICO bloque delimitado. **NO escanea fuera de las marcas.**

        **El ORDEN de los tres primeros chequeos es LOAD-BEARING.** El orquestador elige
        el feedback del reintento **por el TIPO de excepcion**, asi que decirle *"emitiste
        mas de un bloque"* a un modelo que **no emitio ninguna marca** gasta el reintento
        en una instruccion **falsa**, y el mago muere por un bug del algoritmo que existe
        para salvarlo. (Hallado como ``[CRITICAL]`` en revision.)

        Se cuentan **todas** las aperturas y **todos** los cierres, **una sola pasada** y
        **una sola normalizacion por linea**. **No hay "primer cierre" ni "ultimo
        cierre"**: elegir entre ellos seria una regla de desempate, y las reglas de
        desempate son heuristicas -- justo lo que este modulo borra.

        Args:
            text: La salida cruda del agente (entrada **no confiable**).

        Returns:
            El contenido entre las marcas, con el fence markdown quitado si envolvia el
            bloque **entero**.

        Raises:
            MissingVerdictMarkers: No hay ninguna marca (ni apertura ni cierre).
            UnterminatedVerdictBlock: Falta exactamente una de las dos -- firma de una
                salida **truncada**.
            AmbiguousVerdictMarkers: El conteo no es exactamente 1 y 1, o el cierre
                precede a la apertura.
        """
        lines = text.splitlines()

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
                f"close marker precedes the open marker (open at line {opens[0]}, "
                f"close at line {closes[0]})"
            )

        return self._strip_fence(lines[opens[0] + 1 : closes[0]])

    @staticmethod
    def _strip_fence(block_lines: list[str]) -> str:
        """Quita un fence markdown que envuelva el bloque **entero**.

        **Normalizar DENTRO de una region ya delimitada esta permitido; BUSCAR fuera de
        las marcas, jamas.** Si el fence no envuelve el bloque completo, el contenido se
        deja **INTACTO** y decide ``json.loads``: recortar *"lo que estorba"* hasta que
        algo decodifique **seria volver a buscar**.

        El cierre debe ser **del MISMO tipo** que la apertura (` ``` ` con ` ``` `, ``~~~``
        con ``~~~``). Un par **desparejado** no es un fence: es texto — ningun parser
        markdown lo acepta y ningun modelo lo emite.

        Args:
            block_lines: Las lineas de entre las marcas.

        Returns:
            El bloque como texto, sin las dos lineas de fence si eran un par valido.
        """
        body = list(block_lines)
        if len(body) < 2:
            return "\n".join(body).strip()

        opened = _FENCE_OPEN_RE.match(body[0])
        closed = _FENCE_CLOSE_RE.match(body[-1])
        if opened and closed and opened.group(1) == closed.group(1):
            body = body[1:-1]
        return "\n".join(body).strip()
