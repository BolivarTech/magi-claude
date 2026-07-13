# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-13
"""Feedback correctivo del reintento, y su COTA -- fuente UNICA (MS2, R12).

Este modulo existe por una razon muy concreta: la cota del bloque de feedback la necesitan
**dos** consumidores que no se pueden importar entre si.

* ``run_magi`` la usa para **construir** el prompt del reintento.
* ``model_context`` la usa para **reservar ventana** en el guard de contexto (R5b de MS1).

Tenerla dos veces --hardcodeada en uno, derivada en el otro-- es una bomba de relojeria: si
alguien alarga una plantilla, el valor derivado sube y el guard **sigue reservando el viejo**
-> infra-reserva -> **truncamiento silencioso**, que es exactamente el fallo que R5b existe
para prevenir. Una sola fuente lo hace **imposible**.

La cota se **DERIVA** de las plantillas reales. En MS1 esta constante estuvo **mal tres
veces** (1024, 1536, 2048), siempre por asumir un factor de conversion comodo o un texto
fijo que luego cambio. **Una cota que hay que acordarse de actualizar es una cota que se va
a olvidar.**
"""

from __future__ import annotations

import json

from validate import ValidationError
from verdict_markers import (
    AgentIdentityError,
    AmbiguousVerdictMarkers,
    EchoedExampleRejected,
    MissingVerdictMarkers,
    UnterminatedVerdictBlock,
    VERDICT_CLOSE,
    VERDICT_OPEN,
)

#: El mensaje de error se trunca a esta longitud antes de insertarlo en el prompt.
MAX_ERROR_CHARS = 400

#: Las 7 causas. Una por CADA error que el orquestador puede reintentar: un error
#: sin plantilla degrada a un mensaje generico y le quita al modelo justo lo que
#: necesitaba para auto-corregirse -- o sea, un reintento a ciegas.
_CAUSE_MISSING_MARKERS = "missing_markers"
_CAUSE_UNTERMINATED_BLOCK = "unterminated_block"
_CAUSE_AMBIGUOUS_MARKERS = "ambiguous_markers"
_CAUSE_ECHOED_EXAMPLE = "echoed_example"
_CAUSE_AGENT_IDENTITY = "agent_identity"
_CAUSE_INVALID_JSON = "invalid_json"
_CAUSE_SCHEMA = "schema"


def _feedback_template(intro: str, corrective: str) -> str:
    """Build one retry-feedback block from a cause-specific intro + corrective.

    Factored out so the seven templates in :data:`FEEDBACK_TEMPLATES` share ONE
    definition of the envelope (the ``---RETRY-FEEDBACK---`` delimiter and the
    ``{error}`` placement) instead of seven copies that could silently drift apart.

    Args:
        intro: One sentence naming WHAT went wrong, specific to the cause.
        corrective: The instruction telling the model how to fix it.

    Returns:
        A template string with a single ``{error}`` placeholder, ready for
        ``str.format(error=...)``.
    """
    return f"---RETRY-FEEDBACK---\n{intro}\n{{error}}\n\n{corrective}"


#: One entry per verdict-extraction/schema failure cause -- picked by
#: :func:`_retry_feedback_cause` from ``type(error)``, NEVER from the error's message
#: text. Mixing up the instruction is not cosmetic: telling a model that emitted NO
#: markers to "emit exactly one block" is a FALSE instruction that burns the retry
#: and kills the mage for a bug in the algorithm that exists to save it (found as
#: ``[CRITICAL]`` in review). English, like the agent system prompts.
FEEDBACK_TEMPLATES: dict[str, str] = {
    _CAUSE_MISSING_MARKERS: _feedback_template(
        "Your previous response did not include the required verdict markers.",
        f"Re-emit your response wrapping the COMPLETE JSON verdict between two "
        f"marker lines, each ALONE on its own line: {VERDICT_OPEN} immediately "
        f"before the JSON object and {VERDICT_CLOSE} immediately after it. Nothing "
        "else may appear on those two lines.",
    ),
    _CAUSE_UNTERMINATED_BLOCK: _feedback_template(
        "Your previous response opened a verdict block but never closed it -- the "
        "output looks TRUNCATED.",
        f"Re-emit your COMPLETE response, making sure it ends with the closing "
        f"marker {VERDICT_CLOSE} on its own line, immediately after the full JSON "
        "verdict.",
    ),
    _CAUSE_AMBIGUOUS_MARKERS: _feedback_template(
        "Your previous response contained MORE THAN ONE verdict block (or a "
        "closing marker before an opening one).",
        f"Re-emit your response with EXACTLY ONE verdict block: a single "
        f"{VERDICT_OPEN} line, the JSON verdict, then a single {VERDICT_CLOSE} "
        "line. Do not repeat the markers and do not include any other block.",
    ),
    _CAUSE_ECHOED_EXAMPLE: _feedback_template(
        "Your previous response copied the worked EXAMPLE from your system prompt "
        "instead of analyzing the actual input.",
        "Re-emit your response with YOUR OWN verdict: your own summary, your own "
        "reasoning, and your own findings about the input you were given -- not "
        "the placeholder text from the example.",
    ),
    _CAUSE_AGENT_IDENTITY: _feedback_template(
        "Your previous response claimed to be a different agent than the one that was launched.",
        'Re-emit your response with the "agent" field set to YOUR OWN name, '
        "matching the mage you were launched as.",
    ),
    _CAUSE_INVALID_JSON: _feedback_template(
        "The content between your verdict markers was not valid JSON.",
        "Re-emit your response with a single, syntactically valid JSON object "
        "between the verdict markers -- no trailing commas, no unbalanced braces, "
        "no truncation, and no text other than the JSON object itself between "
        "the markers.",
    ),
    _CAUSE_SCHEMA: _feedback_template(
        "Your previous response was rejected by the parsing pipeline.",
        "Re-emit your response as a complete, syntactically valid JSON object "
        "containing ALL seven required top-level keys: agent, verdict, "
        "confidence, summary, reasoning, findings, recommendation. Do not omit "
        "any key, do not truncate, do not emit anything outside the JSON object.",
    ),
}

#: An emoji is 4 UTF-8 bytes, and a byte-level BPE that fails to merge them emits
#: one token per byte -- 4 tokens/char is the TRUE worst case, not 1 and not 3. This
#: exact ratio was guessed wrong twice in MS1 (see ``model_context.MAX_ERROR_CHARS``'s
#: docstring) by assuming a comfortable ratio instead of the worst one that exists.
WORST_TOKENS_PER_CHAR = 4

#: DERIVED, not hardcoded: the largest fixed portion across all seven retry-feedback
#: templates, plus the worst-case cost of the truncated error detail. A cota that
#: must be updated by hand every time a template is added or edited is a cota that
#: WILL be forgotten -- this exact constant was wrong three times in MS1 for exactly
#: that reason. Deriving it from ``FEEDBACK_TEMPLATES`` makes drift impossible: add
#: an eighth cause and this recomputes on its own.
MAX_RETRY_FEEDBACK_TOKENS = (
    max(len(t.encode("utf-8")) for t in FEEDBACK_TEMPLATES.values())
    + MAX_ERROR_CHARS * WORST_TOKENS_PER_CHAR
)


def _retry_feedback_cause(error: ValidationError | json.JSONDecodeError) -> str:
    """Pick the :data:`FEEDBACK_TEMPLATES` key for *error*, by its TYPE alone.

    Order matters: every verdict-marker exception is a :class:`ValidationError`
    subclass (``verdict_markers.VerdictExtractionError``), so the specific checks
    MUST run before the generic schema fallback -- otherwise every cause would
    collapse into ``"schema"`` and the model would get the wrong instruction.

    Args:
        error: The exception that triggered the retry.

    Returns:
        One of the keys of :data:`FEEDBACK_TEMPLATES`.
    """
    if isinstance(error, MissingVerdictMarkers):
        return _CAUSE_MISSING_MARKERS
    if isinstance(error, UnterminatedVerdictBlock):
        return _CAUSE_UNTERMINATED_BLOCK
    if isinstance(error, AmbiguousVerdictMarkers):
        return _CAUSE_AMBIGUOUS_MARKERS
    if isinstance(error, EchoedExampleRejected):
        return _CAUSE_ECHOED_EXAMPLE
    if isinstance(error, AgentIdentityError):
        return _CAUSE_AGENT_IDENTITY
    if isinstance(error, json.JSONDecodeError):
        return _CAUSE_INVALID_JSON
    return _CAUSE_SCHEMA
