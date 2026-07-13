#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.6.0
# Date: 2026-05-23
"""MAGI Orchestrator — async Python replacement for run_magi.sh.

Launches Melchior, Balthasar, and Caspar in parallel using asyncio,
collects their JSON outputs, validates them, and runs synthesis.

Usage:
    python run_magi.py <mode> <input> [--model opus] [--timeout 900] [--output-dir <dir>]

Exit codes:
    0 - Success: synthesis completed and report saved.
    1 - Failure: prerequisites missing, or fewer than 2 agents succeeded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Bootstrap: make sibling modules importable under invocations that do NOT
# auto-inject this directory into sys.path (e.g. ``python -m
# skills.magi.scripts.run_magi``). Direct invocation
# (``python skills/magi/scripts/run_magi.py``) and pytest (via conftest.py)
# already cover this. See CLAUDE.md "Open technical debt /
# synthesize import gap [LOCKED]".
_SCRIPT_DIR = str(Path(__file__).parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model_context import (  # noqa: E402
    ProbeTokensFn,
    compute_required_tokens,
    make_probe,
)
from fallback_policy import (  # noqa: E402
    ENDPOINT_DOWN_LINEAGE_THRESHOLD,
    REJECT_TOO_SMALL,
    REJECT_UNMEASURABLE,
    AgentRotationState,
    LineageRegistry,
    RotationPolicy,
)
from redaction import redact_secrets  # noqa: E402
from models import MODE_DEFAULT_MODELS, MODEL_IDS, VALID_MODELS, resolve_model  # noqa: E402
from parse_agent_output import parse_agent_output as parse_raw_output  # noqa: E402
from sanitize import InvalidInputError, build_user_prompt  # noqa: E402
from status_display import StatusDisplay  # noqa: E402
from stderr_shim import _buffered_stderr_while  # noqa: E402
from synthesize import (  # noqa: E402
    determine_consensus,
    format_report,
    load_agent_output,
)
from backend import AgentBackend  # noqa: E402
from claude_backend import ClaudeBackend  # noqa: E402
from ollama_backend import (  # noqa: E402
    TRANSPORT_CONNECTION_MARKERS,
    TRANSPORT_HTTP_PATTERN,
    OllamaBackend,
)
from ollama_config import ModelSpec, OllamaConfig, OllamaConfigError, resolve_config  # noqa: E402
from ollama_init import write_template  # noqa: E402
from ollama_preflight import (  # noqa: E402
    CONTEXT_GUARD_ENFORCED,
    CONTEXT_GUARD_ESTIMATED,
    OllamaPreflightError,
    PreflightResult,
    preflight,
)
from run_lock import remove_lock, staleness_bound_for_timeout, write_lock  # noqa: E402
from temp_dirs import (  # noqa: E402
    MAGI_DIR_PREFIX,
    cleanup_old_runs,
    create_output_dir,
    project_run_root,
    sweep_legacy_runs_once,
)
from review_context import enrich_code_review_context, resolve_diff  # noqa: E402
from cost import aggregate_cost  # noqa: E402
from input_size import WARN_INPUT_TOKENS, check_input_size, estimate_tokens  # noqa: E402
from finding_validation import parse_diff_ranges, validate_findings  # noqa: E402
from retry_feedback import (  # noqa: E402
    FEEDBACK_TEMPLATES,
    MAX_ERROR_CHARS,
    _retry_feedback_cause,
)
from validate import MAX_INPUT_FILE_SIZE, ValidationError  # noqa: E402
from verdict_markers import (  # noqa: E402
    ECHO_CANARY,
    AgentIdentityError,
    EchoedExampleRejected,
)

# Public star-import contract. Underscore-prefixed symbols from
# ``stderr_shim`` (``_StderrBufferShim``, ``_BinaryStderrBufferShim``,
# ``_buffered_stderr_while``) are intentionally excluded — they are
# private helpers of that module, and tests that need them import from
# ``stderr_shim`` directly. ``_buffered_stderr_while`` is still imported
# here for internal use inside ``run_orchestrator``.
#
# The ``temp_dirs`` symbols (``cleanup_old_runs``, ``create_output_dir``,
# ``MAGI_DIR_PREFIX`` and the underscore-prefixed traversal helpers) are
# re-exported from here so the longstanding ``from run_magi import
# cleanup_old_runs`` pattern in callers and tests continues to work after
# the 2.1.3 split. Future code should import from ``temp_dirs`` directly.
__all__ = [
    "MAGI_DIR_PREFIX",
    "MODEL_IDS",
    "VALID_MODELS",
    "cleanup_old_runs",
    "create_output_dir",
    "resolve_model",
]

# Dispatch/display/report order — deliberately Caspar-first so the adversarial
# critic leads (mirrors the fallback's anti-anchoring ordering). Agents still run
# concurrently (asyncio.gather); this tuple only sets kickoff and stable output
# order. Keep Caspar first.
AGENTS = ("caspar", "melchior", "balthasar")
MAX_HISTORY_RUNS = 5
VALID_MODES = ("code-review", "design", "analysis")


#: Intentos por modelo, por defecto: el original + 1 reintento con feedback correctivo.
DEFAULT_MAX_ATTEMPTS = 2
MIN_ATTEMPTS = 1

#: Cota superior. **No es paranoia:** sin ella, un ``--max-attempts 1000`` (un cero de mas)
#: convierte un mago obstinado en **mil llamadas** -- caras en Ollama (`:cloud` es de pago)
#: y **cientos de dolares en Claude**, con un run que no termina nunca. El proyecto ya
#: valida asi los enteros del TOML (NR3 de MS1: valor invalido -> error, sin fallback
#: silencioso); esto es la misma regla en el flag.
MAX_ATTEMPTS_CAP = 10


def _max_attempts(raw: str) -> int:
    """Valida ``--max-attempts``: entero en ``[MIN_ATTEMPTS, MAX_ATTEMPTS_CAP]``.

    Args:
        raw: El valor tal como llega de la linea de comandos.

    Returns:
        El entero validado.

    Raises:
        argparse.ArgumentTypeError: Si no es un entero, o cae fuera del rango. **Falla
            cerrado**: nunca degrada a un default silencioso.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-attempts must be an integer (got {raw!r})"
        ) from exc
    if not MIN_ATTEMPTS <= value <= MAX_ATTEMPTS_CAP:
        raise argparse.ArgumentTypeError(
            f"--max-attempts must be between {MIN_ATTEMPTS} and {MAX_ATTEMPTS_CAP} (got {value})"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with mode, input, timeout, output_dir.
    """
    parser = argparse.ArgumentParser(description="MAGI Orchestrator")
    parser.add_argument("mode", choices=VALID_MODES, help="Analysis mode")
    parser.add_argument("input", help="Path to file or inline text to analyze")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-agent timeout in seconds (default: 900)",
    )
    parser.add_argument("--output-dir", help="Directory for agent outputs")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="FILE",
        help=(
            "Redirect the human-readable verdict report (the banner + findings) to "
            "FILE, SUPPRESSING it on stdout -- useful when stdout is not captured "
            "(e.g. a remote/phone client). The write is atomic; on a write failure it "
            "warns and falls back to stdout so the verdict is never lost. The "
            "structured per-agent JSON still goes to --output-dir."
        ),
    )
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        default=None,
        help=(
            "LLM model for all agents. When omitted, the default depends "
            "on the mode: opus for code-review and design, sonnet for "
            "analysis. Pass --model explicitly to override."
        ),
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=MAX_HISTORY_RUNS,
        help=(
            f"Maximum number of non-live magi-run-* temp dirs to retain "
            f"(default: {MAX_HISTORY_RUNS}). Live (locked) dirs are excluded "
            f"from the count and never deleted, so the on-disk total can "
            f"exceed this value under concurrent or stale-locked runs. "
            f"``--keep-runs 1`` wipes all prior non-live runs, keeping only "
            f"the current one. ``--keep-runs 0`` is rejected. "
            f"``--keep-runs -1`` disables cleanup entirely."
        ),
    )
    parser.add_argument(
        "--no-status",
        dest="show_status",
        action="store_false",
        help="Disable the live status tree display",
    )
    parser.add_argument(
        "--max-attempts",
        type=_max_attempts,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            f"Attempts per model, {MIN_ATTEMPTS}..{MAX_ATTEMPTS_CAP} "
            f"(default: {DEFAULT_MAX_ATTEMPTS}). With --ollama, the TOML's "
            f"max_attempts_per_model overrides it."
        ),
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base ref for code-review context enrichment (default: main)",
    )
    parser.add_argument(
        "--no-enrich",
        dest="enrich",
        action="store_false",
        help="Disable code-review context enrichment (use for untrusted PRs)",
    )
    parser.add_argument(
        "--enrich-max-chars",
        type=int,
        default=512_000,
        help="Max chars of enriched code-review context (default: 512000)",
    )
    parser.add_argument(
        "--warn-input-tokens",
        type=int,
        default=WARN_INPUT_TOKENS,
        help=(
            f"Warn when estimated input tokens exceed this value "
            f"(default: {WARN_INPUT_TOKENS}). Warning reflects the RAW input "
            f"before enrichment; the estimate is approximate (English chars/4). "
            f"MAGI reviews the input whole; detect-and-warn only, not a hard limit."
        ),
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use the OpenAI-compatible Ollama backend instead of `claude -p`.",
    )
    parser.add_argument(
        "--ollama-init",
        action="store_true",
        help="Scaffold ./.claude/magi-ollama.toml from defaults and exit.",
    )
    parser.set_defaults(show_status=True, enrich=True)
    args = parser.parse_args(argv)
    # ``--keep-runs 0`` is ambiguous: a naive reading is "keep nothing"
    # (wipe), but the legacy contract for ``cleanup_old_runs(keep)`` treats
    # a negative result as "disabled". Rather than bake a surprise into the
    # CLI, we reject 0 explicitly so operators pick the side they mean:
    # ``--keep-runs 1`` to wipe everything except the current run, or
    # ``--keep-runs -1`` to disable cleanup entirely.
    if args.keep_runs == 0:
        parser.error(
            "--keep-runs 0 is ambiguous: use --keep-runs 1 to wipe all prior "
            "runs (keeping only the one about to be created), or --keep-runs "
            "-1 to disable cleanup entirely."
        )
    if args.warn_input_tokens <= 0:
        parser.error("--warn-input-tokens must be a positive integer")
    if args.ollama and args.model is not None:
        parser.error(
            "--model does not apply with --ollama; per-mage models are "
            "configured in magi-ollama.toml / MAGI_OLLAMA_MODEL_*."
        )
    # INVARIANT: --model must stay None when --ollama is set. Do NOT collapse
    # this into `args.model or MODE_DEFAULT_MODELS[...]` — that would silently
    # re-enable `--ollama --model` and feed Ollama a Claude-shaped model name.
    if not args.ollama and args.model is None:
        args.model = MODE_DEFAULT_MODELS[args.mode]
    return args


async def launch_agent(
    agent_name: str,
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    spec: ModelSpec = ModelSpec("opus", "anthropic"),
    backend: AgentBackend | None = None,
) -> dict[str, Any]:
    """Launch one agent via *backend* and return validated output.

    Writes the prompt + raw artifacts, then parses and validates. The
    transport (claude -p, Ollama HTTP, ...) lives in the backend. Defaults
    to ClaudeBackend so existing callers keep 3.x behavior.

    Args:
        agent_name: One of 'melchior', 'balthasar', 'caspar'.
        agents_dir: Directory containing agent prompt .md files.
        prompt: The prompt payload to send to the agent.
        output_dir: Directory for raw and parsed output files.
        timeout: Timeout in seconds per agent.
        spec: The model to run, as a :class:`ModelSpec` (tag + lineage). The
            lineage is carried so the rotation path can condemn it on failure;
            only ``spec.model`` (the bare tag) reaches ``backend.run``. Defaults
            to the opus spec so the ~40 legacy call sites keep 3.x behavior.
        backend: Transport backend to use. Defaults to ClaudeBackend.

    Returns:
        Validated agent output dictionary.

    Raises:
        TimeoutError: If the agent does not respond within timeout. On this
            path the subprocess is killed and reaped (``wait()``) and any
            buffered stderr is persisted to ``{agent_name}.stderr.log`` and
            included in the error message for post-mortem diagnosis.
        RuntimeError: If the subprocess exits with a non-zero code.
        ValidationError: If the agent output fails schema validation. Caught
            and retried by ``run_orchestrator.tracked_launch`` (2.2.0).
        json.JSONDecodeError: If the parsed text is not valid JSON. Raised
            by ``parse_agent_output``, propagated through ``launch_agent``,
            and caught + retried by ``run_orchestrator.tracked_launch``
            (2.2.4).
        ValueError: From ``resolve_model`` for unknown model short names,
            from ``parse_agent_output`` for unrecognised CLI output shapes,
            or when the agent's raw stdout (``{agent_name}.raw.json``)
            exceeds :data:`validate.MAX_INPUT_FILE_SIZE`. NOT retried —
            these are configuration / structural failures that a re-roll
            cannot fix.
        asyncio.CancelledError: If the orchestrating task is cancelled
            while ``launch_agent`` is awaiting the subprocess. Propagated
            unchanged so the cancel reaches the surrounding
            ``asyncio.gather`` in ``run_orchestrator``; ``tracked_launch``
            treats this as a non-retryable failure (the run as a whole is
            shutting down).
    """
    if backend is None:
        backend = ClaudeBackend()

    system_prompt_file = os.path.join(agents_dir, f"{agent_name}.md")
    raw_file = os.path.join(output_dir, f"{agent_name}.raw.json")
    parsed_file = os.path.join(output_dir, f"{agent_name}.json")

    prompt_file = os.path.join(output_dir, f"{agent_name}.prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    stdout = await backend.run(
        agent_name, system_prompt_file, prompt, spec.model, timeout, output_dir
    )

    with open(raw_file, "wb") as f:
        f.write(stdout)

    parse_raw_output(raw_file, parsed_file)
    payload = load_agent_output(parsed_file)

    # --- Los dos guards que corren DESPUES de que el schema valide (MS2) ---
    #
    # Ambos levantan subclases de ``ValidationError``, asi que el guard de reintento del
    # orquestador los captura: el modelo recibe feedback correctivo y puede arreglarse.

    # R6 -- canario anti-eco. El ULTIMO cinturon: el sentinel ya impide que se extraiga
    # cualquier cosa de fuera de las marcas, y el prompt no pone nada valido ENTRE ellas.
    # Queda un solo camino teorico: que el modelo tome el ejemplo trabajado de FUERA y lo
    # envuelva EL MISMO en marcas. Su huella dactilar lo delata.
    if all(payload.get(key) == value for key, value in ECHO_CANARY.items()):
        raise EchoedExampleRejected(
            "the verdict is a verbatim copy of the prompt's example, not your analysis"
        )

    # R10 -- identidad. ``load_agent_output`` valida que ``agent`` este en el ENUM, pero
    # nadie validaba que fuera el mago que se LANZO. Un nombre duplicado mata el run
    # entero; uno unico pero equivocado mete el texto de un mago en el asiento de otro, y
    # el consenso lo cuenta como una perspectiva independiente **que nunca existio**.
    #
    # Case-insensitive: un modelo que escribe "Caspar" EMITIO bien su veredicto. Matarlo
    # por una mayuscula es un reintento regalado, y el enum ya se valida aparte.
    claimed = str(payload["agent"]).strip()
    if claimed.casefold() != agent_name.casefold():
        raise AgentIdentityError(
            f"verdict claims agent {claimed!r} but {agent_name!r} was launched"
        )

    return payload


class _DisplayLogGate:
    """Once-per-run gate that logs the first display-update failure.

    Owns the "has the first failure already been logged" flag that used
    to live as module-level mutable state. A fresh instance is created
    by :func:`run_orchestrator` for every run, so there is no residual
    state across runs and no ``global`` plumbing for tests to reset.
    Each :func:`_safe_display_update` call is threaded through the gate
    belonging to the enclosing orchestrator invocation.
    """

    __slots__ = ("_logged",)

    def __init__(self) -> None:
        self._logged: bool = False

    def emit_once(self, exc: BaseException) -> None:
        """Log *exc* to stderr exactly once for this gate's lifetime.

        Subsequent calls are no-ops. The helper must never propagate a
        new exception — doing so would mask the original shutdown signal
        the caller is already re-raising. Failures inside the ``print``
        itself (stream closed, etc.) are swallowed silently for the same
        reason.
        """
        if self._logged:
            return
        self._logged = True
        try:
            print(
                f"[!] WARNING: status display update failed ({exc!r}) "
                f"\u2014 live tree may be stale for the rest of this run",
                file=sys.stderr,
            )
        except BaseException:  # noqa: BLE001 — never let logging shadow shutdown
            pass


def _safe_display_update(
    display: StatusDisplay | None,
    name: str,
    state: str,
    log_gate: _DisplayLogGate,
) -> None:
    """Update a status display, swallowing any exception on failure.

    During shutdown paths (``KeyboardInterrupt``, ``CancelledError``, event
    loop closing) the display's underlying stream may already be closed or
    in a broken state. In that case a ``display.update`` call can raise,
    and propagating that new exception would mask the original shutdown
    signal. This helper isolates the display update so that the caller's
    ``raise`` statement always preserves the real cause.

    The first exception per run is logged to stderr through *log_gate* so
    the operator knows the live tree is blind; subsequent exceptions stay
    silent to prevent the redraw path from flooding the log on every tick.

    Args:
        display: The status display, or ``None`` to skip the update.
        name: Agent name to update.
        state: New state for the agent row.
        log_gate: Run-scoped gate that enforces the once-per-run log rule.
    """
    if display is None:
        return
    try:
        display.update(name, state)
    except BaseException as exc:  # noqa: BLE001 — see docstring shutdown-path contract
        # Catches ``Exception`` subclasses plus ``CancelledError``,
        # ``KeyboardInterrupt``, and ``SystemExit``. The helper is invoked
        # from ``tracked_launch``'s ``except BaseException`` clause which
        # then re-raises the *original* signal — if we let the display's
        # own BaseException escape here, that outer ``raise`` never runs
        # and the real shutdown reason is lost.
        log_gate.emit_once(exc)


_FAIL_SCHEMA = "schema"
_FAIL_HTTP = "http"
_FAIL_CONNECTION = "connection"
_FAIL_TIMEOUT = "timeout"
_FAIL_UNEXPECTED = "unexpected"

#: The transport-message contract is OWNED by ``ollama_backend`` (the module that raises
#: the messages); ``_classify`` imports the marker constants from there rather than
#: duplicating the strings, so the two cannot drift (MAGI gate, Balthasar -- single source
#: of truth). ``test_classify_matches_the_real_ollama_backend_messages`` pins the coupling.
_HTTP_MESSAGE_RE = re.compile(TRANSPORT_HTTP_PATTERN)
_CONNECTION_MESSAGE_MARKERS = TRANSPORT_CONNECTION_MARKERS


def _classify(exc: BaseException) -> str:
    """Classify a failure to decide retry SCOPE and whether the fast-fail fires.

    Args:
        exc: The exception raised by :func:`launch_agent` / the backend.

    Returns:
        One of ``_FAIL_SCHEMA`` / ``_FAIL_HTTP`` / ``_FAIL_CONNECTION`` /
        ``_FAIL_TIMEOUT`` / ``_FAIL_UNEXPECTED``. A ``RuntimeError`` is transport
        ONLY when its message matches a known backend signature; one matching
        neither falls through to ``_FAIL_UNEXPECTED`` so a genuine coding bug
        surfaces on the spot instead of being retried, rotated and globalized.
    """
    if isinstance(exc, (ValidationError, json.JSONDecodeError)):
        return _FAIL_SCHEMA
    # HTTPError BEFORE URLError: HTTPError is a URLError subclass, and a 5xx must
    # not be read as a connection failure (that would fast-fail a live endpoint).
    if isinstance(exc, urllib.error.HTTPError):
        return _FAIL_HTTP
    if isinstance(exc, urllib.error.URLError):
        # A socket timeout arrives WRAPPED: URLError(socket.timeout()). ``socket.timeout``
        # IS an alias of ``TimeoutError`` on 3.10+ (the project floor is 3.12), so the
        # first check already covers it -- ``socket.timeout`` is named explicitly for a
        # self-documenting predicate that does not rely on the reader knowing the alias
        # (MAGI gate, Caspar). Not classifying it as a timeout would feed a slow-but-alive
        # endpoint into the endpoint-down fast-fail (decisions #50/#98).
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            return _FAIL_TIMEOUT
        return _FAIL_CONNECTION
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError is an alias since 3.11
        return _FAIL_TIMEOUT
    if isinstance(exc, ConnectionError):
        return _FAIL_CONNECTION
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if _HTTP_MESSAGE_RE.search(msg):
            return _FAIL_HTTP
        if any(marker in msg for marker in _CONNECTION_MESSAGE_MARKERS):
            return _FAIL_CONNECTION
        return _FAIL_UNEXPECTED
    return _FAIL_UNEXPECTED


#: The retry causes that map to a KNOWN, distinct corrective instruction. Every
#: :class:`verdict_markers.VerdictExtractionError` subclass gets its own entry, plus
#: the two causes ``verdict_markers`` does not raise: a JSON-decode failure INSIDE an
#: otherwise well-formed block, and the generic 7-key schema contract.


def _build_retry_prompt(
    original_prompt: str,
    error: ValidationError | json.JSONDecodeError,
    *,
    api_key: str | None = None,
) -> str:
    """Return the retry prompt with CAUSE-SPECIFIC corrective feedback appended.

    When :func:`launch_agent` raises a schema-scoped exception (a
    :class:`ValidationError` -- including every :mod:`verdict_markers` extraction
    failure -- or a :class:`json.JSONDecodeError`) on an attempt, :func:`_attempt_model`
    calls this helper to build the replacement prompt for the retry. The original
    user prompt is preserved verbatim so the agent's task is unchanged; the
    template picked by :func:`_retry_feedback_cause` names the SPECIFIC defect --
    missing markers, an unterminated block, more than one block, a copied example,
    a wrong agent identity, undecodable JSON, or a missing schema key -- so the
    model spends its one retry on the instruction that actually applies. Handing a
    model that emitted NO markers the "emit exactly one block" instruction (the old,
    single-template behaviour) wastes the retry on a FALSE diagnosis and the mage
    dies. The envelope delimiter ``---RETRY-FEEDBACK---`` is intentionally distinct
    from user input so the model can identify the corrective block even if the
    original prompt already contains arbitrary markdown.

    Args:
        original_prompt: The exact prompt sent on the failed attempt.
        error: The exception that triggered the retry. A :class:`ValidationError`
            (schema mismatch, or any :mod:`verdict_markers` extraction failure) or
            a :class:`json.JSONDecodeError` (content between the markers is not
            parseable JSON).
        api_key: The Ollama api_key, if any. The error is REDACTED against it
            before being embedded -- the retry prompt is written to
            ``{agent}.prompt.txt``, and NR3b requires the key to appear on no
            surface (MAGI gate, Caspar): the error is redacted at every other
            boundary, so this one must not be the exception.

    Returns:
        A new prompt string that concatenates the original prompt with a
        cause-specific feedback block.
    """
    detail = redact_secrets(str(error), api_key)
    if len(detail) > MAX_ERROR_CHARS:
        detail = detail[:MAX_ERROR_CHARS] + "..."
    cause = _retry_feedback_cause(error)
    feedback = FEEDBACK_TEMPLATES[cause].format(error=detail)
    return f"{original_prompt}\n\n{feedback}"


@dataclass(frozen=True)
class RotationRuntimeConfig:
    """The rotation knobs the orchestrator needs at runtime (a slice of OllamaConfig).

    Kept separate from :class:`OllamaConfig` so the orchestrator does not depend on
    the whole config object (low coupling) and tests can build one in a line. It
    carries EVERY field the rotation path reads -- including the two the context
    guard needs (``output_headroom_tokens``, ``input_margin_pct``): a slice that
    lacked them and was handed to ``compute_required_tokens`` would be a type error
    under mypy --strict (finding by Balthasar, Checkpoint 2).
    """

    max_attempts_per_model: int
    max_probe_attempts: int
    retry_backoff_seconds: float
    strict_context_guard: bool
    output_headroom_tokens: int
    input_margin_pct: int
    probe_timeout_seconds: int
    api_key: str | None  # for redact_secrets at the rotation boundaries

    @classmethod
    def from_config(cls, config: OllamaConfig) -> "RotationRuntimeConfig":
        """Derive the runtime slice from the resolved config. The ONLY constructor.

        Config drift (preflight validating one value while the orchestrator runs
        another) is impossible if exactly one place copies the fields (finding by
        Caspar, Checkpoint 2). Hand-building this object anywhere else re-opens that
        door -- tests included, which is why the fixtures go through
        ``dataclasses.replace`` on a real one.

        Args:
            config: The resolved OllamaConfig (built once, in select_backend).

        Returns:
            The runtime slice the rotation path reads.
        """
        return cls(
            max_attempts_per_model=config.max_attempts_per_model,
            max_probe_attempts=config.max_probe_attempts,
            retry_backoff_seconds=config.retry_backoff_seconds,
            strict_context_guard=config.strict_context_guard,
            output_headroom_tokens=config.output_headroom_tokens,
            input_margin_pct=config.input_margin_pct,
            probe_timeout_seconds=config.probe_timeout_seconds,
            api_key=config.api_key,
        )


@dataclass(frozen=True)
class LaunchEnv:
    """Everything :func:`_attempt_model` needs to actually launch an agent.

    ``tracked_launch`` is a closure inside :func:`run_orchestrator` and reads these
    from the enclosing scope. :func:`_attempt_model` is a MODULE-LEVEL function, so
    it cannot: it must receive them explicitly. Passing them as one frozen value
    object keeps the signature honest without an implicit dependency on the
    orchestrator's locals (findings by Balthasar and Caspar, Checkpoint 2).
    """

    agents_dir: str
    output_dir: str
    timeout: int
    backend: AgentBackend
    api_key: str | None


@dataclass(frozen=True)
class RotationContext:
    """The whole rotation apparatus, in ONE optional parameter.

    Bundling it keeps :func:`run_orchestrator`'s signature back-compatible: the ~40
    existing call sites pass nothing and get exactly the v4 behaviour
    (``rotation=None`` => single-shot schema retry, no rotation).
    """

    registry: LineageRegistry
    policy: RotationPolicy
    config: RotationRuntimeConfig
    probe: ProbeTokensFn
    #: Everything the preflight MEASURED. Carried here so Phase 4 has a defined
    #: source for context_guard / lineage_warnings / token_estimate_delta instead
    #: of inventing them (finding by Balthasar, Checkpoint 2).
    preflight: PreflightResult
    #: Set the moment the endpoint is declared dead. Every mage checks it before
    #: each attempt and each rotation, so the abort is actually FAST. Without it,
    #: "fast-fail" only aborts after the siblings finish burning their own budgets
    #: against the same dead server (finding by Balthasar, Checkpoint 2).
    endpoint_down: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)


class _EndpointDown(RuntimeError):
    """The endpoint itself is unreachable -- rotating cannot help.

    A dedicated type, not a bare RuntimeError: ``gather(return_exceptions=True)``
    turns every exception into a value, so the orchestrator must be able to pick
    THIS one out of the results and re-raise it. A string match would be fragile;
    a type is not.
    """


class _AttemptsExhausted(Exception):
    """A model spent its whole attempt budget. Carries WHY, so the caller can decide
    whether the failure is local (schema) or run-wide (transport).

    Attributes:
        kind: "schema" | "connection" | "http" | "timeout" (see :func:`_classify`).
        detail: Redacted, human-readable cause, for telemetry and stderr.
        http_status: The HTTP status when the failure was an HTTPError, else None.
            R13's structured fallback_reason needs it, and re-parsing it out of the
            message string later would be the stringly-typed fragility that field
            exists to remove.
        attempts: How many attempts the model consumed before giving up.
    """

    def __init__(
        self, kind: str, detail: str, *, http_status: int | None = None, attempts: int = 0
    ) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.http_status = http_status
        self.attempts = attempts


async def _attempt_model(
    name: str,
    spec: ModelSpec,
    ctx: RotationContext,
    env: LaunchEnv,
    prompt: str,
    on_retry: Callable[[], None],
) -> dict[str, Any]:
    """Run ONE model until it succeeds or spends its attempt budget.

    A schema failure gets the corrective feedback (the model answered: it needs to
    be told what was wrong). A transport failure gets a backoff and the ORIGINAL
    prompt (the model never answered: there is nothing to correct, and hammering a
    rate-limited server instantly just burns the budget).

    Every dependency is EXPLICIT: this is a module-level function, so it cannot read
    the orchestrator's closure. ``on_retry`` is injected instead of touching the
    display directly -- the attempt loop must not know a status display exists.

    Args:
        name: Mage name.
        spec: The model to run.
        ctx: Rotation context (attempt budget, backoff).
        env: Launch environment (agents dir, output dir, timeout, backend, api key).
        prompt: The original user prompt.
        on_retry: Called once per retry, for the "retrying" display state.

    Returns:
        The validated agent output.

    Raises:
        _AttemptsExhausted: Every attempt failed. Carries the LAST failure's kind
            and its REDACTED detail, so the caller can scope it (local vs run-wide).
        _EndpointDown: A sibling already proved the endpoint is dead.
        asyncio.CancelledError: Propagated untouched -- a shutdown is not a failure.
        KeyboardInterrupt, SystemExit: Propagated untouched.
    """
    attempt_prompt = prompt
    last: Exception | None = None
    for attempt in range(ctx.config.max_attempts_per_model):
        if ctx.endpoint_down.is_set():
            # A sibling already proved the endpoint is dead. Do not spend an attempt
            # (and up to --timeout seconds) discovering it again.
            raise _EndpointDown("endpoint down (detected by another agent)")
        try:
            return await launch_agent(
                name,
                env.agents_dir,
                attempt_prompt,
                env.output_dir,
                env.timeout,
                spec,
                backend=env.backend,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise  # shutdown: never retried, never swallowed
        except Exception as exc:  # noqa: BLE001 -- classified and re-raised
            if _classify(exc) == _FAIL_UNEXPECTED:
                # A TypeError/AttributeError from OUR code is not an endpoint failure.
                # Retrying it would burn attempts, rotate the mage, and BURY the bug
                # behind a degraded run (Balthasar, Checkpoint 2). Fail loud.
                raise
            last = exc
            if attempt + 1 >= ctx.config.max_attempts_per_model:
                break
            # isinstance (not ``_classify(exc) == _FAIL_SCHEMA``) so mypy narrows exc to
            # the type _build_retry_prompt accepts -- the two are equivalent by
            # construction (_classify returns schema iff it is one of these).
            if isinstance(exc, (ValidationError, json.JSONDecodeError)):
                # the model needs correcting; redact the api_key from the embedded error
                attempt_prompt = _build_retry_prompt(prompt, exc, api_key=env.api_key)
            else:
                attempt_prompt = prompt  # nothing to correct
                await asyncio.sleep(ctx.config.retry_backoff_seconds)
            on_retry()
    if last is None:
        # NOT an assert: ``python -O`` strips asserts, and this would then fall
        # through to a NameError in production (finding by Balthasar, Checkpoint 2).
        raise RuntimeError(
            f"_attempt_model exited its loop without a failure for {name} -- "
            f"max_attempts_per_model must be >= 1 (got {ctx.config.max_attempts_per_model})"
        )
    raise _AttemptsExhausted(
        _classify(last),
        redact_secrets(str(last), env.api_key),
        http_status=getattr(last, "code", None),
        attempts=ctx.config.max_attempts_per_model,
    ) from last


# R13's telemetry enum for fallback_reason.kind -- the ONLY values that may reach the
# report (NR6b: no magic strings). The internal connection/http split from _classify (T9)
# normalizes here: both are "transport". Keeping the split out of telemetry is the point of
# Balthasar's Checkpoint-2 finding -- R13 restricts kind to three values.
_KIND_TRANSPORT = "transport"
_KIND_SCHEMA = "schema"
_KIND_TIMEOUT = "timeout"
_KIND_NO_FITTING_CANDIDATE = "no_fitting_candidate"  # the mage DIED: not a live-rotation kind

#: internal _FAIL_* label (T9) -> R13 telemetry enum value.
_KIND_BY_FAIL = {
    _FAIL_CONNECTION: _KIND_TRANSPORT,
    _FAIL_HTTP: _KIND_TRANSPORT,
    _FAIL_SCHEMA: _KIND_SCHEMA,
    _FAIL_TIMEOUT: _KIND_TIMEOUT,
}


def _reason(
    old: ModelSpec,
    new: ModelSpec | None,
    exc: _AttemptsExhausted,
    state: AgentRotationState,
) -> dict[str, Any]:
    """Build R13's STRUCTURED fallback_reason -- queryable, not greppable.

    Answers the question that makes anyone open the telemetry: *which model is failing,
    and why?* A prose string cannot answer it at scale.

    Args:
        old: The model that failed.
        new: The model rotated to, or None when the mage died with no candidate.
        exc: The exhausted-attempts failure.
        state: The mage's rotation state (for the rotation count).

    Returns:
        The structured reason dict written verbatim into the report. ``kind`` is
        normalized to R13's telemetry enum ({transport, schema, timeout}) when the mage
        rotated -- the internal connection/http distinction NEVER leaks -- and is
        ``no_fitting_candidate`` when the mage died with no candidate.
    """
    kind = (
        _KIND_BY_FAIL.get(exc.kind, exc.kind)  # normalize connection/http -> transport
        if new is not None
        else _KIND_NO_FITTING_CANDIDATE
    )
    return {
        "kind": kind,
        "from_model": old.model,
        "from_lineage": old.lineage,
        "to_model": new.model if new is not None else None,
        "to_lineage": new.lineage if new is not None else None,
        "detail": exc.detail,  # already redacted at construction
        "http_status": exc.http_status,
        "attempts": exc.attempts,
        "rotations_done": state.rotations_done,
    }


def _announce_rotation(
    name: str,
    old: ModelSpec,
    new: ModelSpec,
    exc: _AttemptsExhausted,
    ctx: RotationContext,
) -> None:
    """Say out loud that a mage changed model, and WHY. Never silent (R9).

    A silent fallback is the one failure mode this feature must not have: whoever reads
    the verdict has to know that the mage was not the judge they configured.

    Args:
        name: The rotating mage.
        old: The model that failed.
        new: The model it rotates to.
        exc: The exhausted-attempts failure (kind + ALREADY-redacted detail).
        ctx: Rotation context (for the attempt count).
    """
    print(
        f"[!] {name}: {old.model} failed "
        f"{ctx.config.max_attempts_per_model}x ({exc.kind}: {exc.detail}) "
        f"-> rotating to {new.model} ({new.lineage})",
        file=sys.stderr,
    )


async def _record_failure(
    state: AgentRotationState,
    ctx: RotationContext,
    spec: ModelSpec,
    exc: _AttemptsExhausted,
) -> None:
    """Route a spent model's failure to the right scope, by its NATURE.

    Schema failures stay LOCAL: the model answered -- it just could not satisfy THIS
    mage's contract, and may serve another mage. Transport failures go RUN-WIDE: the
    model is down for everyone, so no other mage should burn attempts against it.

    Args:
        state: The mage's rotation state (schema failures accumulate here).
        ctx: Registry + config.
        spec: The model that exhausted its attempts.
        exc: The exhausted-attempts failure.

    Raises:
        _EndpointDown: When this registration crosses the endpoint-down threshold
            (>= 2 distinct lineages refusing the connection). Rotating is pointless
            if what died is the server. It also SETS ``ctx.endpoint_down`` so the
            siblings stop before burning their own budgets on the same dead server;
            a bare ``RuntimeError`` would be captured by ``gather`` and the abort
            would never fire (R15 -- Caspar, Checkpoint 2: the fast-fail was mute).
    """
    if exc.kind == _FAIL_SCHEMA:
        state.failed_lineages.add(spec.lineage)
        return
    endpoint_down = await ctx.registry.register_transport_failure(
        spec.lineage, connection=(exc.kind == _FAIL_CONNECTION)
    )
    if endpoint_down:
        ctx.endpoint_down.set()  # tell the siblings to stop, NOW
        raise _EndpointDown(
            f"endpoint down: {ENDPOINT_DOWN_LINEAGE_THRESHOLD} distinct lineages "
            f"refused the connection ({exc.detail}). Rotating cannot help when the "
            f"server itself is unreachable."
        )


async def _rotate(
    name: str,
    state: AgentRotationState,
    ctx: RotationContext,
    prompt: str,
) -> ModelSpec | None:
    """Propose a candidate under the lock, VERIFY it with a probe outside it, commit.

    The probe is what makes the guard real: the cached window says a model *should* fit,
    but only the model's own tokenizer knows whether the payload actually does (chars/4
    underestimates by 15-20%). A candidate that would truncate is rejected and the next
    proposed -- never run "and hope", because a truncated verdict is indistinguishable
    from a legitimate one.

    I/O NEVER happens inside the registry lock: claim_next returns, the lock is released,
    and only then do we probe. Holding the lock across a network call would serialize the
    three mages and be a deadlock waiting for a future second lock.

    Args:
        name: The rotating mage.
        state: Its local rotation state (mutated: window_rejected).
        ctx: Registry + policy + runtime config + probe.
        prompt: The exact payload the agent will receive (what we measure).

    Returns:
        The committed ModelSpec, or None if no candidate fits within
        ``max_probe_attempts`` -- in which case the mage dies and the run degrades.

    Raises:
        _EndpointDown: Another agent already proved the endpoint dead.
    """
    for _ in range(ctx.config.max_probe_attempts):
        if ctx.endpoint_down.is_set():
            raise _EndpointDown("endpoint down (detected by another agent)")
        candidate = await ctx.registry.claim_next(name, ctx.policy, state)
        if candidate is None:
            return None  # nothing eligible left

        try:
            exact = await ctx.probe(candidate.model, prompt, ctx.config.probe_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 -- a probe is never fatal (R18)
            # An injected or future probe that raises must not kill a mage over a
            # measurement (R18). Treat as unmeasurable.
            print(
                f"WARNING: probe raised for {candidate.model} "
                f"({redact_secrets(str(exc), ctx.config.api_key)}); treating as unmeasurable",
                file=sys.stderr,
            )
            exact = None
        window = ctx.policy.window_of(candidate.model)

        # THREE epistemic states, never collapsed. Folding "unknown" into a number makes
        # every unknown-window candidate look infinitely too small, so on any endpoint
        # without window data NOTHING is eligible and every mage dies -- the guard becomes
        # the outage.
        if ctx.config.strict_context_guard and (exact is None or window is None):
            state.window_rejected[candidate.model] = REJECT_UNMEASURABLE
            continue

        if window is None:
            # Nothing to compare against: no check is possible. Non-strict => accept,
            # loudly. This mage runs UNMEASURED, so the run-level context_guard must be
            # downgraded to "estimated" (R16 -- MAGI gate Loop 1 pass 2).
            state.ran_unmeasured = True
            return candidate

        # The window IS known. Even without an exact count we check against the ESTIMATE,
        # and reject if it does not fit even then (decision #96: degrading ACCURACY must
        # not degrade PRUDENCE -- accepting blindly re-opens silent truncation, the very
        # failure R5b exists to prevent).
        measured = exact is not None
        # Narrow on ``exact`` directly (not via ``measured``): mypy does not track that
        # ``measured`` implies ``exact is not None``, so ``exact if measured else ...``
        # would leave payload typed ``int | None``.
        payload = exact if exact is not None else estimate_tokens(prompt)
        required = compute_required_tokens(
            payload,
            output_headroom_tokens=ctx.config.output_headroom_tokens,
            input_margin_pct=ctx.config.input_margin_pct,
            exact=measured,  # unmeasured => apply the margin
        )
        if required > window:
            state.window_rejected[candidate.model] = (
                REJECT_TOO_SMALL if measured else REJECT_UNMEASURABLE
            )
            continue

        # Record whether this mage's committed model was EXACTLY measured: an
        # estimate-based accept means the run-level context_guard must read "estimated",
        # not "enforced" (R16 honesty -- MAGI gate Loop 1 pass 2).
        state.ran_unmeasured = not measured
        return candidate  # fits: proven, or estimated + margin

    return None  # probe budget spent


def _load_input_content(input_arg: str) -> tuple[str, str]:
    """Resolve the CLI ``input`` argument to (content, label).

    If *input_arg* is a path to an existing file, the file is read
    with ``encoding="utf-8"`` and ``errors="replace"`` so that a
    cp1252-encoded source (default for Windows tooling that does not
    set an explicit encoding) does not crash MAGI on the first byte
    that is not a valid UTF-8 start byte. Invalid bytes are replaced
    with U+FFFD and the run continues; readable portions of the file
    survive verbatim. The size check still applies.

    If *input_arg* is not a file path, it is returned as inline text
    unchanged — Python str values cannot have an encoding mismatch.

    Known limitation (tracked, future fix): a value that *looks* like a
    path but does not exist (e.g. a typo'd file path) is not distinguished
    from genuine inline text — ``os.path.isfile`` is ``False``, so the
    literal path string becomes the prompt body and is silently reviewed
    as content instead of failing closed. Surfaced by the v3.0.0 Block B
    over-suppression-probe gate run (a missing bundle path was reviewed as
    path-only text). A future fix should detect path-shaped-but-missing
    inputs (no whitespace/newline plus a path separator or known
    extension) and raise instead of treating them as inline text.

    Args:
        input_arg: The raw value from ``argparse`` for the positional
            ``input`` argument. Either a path to a file or inline
            text.

    Returns:
        Tuple ``(content, label)`` where ``content`` is the prompt
        body and ``label`` is the source description used in the
        eventual prompt envelope (``"File: <path>"`` or
        ``"Inline input"``).

    Raises:
        ValueError: If *input_arg* is a path to a file that exceeds
            :data:`validate.MAX_INPUT_FILE_SIZE`.
    """
    if os.path.isfile(input_arg):
        file_size = os.path.getsize(input_arg)
        if file_size > MAX_INPUT_FILE_SIZE:
            raise ValueError(
                f"Input file {input_arg} is {file_size} bytes, "
                f"exceeding maximum of {MAX_INPUT_FILE_SIZE} bytes."
            )
        # ``errors="replace"`` is the cp1252 hardening shipped in
        # 2.2.6. Windows tooling that writes input files without an
        # explicit encoding produces cp1252 bytes; reading those with
        # strict UTF-8 raises ``UnicodeDecodeError`` on the first
        # byte ≥0x80 that is not a valid UTF-8 start byte. The
        # replacement character (U+FFFD) is preferable to crashing
        # the orchestrator before synthesis.
        with open(input_arg, encoding="utf-8", errors="replace") as f:
            return f.read(), f"File: {input_arg}"
    return input_arg, "Inline input"


def _maybe_enrich(
    mode: str,
    content: str,
    *,
    base_ref: str,
    enrich: bool,
    max_chars: int,
    diff: str | None = None,
) -> tuple[str, str | None]:
    """Enrich code-review input; pass-through otherwise. Boundary fail-safe —
    never raises into the orchestrator.

    Only applies enrichment for ``code-review`` mode when ``enrich`` is
    ``True``. All other modes and ``--no-enrich`` receive the original
    content unchanged with ``None`` as the note.

    The *diff* is the run's single resolved diff source (A2): ``main`` resolves
    it once via :func:`review_context.resolve_diff` and threads the same value to
    BOTH this enrichment path and the finding guard, so the two can never diverge
    and the ``git diff`` invocation runs only once per run (lighter read-only
    probes such as ``_git_toplevel`` and ``_tree_is_clean`` may still run
    independently). The value is forwarded to
    :func:`enrich_code_review_context`, which consumes it verbatim instead of
    re-resolving. ``None`` (the default, used by standalone callers and tests
    that do not pre-resolve) tells enrichment to resolve internally via the same
    :func:`resolve_diff` seam.

    Args:
        mode: Analysis mode (e.g. "code-review", "design", "analysis").
        content: The loaded input content to potentially enrich.
        base_ref: Git base ref for diff enrichment (e.g. "main").
        enrich: Whether enrichment is enabled (``False`` when ``--no-enrich``
            was passed).
        max_chars: Maximum characters allowed for the enriched output.
        diff: The run's resolved diff shared with the guard (``""`` when none),
            or ``None`` to let enrichment resolve it internally.

    Returns:
        Tuple ``(content, note)`` where ``content`` is the (possibly
        enriched) prompt body and ``note`` is a human-readable description
        of the enrichment action, or ``None`` if no enrichment occurred.
    """
    if mode != "code-review" or not enrich:
        return content, None
    try:
        return enrich_code_review_context(
            content, repo_root=os.getcwd(), base_ref=base_ref, max_chars=max_chars, diff=diff
        )
    except Exception as exc:  # noqa: BLE001 — boundary fail-safe
        return content, f"enrichment skipped (boundary error: {exc!r})"


async def select_backend(
    args: argparse.Namespace,
    prompt: str,
) -> tuple[AgentBackend, dict[str, ModelSpec], RotationContext | None]:
    """Return (backend, per-agent models, rotation context) for the chosen mode.

    Ollama path: resolve config once, run the preflight (which MEASURES the payload
    and enforces the lineage/capability/window guards), and assemble the whole
    :class:`RotationContext` -- registry, pure policy, measured preflight result and
    a config-bound probe -- once, at setup. Claude path: :class:`ClaudeBackend` with
    the single ``--model`` for all three agents and ``rotation=None`` (no fallback
    list, so v4 single-shot retry is kept untouched).

    F-M invariant (#6 of v4.0.0): ``resolve_config`` is called exactly once here
    (setup time), never inside tracked_launch/retry -- an ``OllamaConfigError`` must
    not be swallowed by the ``(ValidationError, json.JSONDecodeError)`` retry guard.

    Args:
        args: Parsed CLI namespace (requires .ollama and .model attributes).
        prompt: The built user prompt; the Ollama preflight MEASURES it (R5c) to
            size the context guard, so it must be the exact payload the agents get.

    Returns:
        Tuple of (backend instance, per-agent :class:`ModelSpec` map, rotation
        context or ``None`` for the Claude path).
    """
    if not args.ollama:
        return ClaudeBackend(), {n: ModelSpec(args.model, "anthropic") for n in AGENTS}, None

    config = resolve_config()  # EXACTLY once (invariant #6)
    result = await preflight(config, prompt)  # measures, validates, caches
    policy = RotationPolicy(
        fallback=result.fallback,  # absent fallbacks already dropped (R11.1)
        max_rotations=config.max_rotations,
        min_window_tokens=result.min_window_tokens,  # RAW payload -- pre-filter only (C2-1)
        capabilities=result.capabilities,
        strict_context_guard=config.strict_context_guard,
    )
    rotation = RotationContext(
        registry=LineageRegistry(config.models),
        policy=policy,
        preflight=result,
        # from_config is the ONLY constructor -- hand-building the slice here is what
        # from_config exists to prevent (finding by Balthasar, Checkpoint 2).
        config=RotationRuntimeConfig.from_config(config),
        probe=make_probe(config),  # binds config -> (model, prompt, timeout)
    )
    return OllamaBackend(config), dict(config.models), rotation


def _record_extraction_failure(
    tally: dict[str, "Counter[str]"],
    agent: str,
    error: ValidationError | json.JSONDecodeError,
) -> None:
    """Anota la CAUSA del fallo de extraccion en la telemetria de adherencia (R18).

    Reutiliza el mismo dispatcher que elige el feedback del reintento
    (``retry_feedback._retry_feedback_cause``), asi que la telemetria y la instruccion
    correctiva **no pueden discrepar**: si el modelo recibe "te faltaron las marcas", el
    contador que sube es ``missing_markers``. Duplicar la clasificacion seria plantar la
    semilla de que un dia el reporte diga una cosa y el prompt otra.

    Args:
        tally: El acumulador por agente.
        agent: El mago cuyo intento fallo.
        error: La excepcion que lo tumbo.
    """
    tally[agent][_retry_feedback_cause(error)] += 1


async def run_orchestrator(
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    model: str = "opus",
    *,
    agent_models: dict[str, ModelSpec] | None = None,
    backend: AgentBackend | None = None,
    rotation: RotationContext | None = None,
    show_status: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Run all three agents concurrently and synthesize results.

    Launches agents in parallel, collects results, alerts on failures,
    and runs consensus synthesis on successful outputs.

    Note: for ``code-review``, ``main()`` recomputes ``report['consensus']``
    after applying the finding guard; a caller that uses ``run_orchestrator``
    without ``main()`` receives the pre-guard (unguarded) consensus.

    Args:
        agents_dir: Directory containing agent prompt files.
        prompt: The prompt payload.
        output_dir: Directory for output files.
        timeout: Per-agent timeout in seconds.
        model: Model short name for all agents ('opus', 'sonnet', 'haiku').
            Back-compat: kept so the ~40 existing call sites in tests that
            pass ``model=`` keep working untouched. New callers should pass
            ``agent_models`` + ``backend`` from ``select_backend()``.
        agent_models: Per-agent model map. When None, derived from ``model``
            so every agent uses the same model (BDD-30 back-compat).
        backend: Transport backend. When None, defaults to ClaudeBackend()
            to preserve 3.x behavior for existing callers.
        show_status: Render a live status tree while agents run. When the
            stream is not a TTY, plain one-line-per-event output is emitted
            instead.

    Returns:
        Report dict with 'agents', 'consensus', and optionally
        'degraded' and 'failed_agents' when < 3 agents succeed.

    Raises:
        RuntimeError: If fewer than 2 agents succeed.
    """
    # Back-compat (BDD-30): KEEP `model` so the ~40 existing call sites in
    # tests keep working untouched; derive agent_models from it when not
    # supplied. Values are ModelSpec now (launch_agent takes a spec); the
    # legacy Claude path has no lineage of interest, so tag it "anthropic".
    # New callers (select_backend) pass agent_models + backend.
    if agent_models is None:
        agent_models = {name: ModelSpec(model, "anthropic") for name in AGENTS}
    if backend is None:
        backend = ClaudeBackend()
    successful: list[dict[str, Any]] = []
    failed: list[str] = []
    # Telemetry (2.2.1): names of agents whose first attempt raised
    # ValidationError, regardless of whether the retry recovered.
    # Composes with ``failed`` to give downstream consumers two derived
    # cohorts: ``retried - failed`` is "retry recovered",
    # ``retried & failed`` is "retry also failed".
    retried: set[str] = set()
    # R18 -- telemetria de adherencia. R17 mide UNA vez, con los modelos de HOY; los
    # modelos derivan bajo el mismo tag. Sin esto, el dia que uno empiece a omitir las
    # marcas se veria como "MAGI va lento y rota mucho" -- un sintoma que nadie sabria
    # leer. Aditivo y fail-soft: ``consensus`` no lo lee.
    extraction_failures: dict[str, Counter[str]] = defaultdict(Counter)
    # Rotation telemetry (T10): agent -> its final AgentRotationState. Registered
    # up-front in the rotation path so even a mage that DIES appears, giving Task 13 a
    # defined source for model_configured / model_used / fallback_reason.
    rotation_telemetry: dict[str, AgentRotationState] = {}

    # Fresh log gate per run so the first display failure is always
    # surfaced, even in hosts that reuse the module across orchestrator
    # invocations (tests, long-lived services).
    log_gate = _DisplayLogGate()

    # Display lifecycle invariant (structurally enforced by the
    # ``_buffered_stderr_while`` context manager below): while the status
    # display is rendering, ``sys.stderr`` is replaced with a write-buffer, so
    # any diagnostic print that would otherwise collide with the in-place
    # redraw is deferred until after ``display.stop()`` returns.
    #
    # The display itself captures the *real* ``sys.stderr`` reference at
    # construction time (below), so its own writes go straight to the
    # terminal, not through the buffer.
    display: StatusDisplay | None = (
        StatusDisplay(list(AGENTS), stream=sys.stderr) if show_status else None
    )

    async def _legacy_tracked_launch(name: str) -> dict[str, Any]:
        """The v4 path (``rotation=None``): running + single-shot schema retry.

        Extracted verbatim so ``rotation=None`` keeps EXACTLY the v4 behaviour; the
        one adaptation is that ``agent_models[name]`` is now a :class:`ModelSpec`
        (``launch_agent`` takes a spec), passed as the object, not ``spec.model``.

        State machine emitted to the live display:

        * ``running`` once at entry.
        * ``retrying`` iff the first attempt raised :class:`ValidationError`.
          The retry receives the full ``timeout`` budget and a corrective
          feedback block appended by :func:`_build_retry_prompt`.
        * Terminal state (``success`` | ``timeout`` | ``failed``) emitted
          exactly once by the outer handler, regardless of which attempt
          reached the terminal condition. This is why the retry branch
          does **not** install its own terminal handlers — they would
          duplicate the outer ones and risk drifting out of sync.

        Scope of retry: :class:`ValidationError` only. ``TimeoutError``,
        subprocess exit errors, ``asyncio.CancelledError``, and
        ``BaseException`` subclasses (``KeyboardInterrupt``,
        ``SystemExit``) flow through the outer handler unchanged so the
        degraded-mode and signal paths keep the 2.1.x semantics.
        """
        _safe_display_update(display, name, "running", log_gate)
        try:
            # N intentos (MS2/R13): el camino Claude tenia un retry SINGLE-SHOT fijo; ahora
            # lo gobierna ``--max-attempts`` (default 2 = el comportamiento previo, exacto).
            #
            # Dispara ante deriva de schema (ValidationError -- que incluye TODOS los fallos
            # de extraccion del sentinel) y ante JSON no parseable (json.JSONDecodeError).
            # NUNCA ante timeout / fallo de subproceso / cancelacion / ValueError. Cada
            # reintento recibe un ``timeout`` FRESCO (no el residuo del anterior) y lleva el
            # feedback correctivo ESPECIFICO DE LA CAUSA: el tipo de excepcion selecciona la
            # instruccion, asi que un modelo que olvido las marcas recibe "te faltaron las
            # marcas" y no un mensaje generico de schema.
            attempt_prompt = prompt
            for attempt in range(max_attempts):
                try:
                    result = await launch_agent(
                        name,
                        agents_dir,
                        attempt_prompt,
                        output_dir,
                        timeout,
                        agent_models[name],
                        backend=backend,
                    )
                    break
                except (ValidationError, json.JSONDecodeError) as err:
                    _record_extraction_failure(extraction_failures, name, err)
                    if attempt + 1 >= max_attempts:
                        raise
                    retried.add(name)
                    _safe_display_update(display, name, "retrying", log_gate)
                    attempt_prompt = _build_retry_prompt(prompt, err)
        except (asyncio.TimeoutError, TimeoutError):
            _safe_display_update(display, name, "timeout", log_gate)
            raise
        except BaseException:
            # Catches asyncio.CancelledError (which is BaseException in 3.8+),
            # generic Exception subclasses (including a retry that itself
            # raised ValidationError), KeyboardInterrupt, and SystemExit.
            # We always re-raise — the display update is a best-effort side
            # effect (see ``_safe_display_update``) so a stream already closed
            # during shutdown can never mask the real shutdown signal.
            _safe_display_update(display, name, "failed", log_gate)
            raise
        _safe_display_update(display, name, "success", log_gate)
        return result

    async def _rotating_tracked_launch(name: str, ctx: RotationContext) -> dict[str, Any]:
        """The Ollama path (R1/R2/R12): attempts-per-model, then rotate on exhaustion.

        State machine ``running -> retrying -> rotating -> terminal`` with the
        ``LineageRegistry`` mutations of spec 6.3.1. Lineage cleanup is STRUCTURAL:
        ``agent_slot`` releases the mage's lineage on death and conserves it on
        success, keyed SOLELY on ``state.succeeded``.
        """
        spec = agent_models[name]
        state = AgentRotationState(model_configured=spec, model_used=spec, used={spec.model})
        rotation_telemetry[name] = state  # up-front: even a mage that DIES must appear
        env = LaunchEnv(agents_dir, output_dir, timeout, backend, ctx.config.api_key)

        def on_retry() -> None:
            """Surface a retry (schema or transport) in the live status tree."""
            retried.add(name)
            _safe_display_update(display, name, "retrying", log_gate)

        _safe_display_update(display, name, "running", log_gate)
        async with ctx.registry.agent_slot(name, state) as slot:
            while True:
                try:
                    result = await _attempt_model(name, spec, ctx, env, prompt, on_retry)
                except _AttemptsExhausted as exc:
                    # The model spent its budget. Route the failure by scope (schema
                    # local / transport run-wide) then propose the next model.
                    await _record_failure(state, ctx, spec, exc)  # may raise (endpoint down)
                    _safe_display_update(display, name, "rotating", log_gate)
                    next_spec = await _rotate(name, state, ctx, prompt)
                    if next_spec is None:
                        state.fallback_reason = _reason(spec, None, exc, state)
                        _safe_display_update(display, name, "failed", log_gate)
                        raise  # mage dies with no candidate -> degraded mode (R7)
                    _announce_rotation(name, spec, next_spec, exc, ctx)  # R9: never silent
                    state.fallback_reason = _reason(spec, next_spec, exc, state)
                    spec = next_spec
                    state.rotations_done += 1
                    state.used.add(spec.model)
                    state.model_used = spec  # what the verdict is ACTUALLY judged with
                    continue
                except _EndpointDown:
                    # A sibling proved the endpoint dead; rotating cannot help. (A raw
                    # TimeoutError never reaches here -- _attempt_model classifies it and
                    # folds it into _AttemptsExhausted, so there is no separate "timeout"
                    # terminal on the rotation path.)
                    _safe_display_update(display, name, "failed", log_gate)
                    raise
                except BaseException:
                    _safe_display_update(display, name, "failed", log_gate)
                    raise
                # A valid verdict EXISTS. Commit the lineage FIRST (succeeded=True), before
                # any teardown -- from here agent_slot keys on `succeeded`, so the lineage
                # is conserved no matter what teardown does (spec 6.2.1).
                slot.succeeded = True
                # BEST-EFFORT teardown (the 4th broad catch): the post-verdict "success"
                # display update must NEVER cost a verdict that already exists, so a display
                # glitch here is SWALLOWED. This is DISTINCT from a GENUINE late teardown
                # bug, which propagates -- but `succeeded=True` conserves the lineage in
                # either case (spec 6.2.1). T12 pins both sides.
                try:
                    _safe_display_update(display, name, "success", log_gate)
                except Exception as exc:  # noqa: BLE001 -- a display glitch must not cost a verdict
                    print(
                        f"WARNING: post-verdict status-display update failed for {name} "
                        f"({exc}); the verdict stands",
                        file=sys.stderr,
                    )
                return result

    async def tracked_launch(name: str) -> dict[str, Any]:
        """Dispatch to the v4 path (``rotation is None``) or the rotation path."""
        if rotation is None:
            return await _legacy_tracked_launch(name)
        return await _rotating_tracked_launch(name, rotation)

    tasks = {name: tracked_launch(name) for name in AGENTS}

    if display is not None:
        try:
            await display.start()
        except Exception as exc:
            # A display-start failure (event-loop issue, terminal problem) must
            # never block the actual analysis. Drop the display and fall
            # through — tracked_launch closures will see ``display is None``.
            print(
                f"[!] WARNING: status display failed to start ({exc}) "
                f"\u2014 continuing without live status",
                file=sys.stderr,
            )
            display = None

    with _buffered_stderr_while(active=display is not None):
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        finally:
            if display is not None:
                await display.stop()

    # Fast-fail (R15) must WIN over degraded-mode: if the endpoint is dead, two
    # "surviving" mages are two mages that never ran. ``gather(return_exceptions=True)``
    # captured the _EndpointDown as a result, so re-raise it here, before synthesis.
    #
    # A CancelledError is likewise re-raised on the ROTATION path: a cancelled mage means
    # the run is being torn down (timeout, Ctrl-C), and completing with partial results
    # would hide that. The legacy path keeps v4 back-compat (a cancelled sub-agent is a
    # degraded failure, pinned by test_cancelled_error_marks_display_failed) -- gather
    # has already captured it as a result either way, so the choice is local to here.
    for result in results:
        if isinstance(result, _EndpointDown):
            raise result
        if rotation is not None and isinstance(result, asyncio.CancelledError):
            raise result

    for name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            # CancelledError is BaseException in 3.8+ but we treat a cancelled
            # child task as a normal agent failure — the orchestrator itself is
            # not being cancelled, only one sub-agent was. Truly fatal signals
            # (KeyboardInterrupt, SystemExit) still propagate.
            if not isinstance(result, (Exception, asyncio.CancelledError)):
                raise result
            print(
                f"[!] WARNING: Agent '{name}' failed ({result}) \u2014 excluded from synthesis",
                file=sys.stderr,
            )
            failed.append(name)
        else:
            successful.append(result)

    if len(successful) < 2:
        raise RuntimeError(
            f"Only {len(successful)} agent(s) succeeded \u2014 fewer than 2 required for synthesis"
        )

    if failed:
        print(
            f"[!] WARNING: Running synthesis with "
            f"{len(successful)}/{len(AGENTS)} agents "
            f"\u2014 results may be biased",
            file=sys.stderr,
        )

    consensus = determine_consensus(successful)

    report: dict[str, Any] = {
        "agents": successful,
        "consensus": consensus,
    }

    if failed:
        report["degraded"] = True
        report["failed_agents"] = failed

    # Conditional presence mirrors degraded/failed_agents: the field is
    # introduced only when there is something to report so 2.2.0 consumers
    # that ignore unknown keys keep working unchanged.
    if retried:
        report["retried_agents"] = sorted(retried)
    if extraction_failures:
        # R18. ADITIVO y FAIL-SOFT: ``consensus`` no lo lee, y un reporte sin este
        # campo sigue siendo valido (NR3). Es lo unico que va a VER la deriva de un
        # modelo en produccion -- donde la tasa real, con una muestra de 5+2, solo se
        # puede conocer ahi.
        report["extraction_failures"] = {
            agent: dict(causes) for agent, causes in sorted(extraction_failures.items())
        }

    # T13 telemetry (R9/R13/R16): fold each mage's rotation state + the preflight's
    # measured fields into the report. Additive and fail-soft -- absent on the Claude
    # path (rotation is None), so 2.x consumers that ignore unknown keys are unaffected.
    if rotation is not None:
        for agent in successful:
            st = rotation_telemetry.get(agent["agent"])
            if st is None or st.model_configured is None or st.model_used is None:
                continue
            # Serialise the TAG, not the ModelSpec: a dataclass is not JSON-serialisable,
            # so json.dump would die at the LAST step of a successful run (Caspar, C2).
            agent["model_configured"] = st.model_configured.model
            agent["model_used"] = st.model_used.model
            agent["rotations"] = st.rotations_done
            agent["fallback_reason"] = st.fallback_reason
        report["fallback_agents"] = sorted(
            name
            for name, st in rotation_telemetry.items()
            if st.model_configured is not None
            and st.model_used is not None
            and st.model_used.model != st.model_configured.model
        )
        # The preflight computed context_guard from the TRIO. If any SURVIVING mage
        # rotated to a model accepted on an estimate/unknown window, the run was NOT
        # fully enforced -- downgrade the run-level label so it never claims a guarantee
        # the rotation path did not keep (R16 honesty -- MAGI gate Loop 1 pass 2).
        guard = rotation.preflight.context_guard
        if guard == CONTEXT_GUARD_ENFORCED and any(
            (st := rotation_telemetry.get(a["agent"])) is not None and st.ran_unmeasured
            for a in successful
        ):
            guard = CONTEXT_GUARD_ESTIMATED
        report["context_guard"] = guard
        report["lineage_warnings"] = list(rotation.preflight.lineage_warnings)
        report["token_estimate_delta"] = list(rotation.preflight.token_estimate_delta)

    return report


def _enable_utf8_console_io() -> None:
    """Switch ``sys.stdout`` / ``sys.stderr`` to UTF-8 with
    ``errors="backslashreplace"`` on Windows.

    The 2.2.6 hotfix removed the four ``\\u26a0`` warning signs that
    were the immediate trigger for ``UnicodeEncodeError`` crashes on
    cp1252 locales, but the underlying streams were still bound to the
    locale-derived wrapper Python gives child processes on Windows.
    Any future non-cp1252 codepoint emitted through ``print`` — a
    finding title that the LLM rolls with ``→``, ``≥``, or
    any character outside cp1252's 256-codepoint range — would
    re-introduce the same crash mode. This helper is the structural
    fix: it switches the encoding at startup so every output path
    (warnings, ERROR finals, banner, report-to-stdout) tolerates any
    Unicode the LLM emits.

    The ``backslashreplace`` error policy is non-negotiable. ``strict``
    is what crashed in the first place; ``ignore`` would silently drop
    diagnostic content; ``replace`` substitutes U+FFFD which is itself
    non-ASCII and thus pointless under cp1252. ``backslashreplace``
    always produces ASCII output (``\\u26a0``) so the printed bytes
    are guaranteed encodable in any codepage.

    No-op on non-Windows platforms — POSIX shells default to UTF-8 and
    forcing the encoding would change the byte contract for parents
    that captured stdout assuming the locale-derived encoding.

    Streams that lack ``reconfigure`` (custom logger sinks, buffer
    proxies, pytest capture wrappers) are skipped silently rather than
    crashed. Custom streams have already chosen their encoding
    contract; forcing UTF-8 would either fail or violate that
    contract.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(encoding="utf-8", errors="backslashreplace")


def _diff_files_and_ranges(diff: str) -> tuple[set[str], dict[str, set[int]]]:
    """Return (valid_files, changed_ranges) for the guard. Fail-safe -> empty.

    Parses *diff* into the set of touched files and their changed post-image
    line numbers. Any failure degrades to ``(set(), {})`` so the guard becomes
    a no-op rather than crashing the run (R10).

    Args:
        diff: The resolved unified diff text (``""`` when none).

    Returns:
        Tuple ``(files, ranges)`` where ``files`` is the set of normalized
        touched paths and ``ranges`` maps each path to its changed lines.
    """
    try:
        ranges = parse_diff_ranges(diff)
        return set(ranges.keys()), ranges
    except Exception:  # noqa: BLE001 — boundary fail-safe
        return set(), {}


def _apply_finding_guard(
    agents: list[dict[str, Any]],
    mode: str,
    files: set[str],
    ranges: dict[str, set[int]],
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """In code-review, drop/annotate each agent's findings against the diff.

    Hard-drops findings whose ``file`` is not in the diff (hallucination guard)
    and soft-annotates findings whose ``line`` falls outside the changed range,
    per :func:`finding_validation.validate_findings`. A no-op in non-code-review
    modes or when there is no diff (empty *files*). The guard filters the
    findings section only — it never touches an agent's verdict/confidence, so
    the consensus score (computed downstream by ``determine_consensus`` from
    verdict+confidence) is unaffected. Never raises (each agent is guarded
    independently behind a boundary).

    Args:
        agents: The successful agents' validated output dicts.
        mode: Analysis mode; the guard runs only for ``"code-review"``.
        files: Set of valid (diff-present) normalized file paths.
        ranges: Per-file set of changed post-image line numbers.
        summary: Optional out-param (F4). When given, it is populated with the
            guard's observable effect for the report: ``{"active": False}`` when
            the guard is a no-op, else ``{"active": True, "files_in_diff": N,
            "total_dropped": N, "total_annotated": N, "per_agent": {agent:
            {"dropped", "annotated", "dropped_titles"}}}`` with only agents that
            had a drop/annotation. Surfacing this lets the report explain why a
            voting agent shows no Key Findings (the guard never alters the vote).

    Returns:
        A new list of agent dicts with guarded findings (same order). Agents
        for which the guard fails are passed through with original findings.
    """
    if mode != "code-review" or not files:
        if summary is not None:
            summary["active"] = False
        return agents

    if summary is not None:
        summary.update(
            {
                "active": True,
                "files_in_diff": len(files),
                "total_dropped": 0,
                "total_annotated": 0,
                "per_agent": {},
            }
        )

    out: list[dict[str, Any]] = []
    for a in agents:
        try:
            original = a.get("findings", [])
            kept, dropped, annotated = validate_findings(original, files, ranges)
            a = {**a, "findings": kept}
            if dropped or annotated:
                # Compute dropped titles by an order-preserving walk of
                # *original* against *kept*. ``validate_findings`` keeps survivors
                # in original order (annotated ones replaced by new dicts with the
                # same title/file/line, only ``detail`` changed) and removes the
                # dropped ones, so a two-pointer match by (title, file, line)
                # identifies exactly which originals survived. A title-set diff
                # would wrongly hide a dropped finding whose title is shared by a
                # kept one (duplicate titles across different files).
                kept_idx = 0
                dropped_titles = []
                for orig in original:
                    if kept_idx < len(kept) and (
                        kept[kept_idx].get("title") == orig.get("title")
                        and kept[kept_idx].get("file") == orig.get("file")
                        and kept[kept_idx].get("line") == orig.get("line")
                    ):
                        kept_idx += 1  # this original survived (possibly annotated)
                    else:
                        dropped_titles.append(str(orig.get("title", "")))
                print(
                    f"[guard] {a['agent']}: dropped {dropped} "
                    f"titles={dropped_titles}, annotated {annotated}",
                    file=sys.stderr,
                )
                if summary is not None:
                    summary["per_agent"][a["agent"]] = {
                        "dropped": dropped,
                        "annotated": annotated,
                        "dropped_titles": dropped_titles,
                    }
                    summary["total_dropped"] += dropped
                    summary["total_annotated"] += annotated
        except Exception as exc:  # noqa: BLE001 — boundary fail-safe
            print(f"WARNING: finding guard failed for {a['agent']}: {exc}", file=sys.stderr)
        out.append(a)
    return out


def _resolve_project_root() -> str:
    """Return the git toplevel of the cwd, or the realpath of cwd if not a repo.

    Used to derive the per-project temp namespace key. A missing ``git``
    binary or a non-repository cwd falls back to the realpath of the
    current directory.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            top = completed.stdout.strip()
            if top:
                return top
    except (OSError, subprocess.SubprocessError):
        pass
    return os.path.realpath(os.getcwd())


def _write_report_file(report_text: str, out_path: str) -> None:
    """Atomically write the human-readable *report_text* to *out_path* (for -o/--out).

    Atomic (temp file + ``os.replace``): a failure mid-write never leaves a TRUNCATED
    report at *out_path* -- a partial verdict there is indistinguishable from a whole
    one to a file-only consumer (the project's worst-case, on the output side). The
    temp file carries the PID so two concurrent runs writing the same target never
    clobber each other's temp (MAGI gate, Balthasar).

    ``errors="backslashreplace"`` is load-bearing: ``report_text`` can contain a LONE
    SURROGATE (``json.loads`` decodes a ``\\uD800``-style escape in a model's output to
    one), and a plain utf-8 write would then raise ``UnicodeEncodeError`` -- NOT an
    ``OSError`` -- crashing the process AFTER the verdict was computed and bypassing the
    caller's stdout fallback (MAGI gate, Balthasar). Escaping keeps the write total.

    Args:
        report_text: The rendered report (banner + findings).
        out_path: The target file path.

    Raises:
        OSError: On a real filesystem failure. The partial temp is removed first, so
            *out_path* is left untouched; the caller then falls back to stdout.
    """
    tmp = f"{out_path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", errors="backslashreplace") as out_file:
            out_file.write(report_text + "\n")
        os.replace(tmp, out_path)
    except OSError:
        try:
            os.unlink(tmp)  # drop any partial temp; the target is untouched
        except OSError:
            pass
        raise


def main() -> None:
    """CLI entry point for MAGI orchestrator."""
    # Must run BEFORE any ``print`` or ``sys.exit`` — every output
    # path past this line assumes UTF-8 + backslashreplace on
    # Windows. A later call site cannot fix a crash that already
    # happened on an earlier print.
    _enable_utf8_console_io()

    # Short-circuit: --ollama-init scaffolds the repo TOML and exits before
    # parse_args() so that mode/input positional arguments are not required.
    # We screen sys.argv directly; the flag is unambiguous (no value follows it).
    if "--ollama-init" in sys.argv[1:]:
        try:
            path = write_template()
        except FileExistsError as exc:
            print(
                f"Config already exists at {exc}; not overwriting.",
                file=sys.stderr,
            )
            sys.exit(0)
        print(f"Wrote Ollama config template to {path}")
        sys.exit(0)

    args = parse_args()

    try:
        input_content, input_label = _load_input_content(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Input-size telemetry: estimate token footprint and flag oversized inputs.
    # Pure/total — never raises. Runs after load so the enriched content is NOT
    # measured here (enrichment happens below); we measure the raw user input.
    est_tokens, oversize = check_input_size(input_content, args.warn_input_tokens)
    raw_input_chars = len(input_content)  # capture BEFORE _maybe_enrich reassigns input_content

    # A2: resolve the review diff ONCE (code-review only) and thread the same
    # value to BOTH the enrichment path and the finding guard so they can never
    # diverge. ``resolve_diff`` is TOTAL (returns "" on any failure); "" makes
    # the guard a no-op.
    review_diff = (
        resolve_diff(input_content, os.getcwd(), args.base) if args.mode == "code-review" else ""
    )

    input_content, enrich_note = _maybe_enrich(
        args.mode,
        input_content,
        base_ref=args.base,
        enrich=args.enrich,
        max_chars=args.enrich_max_chars,
        diff=review_diff,
    )

    try:
        prompt = build_user_prompt(args.mode, input_content)
    except InvalidInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)
    agents_dir = os.path.join(skill_dir, "agents")

    # Hard prerequisite check: only required for the Claude path. When --ollama
    # is set the claude CLI is not used, so skip the gate entirely to allow the
    # Ollama backend to run even when claude is absent from PATH.
    if not args.ollama and not shutil.which("claude"):
        print("ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    try:
        # select_backend is async now (T9): the Ollama preflight MEASURES the
        # payload. Its own asyncio.run runs here, BEFORE the output dir is created,
        # so a config/preflight error exits without leaking a temp run dir -- the
        # exact ordering of the v4 sync call it replaces. The rotation's asyncio.Event
        # is only constructed (never awaited) in this loop, so it binds lazily to the
        # run_orchestrator loop on its first wait().
        backend, agent_models, rotation = asyncio.run(select_backend(args, prompt))
    except (OllamaConfigError, OllamaPreflightError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    is_temp_dir = args.output_dir is None
    if is_temp_dir:
        # One-shot removal of pre-2.6.0 dirs directly under temp.
        sweep_legacy_runs_once()
        # Per-project namespace so concurrent runs from other projects are
        # isolated and never see each other's run dirs.
        run_root = project_run_root(_resolve_project_root())
        # Prune to ``keep_runs - 1`` existing dirs so the run about to be
        # created below brings the total to exactly ``keep_runs``. Live
        # dirs (locked by a running session) are excluded from the budget.
        cleanup_old_runs(args.keep_runs - 1, run_root)
        output_dir = create_output_dir(None, run_root)
        # Mark this run live with a per-run staleness bound derived from
        # --timeout (closes F9) so a concurrent session's cleanup skips it.
        write_lock(output_dir, staleness_bound_for_timeout(args.timeout))
    else:
        output_dir = create_output_dir(args.output_dir)

    print("+==================================================+")
    print("|          MAGI SYSTEM -- INITIALIZING              |")
    print("+==================================================+")
    print(f"|  Mode: {args.mode}")
    print(f"|  Input: {input_label}")
    if enrich_note is not None:
        print(f"|  Context: {enrich_note}")
    if args.ollama:
        model_label = "ollama/" + "/".join(sorted({s.model for s in agent_models.values()}))
        print(f"|  Model: {model_label}")
    else:
        print(f"|  Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"|  Timeout: {args.timeout}s")
    print(f"|  Output: {output_dir}")
    print("+==================================================+")
    print(flush=True)

    # ``BaseException`` rather than ``Exception`` so KeyboardInterrupt and
    # SystemExit also trigger the temp-dir cleanup — otherwise Ctrl-C mid
    # run leaves orphaned ``magi-run-*`` dirs that ``cleanup_old_runs``
    # only prunes opportunistically on the *next* run.
    report: dict[str, Any] | None = None
    try:
        report = asyncio.run(
            run_orchestrator(
                agents_dir,
                prompt,
                output_dir,
                args.timeout,
                agent_models=agent_models,
                backend=backend,
                rotation=rotation,
                show_status=args.show_status,
                max_attempts=args.max_attempts,
            )
        )
    except BaseException:
        if is_temp_dir:
            try:
                shutil.rmtree(output_dir)
            except OSError as cleanup_exc:
                print(
                    f"WARNING: Failed to clean up {output_dir}: {cleanup_exc}",
                    file=sys.stderr,
                )
        raise

    # A2 + R8: apply the diff-grounded finding guard to each agent BEFORE the
    # consensus that ends up in the report. ``determine_consensus`` stays
    # mode-agnostic (it never receives the diff); the guard runs here, on the
    # successful agents, using the single resolved ``review_diff`` shared with
    # enrichment. ``files`` empty (non-code-review or no diff) makes it a no-op.
    files, ranges = _diff_files_and_ranges(review_diff)
    # FIX 3b: emit ONE stderr line in code-review so a no-diff no-op is visible.
    if args.mode == "code-review":
        if files:
            print(f"[guard] active: {len(files)} file(s) in diff", file=sys.stderr)
        else:
            print("[guard] skipped: no resolvable diff", file=sys.stderr)
    # F4: collect the guard's observable effect into the report so an agent that
    # votes but has all its findings dropped is explained in the audit artifact.
    guard_summary: dict[str, Any] = {}
    report["agents"] = _apply_finding_guard(
        report["agents"], args.mode, files, ranges, summary=guard_summary
    )
    report["guard"] = guard_summary

    # A5: outside code-review there is no diff to ground file/line against, so
    # strip them to ``None`` — this forces title-based dedup for design/analysis
    # regardless of what the agent emitted, keeping their behaviour identical to
    # the pre-3.0.0 contract.
    if args.mode != "code-review":
        for a in report["agents"]:
            for fnd in a.get("findings", []):
                fnd["file"] = None
                fnd["line"] = None

    # Recompute the consensus on the guarded agents so the rendered report's
    # findings section reflects the filtering. The score/verdict/label are
    # invariant under the guard (it only touches the findings section, never an
    # agent's verdict or confidence — pinned by the BDD-14 score-invariance
    # test); only the deduplicated ``findings`` list changes. Guarded by the
    # ``>= 2`` precondition of ``determine_consensus`` — real runs always reach
    # here with >= 2 agents (the orchestrator raised otherwise), so this only
    # skips the refresh under stubbed/degenerate agent lists.
    if len(report["agents"]) >= 2:
        report["consensus"] = determine_consensus(report["agents"])

    report_text = format_report(
        report["agents"],
        report["consensus"],
        context_guard=report.get("context_guard"),
        lineage_warnings=report.get("lineage_warnings"),
    )
    # -o/--out REDIRECTS the report to a file and SUPPRESSES it on stdout. A write
    # failure is not allowed to lose the verdict: it warns LOUDLY on stderr and falls
    # back to printing the report on stdout.
    if args.out:
        try:
            _write_report_file(report_text, args.out)
        except OSError as exc:
            print(
                f"WARNING: could not write report to {args.out} ({exc}); "
                "printing it to stdout instead so the verdict is not lost",
                file=sys.stderr,
            )
            print(report_text)
    else:
        print(report_text)

    # A1: aggregate per-run cost into the report BEFORE it is serialized so the
    # saved magi-report.json carries the ``cost`` block. Aggregate over all
    # canonical agent names (AGENTS), not just report["agents"], so a failed or
    # timed-out agent that wrote its raw envelope still contributes to the total.
    # Fail-safe: a missing or corrupt envelope contributes 0 for that agent.
    report["cost"] = aggregate_cost(output_dir, list(AGENTS))
    # FIX 4: if the aggregated cost is $0.00 despite having at least one agent,
    # the CLI may have renamed or relocated ``total_cost_usd`` — emit a single
    # warning so the silent mis-reporting is visible in operator logs.
    # Skipped on --ollama: Ollama responses carry no total_cost_usd field so
    # $0.00 is always the correct aggregated value, not a mis-reporting signal.
    if not args.ollama and report["cost"]["total_usd"] == 0.0 and report["agents"]:
        print(
            "[!] WARNING: per-run cost resolved to $0.00; the CLI may have "
            "renamed the total_cost_usd field — check raw envelopes.",
            file=sys.stderr,
        )

    # Input-size telemetry: record the raw-input footprint in the report so the
    # saved magi-report.json carries observable per-run size data (mirrors the
    # ``cost`` block discipline: set BEFORE json.dump). ``est_tokens``,
    # ``oversize``, and ``raw_input_chars`` were all computed right after
    # _load_input_content, before _maybe_enrich could reassign input_content.
    report["input_size"] = {
        "chars": raw_input_chars,
        "est_tokens": est_tokens,
        "oversize": oversize,
        "warn_threshold_tokens": args.warn_input_tokens,
    }

    report_path = os.path.join(output_dir, "magi-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")
    print(f"Cost: ${report['cost']['total_usd']:.4f} ({len(report['agents'])} agents)")
    print(f"Input size: ~{est_tokens} tokens ({raw_input_chars} chars)")
    if oversize:
        print(
            f"[!] WARNING: input ~{est_tokens} tokens is very large; MAGI reviews it whole "
            "(no map-reduce). Consider splitting into smaller PRs for sharper review.",
            file=sys.stderr,
        )

    if is_temp_dir:
        # Run completed: drop the liveness lock so this dir becomes
        # ordinary podable history for future cleanups. The failure path
        # (except BaseException -> rmtree) already removes it with the dir.
        remove_lock(output_dir)


if __name__ == "__main__":
    main()
