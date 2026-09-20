#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.10.3
# Date: 2026-09-19
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
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable, Mapping

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
from backoff import next_backoff, parse_retry_after  # noqa: E402
from claude_backend import ClaudeBackend  # noqa: E402
from trio_executor import ClaudeTrio, TrioExecutor  # noqa: E402
from bundle_segmenter import (  # noqa: E402
    CONSULT_INPUT_CEILING_BYTES,
    BundleSegmenter,
    SegmentPlan,
    SegmentTooLargeError,
    mentions,
)
from magi_rs_paths import bin_dir, pin_path, plugin_root  # noqa: E402
from magi_rs_pin import Pin  # noqa: E402
from magi_rs_result import ConsultResult  # noqa: E402
from magi_rs_trio import (  # noqa: E402
    FAIL_CLOSED_ERRORS,
    MagiRsContractError,
    MagiRsProcessError,
    MagiRsTrio,
)
from magi_rs_trio import resolve_timeout as resolve_magi_rs_timeout  # noqa: E402
from ollama_backend import (  # noqa: E402
    TRANSPORT_CONNECTION_MARKERS,
    TRANSPORT_HTTP_PATTERN,
    OllamaBackend,
)
from ollama_config import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    ModelSpec,
    OllamaConfig,
    OllamaConfigError,
    resolve_config,
)
from ollama_init import run_ollama_init  # noqa: E402
from ollama_preflight import (  # noqa: E402
    CONTEXT_GUARD_ENFORCED,
    CONTEXT_GUARD_ESTIMATED,
    OllamaPreflightError,
    PreflightResult,
    _is_cloud_tag,
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
from served_model import read_served_models  # noqa: E402
from finding_validation import parse_diff_ranges, validate_findings  # noqa: E402
from prompt_guard import AgentPromptGuard, PromptContractError  # noqa: E402
from retry_feedback import (  # noqa: E402
    FEEDBACK_TEMPLATES,
    MAX_ERROR_CHARS,
    retry_feedback_cause,
)
from validate import (  # noqa: E402
    MAX_ATTEMPTS_CAP,
    MAX_INPUT_FILE_SIZE,
    MIN_ATTEMPTS,
    ValidationError,
)
from verdict_markers import (  # noqa: E402
    ECHO_CANARY,
    VerdictSentinel,
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


#: Attempts per model, by default: the original + 1 retry with corrective feedback.
DEFAULT_MAX_ATTEMPTS = 2

#: The bounds themselves live in ``validate`` and are RE-EXPORTED here, because the budget has
#: two doors -- this flag and the Ollama TOML's ``max_attempts_per_model`` -- and they must not
#: be able to disagree. **This is not paranoia:** ``--max-attempts 1000`` (one zero too many)
#: turns a stubborn mage into a thousand calls -- expensive on Ollama, where `:cloud` is a paid
#: tier, and hundreds of dollars on Claude. The comment that used to sit here claimed the TOML
#: "already validates this way". It did not: it accepted 1000 (MAGI gate, Balthasar).


def _max_attempts(raw: str) -> int:
    """Validate ``--max-attempts``: an integer in ``[MIN_ATTEMPTS, MAX_ATTEMPTS_CAP]``.

    Args:
        raw: The value exactly as it arrives from the command line.

    Returns:
        The validated integer.

    Raises:
        argparse.ArgumentTypeError: If it is not an integer, or falls outside the range.
            **Fails closed**: it never degrades to a silent default.
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


def _positive_timeout(raw: str) -> int:
    """Validate ``--timeout``: a positive integer of seconds (``>= 1``).

    The FLAG must be floored too (gate CP2 plan loop 5, Caspar): the TOML path is
    floored by config validation (``ollama_config._MIN_TIMEOUT_SECONDS``), but a bare
    ``type=int`` would let ``--timeout 0`` through to a 0-second timeout (immediate
    failure on every request). Validated at the input boundary, matching the
    ``_max_attempts`` convention above.

    Args:
        raw: The value exactly as it arrives from the command line.

    Returns:
        The validated integer.

    Raises:
        argparse.ArgumentTypeError: If it is not an integer, or is ``< 1``.
            **Fails closed**: it never degrades to a silent default.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--timeout must be an integer (got {raw!r})") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"--timeout must be >= 1 (got {value})")
    return value


def _resolve_timeout(flag: int | None, toml: float | None) -> float:
    """Resolve the per-agent timeout: flag ``--timeout`` > TOML ``timeout`` > default.

    Pure and total -- no I/O, never raises. The precedence matches every other
    per-key config resolution in the project (env > repo > global > built-in default
    collapses, here, to just flag > TOML > default since ``--timeout`` has no env
    tier of its own).

    Args:
        flag: The parsed ``--timeout`` value, or ``None`` if the user did not pass it.
        toml: The Ollama TOML's ``timeout`` (``OllamaConfig.timeout``), or ``None`` on
            the Claude path (which has no TOML).

    Returns:
        The resolved timeout in seconds.

    Example:
        >>> _resolve_timeout(300, 600.0)
        300.0
        >>> _resolve_timeout(None, 600.0)
        600.0
        >>> _resolve_timeout(None, None)
        900.0
    """
    if flag is not None:
        return float(flag)
    if toml is not None:
        return toml
    return DEFAULT_TIMEOUT_SECONDS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with mode, input, timeout, output_dir.
    """
    parser = argparse.ArgumentParser(description="MAGI Orchestrator")
    # Optional AT PARSE TIME, required in practice: ``--check-prompts`` is a dry run that needs
    # neither a mode nor an input, and making argparse enforce the positionals would force that
    # flag to be screened out of ``sys.argv`` by hand -- which silently breaks argparse's own
    # abbreviations (``--check`` would expand to ``--check-prompts`` and then be ignored, giving
    # the user a normal run they did not ask for; MAGI gate, Balthasar). Enforced below instead.
    parser.add_argument("mode", nargs="?", choices=VALID_MODES, help="Analysis mode")
    parser.add_argument("input", nargs="?", help="Path to file or inline text to analyze")
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        # None = "the user did not pass it" -> lets the TOML `timeout` win over the
        # 900s default (MS3, R6); collapsing this to a hardcoded 900 would make
        # `flag is not None` always true and the TOML value dead on arrival.
        default=None,
        help=(
            "Per-agent timeout in seconds, >= 1. Claude path: default 900. With --ollama: "
            "default 1800, and magi-rs also derives each mage's per-attempt budget from it."
        ),
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
            "LLM model for all agents. When omitted, the per-mode default "
            "from models.MODE_DEFAULT_MODELS applies (currently opus for all "
            "three modes). Pass --model explicitly to override."
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
        "--check-prompts",
        action="store_true",
        help=(
            "Validate the agent prompts against the verdict-marker contract and exit. "
            "Costs no tokens; use it after customising a prompt. Handled before the "
            "positional arguments are required (see main)."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=_max_attempts,
        # A None sentinel, not the default value: "was it passed?" and "what is it worth?"
        # are different questions, and answering the first with ``!= DEFAULT`` gets it wrong
        # for exactly one value -- the default -- so ``--ollama --max-attempts 2`` would be
        # overridden by the TOML in silence, which is the polite lie the warning exists to
        # kill. Resolved to DEFAULT_MAX_ATTEMPTS below, so callers still read an int.
        default=None,
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
        help=(
            "Run the open-weight trio through magi-rs (one model per mage, from "
            ".magi/magi.toml) instead of `claude -p`. Mutually exclusive with --model."
        ),
    )
    parser.add_argument(
        "--ollama-init",
        action="store_true",
        help="Initialise this project for magi-rs (`magi-rs init -w <cwd>`, creates .magi/) and exit.",
    )
    # ``model_source`` is declared here with its siblings, not only assigned in
    # the body below: a hand-built Namespace (tests, embedders) would otherwise
    # reach the report step without it and crash AFTER the verdict exists.
    parser.set_defaults(show_status=True, enrich=True, model_source=None)
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
    if not args.check_prompts and (args.mode is None or args.input is None):
        parser.error("the following arguments are required: mode, input")
    if args.ollama and args.model is not None:
        parser.error(
            "--model does not apply with --ollama; the trio is declared in .magi/magi.toml "
            "(run --ollama-init once per project)."
        )
    if args.ollama and args.max_attempts is not None:
        # R13: with --ollama the TOML wins. It used to win SILENTLY, which is a polite
        # lie: the user who passed the flag believes they configured something. Warn, do
        # not error -- the flag is legal, it is simply overridden.
        print(
            f"[!] WARNING: --max-attempts {args.max_attempts} is overridden by "
            "max_attempts_per_model in magi-ollama.toml (it governs the Ollama backend)",
            file=sys.stderr,
        )
    if args.max_attempts is None:
        args.max_attempts = DEFAULT_MAX_ATTEMPTS
    # INVARIANT: --model must stay None when --ollama is set. Do NOT collapse
    # this into `args.model or MODE_DEFAULT_MODELS[...]` — that would silently
    # re-enable `--ollama --model` and feed Ollama a Claude-shaped model name.
    #
    # ``args.mode is not None`` guards the dry run, which HAS no mode: the model default is
    # keyed by mode, and a lookup of ``None`` is a ``KeyError`` -- i.e. ``--check-prompts``
    # would die inside argument parsing, before the guard it exists to run.
    #
    # ``model_source`` is set BEFORE the fill, because afterwards the operator's
    # choice and the mode's default are the same value and the difference is
    # unrecoverable. Downstream telemetry records what was chosen AND who chose
    # it; without this, "requested: opus" cannot be told apart from "nobody
    # asked, this mode defaults to opus". ``None`` means no Claude model was
    # chosen at all (--ollama, or the mode-less dry run).
    args.model_source = "flag" if args.model is not None else None
    if not args.ollama and args.model is None and args.mode is not None:
        args.model = MODE_DEFAULT_MODELS[args.mode]
        args.model_source = "mode-default"
    return args


def _model_block(
    args: argparse.Namespace, served: dict[str, list[str]], unattributed: list[str]
) -> dict[str, object]:
    """Build the report's model-attribution block.

    One shape for both backends. Two shapes under a single key would push a
    conditional into every consumer -- the renderer, the corpus collector, and
    any later analysis over archived gates -- so the fields that do not apply
    are explicitly ``null`` rather than absent.

    ``served`` is ``null`` (not ``{}``) on the Ollama path, because an empty
    mapping would read as "attribution ran and found nothing" when the truth is
    that there is nothing to attribute here at all: models there are per-mage
    and a mage may have rotated, and both reach the report through the rotation
    telemetry. Distinguishing *not applicable* from *not known* is the whole
    point of the pair.

    Args:
        args: Parsed arguments; reads ``ollama``, ``model`` and ``model_source``.
        served: Agent → confirmed served model IDs, from
            :func:`read_served_models`.
        unattributed: Agents with no confirmed identity. Sorted here rather than
            trusted from the caller: the block IS the archived artifact, and a
            stable field order should not depend on every call site remembering
            it. Empty means every agent was attributed.

    Returns:
        ``{backend, short_name, source, resolved, served, unattributed}``.
        ``short_name`` is the post-default short name and ``source`` says who
        chose it, so the block never presents a mode default as an operator
        request.
    """
    if args.ollama:
        return {
            "backend": "ollama",
            "short_name": None,
            "source": None,
            "resolved": None,
            "served": None,
            "unattributed": None,
        }
    return {
        "backend": "claude",
        "short_name": args.model,
        "source": getattr(args, "model_source", None),
        "resolved": resolve_model(args.model),
        "served": served,
        "unattributed": sorted(unattributed),
    }


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

    # --- The two guards that run AFTER the schema validates (MS2) ---
    #
    # Both raise ``ValidationError`` subclasses, so the orchestrator's retry guard catches
    # them: the model gets corrective feedback and can fix itself.

    # R6 -- anti-echo canary. The LAST belt: the sentinel already prevents anything from
    # outside the markers being extracted, and the prompt puts nothing valid BETWEEN them.
    # One theoretical path is left: the model taking the worked example from OUTSIDE and
    # wrapping it in markers ITSELF. Its fingerprint gives it away.
    if all(payload.get(key) == value for key, value in ECHO_CANARY.items()):
        raise EchoedExampleRejected(
            "the verdict is a verbatim copy of the prompt's example, not your analysis"
        )

    # R10 -- identity. ``load_agent_output`` validates that ``agent`` is in the ENUM, but
    # nobody validated that it was the mage that was LAUNCHED. A duplicate name kills the
    # whole run; a unique but wrong one puts one mage's text in another's seat, and the
    # consensus counts it as an independent perspective **that never existed**.
    #
    # Case-insensitive: a model that writes "Caspar" DID emit its verdict correctly. Killing
    # it over a capital letter is a retry given away for free, and the enum is validated
    # separately anyway.
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


#: HTTP statuses where WAITING can clear the condition -> exponential backoff (MS3, R5).
#: A permanent 500 just exhausts max_attempts and rotates; it never loops here.
_TRANSIENT_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _retry_wait(
    exc: Exception, classified: str, attempt: int, config: "RotationRuntimeConfig"
) -> float:
    """Seconds to sleep before the next retry, exhaustive over ``_classify``.

    HTTP transient status -> exponential (honoring Retry-After); EVERYTHING else
    retryable (timeout, connection reset, DNS, a non-transient HTTP status) ->
    flat ``retry_backoff_seconds``. A schema failure never reaches here (the
    loop's ``isinstance`` branch routes it to the feedback path with NO backoff,
    BEFORE this is called). Falling to flat for any unrecognized category keeps
    the branch total even if ``_classify`` grows a new category later.

    Args:
        exc: The failed attempt's exception (for status/retry_after/receipt).
        classified: The ALREADY-computed ``_classify(exc)`` result -- the loop
            computes it once and passes it, so ``_classify`` is never called
            twice per exception (gate CP2 loop 1, Melchior).
        attempt: 1-based attempt number within the model's budget.
        config: The rotation runtime config (base, ceilings).

    Returns:
        Seconds to sleep (``>= 0``).
    """
    # Duck-typed (getattr) not isinstance(OllamaHTTPError): the fields ride on the
    # exception the Ollama backend raises, and any exc WITHOUT them (a hypothetical
    # non-OllamaHTTPError transport RuntimeError) simply falls to flat backoff below --
    # fail-CLOSED, never fail-open (gate CP2/§6 loop 1 note).
    status = getattr(exc, "status", None)
    if classified == _FAIL_HTTP and status in _TRANSIENT_HTTP_STATUS:
        receipt = getattr(exc, "receipt", None) or datetime.now(timezone.utc)
        # The cap (retry_after_max_seconds) is an ANTI-HANG defense against a hostile/
        # buggy server (e.g. `Retry-After: 999999`), NOT a limit on the user's model.
        # parse_retry_after applies it (§0.1 mandated inline comment).
        retry_after = parse_retry_after(
            getattr(exc, "retry_after", None), receipt, config.retry_after_max_seconds
        )
        wait = next_backoff(
            attempt,
            config.retry_backoff_seconds,
            config.retry_backoff_max_seconds,
            retry_after,
        )
        source = "Retry-After" if retry_after is not None else "formula"
        # parse caps at min(seconds, cap), so the capped value can only be <= cap and
        # equals cap iff the server asked for >= cap. The log says "at/over cap" (NOT
        # "truncated"): a server asking exactly cap is not truncated, so an "exceeded"
        # claim would be a false positive (spec R7; gate CP2 plan loop 2, Caspar).
        if retry_after is not None and retry_after >= config.retry_after_max_seconds:
            print(
                f"[backoff] server Retry-After >= cap; using "
                f"cap={config.retry_after_max_seconds:.0f}s",
                file=sys.stderr,
            )
    else:
        wait = config.retry_backoff_seconds  # flat: timeout + non-HTTP transport
        source = "flat-timeout"
    # Fields are non-secret (source literal, wait/status/attempt numeric); matches the
    # project's stderr-print logging pattern (no `logging` module in run_magi.py).
    print(
        f"[backoff] {source} wait={wait:.1f}s status={status} attempt={attempt}",
        file=sys.stderr,
    )
    return wait


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
    template picked by :func:`retry_feedback_cause` names the SPECIFIC defect --
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
    cause = retry_feedback_cause(error)
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
    retry_backoff_max_seconds: float  # MS3: ceiling for the exponential FORMULA
    retry_after_max_seconds: float  # MS3: ceiling for a server-sent Retry-After

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
            retry_backoff_max_seconds=config.retry_backoff_max_seconds,
            retry_after_max_seconds=config.retry_after_max_seconds,
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
    #: The per-run digest lookup (model -> digest; Task 5b, R5a/R5b/R5c). Lives
    #: HERE, not on ``LineageRegistry`` -- the registry's shared-state SHAPE is
    #: invariant (spec Sec.8.5) and this is not lineage/reservation state, it is
    #: preflight-adjacent identity data. Seeded from ``preflight.digest_by_model``
    #: (the trio) in :func:`select_backend`; grows append-only, with ZERO extra
    #: I/O, as :func:`_rotate` resolves each candidate's digest (see
    #: ``fallback_policy._resolve_digest``). NEVER copied into the report or the
    #: 7-key agent JSON -- internal-only, like the rest of the preflight cache.
    digest_by_model: dict[str, str] = field(default_factory=dict)
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
    on_extraction_failure: Callable[[ValidationError | json.JSONDecodeError], None],
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
        on_extraction_failure: Called once per SCHEMA failure (R18 adherence telemetry),
            including the last one -- the attempt that kills the model is a data point
            like any other. Injected, not read from a closure, for the same reason
            ``on_retry`` is: this loop must not know what a report or a display is.

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
            classified = _classify(exc)  # MS3: computed ONCE, reused below and by _retry_wait
            if classified == _FAIL_UNEXPECTED:
                # A TypeError/AttributeError from OUR code is not an endpoint failure.
                # Retrying it would burn attempts, rotate the mage, and BURY the bug
                # behind a degraded run (Balthasar, Checkpoint 2). Fail loud.
                raise
            last = exc
            # isinstance (not ``_classify(exc) == _FAIL_SCHEMA``) so mypy narrows exc to
            # the type _build_retry_prompt and on_extraction_failure accept -- the two are
            # equivalent by construction (_classify returns schema iff it is one of these).
            if isinstance(exc, (ValidationError, json.JSONDecodeError)):
                # R18: tally BEFORE the budget check. The attempt that EXHAUSTS the model
                # is exactly the one a marker-adherence gate must see; counting only the
                # attempts that got retried reports a rate that is systematically
                # optimistic -- the instrument would be blind to the failures that matter.
                on_extraction_failure(exc)
                if attempt + 1 >= ctx.config.max_attempts_per_model:
                    break
                # the model needs correcting; redact the api_key from the embedded error
                attempt_prompt = _build_retry_prompt(prompt, exc, api_key=env.api_key)
            elif attempt + 1 >= ctx.config.max_attempts_per_model:
                break
            else:
                attempt_prompt = prompt  # nothing to correct
                # attempt+1 is 1-based; `attempt` is local to THIS model's loop
                # (range(max_attempts_per_model)), so the backoff counter RESETS per
                # model on rotation -- a fallback starts at attempt=1, giving that
                # model's rate-limit a fresh window (§0.1 mandated inline comment).
                await asyncio.sleep(_retry_wait(exc, classified, attempt + 1, ctx.config))
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

    ``claim_next`` also enforces model-DIGEST uniqueness across the mages ACTIVE right
    now (Task 5b, R5a/R5b/R5c) -- INSIDE that same lock, since the commit itself is the
    only point where "who is active" cannot change out from under the check. A
    digest-unsafe candidate is rejected and re-proposed by ``claim_next`` internally, so
    by the time this function sees a non-``None`` candidate it is already digest-distinct
    from every other active mage; only the WINDOW verification remains this function's job.

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
        candidate = await ctx.registry.claim_next(
            name, ctx.policy, state, ctx.digest_by_model, is_cloud=_is_cloud_tag
        )
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
) -> tuple[AgentBackend, dict[str, ModelSpec], RotationContext | None, float | None]:
    """Return (backend, per-agent models, rotation context, TOML timeout) for the mode.

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
        context or ``None`` for the Claude path, and the resolved TOML ``timeout``
        (MS3, R6) or ``None`` on the Claude path -- the caller resolves the final
        per-agent timeout via :func:`_resolve_timeout`, flag > TOML > default).
    """
    if not args.ollama:
        return (
            ClaudeBackend(),
            {n: ModelSpec(args.model, "anthropic") for n in AGENTS},
            None,
            None,
        )

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
        # Task 5b: seed the per-run digest lookup from the trio (a COPY -- the
        # frozen PreflightResult.digest_by_model must never be mutated by rotation).
        digest_by_model=dict(result.digest_by_model),
    )
    return OllamaBackend(config), dict(config.models), rotation, config.timeout


def _select_executor(args: argparse.Namespace, timeout: int) -> MagiRsTrio | None:
    """Choose the trio executor for the open-weight path.

    ``--ollama`` names the open-weight path, and since v6 that path is executed
    by ``magi-rs`` (R18): one ``consult`` runs the three mages. Everything
    below the raw verdicts does not fork, so the ONLY thing the flag decides is
    which :class:`TrioExecutor` :func:`run_orchestrator` runs.

    Args:
        args: Parsed CLI namespace (``.ollama`` and ``.mode`` are read).
        timeout: The wall-clock ceiling already resolved for this run.

    Returns:
        A :class:`MagiRsTrio` bound to the current working directory (the
        project root, exactly as ``--ollama-init`` resolves it -- R6), the
        explicit mode (R10), the resolved timeout and the plugin's ``bin/`` +
        pin (R3); ``None`` on the Claude path, whose :class:`ClaudeTrio` is
        built inside :func:`run_orchestrator` because it binds to that
        function's own telemetry (R20 -- untouched).

    Raises:
        PluginRootError: ``CLAUDE_PLUGIN_ROOT`` is set but is not a directory.
        PinFormatError: The shipped pin does not parse.
    """
    if not args.ollama:
        return None
    root = plugin_root()
    return MagiRsTrio(
        workdir=os.getcwd(),
        mode=args.mode,
        timeout=timeout,
        bin_dir=bin_dir(root),
        pin=Pin.load(pin_path(root)),
    )


# --- MS4b: segmentation of a bundle that does not fit the consult ceiling ----------------
#
# The transport ceiling is a measured fact (MS0: 262 144 bytes of stdin, inclusive), and a
# verdict on a truncated payload is indistinguishable from a legitimate one. So a bundle
# that does not fit under the headroom is split at SECTION boundaries, each segment is
# evaluated as its own run, ONE AT A TIME (an Ollama invocation takes all three slots),
# and the gate is approved only when EVERY segment is (CLAUDE.local.md §6.2).

#: A markdown heading of depth 1-3 at column zero opens a section. Depth 3 is
#: included because the code-review enrichment nests every touched file at
#: ``### path`` under two ``##`` wrappers -- with ``##`` only, a real bundle was
#: at most three sections, each far larger than the headroom.
_SECTION_HEADING = re.compile(r"^(#{1,3}) (.+?)\s*$")
#: A per-file boundary of a git diff, the shape of a raw code-review payload
#: (no headings at all) and of the ```diff fences the project's own bundles carry.
_DIFF_FILE_BOUNDARY = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$")
#: The fence info string under which per-file diff boundaries split a fence.
_DIFF_FENCE_INFO = "diff"
#: Section name of one file's diff.
_DIFF_SECTION_NAME = "diff: {path}"
#: An optional ownership declaration right under a heading: ``<!-- defines: A, B -->``.
_SECTION_DEFINES = re.compile(r"^<!--\s*defines:\s*(.+?)\s*-->\s*$")
#: A markdown code fence, at column zero: headings inside it are not sections.
_CODE_FENCE = "```"
#: The name of the text before the first heading.
_PREAMBLE_SECTION = "(preamble)"
#: The extra segment that re-reads the contracts a split cut through.
_INTEGRATION_SECTION = "(integration)"
#: Per-section safety margin charged to the planner. Sections are re-assembled by
#: plain concatenation and each body already carries its own heading line, so this
#: is not a measured rendering cost but a deliberate over-count (a long path heading
#: is ~50 bytes) that keeps the planner conservative.
_SECTION_RENDER_OVERHEAD_BYTES = 128
#: Sub-directory of the run holding one segment's artefacts.
_SEGMENT_DIR = "segment-{order}"
#: Consensus labels that count as a PASS for one segment (§6: >= GO WITH CAVEATS).
_PASSING_LABEL_PREFIXES = ("STRONG GO", "GO")
#: The label a degraded segment is tabled under: a degraded run approves nothing.
_INVALID_SEGMENT_LABEL = "INVALID (degraded)"
#: The label a segment whose run raised is tabled under (§6.2.5: tabled, not aborted).
_FAILED_SEGMENT_LABEL = "FAILED ({reason})"
#: The label of an integration segment that could not be run because it does not fit.
_NOT_RUN_SEGMENT_LABEL = "NOT RUN ({reason})"
#: What one segment's run may raise without aborting the others: the seat died,
#: answered off-contract, or fewer than two seats survived -- ``RuntimeError``
#: is what ``run_orchestrator`` raises for that last case (its documented
#: contract), and a broader one still lands in the table with its message
#: rather than killing the report, which is what §6.2.5 asks for. The
#: bootstrap's fail-closed errors are NOT here -- a digest that does not
#: verify fails identically for every later segment, so they abort the run.
_SEGMENT_RUN_FAILURES = (MagiRsProcessError, MagiRsContractError, RuntimeError)


def split_sections(text: str) -> list[dict[str, Any]]:
    """Split a bundle into the sections the segmenter plans over.

    A section is opened by a markdown heading of depth 1-3 at column zero (its
    name is the heading text) or by a ``diff --git`` file boundary (named
    ``diff: <path>``). Text before the first boundary is the ``(preamble)``
    section. A section may declare the shared interfaces it OWNS with one
    ``<!-- defines: A, B -->`` line RIGHT UNDER its heading (§6.2.3: the
    segmenter never infers ownership). Deeper headings stay inside their
    section. Names are made unique (``name (2)``) so two sections can never
    collapse into one.

    Code fences are opaque to headings (a ``# Author:`` line inside a pasted
    source is not a heading). The one split allowed INSIDE a fence is the
    per-file boundary of a ```` ```diff ```` fence: each piece closes and
    reopens the fence, so every segment's markdown still parses.

    Args:
        text: The bundle, exactly as the run would send it.

    Returns:
        ``[{"name": str, "body": str, "defines": [str]}, ...]`` in bundle order.
        Outside a diff fence the bodies concatenate back to the input; inside
        one, only the reopened fence markers are added.

    Complexity:
        O(n) in the number of lines, once per run.
    """
    sections: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    name = _PREAMBLE_SECTION
    lines: list[str] = []
    defines: list[str] = []
    fence_info: str | None = None  # None outside a fence, else the info string
    expect_defines = False

    def close() -> None:
        if not lines:
            return
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique = name if count == 1 else f"{name} ({count})"
        sections.append({"name": unique, "body": "".join(lines), "defines": list(defines)})

    for line in text.splitlines(keepends=True):
        if line.startswith(_CODE_FENCE):
            fence_info = None if fence_info is not None else line[len(_CODE_FENCE) :].strip()
            lines.append(line)
            expect_defines = False
            continue
        if fence_info is None:
            heading = _SECTION_HEADING.match(line)
            boundary = _DIFF_FILE_BOUNDARY.match(line)
            if heading is not None:
                close()
                name, lines, defines = heading.group(2), [line], []
                expect_defines = True
                continue
            if boundary is not None:
                close()
                name = _DIFF_SECTION_NAME.format(path=boundary.group(2))
                lines, defines = [line], []
                expect_defines = False
                continue
            if expect_defines:
                declared = _SECTION_DEFINES.match(line)
                if declared:
                    defines = [n.strip() for n in declared.group(1).split(",") if n.strip()]
                expect_defines = False
        elif fence_info == _DIFF_FENCE_INFO:
            boundary = _DIFF_FILE_BOUNDARY.match(line)
            if boundary:
                if lines and lines[-1].startswith(_CODE_FENCE):
                    lines.pop()  # the fence had nothing in it yet: no empty fence left behind
                else:
                    lines.append(f"{_CODE_FENCE}\n")  # close the fence in the piece that ends
                close()
                name = _DIFF_SECTION_NAME.format(path=boundary.group(2))
                lines, defines = [f"{_CODE_FENCE}{_DIFF_FENCE_INFO}\n", line], []
                continue
        lines.append(line)
    close()
    return sections


def announce_segmentation(plan: SegmentPlan, payload_bytes: int, ceiling: int) -> None:
    """Say on stderr that the bundle was split, into how many, and why.

    A user who does not know their bundle was split cannot read the verdicts
    they get back: each one judged a PART, not the whole.

    Args:
        plan: The segmentation plan.
        payload_bytes: The size of the whole payload, in bytes.
        ceiling: The transport ceiling it did not fit under.

    Returns:
        None.

    Side effects:
        Writes one line to stderr.
    """
    print(
        f"[!] WARNING: the bundle was SPLIT into {len(plan.segments)} segments: "
        f"{payload_bytes} bytes do not fit under magi-rs's {ceiling}-byte input ceiling "
        f"({plan.headroom:.0%} headroom). Each verdict below judges ONE segment; the gate "
        "is approved only when every segment is.",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class SegmentationVerdict:
    """The gate's decision over the per-segment verdicts (§6.2.6).

    Attributes:
        approved: True only when at least one segment was evaluated and EVERY
            one of them passed.
        failing: The labels that sank the gate, verbatim.
    """

    approved: bool
    failing: list[str]


def segmentation_verdict(labels: list[str]) -> SegmentationVerdict:
    """Approve the gate only when EVERY segment passed.

    Two of three segments passing reads like "mostly approved", and it is not
    approved at all. Nothing evaluated approves nothing.

    Args:
        labels: One consensus label per segment, verbatim (a degraded segment
            arrives as ``INVALID (degraded)``).

    Returns:
        The decision and the labels that failed.
    """
    failing = [label for label in labels if not label.startswith(_PASSING_LABEL_PREFIXES)]
    return SegmentationVerdict(approved=bool(labels) and not failing, failing=failing)


def needs_integration_pass(plan: SegmentPlan) -> bool:
    """Whether the split crossed a shared interface.

    Args:
        plan: The segmentation plan.

    Returns:
        True when any segment references a name another segment defines --
        i.e. the split cut through a contract. False for a clean split, where
        the extra pass would be cost without a defect to catch, and False when
        nothing was split at all.
    """
    if not plan.is_segmented:
        return False
    return any(segment.references for segment in plan.segments)


def _integration_text(sections: list[dict[str, Any]], plan: SegmentPlan) -> str:
    """The contracts a split cut through: every line that names a shared interface.

    Not the bodies -- those were reviewed in their own segment -- but the
    signatures, the calls and the prose that mention each declared name, from
    every section, in bundle order.

    Args:
        sections: All sections.
        plan: The segmentation plan (its ``defines`` are the names).

    Returns:
        A markdown document, one ``##`` per section that mentions a name.
    """
    names = [n for segment in plan.segments for n in segment.defines]
    parts: list[str] = [
        "# Integration pass: the contracts the segmentation cut through\n\n"
        f"Shared interfaces: {', '.join(names)}. Only the lines that mention them, per section.\n"
    ]
    for section in sections:
        hits = [
            line
            for line in str(section["body"]).splitlines()
            if any(mentions(n, line) for n in names)
        ]
        if hits:
            parts.append(f"\n## {section['name']}\n\n" + "\n".join(hits) + "\n")
    return "".join(parts)


def _write_segmentation_map(output_dir: str, report: dict[str, Any]) -> None:
    """Write the run's top-level report -- first with the map only, then with every result.

    Called BEFORE the first segment runs (the map is the decision, not a record
    of what happened: if the run dies halfway, the file still says what the
    split was going to be) and again after each segment.

    Args:
        output_dir: The run's output directory.
        report: The top-level report (``segmentation`` block inside).

    Side effects:
        Writes ``magi-report.json``.
    """
    with open(os.path.join(output_dir, "magi-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _evaluate_segment(
    order: int,
    text: str,
    args: argparse.Namespace,
    output_dir: str,
    agents_dir: str,
    timeout: int,
) -> dict[str, Any]:
    """Run ONE segment as its own gate run, into its own directory.

    Args:
        order: The segment's 1-based position (names the directory).
        text: The segment's content, exactly as a whole bundle would be given.
        args: The parsed CLI namespace.
        output_dir: The run's output directory; the segment gets a child.
        agents_dir: The shipped agent prompts.
        timeout: The resolved wall timeout.

    Returns:
        The segment's report, as ``run_orchestrator`` returns it plus the
        ``magi_rs`` block and the unpriced cost; also written to
        ``segment-N/magi-report.json``.

    Raises:
        SegmentTooLargeError: The rendered prompt exceeds the transport
            ceiling (tabled as ``NOT RUN`` by the caller; nothing is launched).
        Whatever ``run_orchestrator`` raises otherwise.

    Side effects:
        Launches one magi-rs run; writes the segment's artefacts.
    """
    segment_dir = os.path.join(output_dir, _SEGMENT_DIR.format(order=order))
    os.makedirs(segment_dir, exist_ok=True)
    executor = _select_executor(args, timeout)
    prompt = build_user_prompt(args.mode, text)
    # The planner's arithmetic says a segment plus the wrapper always fits; the
    # guarantee is EXECUTED anyway, on the real prompt, right before the launch.
    # An oversized prompt never reaches the transport, where it would be rejected
    # fail-closed AFTER the seats had been reserved.
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > CONSULT_INPUT_CEILING_BYTES:
        raise SegmentTooLargeError(
            f"segment-{order} renders to {prompt_bytes} bytes, over the "
            f"{CONSULT_INPUT_CEILING_BYTES}-byte transport ceiling; not sent"
        )
    report: dict[str, Any] = asyncio.run(
        run_orchestrator(
            agents_dir,
            prompt,
            segment_dir,
            timeout,
            executor=executor,
            show_status=args.show_status,
        )
    )
    report["cost"] = unpriced_cost()
    result: ConsultResult | None = getattr(executor, "last_result", None)
    if result is not None:
        report.update(build_report_block(result))
    with open(os.path.join(segment_dir, "magi-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if result is not None:
        announce_magi_rs_telemetry(result, degraded_locally=bool(report.get("degraded", False)))
    return report


def run_segmented(
    plan: SegmentPlan,
    sections: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: str,
    agents_dir: str,
    timeout: int,
    wrapper_bytes: int = 0,
) -> dict[str, Any]:
    """Evaluate a split bundle: the map first, then one segment at a time, then the gate.

    Segments run IN SERIES and in plan order (§Concurrencia: one Ollama
    invocation holds all three slots). A segment whose run raises does not
    abort the run -- it is tabled as ``FAILED (<reason>)`` and the next one
    runs, so the report is complete, not partial (§6.2.5); only the bootstrap's
    fail-closed errors abort, because they would recur identically. The gate is
    approved only when every segment passed (§6.2.6). When the split cut
    through a shared interface, an extra integration segment re-reads those
    contracts, planned against the same ceiling; if it does not fit it is
    tabled as ``NOT RUN`` and sinks the gate.

    Args:
        plan: The segmentation plan.
        sections: The bundle's sections, in order.
        args: The parsed CLI namespace.
        output_dir: The run's output directory.
        agents_dir: The shipped agent prompts.
        timeout: The resolved wall timeout.
        wrapper_bytes: What the sanitize wrapper adds to every segment, charged
            against the transport ceiling (recorded in the map).

    Returns:
        The top-level report: ``{"segmented": True, "segmentation": {map...,
        "transport_ceiling_bytes", "wrapper_bytes", "results": [{order,
        section_names, consensus, degraded, agents, output_dir}],
        "integration_pass": bool, "approved": bool, "failing": [str]}}``.

    Raises:
        The bootstrap's fail-closed errors (``FAIL_CLOSED_ERRORS``), unchanged.

    Side effects:
        Writes ``magi-report.json`` before the first segment and after each;
        prints each segment's report and a final table on stdout.
    """
    by_name = {section["name"]: section for section in sections}
    jobs: list[tuple[int, list[str], str | None, str]] = [
        (
            seg.order,
            seg.section_names,
            "".join(str(by_name[n]["body"]) for n in seg.section_names),
            "",
        )
        for seg in plan.segments
    ]
    integration = needs_integration_pass(plan)
    if integration:
        jobs.extend(_integration_jobs(sections, plan, len(jobs), wrapper_bytes))

    segmentation: dict[str, Any] = dict(plan.as_map())
    segmentation.update(
        transport_ceiling_bytes=CONSULT_INPUT_CEILING_BYTES,
        wrapper_bytes=wrapper_bytes,
        results=[],
        integration_pass=integration,
        approved=None,
        failing=[],
    )
    report: dict[str, Any] = {"segmented": True, "segmentation": segmentation}
    _write_segmentation_map(output_dir, report)  # the decision, before anything runs

    for order, names, text, not_run_reason in jobs:
        print(f"\n=== segment-{order}: {', '.join(names)} ===")
        entry: dict[str, Any] = {
            "order": order,
            "section_names": names,
            "degraded": False,
            "agents": 0,
            "output_dir": os.path.join(output_dir, _SEGMENT_DIR.format(order=order)),
        }
        if text is None:
            entry["consensus"] = _NOT_RUN_SEGMENT_LABEL.format(reason=not_run_reason)
        else:
            try:
                segment_report = _evaluate_segment(
                    order, text, args, output_dir, agents_dir, timeout
                )
            except SegmentTooLargeError as exc:
                # Refused before the launch: tabled, and the next segment still runs.
                print(f"[!] WARNING: segment-{order} not run: {exc}", file=sys.stderr)
                entry["consensus"] = _NOT_RUN_SEGMENT_LABEL.format(reason=exc)
            except _SEGMENT_RUN_FAILURES as exc:
                # Tabled, not fatal: the next segment still runs (§6.2.5).
                print(f"[!] WARNING: segment-{order} failed: {exc}", file=sys.stderr)
                entry["consensus"] = _FAILED_SEGMENT_LABEL.format(reason=exc)
            else:
                degraded = bool(segment_report.get("degraded", False))
                entry["degraded"] = degraded
                entry["agents"] = len(segment_report.get("agents", []))
                entry["consensus"] = (
                    _INVALID_SEGMENT_LABEL
                    if degraded
                    else str(segment_report.get("consensus", {}).get("consensus", ""))
                )
                print(format_report(segment_report["agents"], segment_report["consensus"]))
        segmentation["results"].append(entry)
        _write_segmentation_map(output_dir, report)  # every result lands as it comes

    verdict = segmentation_verdict([r["consensus"] for r in segmentation["results"]])
    segmentation["approved"] = verdict.approved
    segmentation["failing"] = verdict.failing
    _write_segmentation_map(output_dir, report)
    print(_format_segmentation_table(segmentation))
    return report


def _integration_jobs(
    sections: list[dict[str, Any]], plan: SegmentPlan, first_order: int, wrapper_bytes: int
) -> list[tuple[int, list[str], str | None, str]]:
    """Plan the integration pass against the same ceiling as every other segment.

    Args:
        sections: All sections.
        plan: The segmentation plan.
        first_order: The number of ordinary segments (the integration jobs follow).
        wrapper_bytes: The per-segment wrapper cost.

    Returns:
        ``(order, names, text, not_run_reason)`` jobs: one or more integration
        segments when the contract lines fit, or ONE ``text=None`` job carrying
        the reason when even a single piece does not.
    """
    parts = split_sections(_integration_text(sections, plan))
    try:
        pieces = BundleSegmenter.plan(
            parts,
            CONSULT_INPUT_CEILING_BYTES - wrapper_bytes,
            overhead_bytes=_SECTION_RENDER_OVERHEAD_BYTES,
        )
    except SegmentTooLargeError as exc:
        return [(first_order + 1, [_INTEGRATION_SECTION], None, str(exc))]
    by_name = {part["name"]: part for part in parts}
    return [
        (
            first_order + index,
            [_INTEGRATION_SECTION],
            "".join(str(by_name[n]["body"]) for n in piece.section_names),
            "",
        )
        for index, piece in enumerate(pieces.segments, start=1)
    ]


def _format_segmentation_table(segmentation: dict[str, Any]) -> str:
    """Render the closing table of a segmented run, verdicts verbatim (§6.2.6).

    Args:
        segmentation: The ``segmentation`` block of the top-level report.

    Returns:
        A markdown table plus one line with the gate's decision.
    """
    rows = [
        "",
        "## Segmented run",
        "",
        "| Segment | Sections | Verdict | Passes |",
        "|---|---|---|---|",
    ]
    for entry in segmentation["results"]:
        passes = "yes" if entry["consensus"].startswith(_PASSING_LABEL_PREFIXES) else "NO"
        rows.append(
            f"| segment-{entry['order']} | {', '.join(entry['section_names'])} | "
            f"{entry['consensus']} | {passes} |"
        )
    decision = (
        "APPROVED: every segment passed"
        if segmentation["approved"]
        else ("NOT APPROVED: " + ", ".join(segmentation["failing"]) + " -- stop and escalate")
    )
    rows.append("")
    rows.append(f"GATE {decision}")
    return "\n".join(rows)


def _default_agents_dir() -> str:
    """The shipped ``agents/`` directory, next to this script's package.

    ONE definition, used by both the run and the ``--check-prompts`` dry run: two copies of
    this path is how the dry run ends up validating a directory the run never reads.

    Returns:
        Absolute path to ``skills/magi/agents``.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), "agents")


def check_prompts(agents_dir: str) -> None:
    """Validate a directory of agent prompts and exit. The dry run of the startup guard.

    The guard is strict on purpose: a prompt carrying a complete verdict BETWEEN the markers
    lets the model copy it, and the copy would be accepted -- fabrication, reintroduced in the
    user's own installation, where none of this repo's tests can see it. But strictness owes
    the user a way to check their work: until now the only way to discover that an edit was
    rejected was to START A RUN and watch it abort (MAGI gate, Balthasar, four cycles running).

    This costs no tokens, touches no network, and is the same guard the run itself uses -- not
    a second implementation that could drift from it.

    Args:
        agents_dir: The directory of ``{mage}.md`` prompts to validate.

    Raises:
        SystemExit: Always. ``0`` if every prompt honours the contract, ``1`` otherwise (with
            the offending file and the reason on stderr).
    """
    try:
        AgentPromptGuard(Path(agents_dir), VerdictSentinel()).check()
    except PromptContractError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: the prompts in {agents_dir} honour the verdict-marker contract.")
    sys.exit(0)


#: D3: the remedy for an unmeasured context window lives in the USER's file, which the
#: plugin never opens (R7) -- so the warning names it instead.
_UNMEASURED_REMEDY = "set `strict_context_guard = true` in .magi/magi.toml"


#: R20: the open-weight path exposes no usage per mage, so the cost is NOT
#: TARIFFED -- which is not zero. ``null`` fails a consumer that sums it loudly;
#: ``0.0`` sums in silence and lies; an absent key breaks each consumer differently.
COST_SOURCE_UNPRICED = "unpriced"


def unpriced_cost() -> dict[str, Any]:
    """The ``cost`` block of a run whose backend reports no usage (R20).

    Returns:
        ``{"total_usd": None, "per_agent": {}, "cost_source": "unpriced"}`` --
        the same shape the Claude path writes, with the unconfirmable fields
        explicitly null, never absent (the ``model`` block's rule).
    """
    return {"total_usd": None, "per_agent": {}, "cost_source": COST_SOURCE_UNPRICED}


def build_report_block(result: ConsultResult) -> dict[str, Any]:
    """Build the ``magi_rs`` block for ``magi-report.json``.

    ADDITIVE and fail-soft, exactly like v5.4.0's ``model`` block: ``consensus``
    and ``reporting`` ignore it, and a report without it is still valid --
    which is what the Claude path produces.

    Args:
        result: The parsed consult result.

    Returns:
        ``{"magi_rs": {...}}`` carrying the telemetry VERBATIM (every field of
        ``ConsultResult.telemetry``, empty values included: an empty
        ``rotations`` is the positive certificate that nobody rotated) plus:
        ``capped_agents`` (D2); ``model_used`` -- the tag each seat's accepted
        attempt ran on, under a name that says what it is and never the
        ``model`` block's ``served`` (D8); ``run_id`` -- the pointer into
        magi-rs's daily log, ``None`` when not emitted (R15); and
        ``gate_ran_on_unverified_context`` -- ``True`` only when EVERY seat ran
        unmeasured (D11 / BDD-34), the marker a reader of the report needs
        because stderr does not survive the run.

    Raises:
        Nothing -- see :func:`announce_magi_rs_telemetry`.
    """
    block: dict[str, Any] = dict(result.telemetry)
    block["capped_agents"] = result.capped_agents
    block["model_used"] = _model_used(result.telemetry.get("completions"))
    block["run_id"] = result.run_id
    block["gate_ran_on_unverified_context"] = _every_seat_unmeasured(
        result.telemetry.get("ran_unmeasured")
    )
    return {"magi_rs": block}


def _model_used(completions: Any) -> dict[str, Any]:
    """The tag of each seat's LAST attempt, from the measured ``completions`` shape.

    Args:
        completions: ``{mage: [{model, cap, finish, ...}, ...]}`` or anything
            else (telemetry is fail-soft).

    Returns:
        ``{mage: tag}`` for every seat with at least one attempt; empty when
        the telemetry is absent or malformed.
    """
    if not isinstance(completions, dict):
        return {}
    used: dict[str, Any] = {}
    for mage, attempts in completions.items():
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
            used[str(mage)] = attempts[-1].get("model")
    return used


def _every_seat_unmeasured(unmeasured: Any) -> bool:
    """Whether EVERY seat of the trio ran with an unmeasured context window.

    The ONE predicate behind both the report marker
    (``gate_ran_on_unverified_context``) and the raised tone of the stderr
    warning: two definitions would let the message and the artefact disagree.

    Args:
        unmeasured: The ``ran_unmeasured`` telemetry value, or anything else
            (fail-soft).

    Returns:
        True only when the set of names covers all of ``AGENTS``; duplicates
        do not count twice, an unknown name does not count at all.
    """
    if not isinstance(unmeasured, list):
        return False
    return set(AGENTS) <= {str(name) for name in unmeasured}


def announce_magi_rs_telemetry(
    result: ConsultResult, stream: IO[str] | None = None, *, degraded_locally: bool = False
) -> None:
    """Warn about anything that degrades a gate, on a channel a human reads.

    Emits at most four warnings, and NONE of them on a clean run -- a channel
    that speaks every time is a channel nobody reads, and then the run that
    mattered scrolls past unread. Sibling of :func:`announce_extraction_failures`
    and of the ``$0.00`` / unattributed warnings of v5.4.0, for the same reason:
    a field in a file nobody opens is found too late.

    Args:
        result: The parsed consult result.
        stream: Where to write; ``sys.stderr`` when None, resolved at CALL
            time (a default bound at definition time would outlive a
            redirected stderr). Injected so a test can capture it.
        degraded_locally: The orchestrator's own ``degraded`` flag. magi-rs
            reports ITS degradation (a seat that delivered nothing); a verdict
            that reached us and then failed the seven-key contract degrades the
            run on THIS side with magi-rs none the wiser (D6/BDD-28) -- the
            "gate is INVALID" warning must fire for either.

    Returns:
        None.

    Raises:
        Nothing. Telemetry must not be able to destroy the artifact it
        annotates -- the same fail-soft contract ``read_served_models`` has,
        and for the same reason: this runs immediately before ``json.dump``.

    Side effects:
        Writes to ``stream``.
    """
    out = sys.stderr if stream is None else stream
    try:
        _announce_magi_rs_telemetry(result, out, degraded_locally)
    except Exception as exc:  # noqa: BLE001 -- fail-soft by contract (see the docstring)
        print(f"[!] WARNING: could not render the magi-rs telemetry warnings ({exc})", file=out)


def _announce_magi_rs_telemetry(
    result: ConsultResult, stream: IO[str], degraded_locally: bool
) -> None:
    """The four warnings of :func:`announce_magi_rs_telemetry`, without the fail-soft wrap."""
    telemetry = result.telemetry
    if result.degraded or degraded_locally:
        cause = (
            "a seat delivered no verdict"
            if result.degraded
            else "a delivered verdict failed the seven-key contract here"
        )
        print(
            f"[!] WARNING: the run is DEGRADED ({cause}). "
            "The gate is INVALID whatever the verdict says -- resolve the cause and re-run.",
            file=stream,
        )
    capped = result.capped_agents
    if capped:
        completions = telemetry.get("completions") or {}
        detail = ", ".join(
            f"{mage} ({completions[mage][-1].get('model')}, cap {completions[mage][-1].get('cap')})"
            for mage in capped
        )
        print(
            f"[!] WARNING: the accepted attempt hit the completion cap for {detail}. The verdict "
            "counts (D2: the cut fell after the closing marker), but trailing prose was lost.",
            file=stream,
        )
    unmeasured = telemetry.get("ran_unmeasured")
    if isinstance(unmeasured, list) and unmeasured:
        seats = ", ".join(str(name) for name in sorted(unmeasured))
        if _every_seat_unmeasured(unmeasured):
            print(
                f"[!] WARNING: NOBODY MEASURED the context window: {seats} all ran unmeasured, "
                "so the input may have been silently truncated for every seat. "
                f"To fail closed instead, {_UNMEASURED_REMEDY}.",
                file=stream,
            )
        else:
            print(
                f"[!] WARNING: the context window was not measured for {seats}; the guard "
                f"estimated it and ran. To fail closed instead, {_UNMEASURED_REMEDY}.",
                file=stream,
            )
    failures = telemetry.get("extraction_failures")
    if isinstance(failures, dict):
        records = [
            (mage, rec)
            for mage, recs in sorted(failures.items())
            if isinstance(recs, list)
            for rec in recs
            if isinstance(rec, dict)
        ]
        if records:
            detail = "; ".join(
                f"{mage}: attempt {rec.get('attempt')} {rec.get('cause')} ({rec.get('model')})"
                for mage, rec in records
            )
            print(
                f"[!] WARNING: {len(records)} verdict extraction failure(s) upstream -- {detail}. "
                "magi-rs retried (and may have rotated); a seat that keeps doing this is a "
                "model drifting away from the marker contract.",
                file=stream,
            )


def announce_extraction_failures(failures: Mapping[str, Mapping[str, int]]) -> None:
    """Say out loud that a mage failed to deliver its verdict on some attempt (R18).

    The counts already reach ``magi-report.json``. That was not enough, and the proof is a
    real run: a gate cycle recorded ``caspar: {missing_markers: 1}`` -- the mage forgot the
    markers, the retry recovered it, the run came out valid, and the only trace was a field
    in a file nobody opens. A model drifting under the same tag looks exactly like that, run
    after run, until it stops being recoverable. **A counter that has to be gone looking for
    is a counter that gets found too late** (MAGI gate, Caspar).

    Silent on a clean run: a warning that fires every time is a warning nobody reads.

    Args:
        failures: The per-agent, per-cause tally, exactly as it reaches the report.
    """
    if not failures:
        return

    detail = "; ".join(
        f"{agent}: " + ", ".join(f"{cause}={count}" for cause, count in sorted(causes.items()))
        for agent, causes in sorted(failures.items())
    )
    print(
        f"[!] WARNING: verdict extraction failed on some attempts -- {detail}. "
        "The retry (and rotation) may have recovered it, but a seat that keeps doing this is "
        "a model drifting away from the marker contract. See docs/ollama-backend.md "
        "(what each cause means and what to do about it).",
        file=sys.stderr,
    )


def _record_extraction_failure(
    tally: dict[str, "Counter[str]"],
    agent: str,
    error: ValidationError | json.JSONDecodeError,
) -> None:
    """Record the CAUSE of the extraction failure in the adherence telemetry (R18).

    Reuses the same dispatcher that picks the retry's feedback
    (``retry_feedback.retry_feedback_cause``), so the telemetry and the corrective
    instruction **cannot disagree**: if the model is told "you were missing the markers",
    the counter that goes up is ``missing_markers``. Duplicating the classification would
    plant the seed of the report saying one thing and the prompt another, one day.

    Args:
        tally: The per-agent accumulator.
        agent: The mage whose attempt failed.
        error: The exception that took it down.
    """
    tally[agent][retry_feedback_cause(error)] += 1


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
    executor: TrioExecutor | None = None,
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
        executor: The :class:`TrioExecutor` that produces the raw verdicts
            when ``rotation`` is None. When None (every v5 caller), a
            :class:`ClaudeTrio` bound to this run's telemetry is built here;
            ``main()`` passes a :class:`MagiRsTrio` on ``--ollama`` (v6).
        show_status: Render a live status tree while agents run. When the
            stream is not a TTY, plain one-line-per-event output is emitted
            instead.

    Returns:
        Report dict with 'agents', 'consensus', and optionally
        'degraded' and 'failed_agents' when < 3 agents succeed.

    Raises:
        RuntimeError: If fewer than 2 agents succeed, or if ``max_attempts`` is < 1.
    """
    # Validated HERE, at the entry point, not inside the per-agent coroutine: raised in
    # there it would be captured by the ``gather`` and reported as "Only 0 agent(s)
    # succeeded", burying its own cause. NOT an assert (``python -O`` strips those, and the
    # attempt loop would then fall through to an UnboundLocalError on ``result``). The CLI
    # cannot produce this -- ``_max_attempts`` enforces the range -- but ``run_orchestrator``
    # is a public entry point with many direct callers.
    if not MIN_ATTEMPTS <= max_attempts <= MAX_ATTEMPTS_CAP:
        # BOTH ends. The CLI flag and the Ollama TOML are both bounded above; a direct Python
        # caller was the last unbounded door, and the budget is spent on PAID calls (MAGI gate,
        # Balthasar). NOT an assert: ``python -O`` strips those, and the loop below would then
        # fall through to an UnboundLocalError on ``result``.
        raise RuntimeError(
            f"max_attempts must be between {MIN_ATTEMPTS} and {MAX_ATTEMPTS_CAP} "
            f"(got {max_attempts})"
        )

    # The prompt-contract guard lives HERE, not only in ``main()`` (MAGI gate finding,
    # Balthasar): this is the function that actually hands the ``.md`` files to a model, so
    # this is the door that has to be shut. With the guard only in the CLI, any other caller
    # -- a test, an integration -- ran with stale prompts or with a fabricable verdict
    # BETWEEN the markers, which is precisely the last fabrication path MS2 closes.
    # ``main()`` still calls it earlier to abort as soon as possible (without creating the
    # temp dir or paying for the preflight); repeating it here costs three file reads and is
    # idempotent.
    AgentPromptGuard(Path(agents_dir), VerdictSentinel()).check()

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
    # R18 -- adherence telemetry. R17 measures ONCE, with TODAY's models; models drift under
    # the same tag. Without this, the day one starts omitting the markers it would look like
    # "MAGI is slow and rotates a lot" -- a symptom nobody would know how to read. Additive
    # and fail-soft: ``consensus`` does not read it.
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
            # N attempts (MS2/R13): the Claude path had a fixed SINGLE-SHOT retry; it is now
            # governed by ``--max-attempts`` (default 2 = the previous behaviour, exactly).
            #
            # It fires on schema drift (ValidationError -- which includes ALL the sentinel's
            # extraction failures) and on unparseable JSON (json.JSONDecodeError). NEVER on
            # timeout / subprocess failure / cancellation / ValueError. Each retry gets a
            # FRESH ``timeout`` (not the leftover of the previous one) and carries the
            # CAUSE-SPECIFIC corrective feedback: the exception type selects the instruction,
            # so a model that forgot the markers is told "you were missing the markers" and
            # not a generic schema message.
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

        def on_extraction_failure(err: ValidationError | json.JSONDecodeError) -> None:
            """Tally a schema failure of THIS mage into the run's adherence telemetry (R18).

            The rotation path is the one R18 exists for: a model that starts omitting the
            markers shows up here as a cause, instead of as an unreadable "MAGI is slow
            and rotates a lot". The mage's seat is what is counted -- a rotated model's
            failures belong to the same seat.
            """
            _record_extraction_failure(extraction_failures, name, err)

        _safe_display_update(display, name, "running", log_gate)
        async with ctx.registry.agent_slot(name, state) as slot:
            while True:
                try:
                    result = await _attempt_model(
                        name, spec, ctx, env, prompt, on_retry, on_extraction_failure
                    )
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

    # The Claude path's executor is chosen HERE, not in ``select_backend``: that
    # one returns a ``ClaudeBackend`` and ``test_run_magi.py:5065`` asserts it --
    # a file BDD-31 protects. ``backend`` still arrives and is still what the
    # rotation branch uses; the Claude branch simply no longer consults it.
    #
    # This is the design rather than a detour: ``select_backend`` resolves
    # CONFIG -- models, rotation, timeout -- not TRANSPORT. Having it also build
    # the backend was a coupling MS5 would have had to undo anyway.
    # v6: the open-weight path hands its executor in. ``MagiRsTrio`` reports
    # extraction failures through the envelope's telemetry (MS4), not through
    # this callback, so only the Claude executor is wired to it.
    trio: TrioExecutor = (
        executor
        if executor is not None
        else ClaudeTrio(
            on_extraction_failure=lambda name, err: _record_extraction_failure(
                extraction_failures, name, err
            ),
        )
    )

    def _on_state(name: str, state: str) -> None:
        """Bridge the executor's state callback to the display AND the telemetry.

        ``retried`` is the ONLY channel ``retried_agents`` has under the new
        contract: the v5 ``retried.add(name)`` lived inside
        ``_legacy_tracked_launch``, which moved into ``ClaudeTrio``. The callback
        already fires exactly when a retry happens, so no extra parameter is
        needed -- but the recording has to be HERE, in code. Lines 1683-1684
        derive two cohorts from this set, and ``test_run_magi.py`` checks them in
        a file BDD-31 forbids editing.

        Args:
            name: The mage the state belongs to.
            state: One of ``running``, ``retrying``, ``success``, ``timeout``,
                ``failed``.

        Returns:
            None.

        Side effects:
            Records into ``retried``; updates the live display.
        """
        # Record FIRST, draw second. If the display raises and the order were
        # reversed, the mage would drop out of ``retried_agents`` silently --
        # telemetry reporting fewer retries than happened, which is worse than
        # reporting none. The display is decoration; the set is data.
        if state == "retrying":
            retried.add(name)
        _safe_display_update(display, name, state, log_gate)

    if display is not None:
        try:
            await display.start()
        except Exception as exc:
            # A display-start failure (event-loop issue, terminal problem) must
            # never block the actual analysis. Drop the display and fall
            # through — the executor's callback will see ``display is None``.
            print(
                f"[!] WARNING: status display failed to start ({exc}) "
                f"\u2014 continuing without live status",
                file=sys.stderr,
            )
            display = None

    with _buffered_stderr_while(active=display is not None):
        try:
            if rotation is None:
                # Executor path: it owns the gather AND any per-mage retry.
                verdicts = await trio.run_trio(
                    prompt,
                    agents_dir,
                    output_dir,
                    timeout,
                    # Inert on the magi-rs path: MagiRsTrio ignores ``models`` (the
                    # trio is declared in .magi/magi.toml), so the "opus" defaults
                    # above never reach anything there.
                    {name: spec.model for name, spec in agent_models.items()},
                    max_attempts,
                    on_state=_on_state,
                )
                # One entry per mage: the verdict or ITS exception, the exact
                # object. The loop below branches on subtype, so no wrapper.
                results = [verdicts[name] for name in AGENTS]
            else:
                tasks = {name: _rotating_tracked_launch(name, rotation) for name in AGENTS}
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

    # ``zip(AGENTS, ...)`` and not ``tasks.keys()``: ``tasks`` only exists on the
    # rotation branch now, and ``results`` is built in AGENTS order on BOTH.
    for name, result in zip(AGENTS, results):
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
        # R18 has to SURVIVE the run's death, and this is the only place where it can. A run
        # that dies below the 2-mage floor does NOT write ``magi-report.json`` -- and with it
        # would go the only data that explains the death. The day a model starts omitting the
        # markers, MAGI would die saying "only 0 mages succeeded" and nobody would know why:
        # exactly the unreadable symptom this telemetry exists to eliminate. It goes to
        # stderr because that is the only channel left when there is no report.
        if extraction_failures:
            causes = {agent: dict(counts) for agent, counts in sorted(extraction_failures.items())}
            print(
                f"[!] extraction_failures (R18, the run died before it could be reported): "
                f"{json.dumps(causes, sort_keys=True)}",
                file=sys.stderr,
            )
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
        # R18. ADDITIVE and FAIL-SOFT: ``consensus`` does not read it, and a report without
        # this field is still valid (NR3). It is the only thing that will SEE a model's
        # drift in production -- which, with a sample of 5+2, is the only place the real
        # rate can be known.
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

    # Short-circuit: --ollama-init delegates to ``magi-rs init`` and exits
    # before parse_args() so that mode/input positional arguments are not
    # required. We screen sys.argv directly; the flag is unambiguous (no value
    # follows it). The whole command is ONE call (R5): no config of the
    # plugin's own is written, nothing under .claude/ is read.
    if "--ollama-init" in sys.argv[1:]:
        try:
            output = run_ollama_init(os.getcwd())
        except Exception as exc:  # noqa: BLE001 - every bootstrap error is fatal here
            # The bootstrap's fail-closed errors are siblings of Exception on
            # purpose; at the CLI boundary each of them is a [FATAL] with its
            # own actionable message, never a traceback.
            print(f"[FATAL] {exc}", file=sys.stderr)
            sys.exit(1)
        print(output, end="" if output.endswith("\n") else "\n")
        sys.exit(0)

    args = parse_args()

    # A dry run of the prompt guard: no mode, no input, no tokens. Decided from the PARSED
    # args, not by scanning sys.argv -- a raw scan duplicates the flag name as a magic string
    # and bypasses argparse's abbreviation handling, so ``--check`` would be accepted, expanded,
    # and then quietly ignored (MAGI gate, Balthasar).
    if args.check_prompts:
        check_prompts(_default_agents_dir())

    # v6: the open-weight executor is chosen and verified HERE, before any input is
    # read -- see _select_executor. ``executor`` stays None on the Claude path.
    executor: TrioExecutor | None = None
    if args.ollama:
        # v6: the open-weight path is executed by magi-rs (R18). Nothing of the
        # v5 subsystem runs -- no plugin TOML, no preflight of our own: the trio,
        # its rotation and its context guard are declared in .magi/magi.toml and
        # owned by magi-rs (R7). The v5 backend factory is the Claude path's only.
        backend, agent_models, rotation = None, None, None
        # R12: --timeout is BOTH the wall ceiling and the source magi-rs derives
        # each mage's attempt budget from; the default is the measured 1800 and
        # a low explicit value is honoured with a warning, never silently.
        try:
            resolved_timeout, timeout_warning = resolve_magi_rs_timeout(args.timeout)
        except MagiRsContractError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        if timeout_warning is not None:
            print(f"[!] WARNING: {timeout_warning}", file=sys.stderr)
        try:
            executor = _select_executor(args, resolved_timeout)
            # Bootstrap + project preflight BEFORE the payload is built (spec
            # §3.2): both are fail-closed and free; a bundle built for a run
            # that cannot start is pure waste. (``None`` is the Claude path's
            # arm of the selector and cannot occur under ``--ollama``.)
            if executor is not None:
                executor.preflight()
        except Exception as exc:  # noqa: BLE001 -- every bootstrap/preflight error is fatal here
            # Fail-closed siblings of Exception (PluginRootError, PinFormatError,
            # the bootstrap's, ProjectNotInitialisedError): at the CLI boundary
            # each is a [FATAL] with its own message, never a traceback, exactly
            # like --ollama-init.
            print(f"[FATAL] {exc}", file=sys.stderr)
            sys.exit(1)

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

    agents_dir = _default_agents_dir()

    # Hard prerequisite check: only required for the Claude path. When --ollama
    # is set the claude CLI is not used, so skip the gate entirely to allow the
    # Ollama backend to run even when claude is absent from PATH.
    if not args.ollama and not shutil.which("claude"):
        print("ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    # --- Prompt-contract startup guard (R9, MS2) -------------------------------------
    #
    # Runs BEFORE spending a single token, against the ``agents_dir`` the run will ACTUALLY
    # use (not a fixed path). It covers what no test of ours can see: **the user's
    # installation**. The anchoring test runs in the developer's repo; the stale-copy bug
    # (``mklink /D`` silently degrading to a copy on Windows) produces old prompts with a
    # new parser on the user's machine.
    #
    # And it closes the LAST fabrication path: a user who "improves" the prompt by putting a
    # complete example BETWEEN the markers lets the model copy it and lets that copy be
    # accepted as a verdict -- neither the canary (it is not the shipped example) nor the
    # anchoring test (it runs in OUR repo) would see it.
    #
    # ``PromptContractError`` is a SIBLING of ``ValidationError``, not a child: a stale
    # prompt is NOT fixed by retrying, so the retry guard must not swallow it. Here it is
    # caught and the run aborts -- exactly like ``OllamaConfigError``.
    try:
        AgentPromptGuard(Path(agents_dir), VerdictSentinel()).check()
    except PromptContractError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.ollama:
        try:
            # select_backend is async now (T9): the Ollama preflight MEASURES the
            # payload. Its own asyncio.run runs here, BEFORE the output dir is created,
            # so a config/preflight error exits without leaking a temp run dir -- the
            # exact ordering of the v4 sync call it replaces. The rotation's asyncio.Event
            # is only constructed (never awaited) in this loop, so it binds lazily to the
            # run_orchestrator loop on its first wait().
            backend, agent_models, rotation, toml_timeout = asyncio.run(
                select_backend(args, prompt)
            )
        except (OllamaConfigError, OllamaPreflightError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        # MS3 (R6): flag --timeout > TOML timeout > 900s default. On the Claude path
        # ``toml_timeout`` is always None, so this is exactly the v4 behaviour (flag
        # or 900). AUDITED: no other read of ``args.timeout`` survives below --
        # every per-agent timeout use goes through this single resolution.
        resolved_timeout = int(_resolve_timeout(args.timeout, toml_timeout))

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
        write_lock(output_dir, staleness_bound_for_timeout(resolved_timeout))
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
        # The trio is magi-rs's business, declared in the project's own file.
        print("|  Model: magi-rs (.magi/magi.toml)")
    else:
        print(f"|  Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"|  Timeout: {resolved_timeout}s")
    print(f"|  Output: {output_dir}")
    print("+==================================================+")
    print(flush=True)

    # MS4b: a bundle that does not fit under magi-rs's input ceiling (with headroom)
    # is split at section boundaries and evaluated one segment at a time. The whole
    # prompt is what travels, so the wrapper sanitize adds is charged against the
    # ceiling. Decided BEFORE anything runs; a section too large to fit is refused.
    if executor is not None:
        sections = split_sections(input_content)
        wrapper_bytes = len(build_user_prompt(args.mode, "").encode("utf-8"))
        try:
            segment_plan = BundleSegmenter.plan(
                sections,
                CONSULT_INPUT_CEILING_BYTES - wrapper_bytes,
                overhead_bytes=_SECTION_RENDER_OVERHEAD_BYTES,
            )
        except SegmentTooLargeError as exc:
            if is_temp_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
            print(f"[FATAL] {exc}", file=sys.stderr)
            sys.exit(1)
        if segment_plan.is_segmented:
            announce_segmentation(
                segment_plan,
                payload_bytes=len(prompt.encode("utf-8")),
                ceiling=CONSULT_INPUT_CEILING_BYTES,
            )
            try:
                run_segmented(
                    segment_plan,
                    sections,
                    args,
                    output_dir,
                    agents_dir,
                    resolved_timeout,
                    wrapper_bytes=wrapper_bytes,
                )
            except BaseException as exc:
                if is_temp_dir:
                    shutil.rmtree(output_dir, ignore_errors=True)
                # Only the bootstrap's fail-closed family reaches here: a segment's
                # own failures are tabled inside run_segmented (§6.2.5).
                if isinstance(exc, FAIL_CLOSED_ERRORS):
                    print(f"[FATAL] {exc}", file=sys.stderr)
                    sys.exit(1)
                raise
            print(f"\nFull report saved to: {os.path.join(output_dir, 'magi-report.json')}")
            if is_temp_dir:
                remove_lock(output_dir)
            return

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
                resolved_timeout,
                agent_models=agent_models,
                backend=backend,
                rotation=rotation,
                executor=executor,
                show_status=args.show_status,
                max_attempts=args.max_attempts,
            )
        )
    except BaseException as exc:
        if is_temp_dir:
            try:
                shutil.rmtree(output_dir)
            except OSError as cleanup_exc:
                print(
                    f"WARNING: Failed to clean up {output_dir}: {cleanup_exc}",
                    file=sys.stderr,
                )
        if isinstance(exc, FAIL_CLOSED_ERRORS):
            # The preflight run_trio repeats (R4) failed differently from the
            # one main() ran: same rendering as the first, never a traceback.
            print(f"[FATAL] {exc}", file=sys.stderr)
            sys.exit(1)
        if isinstance(exc, (MagiRsProcessError, MagiRsContractError)):
            # BDD-14a/14b/14d: the trio never formed (the child died, timed out,
            # or answered off-contract). No report, the cause relayed, exit 1 --
            # and no traceback, so the operator can tell this from a plugin bug.
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
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
    # R20: the open-weight path has no usage to aggregate -- NOT tariffed, which
    # is not zero. The Claude path keeps its per-envelope aggregation.
    report["cost"] = unpriced_cost() if args.ollama else aggregate_cost(output_dir, list(AGENTS))
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

    # Attribution (5.4.0): record which model produced this verdict, same
    # set-before-json.dump discipline as the two blocks above.
    #
    # ``resolved`` is only what the CLI was HANDED, and every Claude model ID is
    # now a floating alias -- so on its own it cannot tell two servings of
    # ``claude-opus-5`` apart, which is the very drift this block exists to make
    # visible. ``served`` closes that: it is read back out of each agent's raw
    # envelope, i.e. what actually answered.
    served = read_served_models(output_dir, list(AGENTS))
    unattributed = sorted(set(AGENTS) - set(served))
    report["model"] = _model_block(args, served, unattributed)
    # Symmetric with the $0.00 cost warning above, and for the same reason: a
    # telemetry field that quietly stopped working is indistinguishable from one
    # with nothing to report. Attributing NONE of three agents on the Claude
    # path means the envelope contract moved -- say so once, on stderr.
    if not args.ollama and report["agents"] and not served:
        print(
            "[!] WARNING: could not attribute any agent to a served model; the "
            "CLI may have renamed modelUsage/canonicalModel — check raw envelopes.",
            file=sys.stderr,
        )

    # MS4: the magi-rs telemetry travels VERBATIM into the report (R15/R16),
    # from the executor's last parsed envelope -- same set-before-json.dump
    # discipline as the blocks above. Absent on the Claude path (additive).
    magi_rs_result: ConsultResult | None = getattr(executor, "last_result", None)
    if magi_rs_result is not None:
        report.update(build_report_block(magi_rs_result))

    report_path = os.path.join(output_dir, "magi-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # And what degrades a gate is SHOUTED, after the report exists so the message
    # and the file it points to always agree (D2, D3, D11, R17).
    if magi_rs_result is not None:
        announce_magi_rs_telemetry(
            magi_rs_result, degraded_locally=bool(report.get("degraded", False))
        )

    # R18 made ACTIVE: filing the counts was not enough (MAGI gate, Caspar). Announced here,
    # after the report exists, so the message and the file it points to always agree.
    announce_extraction_failures(report.get("extraction_failures", {}))

    print(f"\nFull report saved to: {report_path}")
    total_usd = report["cost"]["total_usd"]
    cost_label = "unpriced (magi-rs exposes no usage)" if total_usd is None else f"${total_usd:.4f}"
    print(f"Cost: {cost_label} ({len(report['agents'])} agents)")
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
