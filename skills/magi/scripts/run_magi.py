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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Bootstrap: make sibling modules importable under invocations that do NOT
# auto-inject this directory into sys.path (e.g. ``python -m
# skills.magi.scripts.run_magi``). Direct invocation
# (``python skills/magi/scripts/run_magi.py``) and pytest (via conftest.py)
# already cover this. See CLAUDE.md "Open technical debt /
# synthesize import gap [LOCKED]".
_SCRIPT_DIR = str(Path(__file__).parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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
from subprocess_utils import (  # noqa: E402
    format_stderr_excerpt as _format_stderr_excerpt,
    reap_and_drain_stderr as _reap_and_drain_stderr,
    write_stderr_log as _write_stderr_log,
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
from input_size import WARN_INPUT_TOKENS, check_input_size  # noqa: E402
from finding_validation import parse_diff_ranges, validate_findings  # noqa: E402
from validate import MAX_INPUT_FILE_SIZE, ValidationError  # noqa: E402

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

AGENTS = ("melchior", "balthasar", "caspar")
MAX_HISTORY_RUNS = 5
VALID_MODES = ("code-review", "design", "analysis")


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
            f"(default: {WARN_INPUT_TOKENS}). MAGI reviews the input whole; "
            f"this is a detect-and-warn guard, not a hard limit."
        ),
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
    # Per-mode default model resolution (2.2.3). ``argparse`` cannot express
    # "default depends on another arg" cleanly, so we resolve here. The mode
    # has already been validated by ``choices=VALID_MODES`` above, so the
    # ``MODE_DEFAULT_MODELS`` lookup is total — no KeyError path is reachable
    # while VALID_MODES and MODE_DEFAULT_MODELS stay in lockstep (a guarantee
    # the test suite pins).
    if args.model is None:
        args.model = MODE_DEFAULT_MODELS[args.mode]
    return args


async def launch_agent(
    agent_name: str,
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    model: str = "opus",
) -> dict[str, Any]:
    """Launch a single agent subprocess and return validated output.

    Runs ``claude -p`` with the agent's system prompt, applies timeout,
    parses the raw output, and validates against the agent JSON schema.
    The user prompt is sent via stdin to avoid OS CLI argument length
    limits.  A copy is also saved to ``{agent_name}.prompt.txt`` in
    *output_dir* as a debug artifact.

    Args:
        agent_name: One of 'melchior', 'balthasar', 'caspar'.
        agents_dir: Directory containing agent prompt .md files.
        prompt: The prompt payload to send to the agent.
        output_dir: Directory for raw and parsed output files.
        timeout: Timeout in seconds per agent.
        model: Model short name ('opus', 'sonnet', 'haiku').

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
    model_id = resolve_model(model)

    system_prompt_file = os.path.join(agents_dir, f"{agent_name}.md")
    raw_file = os.path.join(output_dir, f"{agent_name}.raw.json")
    parsed_file = os.path.join(output_dir, f"{agent_name}.json")

    # Write user prompt to a temp file and pass via stdin to avoid
    # OS CLI argument length limits (~32K on Windows).
    prompt_file = os.path.join(output_dir, f"{agent_name}.prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        model_id,
        "--system-prompt-file",
        system_prompt_file,
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        stderr_buffered = await _reap_and_drain_stderr(proc)
        # Persisting the log is best-effort. If it fails (disk full,
        # permission denied), surface a warning but do not let the
        # OSError shadow the TimeoutError the caller actually needs.
        try:
            _write_stderr_log(output_dir, agent_name, stderr_buffered)
        except OSError as log_exc:
            print(
                f"WARNING: Failed to persist {agent_name}.stderr.log on timeout: {log_exc}",
                file=sys.stderr,
            )
        raise TimeoutError(
            f"Agent '{agent_name}' timed out after {timeout}s"
            f"{_format_stderr_excerpt(stderr_buffered)}"
        ) from None

    with open(raw_file, "wb") as f:
        f.write(stdout)

    # The stderr log is a diagnostic artefact, not load-bearing. A disk
    # error here (disk full, permission drop, antivirus lock on Windows)
    # must not turn an otherwise-successful agent into a reported
    # failure. Mirror the timeout-path pattern: warn and continue.
    try:
        _write_stderr_log(output_dir, agent_name, stderr)
    except OSError as log_exc:
        print(
            f"WARNING: Failed to persist {agent_name}.stderr.log: {log_exc}",
            file=sys.stderr,
        )

    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else "no stderr"
        raise RuntimeError(
            f"Agent '{agent_name}' exited with code {proc.returncode}: {stderr_text}"
        )

    parse_raw_output(raw_file, parsed_file)
    return load_agent_output(parsed_file)


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


def _build_retry_prompt(original_prompt: str, error: ValidationError | json.JSONDecodeError) -> str:
    """Return the retry prompt with corrective feedback appended.

    When :func:`launch_agent` raises :class:`ValidationError` (schema
    fail) or :class:`json.JSONDecodeError` (output is not parseable JSON)
    on the first attempt, :func:`run_orchestrator` calls this helper to
    build the replacement prompt for the single retry. The original
    user prompt is preserved verbatim so the agent's task is unchanged;
    the parser/validator error message is appended so the model can
    self-correct the specific defect — a missing key, a stray comma, a
    truncated output, an unbalanced brace, etc. The envelope delimiter
    ``---RETRY-FEEDBACK---`` is intentionally distinct from user input
    so the model can identify the corrective block even if the original
    prompt already contains arbitrary markdown.

    Args:
        original_prompt: The exact prompt sent on the first attempt.
        error: The exception that triggered the retry. Currently either
            :class:`ValidationError` (schema mismatch) or
            :class:`json.JSONDecodeError` (output not parseable as JSON).

    Returns:
        A new prompt string that concatenates the original prompt with a
        feedback block describing the failure and restating the schema
        contract.
    """
    return (
        f"{original_prompt}\n\n"
        f"---RETRY-FEEDBACK---\n"
        f"Your previous response was rejected by the parsing pipeline:\n"
        f"{error}\n\n"
        f"Re-emit your response as a complete, syntactically valid JSON "
        f"object containing ALL seven required top-level keys: agent, "
        f"verdict, confidence, summary, reasoning, findings, "
        f"recommendation. Do not omit any key, do not truncate, do not "
        f"emit anything outside the JSON object."
    )


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


async def run_orchestrator(
    agents_dir: str,
    prompt: str,
    output_dir: str,
    timeout: int,
    model: str = "opus",
    *,
    show_status: bool = True,
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
        model: Model short name ('opus', 'sonnet', 'haiku').
        show_status: Render a live status tree while agents run. When the
            stream is not a TTY, plain one-line-per-event output is emitted
            instead.

    Returns:
        Report dict with 'agents', 'consensus', and optionally
        'degraded' and 'failed_agents' when < 3 agents succeed.

    Raises:
        RuntimeError: If fewer than 2 agents succeed.
    """
    successful: list[dict[str, Any]] = []
    failed: list[str] = []
    # Telemetry (2.2.1): names of agents whose first attempt raised
    # ValidationError, regardless of whether the retry recovered.
    # Composes with ``failed`` to give downstream consumers two derived
    # cohorts: ``retried - failed`` is "retry recovered",
    # ``retried & failed`` is "retry also failed".
    retried: set[str] = set()

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

    async def tracked_launch(name: str) -> dict[str, Any]:
        """Launch an agent with live status updates and one retry on schema fail.

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
            try:
                result = await launch_agent(name, agents_dir, prompt, output_dir, timeout, model)
            except (ValidationError, json.JSONDecodeError) as err:
                # Single-shot retry (2.2.0 + 2.2.4): fires on schema
                # drift (ValidationError, 2.2.0 scope) AND on JSON parse
                # failures (json.JSONDecodeError, 2.2.4 scope expansion).
                # Never on timeout / subprocess failure / cancellation /
                # ValueError (config or parser-shape errors). The retry
                # gets a fresh ``timeout`` budget (not the residual of
                # the first attempt) and carries the parser/validator
                # text so the model can target the specific defect —
                # missing key, truncated output, unbalanced brace, etc.
                retried.add(name)
                _safe_display_update(display, name, "retrying", log_gate)
                result = await launch_agent(
                    name,
                    agents_dir,
                    _build_retry_prompt(prompt, err),
                    output_dir,
                    timeout,
                    model,
                )
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
    agents: list[dict[str, Any]], mode: str, files: set[str], ranges: dict[str, set[int]]
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

    Returns:
        A new list of agent dicts with guarded findings (same order). Agents
        for which the guard fails are passed through with original findings.
    """
    if mode != "code-review" or not files:
        return agents

    out: list[dict[str, Any]] = []
    for a in agents:
        try:
            original = a.get("findings", [])
            kept, dropped, annotated = validate_findings(original, files, ranges)
            a = {**a, "findings": kept}
            if dropped or annotated:
                # Compute dropped titles by title-set difference so annotated
                # (soft-annotated, KEPT) findings — whose original dict object
                # is replaced by a new dict — are not mistakenly listed as
                # dropped. Annotated findings retain their original title, so
                # any title still present in *kept* belongs to a kept finding.
                kept_titles = {str(f.get("title", "")) for f in kept}
                dropped_titles = [
                    str(f.get("title", ""))
                    for f in original
                    if str(f.get("title", "")) not in kept_titles
                ]
                print(
                    f"[guard] {a['agent']}: dropped {dropped} "
                    f"titles={dropped_titles}, annotated {annotated}",
                    file=sys.stderr,
                )
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


def main() -> None:
    """CLI entry point for MAGI orchestrator."""
    # Must run BEFORE any ``print`` or ``sys.exit`` — every output
    # path past this line assumes UTF-8 + backslashreplace on
    # Windows. A later call site cannot fix a crash that already
    # happened on an earlier print.
    _enable_utf8_console_io()
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

    # Hard prerequisite check runs **before** any filesystem setup so a
    # missing CLI cannot leak a half-initialised temp directory on disk.
    if not shutil.which("claude"):
        print("ERROR: 'claude' CLI not found in PATH", file=sys.stderr)
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
                args.model,
                show_status=args.show_status,
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
    report["agents"] = _apply_finding_guard(report["agents"], args.mode, files, ranges)

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

    print(format_report(report["agents"], report["consensus"]))

    # A1: aggregate per-run cost into the report BEFORE it is serialized so the
    # saved magi-report.json carries the ``cost`` block. Aggregate over all
    # canonical agent names (AGENTS), not just report["agents"], so a failed or
    # timed-out agent that wrote its raw envelope still contributes to the total.
    # Fail-safe: a missing or corrupt envelope contributes 0 for that agent.
    report["cost"] = aggregate_cost(output_dir, list(AGENTS))
    # FIX 4: if the aggregated cost is $0.00 despite having at least one agent,
    # the CLI may have renamed or relocated ``total_cost_usd`` — emit a single
    # warning so the silent mis-reporting is visible in operator logs.
    if report["cost"]["total_usd"] == 0.0 and report["agents"]:
        print(
            "[!] WARNING: per-run cost resolved to $0.00; the CLI may have "
            "renamed the total_cost_usd field — check raw envelopes.",
            file=sys.stderr,
        )

    # Input-size telemetry: record the raw-input footprint in the report so the
    # saved magi-report.json carries observable per-run size data (mirrors the
    # ``cost`` block discipline: set BEFORE json.dump). ``est_tokens`` and
    # ``oversize`` were computed right after _load_input_content above.
    report["input_size"] = {
        "chars": len(input_content),
        "est_tokens": est_tokens,
    }

    report_path = os.path.join(output_dir, "magi-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")
    print(f"Cost: ${report['cost']['total_usd']:.4f} ({len(report['agents'])} agents)")
    print(f"Input size: ~{est_tokens} tokens ({len(input_content)} chars)")
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
