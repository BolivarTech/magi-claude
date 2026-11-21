# Author: Julian Bolivar
# Version: 7.1.0
# Date: 2026-11-19
"""Run the MAGI trio by delegating to the pinned ``magi-rs`` binary.

This is the second implementer of :class:`TrioExecutor` -- the one that
justified the abstraction existing at all. The plugin still builds the payload
and judges the result; ``magi-rs`` only runs the three mages.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any

from cli_commands import VAULT_SET_FLAG, command
from magi_key import KeyExposedError, KeyMissingError, KeyStore, MagiKey
from magi_key_permissions import default_key_store
from magi_rs_bootstrap import (
    BinaryBootstrap,
    DigestMismatchError,
    DownloadError,
    UnsafeArchiveError,
    UnsafeBinDirectoryError,
    VersionMismatchError,
    verified_executable,
)
from magi_rs_env import PASSPHRASE_VARIABLE, EnvironmentPolicy
from magi_rs_paths import PluginRootError
from magi_rs_pin import Pin, PinFormatError, UnsupportedPlatformError
from magi_rs_embedder import EmbedderUnreachableError
from magi_rs_preflight import ProjectNotInitialisedError, ProjectPreflight
from magi_rs_result import ConsultResult, MagiRsContractError
from synthesize import ValidationError, load_agent_output
from trio_executor import TrioExecutor

__all__ = [
    "MagiRsContractError",
    "MagiRsProcessError",
    "MagiRsTrio",
    "build_argv",
    "resolve_timeout",
]

_VALID_MODES = ("code-review", "design", "analysis")

_SUBCOMMAND = "consult"
"""The direct route, which FORCES the three-perspective MAGI pass. NOT
``query --consult``, which folds the report into the agent's tool loop."""

_WORKDIR_FLAG = "-w"
"""The walk-up base for ``.magi/`` AND the sandbox root (R6). Never inherited
from the cwd: the plugin may be invoked from any directory."""

_OUTPUT_FORMAT_FLAGS = ("--output-format", "json")
"""Without JSON there is nothing to parse (R8)."""

_STRUCTURED_VERDICTS_FLAG = "--structured-verdicts"
"""Mandatory: without it the verdicts arrive only as markdown and R14 is
impossible. Forgetting it does not degrade gracefully."""

_MODE_FLAG = "--mode"
"""Always explicit (R10): omitting it costs ``magi-rs`` a classifying model
call and lets the content under review choose the lens it is reviewed under."""

_UNTRUSTED_CONTENT_FLAG = "--untrusted-content"
"""The payload is third-party code (R11). It also turns an omitted ``--mode``
into an error, closing by construction the path R10 closes by convention."""

_TIMEOUT_FLAG = "--timeout"
"""Wall-clock ceiling (R12). ``magi-rs`` also DERIVES the per-mage budget from
it: with magi-rs 0.21.0, 0.24.1 and 0.25.0, 16820 derives a 2335 s per-mage ceiling, a
1401 s operation budget and a 700 s per-request clock (measured 2026-09-29,
2026-10-13 and 2026-11-18)."""

_TRIO_NODE = "magi-rs consult · 3 mages"
"""The single status node. D5/BDD-27: the plugin launches ONE child and cannot
observe individual mages, so three nodes would invent progress."""

_RAW_ARTIFACT = "magi-rs-raw.json"
#: The three seats. A mage magi-rs leaves out of ``agents`` (it is in
#: ``failed_agents`` instead) still gets an entry: its failure.
_MAGES = ("melchior", "balthasar", "caspar")
_NO_VERDICT_CAUSE = "absent from agents and from failed_agents"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 16820
"""magi-rs 0.21.0's published headless gate value; derives its 2335 s default
ceiling (700 s per request, 1401 s operation budget). Source: 0.21.0 release
notes and docs/REASONING-BUDGET.md, replay B (2026-09-27); confirmed locally
on 2026-09-29 against the pinned binary, and unchanged with 0.24.1 on
2026-10-13 and with 0.25.0 on 2026-11-18. Re-measured at every pin move. Public: tools/measure_marker_adherence.py reads it."""
_RECOMMENDED_MIN_TIMEOUT_SECONDS = DEFAULT_OLLAMA_TIMEOUT_SECONDS
"""Below the default the derived ceiling falls under the only value measured
with the 65 536-token output cap, so a lower value warns (D2)."""
_VAULT_HINT = (
    "If this is an authentication failure, load the API keys the config needs with "
    f"`{command(VAULT_SET_FLAG, '<NAME>')}`."
)
"""Every non-zero exit carries it: magi-rs offers no free check for a missing key (REQ-13d)."""
_STDERR_EXCERPT_CHARS = 2000
_STDERR_ELISION_MARKER = "[... earlier stderr elided ...] "
"""Prefixed to a failure's stderr excerpt when its head was cut. Since 0.20 magi-rs
writes WARN lines at startup and the real cause of a failure comes last, so the
excerpt keeps the tail (v6.2.0 R14); the marker says the head is missing."""


#: Every fail-closed error the bootstrap and the preflight can raise. All are
#: siblings of ``Exception`` (retrying does not fix any of them), so the CLI
#: boundary renders each as one ``[FATAL]`` line -- wherever it surfaces: the
#: early ``preflight()`` in ``main()`` or the one ``run_trio`` repeats.
FAIL_CLOSED_ERRORS: tuple[type[Exception], ...] = (
    PluginRootError,
    PinFormatError,
    UnsupportedPlatformError,
    UnsafeBinDirectoryError,
    DownloadError,
    UnsafeArchiveError,
    DigestMismatchError,
    VersionMismatchError,
    ProjectNotInitialisedError,
    KeyMissingError,
    KeyExposedError,
    EmbedderUnreachableError,
)


class MagiRsProcessError(ValidationError):
    """The ``magi-rs`` process died, timed out, or was killed.

    This one DOES inherit from ``ValidationError``, and for the reason the rule
    gives: a child that timed out or was OOM-killed may well succeed on a
    second attempt -- it is the transient class of the taxonomy. Note what
    that does and does not buy today: on the magi-rs path NOTHING retries it
    (``run_orchestrator`` runs the executor once; the per-mage retry lives
    inside ``ClaudeTrio``, and magi-rs owns its own retries and rotation), so
    the inheritance keeps the project's error taxonomy honest -- transient vs
    deterministic -- rather than driving a loop. ``main()`` renders it as one
    ``ERROR:`` line and exits 1.
    """


def build_argv(executable: str, workdir: str, mode: str, timeout: int) -> list[str]:
    """Build the exact command line, with every flag justified.

    Args:
        executable: Path of the verified binary.
        workdir: Project root -- the walk-up base for ``.magi/`` and the
            sandbox root. Never inherited from the cwd.
        mode: One of ``code-review``, ``design``, ``analysis``. ALWAYS explicit:
            omitting it costs ``magi-rs`` an extra model call to classify, and
            lets the content under review choose the lens it is reviewed under.
        timeout: Wall-clock ceiling in seconds. ``magi-rs`` also DERIVES the
            per-mage budget from it (16820 -> 2335 s per mage with 0.25.0).

    Returns:
        The argv. The prompt is NOT in it -- it travels on stdin, because the
        ~32K argv limit on Windows is still in force. No ``-p`` (R5a), no
        ``--max-tool-calls`` (this path has no tool loop).

    Raises:
        MagiRsContractError: ``mode`` is not one of the three. Deterministic --
            retrying with the same mode fails the same way.
    """
    if mode not in _VALID_MODES:
        raise MagiRsContractError(f"unknown mode {mode!r}; expected one of {_VALID_MODES}")
    return [
        executable,
        _SUBCOMMAND,
        _WORKDIR_FLAG,
        workdir,
        *_OUTPUT_FORMAT_FLAGS,
        _STRUCTURED_VERDICTS_FLAG,
        _MODE_FLAG,
        mode,
        _UNTRUSTED_CONTENT_FLAG,
        _TIMEOUT_FLAG,
        str(timeout),
    ]


def resolve_timeout(flag_value: int | None) -> tuple[int, str | None]:
    """Resolve the wall timeout and warn when it starves the per-mage budget.

    ``magi-rs`` DERIVES each mage's budget from this value; with 0.25.0,
    ``--timeout 16820`` derives the 2335 s per-mage ceiling that magi-rs
    measured with its 65 536-token output cap (confirmed 2026-09-29, 2026-10-13, 2026-11-18). The open-weight
    path defaults to that value, and a lower explicit value is honoured with a
    warning rather than silently accepted.

    Args:
        flag_value: The operator's ``--timeout``, or None when omitted.

    Returns:
        A pair of (seconds, warning-or-None).

    Raises:
        MagiRsContractError: The value is not a positive integer. Deterministic.
    """
    if flag_value is None:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS, None
    if isinstance(flag_value, bool) or not isinstance(flag_value, int) or flag_value <= 0:
        raise MagiRsContractError(f"--timeout must be a positive integer, got {flag_value!r}")
    if flag_value < _RECOMMENDED_MIN_TIMEOUT_SECONDS:
        return flag_value, (
            f"--timeout {flag_value} is below the recommended "
            f"{_RECOMMENDED_MIN_TIMEOUT_SECONDS}s: magi-rs derives each mage's attempt budget "
            "from it, so a low wall timeout also cuts the reasoning budget of every seat"
        )
    return flag_value, None


def write_raw_artifact(output_dir: str, raw: str | bytes) -> str:
    """Persist magi-rs's stdout VERBATIM, under its own name (R16).

    Two producers under one name is ambiguity nobody disambiguates later, so
    this writes ``magi-rs-raw.json`` and never ``magi-report.json``.

    Args:
        output_dir: The run's output directory.
        raw: The child's stdout, exactly as received. NOT re-serialised -- a
            round-trip through json would silently normalise key order and
            float formatting, and this file is the audit artifact. ``str`` is
            written as UTF-8.

    Returns:
        The path written.

    Raises:
        OSError: The file cannot be written. NOT swallowed here: the strict
            primitive. :meth:`MagiRsTrio._persist_raw` is the fail-soft caller
            that turns it into a warning, because an audit artifact must not
            be able to destroy the verdicts it records.

    Side effects:
        Writes one file.
    """
    path = os.path.join(output_dir, _RAW_ARTIFACT)
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


def _emit(on_state: Callable[[str, str], None] | None, node: str, state: str) -> None:
    """Forward one status transition, tolerating a callback that raises.

    Args:
        on_state: The caller's callback, or None when status is off.
        node: The node's name. ONE node in this executor -- see run_trio.
        state: ``running``, ``success``, ``failed`` or ``timeout``.

    Returns:
        None.

    Note:
        A status display that raises must not take the run down with it: the
        verdicts are the product, the tree is decoration. Same fail-soft
        contract ``_safe_display_update`` already has on the Claude path.
    """
    if on_state is None:
        return
    try:
        on_state(node, state)
    except Exception as exc:  # noqa: BLE001 - fail-soft by contract (see Note)
        print(f"WARNING: status display failed for {node!r}: {exc}", file=sys.stderr)


def _stderr_tail(stderr: bytes) -> str:
    """Return the last ``_STDERR_EXCERPT_CHARS`` characters of a child's stderr.

    Args:
        stderr: The raw bytes the child wrote; need not be valid UTF-8.

    Returns:
        The decoded text (undecodable bytes replaced), whole when it fits, else
        its tail prefixed with ``_STDERR_ELISION_MARKER``.

    Example:
        >>> _stderr_tail(b"boom")
        'boom'

    Note:
        The tail, not the head: magi-rs's startup WARN lines come first and the
        cause of a failure comes last (v6.2.0 R14). O(n) in the stderr size.
    """
    text = stderr.decode("utf-8", errors="replace")
    if len(text) <= _STDERR_EXCERPT_CHARS:
        return text
    return _STDERR_ELISION_MARKER + text[-_STDERR_EXCERPT_CHARS:]


class MagiRsTrio(TrioExecutor):
    """Run the three mages by delegating to one ``magi-rs consult``.

    This is the SECOND implementer of :class:`TrioExecutor`, and the one that
    justified the abstraction existing. Where :class:`ClaudeTrio` launches three
    processes concurrently, this one launches ONE that runs the trio itself --
    which is precisely why the boundary sits at the trio and not at the mage.

    Example:
        >>> trio = MagiRsTrio(workdir="/repo", mode="code-review", timeout=16820,
        ...                   bin_dir="/plugin/bin", pin=Pin.load(pin_path))
        >>> verdicts = await trio.run_trio(prompt=payload, agents_dir="agents",
        ...                                output_dir=out, timeout=16820,
        ...                                models={}, max_attempts=1)

    Side effects:
        Launches one process; writes ``magi-rs-raw.json`` and one
        ``{mage}.json`` per verdict into ``output_dir``.
    """

    def __init__(
        self,
        workdir: str,
        mode: str,
        timeout: int,
        bin_dir: str,
        pin: Pin,
        key_store: KeyStore | None = None,
    ) -> None:
        """Bind the executor to one project and one invocation.

        Args:
            workdir: Project root. Passed as ``-w`` and used for the preflight.
                Resolved by the CALLER (Task 5) -- never inherited from the cwd,
                because the plugin can be invoked from any directory.
            mode: ``code-review``, ``design`` or ``analysis``. Always explicit.
            timeout: Wall-clock ceiling, already resolved by ``resolve_timeout``.
            bin_dir: The plugin's ``bin/``.
            pin: The parsed pin, for the bootstrap.
            key_store: Where the key is loaded from. ``None`` means the
                platform store from :func:`default_key_store`.
        """
        self._workdir = workdir
        self._mode = mode
        self._timeout = timeout
        self._bin_dir = bin_dir
        self._pin = pin
        self._key_store = key_store if key_store is not None else default_key_store()
        #: The key the last preflight loaded; injected into the child, never logged.
        self._key: MagiKey | None = None
        #: The parsed envelope of the LAST run, for the report's ``magi_rs``
        #: block and the stderr warnings (MS4). ``None`` until a run completes
        #: past the contract check; the verdicts themselves travel through
        #: ``run_trio``'s return value, exactly as ``ClaudeTrio``'s do.
        self.last_result: ConsultResult | None = None

    async def run_trio(
        self,
        prompt: str,
        agents_dir: str,
        output_dir: str,
        timeout: int,
        models: Mapping[str, str],
        max_attempts: int,
        on_state: Callable[[str, str], None] | None = None,
    ) -> dict[str, dict[str, Any] | BaseException]:
        """Run the trio and return one validated verdict per mage.

        The order is fail-before-spending, and it is the order of the spec's own
        pipeline: verify the binary, check the project, THEN launch. Every one of
        those can fail for free; the launch cannot.

        ``agents_dir``, ``models`` and ``max_attempts`` are accepted and unused:
        the ABC is shared with :class:`ClaudeTrio`, and the trio's composition
        is ``magi-rs``'s business, declared in ``.magi/magi.toml``. Silently
        DROPPING them would be wrong; ignoring them explicitly, with this
        sentence, is the honest shape.

        Args:
            prompt: The payload built by ``sanitize``/``review_context``.
                Travels on STDIN -- the ~32K argv limit on Windows is in force.
            agents_dir: Unused here. See above.
            output_dir: Where ``magi-rs-raw.json`` and the ``{mage}.json``
                verdict files are written.
            timeout: IGNORED -- the deadline is the one resolved in the
                constructor, so the argv bound and the wall bound cannot
                disagree. Two sources of truth for one deadline lose a
                SILENT race: the wall bound fires first and kills a run
                magi-rs still believed it had time for.
            models: Unused here. See above.
            max_attempts: Unused here -- magi-rs owns its own retry and rotation.
            on_state: Optional progress callback, ``(node, state)``.

        Returns:
            One entry per mage: the validated verdict dict, or the exception
            that mage failed with. Same shape ``ClaudeTrio`` returns, because
            everything below the raw verdicts does NOT fork.

        Raises:
            UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError:
                The binary did not verify. It is NOT executed (R4).
            ProjectNotInitialisedError: No ``.magi/magi.toml``. Names the command.
            MagiRsProcessError: The child died, timed out, or exited non-zero.
            MagiRsContractError: The output is not JSON, ``consult`` is null, or
                ``agents`` is absent.

        Side effects:
            May download and install the binary; launches one process; writes
            ``magi-rs-raw.json`` and the per-mage verdict files.
        """
        del agents_dir, models, max_attempts, timeout  # accepted, unused: see the docstring
        # Repeated on purpose: ``main()`` already ran it before the payload, but
        # R4 says EVERY run verifies right before executing. Sub-100 ms for a
        # 12 MB hash plus one ``--version`` spawn -- accepted redundancy.
        executable = self.preflight()
        argv = build_argv(executable, self._workdir, self._mode, self._timeout)
        stdout = await self._launch(argv, prompt, self._timeout, on_state, output_dir)
        self.last_result = ConsultResult.from_json(stdout)
        return self._validate(self.last_result, output_dir)

    def preflight(self) -> str:
        """Verify the binary and the project, spending nothing.

        Idempotent and cheap, so ``main()`` calls it BEFORE building the
        payload (the spec's own order: a bundle built for a run that cannot
        start is pure waste) and :meth:`run_trio` calls it again right before
        launching (R4: every run, not only on install).

        Returns:
            The verified executable's path.

        Raises:
            UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
            DownloadError, UnsupportedPlatformError: The binary could not be
                obtained or did not verify. It is NOT executed.
            ProjectNotInitialisedError: No ``.magi/magi.toml`` under the
                workdir. Names ``run_magi.py --ollama-init``; never initialises.
            KeyMissingError, KeyUnverifiableError, KeyExposedError: The key is
                absent, its permissions could not be verified, or they are
                loosened (not owner-only). Fail-closed.

        Side effects:
            May download and install the binary into ``bin_dir``; stores the
            loaded key for the launch.
        """
        executable = verified_executable(BinaryBootstrap(self._pin, self._bin_dir))
        self._key = ProjectPreflight.require_ready(self._workdir, self._key_store)
        return executable

    async def _launch(
        self,
        argv: list[str],
        prompt: str,
        timeout: int,
        on_state: Callable[[str, str], None] | None,
        output_dir: str,
    ) -> str:
        """Run the child, feeding the prompt on stdin and draining both pipes.

        Args:
            argv: The command line from :func:`build_argv`.
            prompt: The payload, written to stdin.
            timeout: Wall-clock ceiling.
            on_state: Optional progress callback.
            output_dir: Where the raw envelope is persisted.

        Returns:
            The child's stdout, decoded as UTF-8.

        Raises:
            MagiRsProcessError: Non-zero exit, the timeout elapsed, or the
                binary could not be started. The message relays the exit code
                and stderr VERBATIM, names the environment allowlist as a
                suspect and lists the parent variables it withheld (names
                only), and nothing is recovered from partial output.
            MagiRsContractError: The payload cannot be encoded as UTF-8 (a lone
                surrogate). Refused before the child starts; deterministic.

        Side effects:
            Launches a process with an allowlisted environment; writes
            ``magi-rs-raw.json`` (fail-soft: a failed write warns).

        Note:
            The outer clock is exactly ``timeout``, with no margin. Measured on
            2026-09-29 against magi-rs 0.21.0 (``--timeout 300``, launched
            directly): magi-rs derives its per-request and per-operation clocks
            from ``--timeout``, gave up on a timed-out seat, and emitted its
            partial JSON 250 s before the deadline. Nothing is lost by stopping
            at ``timeout``. magi-rs's deadline behaviour is internal to it, so
            this is re-measured on every pin bump.
        """
        env = EnvironmentPolicy.build(os.environ, passphrase=self._key)
        _emit(on_state, _TRIO_NODE, "running")
        # Encode BEFORE anything starts: a lone surrogate (argv or a file that
        # arrived via surrogateescape) is a deterministic payload defect, not a
        # transport failure -- refused as a contract error, never a traceback
        # from inside communicate(), and never an orphaned child.
        try:
            payload = prompt.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise MagiRsContractError(
                f"the payload cannot be encoded as UTF-8 for magi-rs's stdin: {exc}"
            ) from exc
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            _emit(on_state, _TRIO_NODE, "failed")
            raise MagiRsProcessError(f"could not start {argv[0]}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            process.kill()
            await process.wait()
            _emit(on_state, _TRIO_NODE, "timeout")
            # A withheld proxy variable makes the child HANG on connect rather
            # than exit non-zero, so the allowlist diagnosis belongs here too.
            raise MagiRsProcessError(
                f"magi-rs did not finish within --timeout {timeout}s; nothing is recovered "
                f"from its partial output. {self._allowlist_hint()}"
            ) from exc
        # The ONE redaction point: nothing the child said reaches a message, the
        # stderr tail, the raw artifact or the parser with the key still in it.
        stdout, stderr = self._redact(stdout), self._redact(stderr)
        self._persist_raw(output_dir, stdout)
        if process.returncode != 0:
            _emit(on_state, _TRIO_NODE, "failed")
            raise MagiRsProcessError(
                f"magi-rs exited {process.returncode}; stderr: "
                f"{_stderr_tail(stderr)!r}. "
                f"{_VAULT_HINT} {self._allowlist_hint()}"
            )
        _emit(on_state, _TRIO_NODE, "success")
        return stdout.decode("utf-8", errors="replace")

    def _redact(self, captured: bytes) -> bytes:
        """Remove the key from bytes captured from the child.

        Args:
            captured: The child's stdout or stderr.

        Returns:
            ``captured`` with every exact occurrence of the key replaced by
            ``REDACTION_MARKER`` (:meth:`MagiKey.redact_bytes`); unchanged when
            no key is loaded.

        Complexity:
            O(n) in the captured size, once per stream per launch.
        """
        if self._key is None:
            return captured
        return self._key.redact_bytes(captured)

    @staticmethod
    def _allowlist_hint() -> str:
        """The diagnosis every process failure carries: what the allowlist withheld.

        Returns:
            One sentence naming the allowlist as a suspect and listing the
            parent variables (names only, never values) that did not travel;
            ``MAGI_PASSPHRASE`` is excluded because the plugin injects it.
        """
        # The plugin injects MAGI_PASSPHRASE itself, so it is never "withheld" from the child.
        withheld = [
            name for name in EnvironmentPolicy.withheld(os.environ) if name != PASSPHRASE_VARIABLE
        ]
        return (
            "The child ran with an allowlisted environment (see magi_rs_env.py); if the "
            "failure looks like a missing certificate, proxy or host, the allowlist is "
            f"a suspect -- it withheld: {withheld}"
        )

    @staticmethod
    def _persist_raw(output_dir: str, stdout: bytes) -> None:
        """Write the raw envelope under its own name (R16); fail-soft.

        Args:
            output_dir: Destination directory.
            stdout: The child's stdout, verbatim.

        Returns:
            None.

        Side effects:
            Writes ``magi-rs-raw.json``; a failed write only warns, because an
            audit artifact must not be able to destroy the verdicts it records.
        """
        try:
            write_raw_artifact(output_dir, stdout)
        except OSError as exc:
            print(f"WARNING: failed to persist {_RAW_ARTIFACT}: {exc}", file=sys.stderr)

    @staticmethod
    def _validate(
        result: ConsultResult, output_dir: str
    ) -> dict[str, dict[str, Any] | BaseException]:
        """Push each verdict through the EXISTING validator, unchanged.

        Each verdict is written to ``{mage}.json`` -- the same per-mage artifact
        the Claude path leaves -- and read back by ``load_agent_output``, whose
        contract is a file path.

        Args:
            result: The parsed consult result.
            output_dir: Where the per-mage files are written.

        Returns:
            One entry per mage of the trio: the validated dict, the
            ``ValidationError`` it failed with, or -- for a mage magi-rs did not
            deliver at all -- a ``RuntimeError`` carrying magi-rs's own cause
            from ``failed_agents`` (the shape the orchestrator's degraded path
            already handles, BDD-15/D7). A verdict that breaks the seven-key
            contract is rejected EXACTLY as today (BDD-14c) -- no new validation
            code, and no relaxing of the contract.

        Side effects:
            Writes one ``{mage}.json`` per verdict.
        """
        failed = result.telemetry.get("failed_agents") or {}
        verdicts: dict[str, dict[str, Any] | BaseException] = {
            mage: RuntimeError(
                f"magi-rs delivered no verdict for {mage}: "
                f"{failed.get(mage, _NO_VERDICT_CAUSE) if isinstance(failed, dict) else failed}"
            )
            for mage in _MAGES
        }
        for index, verdict in enumerate(result.agents):
            name = verdict.get("agent") if isinstance(verdict, dict) else None
            # The name is data from the child: only a seat of the trio may name
            # the artifact, so a crafted name cannot write outside output_dir.
            mage = name if name in _MAGES else f"agent-{index}"
            path = os.path.join(output_dir, f"{mage}.json")
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(verdict, handle, ensure_ascii=False, indent=2)
                verdicts[mage] = load_agent_output(path)
            except (ValidationError, json.JSONDecodeError, OSError, UnicodeEncodeError) as exc:
                # UnicodeEncodeError: json.loads yields a lone surrogate from a
                # JSON escape that the UTF-8 file write cannot take -- that
                # seat's failure, never the run's.
                verdicts[mage] = exc
        return verdicts
