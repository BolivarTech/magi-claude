# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-09-19
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

from magi_rs_bootstrap import (
    BinaryBootstrap,
    DigestMismatchError,
    DownloadError,
    UnsafeArchiveError,
    UnsafeBinDirectoryError,
    VersionMismatchError,
)
from magi_rs_env import EnvironmentPolicy
from magi_rs_paths import PluginRootError
from magi_rs_pin import Pin, PinFormatError, UnsupportedPlatformError
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

ALLOWS_RUNNING_WITHOUT_PASSPHRASE = "--no-memory"
"""Named for what it DOES, not what its name suggests (D10). This flag turns
"missing passphrase => fatal" into "missing passphrase => proceed with the
environment" -- it is what lets the plugin run with no passphrase, no key and
no config of its own (R13). It does NOT isolate memory: ``consult`` never
touches the memory layer by construction, so every run is stateless anyway."""

_TIMEOUT_FLAG = "--timeout"
"""Wall-clock ceiling (R12). ``magi-rs`` also DERIVES the per-mage attempt
budget from it: MS0 measured 149 s per attempt at 1800."""

_TRIO_NODE = "magi-rs consult · 3 mages"
"""The single status node. D5/BDD-27: the plugin launches ONE child and cannot
observe individual mages, so three nodes would invent progress."""

_RAW_ARTIFACT = "magi-rs-raw.json"
_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 1800
"""Measured-healthy: MS0 saw 149 s per mage attempt at this value. The plugin's
historical 900 would roughly halve it, back to the 72 s ceiling that made E-B
abort with one seat of three."""
_MIN_HEALTHY_TIMEOUT_SECONDS = 1200
_STDERR_EXCERPT_CHARS = 2000


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
)


class MagiRsProcessError(ValidationError):
    """The ``magi-rs`` process died, timed out, or was killed.

    This one DOES inherit from ``ValidationError``, and for the reason the rule
    gives: a child that timed out or was OOM-killed may well succeed on a
    second attempt, so the orchestrator's existing retry is exactly right.
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
            per-mage budget from it (MS0 measured 149 s per attempt at 1800).

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
        ALLOWS_RUNNING_WITHOUT_PASSPHRASE,
        _TIMEOUT_FLAG,
        str(timeout),
    ]


def resolve_timeout(flag_value: int | None) -> tuple[int, str | None]:
    """Resolve the wall timeout and warn when it starves the per-mage budget.

    ``magi-rs`` DERIVES each mage's attempt budget from this value; MS0 measured
    149 s per attempt at ``--timeout 1800``. The plugin's historical default of
    900 would halve that to roughly the 72 s ceiling that made E-B abort with
    one seat of three -- so the open-weight path defaults to 1800, and a lower
    explicit value is honoured with a warning rather than silently accepted.

    Args:
        flag_value: The operator's ``--timeout``, or None when omitted.

    Returns:
        A pair of (seconds, warning-or-None).

    Raises:
        MagiRsContractError: The value is not a positive integer. Deterministic.
    """
    if flag_value is None:
        return _DEFAULT_OLLAMA_TIMEOUT_SECONDS, None
    if isinstance(flag_value, bool) or not isinstance(flag_value, int) or flag_value <= 0:
        raise MagiRsContractError(f"--timeout must be a positive integer, got {flag_value!r}")
    if flag_value < _MIN_HEALTHY_TIMEOUT_SECONDS:
        return flag_value, (
            f"--timeout {flag_value} is below the measured-healthy "
            f"{_MIN_HEALTHY_TIMEOUT_SECONDS}s: magi-rs derives each mage's attempt budget "
            "from it, so a low wall timeout also cuts the reasoning budget of every seat"
        )
    return flag_value, None


def _emit(on_state: Callable[[str, str], None] | None, node: str, state: str) -> None:
    """Forward one status transition, tolerating a callback that raises.

    Args:
        on_state: The caller's callback, or None when status is off.
        node: The node's name. ONE node in this executor -- see run_trio.
        state: ``running``, ``success`` or ``failed``.

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


class MagiRsTrio(TrioExecutor):
    """Run the three mages by delegating to one ``magi-rs consult``.

    This is the SECOND implementer of :class:`TrioExecutor`, and the one that
    justified the abstraction existing. Where :class:`ClaudeTrio` launches three
    processes concurrently, this one launches ONE that runs the trio itself --
    which is precisely why the boundary sits at the trio and not at the mage.

    Example:
        >>> trio = MagiRsTrio(workdir="/repo", mode="code-review", timeout=1800,
        ...                   bin_dir="/plugin/bin", pin=Pin.load(pin_path))
        >>> verdicts = await trio.run_trio(prompt=payload, agents_dir="agents",
        ...                                output_dir=out, timeout=1800,
        ...                                models={}, max_attempts=1)

    Side effects:
        Launches one process; writes ``magi-rs-raw.json`` and one
        ``{mage}.json`` per verdict into ``output_dir``.
    """

    def __init__(self, workdir: str, mode: str, timeout: int, bin_dir: str, pin: Pin) -> None:
        """Bind the executor to one project and one invocation.

        Args:
            workdir: Project root. Passed as ``-w`` and used for the preflight.
                Resolved by the CALLER (Task 5) -- never inherited from the cwd,
                because the plugin can be invoked from any directory.
            mode: ``code-review``, ``design`` or ``analysis``. Always explicit.
            timeout: Wall-clock ceiling, already resolved by ``resolve_timeout``.
            bin_dir: The plugin's ``bin/``.
            pin: The parsed pin, for the bootstrap.
        """
        self._workdir = workdir
        self._mode = mode
        self._timeout = timeout
        self._bin_dir = bin_dir
        self._pin = pin

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
        return self._validate(ConsultResult.from_json(stdout), output_dir)

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
                workdir. Names ``/magi --ollama-init``; never initialises.

        Side effects:
            May download and install the binary into ``bin_dir``.
        """
        bootstrap = BinaryBootstrap(self._pin, self._bin_dir)
        bootstrap.ensure()
        executable = bootstrap.verify()
        ProjectPreflight.require_initialised(self._workdir)
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

        Side effects:
            Launches a process with an allowlisted environment; writes
            ``magi-rs-raw.json`` (fail-soft: a failed write warns).
        """
        env = EnvironmentPolicy.build(os.environ)
        _emit(on_state, _TRIO_NODE, "running")
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
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=timeout
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            process.kill()
            await process.wait()
            _emit(on_state, _TRIO_NODE, "timeout")
            raise MagiRsProcessError(
                f"magi-rs did not finish within --timeout {timeout}s; nothing is recovered "
                "from its partial output"
            ) from exc
        self._persist_raw(output_dir, stdout)
        if process.returncode != 0:
            _emit(on_state, _TRIO_NODE, "failed")
            withheld = EnvironmentPolicy.withheld(os.environ)
            raise MagiRsProcessError(
                f"magi-rs exited {process.returncode}; stderr: "
                f"{stderr.decode('utf-8', errors='replace')[:_STDERR_EXCERPT_CHARS]!r}. "
                "The child ran with an allowlisted environment (see magi_rs_env.py); if the "
                "failure looks like a missing certificate, proxy or host, the allowlist is "
                f"a suspect -- it withheld: {withheld}"
            )
        _emit(on_state, _TRIO_NODE, "success")
        return stdout.decode("utf-8", errors="replace")

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
            with open(os.path.join(output_dir, _RAW_ARTIFACT), "wb") as handle:
                handle.write(stdout)
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
            One entry per mage present in ``agents``: the validated dict, or the
            ``ValidationError`` it failed with. A verdict that breaks the
            seven-key contract is rejected EXACTLY as today (BDD-14c) -- no new
            validation code, and no relaxing of the contract.

        Side effects:
            Writes one ``{mage}.json`` per verdict.
        """
        verdicts: dict[str, dict[str, Any] | BaseException] = {}
        for index, verdict in enumerate(result.agents):
            name = verdict.get("agent") if isinstance(verdict, dict) else None
            mage = str(name) if name else f"agent-{index}"
            path = os.path.join(output_dir, f"{mage}.json")
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(verdict, handle, ensure_ascii=False, indent=2)
                verdicts[mage] = load_agent_output(path)
            except (ValidationError, json.JSONDecodeError, OSError) as exc:
                verdicts[mage] = exc
        return verdicts
