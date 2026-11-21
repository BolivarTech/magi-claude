# Author: Julian Bolivar
# Version: 8.0.0
# Date: 2026-11-21
"""Run ONE ``magi-rs code-review`` child and judge what it prints (REQ-14).

The third implementer of :class:`TrioExecutor`. magi-rs batches the change,
enriches each batch, runs the trio and prints one envelope; the plugin only
builds the argv, keeps the key in the child's environment, waits for the child
to finish recording the review history and validates the three verdicts.

One magi-rs instance per ``.magi/`` at a time: two processes on the same
database may collide on its lock (PM-S-4); this is documented, not enforced.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from magi_key import KeyStore, MagiKey
from magi_key_permissions import default_key_store
from magi_rs_bootstrap import BinaryBootstrap, verified_executable
from magi_rs_embedder import EmbedderProbe
from magi_rs_env import EnvironmentPolicy
from magi_rs_pin import Pin
from magi_rs_preflight import ProjectPreflight
from magi_rs_result import CodeReviewResult
from magi_rs_targets import ReviewTarget
from magi_rs_trio import (
    _VAULT_HINT,
    MagiRsContractError,
    MagiRsProcessError,
    MagiRsTrio,
    _emit,
    _stderr_tail,
    redact_captured,
)
from subprocess_utils import reap_and_drain_stderr
from trio_executor import TrioExecutor

__all__ = [
    "HISTORY_WARNING_MARKER",
    "HISTORY_WINDOW_SECONDS",
    "TIMEOUT_REJECTION_MARKER",
    "MagiRsCodeReview",
    "build_review_argv",
]

HISTORY_WINDOW_SECONDS = 45
"""Seconds the child gets past ``--timeout`` to record the review history.
magi-rs waits up to 30 s for it after printing the report; killing it earlier
could cut a SQLite write (PM-S-2). The rest is margin."""

TIMEOUT_REJECTION_MARKER = "pass --timeout"
"""Part of magi-rs's budget rejection (measured M5, 0.25.0): ``... pass
--timeout <114*N> or more ...``, exit 2, before any model is called. A later
pin may reword it (PM-MS2-3): a test pins this constant, a smoke the wording."""

HISTORY_WARNING_MARKER = "warning: the review history was not updated"
"""Prefix of the stderr warning magi-rs prints when it could not record the
review history (src/tools/code_review.rs:1500, followed by ``: <cause>``)."""

_SUBCOMMAND = "code-review"
_TRIO_NODE = "magi-rs code-review · 3 mages"
_STREAM_CHUNK_BYTES = 65536
_TIMEOUT_REJECTED_MESSAGE = (
    "magi-rs rejected --timeout {timeout}s: it is too short for the number of batches in "
    "this review; pass a larger --timeout (not retried). stderr: {stderr!r}"
)


def build_review_argv(
    executable: str, workdir: str, target: ReviewTarget, timeout: int
) -> list[str]:
    """Build the ``magi-rs code-review`` command line.

    Args:
        executable: Path of the verified binary.
        workdir: Project root, passed as ``-w``.
        target: The validated review target; its argv fragment is appended as is.
        timeout: Wall-clock ceiling in seconds, passed to magi-rs as the review total.

    Returns:
        The argv. Never ``--no-memory`` and never the key (it travels in the
        environment, REQ-10).
    """
    return [
        executable,
        _SUBCOMMAND,
        *target.argv,
        "--output-format",
        "json",
        "--structured-verdicts",
        "-w",
        workdir,
        "--timeout",
        str(timeout),
    ]


class MagiRsCodeReview(TrioExecutor):
    """Run one ``magi-rs code-review`` and return the three validated verdicts.

    Example:
        >>> executor = MagiRsCodeReview(workdir, target, 16820, bin_dir, pin)
        >>> executable = executor.preflight()
        >>> verdicts = await executor.run_trio("", "", out_dir, 16820, {}, 1)

    Side effects:
        Launches one process; writes ``magi-rs-raw.json`` and one ``{mage}.json``
        per verdict into ``output_dir``.
    """

    def __init__(
        self,
        workdir: str,
        target: ReviewTarget,
        timeout: int,
        bin_dir: str,
        pin: Pin,
        key_store: KeyStore | None = None,
        probe: EmbedderProbe | None = None,
    ) -> None:
        """Bind the executor to one project, one target and one invocation.

        Args:
            workdir: Project root (``-w`` and the preflight base).
            target: The validated review target.
            timeout: Wall-clock ceiling, already resolved by the caller.
            bin_dir: The plugin's ``bin/``.
            pin: The parsed pin, for the bootstrap.
            key_store: Where the key is loaded from; None selects
                :func:`default_key_store`.
            probe: The embedder probe; None selects the production
                :class:`EmbedderProbe`.
        """
        self._workdir = workdir
        self._target = target
        self._timeout = timeout
        self._bin_dir = bin_dir
        self._pin = pin
        self._key_store = key_store if key_store is not None else default_key_store()
        self._probe = probe if probe is not None else EmbedderProbe()
        #: The key the last preflight loaded; injected into the child, never logged.
        self._key: MagiKey | None = None
        self._executable: str | None = None
        #: The parsed envelope of the last run; None until one passes the contract.
        self.last_result: CodeReviewResult | None = None
        #: True only when the child exited inside the window with no history warning.
        self.history_recorded: bool | None = None

    def preflight(self) -> str:
        """Verify the binary, the project, the key and the embedder, spending nothing.

        Returns:
            The verified executable's path.

        Raises:
            The fail-closed errors of the bootstrap and of
            :meth:`ProjectPreflight.require_ready` (all in ``FAIL_CLOSED_ERRORS``).

        Side effects:
            May install the binary; stores the key and the executable for the launch.
        """
        self._executable = verified_executable(BinaryBootstrap(self._pin, self._bin_dir))
        self._key = ProjectPreflight.require_ready(self._workdir, self._key_store, self._probe)
        return self._executable

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
        """Run the review and return one validated verdict per mage.

        ``prompt``, ``agents_dir``, ``timeout``, ``models`` and ``max_attempts``
        are accepted for the shared ABC and unused: the target is the input, the
        deadline is the constructor's (one source of truth) and magi-rs owns the
        trio and its retries. The caller runs :meth:`preflight` first.

        Args:
            prompt: Unused; the child's stdin is closed.
            agents_dir: Unused.
            output_dir: Where ``magi-rs-raw.json`` and the ``{mage}.json`` files go.
            timeout: Unused (see above).
            models: Unused.
            max_attempts: Unused.
            on_state: Optional progress callback, ``(node, state)``.

        Returns:
            The validated verdicts, as :meth:`MagiRsTrio._validate` returns them.

        Raises:
            MagiRsContractError: :meth:`preflight` has not run, or the envelope
                breaks the contract (``MemoryContractError`` included).
            UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
            KeyMissingError, KeyUnverifiableError, KeyExposedError: The binary or the key
                no longer verifies (re-checked right before the launch, R4).
            MagiRsProcessError: The child could not start, exited non-zero
                (a ``--timeout`` rejection names the flag and is not retried), or
                was killed before printing a complete envelope.
        """
        del prompt, agents_dir, models, max_attempts, timeout  # accepted, unused: see above
        if self._executable is None:
            raise MagiRsContractError("preflight() must run before run_trio()")
        # R4, as MagiRsTrio does: every run verifies right before executing. The embedder
        # probe is NOT repeated -- it ran in preflight() and costs a network request.
        self._executable = verified_executable(BinaryBootstrap(self._pin, self._bin_dir))
        self._key = ProjectPreflight.require_ready(self._workdir, self._key_store)
        argv = build_review_argv(self._executable, self._workdir, self._target, self._timeout)
        stdout, stderr, overran = await self._launch(argv, on_state)
        MagiRsTrio._persist_raw(output_dir, stdout)
        text = stdout.decode("utf-8", errors="replace")
        if overran:
            self._require_complete_envelope(text)
        self.last_result = CodeReviewResult.from_json(text)
        warned = HISTORY_WARNING_MARKER in stderr.decode("utf-8", errors="replace")
        self.history_recorded = not overran and not warned
        return MagiRsTrio._validate(self.last_result.consult, output_dir)

    async def _launch(
        self, argv: list[str], on_state: Callable[[str, str], None] | None
    ) -> tuple[bytes, bytes, bool]:
        """Run the child to completion or to its deadline.

        The deadline is ``timeout + HISTORY_WINDOW_SECONDS`` and bounds both the
        reading and the wait, so a child that never closes stdout is killed too.
        Output read up to that point is kept.

        Args:
            argv: The command line.
            on_state: Optional progress callback.

        Returns:
            ``(stdout, stderr, overran)``, both streams redacted of the key;
            ``overran`` is True when the child had to be killed.

        Raises:
            MagiRsProcessError: The child could not start, or exited non-zero.
        """
        env = EnvironmentPolicy.build(os.environ, passphrase=self._key)
        _emit(on_state, _TRIO_NODE, "running")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            _emit(on_state, _TRIO_NODE, "failed")
            raise MagiRsProcessError(f"could not start {argv[0]}: {exc}") from exc
        out, err = bytearray(), bytearray()

        async def _pump(stream: asyncio.StreamReader, sink: bytearray) -> None:
            while chunk := await stream.read(_STREAM_CHUNK_BYTES):
                sink.extend(chunk)

        async def _finish() -> None:
            assert process.stdout is not None and process.stderr is not None
            await asyncio.gather(
                _pump(process.stdout, out), _pump(process.stderr, err), process.wait()
            )

        overran = False
        try:
            await asyncio.wait_for(_finish(), timeout=self._timeout + HISTORY_WINDOW_SECONDS)
        except (asyncio.TimeoutError, TimeoutError):
            overran = True
            # Only past the history window (PM-S-2). The helper kills the whole
            # tree on Windows (git children of magi-rs can hold the pipes), reaps
            # with a bounded wait that warns instead of raising, and hands back
            # what stderr still held.
            err.extend(await reap_and_drain_stderr(process))
            _emit(on_state, _TRIO_NODE, "timeout")
        stdout, stderr = (
            redact_captured(self._key, bytes(out)),
            redact_captured(self._key, bytes(err)),
        )
        if not overran and process.returncode != 0:
            _emit(on_state, _TRIO_NODE, "failed")
            raise MagiRsProcessError(self._failure_message(process.returncode, stderr))
        if not overran:
            _emit(on_state, _TRIO_NODE, "success")
        return stdout, stderr, overran

    def _failure_message(self, returncode: int | None, stderr: bytes) -> str:
        """Render a non-zero exit as one line; a ``--timeout`` rejection names the flag.

        Args:
            returncode: The child's exit code.
            stderr: Its redacted stderr.

        Returns:
            The message, free of the key (stderr was redacted by the caller).
        """
        tail = _stderr_tail(stderr)
        if TIMEOUT_REJECTION_MARKER in tail:
            return _TIMEOUT_REJECTED_MESSAGE.format(timeout=self._timeout, stderr=tail)
        return (
            f"magi-rs exited {returncode}; stderr: {tail!r}. "
            f"{_VAULT_HINT} {MagiRsTrio._allowlist_hint()}"
        )

    def _require_complete_envelope(self, text: str) -> None:
        """Refuse the output of a killed child unless it is one whole JSON document.

        Args:
            text: What the child had printed when it was killed.

        Raises:
            MagiRsProcessError: The output is cut off: nothing is recovered from it.
        """
        try:
            json.loads(text)
        except ValueError as exc:
            raise MagiRsProcessError(
                f"magi-rs did not finish within --timeout {self._timeout}s plus the "
                f"{HISTORY_WINDOW_SECONDS}s history window and its output is incomplete; "
                f"nothing is recovered from it. {MagiRsTrio._allowlist_hint()}"
            ) from exc
