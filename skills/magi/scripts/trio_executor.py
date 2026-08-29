#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.1.0
# Date: 2026-08-29
"""Trio-level execution contract: run the three mages, return validated verdicts.

The abstraction sits at the TRIO level, not at the individual mage, because the
two transports have different shapes: `claude -p` is three concurrent
subprocesses, while `magi-rs consult` is one process that answers for all three.
A per-mage contract can only express the first honestly.

The contract covers VALIDATED verdicts rather than raw bytes, and the reason is
the retry. The Claude transport re-prompts a single mage with the sentinel's
cause-specific feedback when its verdict fails the schema, so it must know what
a valid verdict is; the magi-rs transport has already spent its own retries
before it answers. Either way the caller receives three verdicts that passed
`load_agent_output`, and never has to know which transport retried what.

Everything downstream -- consensus, reporting, artifacts -- is transport-agnostic
and lives in run_magi.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from models import resolve_model
from parse_agent_output import parse_agent_output
from subprocess_utils import (
    format_stderr_excerpt as _format_stderr_excerpt,
    reap_and_drain_stderr as _reap_and_drain_stderr,
    write_stderr_log as _write_stderr_log,
)
from synthesize import ValidationError, load_agent_output
from verdict_markers import (
    ECHO_CANARY,
    AgentIdentityError,
    EchoedExampleRejected,
)

#: Rejection message for an out-of-range ``max_attempts``. Named once so the
#: up-front check in :meth:`ClaudeTrio.run_trio` and the guard on
#: :meth:`ClaudeTrio._run_with_retry` cannot drift apart.
_MAX_ATTEMPTS_ERROR = "max_attempts must be >= 1"


def _write_text(output_dir: str, filename: str, content: str) -> None:
    """Write one per-mage artifact, warning instead of failing on I/O error.

    An artifact is diagnostic, not the verdict: a full disk must not cost a
    mage. Mirrors how ``launch_agent`` and ``write_stderr_log`` already treat
    their writes.

    Args:
        output_dir: Directory the artifact goes in.
        filename: Artifact file name, e.g. ``melchior.raw.json``.
        content: Text to write, UTF-8.
    """
    # The write itself raises, and the caller turns the failure into a warning
    # -- exactly the split ``write_stderr_log`` already uses and this docstring
    # names. Swallowing it here as well would make the caller's guard, which is
    # what the fail-soft test actually exercises, unreachable.
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _reject_echoed_example(payload: dict[str, Any]) -> None:
    """Refuse a verdict that is a verbatim copy of the prompt's worked example.

    R6 -- anti-echo canary. The LAST belt: the sentinel already prevents anything
    from outside the markers being extracted, and the prompt puts nothing valid
    BETWEEN them. One theoretical path is left: the model taking the worked
    example from OUTSIDE and wrapping it in markers ITSELF. Its fingerprint gives
    it away.

    Args:
        payload: The verdict that already passed ``load_agent_output``.

    Returns:
        None.

    Raises:
        EchoedExampleRejected: The payload matches ``ECHO_CANARY`` field for
            field. A ``ValidationError`` subclass, so the retry loop catches it
            and the model gets corrective feedback.
    """
    if all(payload.get(key) == value for key, value in ECHO_CANARY.items()):
        raise EchoedExampleRejected(
            "the verdict is a verbatim copy of the prompt's example, not your analysis"
        )


def _require_claimed_identity(payload: dict[str, Any], agent_name: str) -> None:
    """Refuse a verdict that claims to come from a different mage.

    R10 -- identity. ``load_agent_output`` validates that ``agent`` is in the
    ENUM, but nobody validated that it was the mage that was LAUNCHED. A
    duplicate name kills the whole run; a unique but wrong one puts one mage's
    text in another's seat, and the consensus counts it as an independent
    perspective **that never existed**.

    Case-insensitive: a model that writes "Caspar" DID emit its verdict
    correctly. Killing it over a capital letter is a retry given away for free,
    and the enum is validated separately anyway.

    Args:
        payload: The verdict that already passed ``load_agent_output``.
        agent_name: The mage that was actually launched.

    Returns:
        None.

    Raises:
        AgentIdentityError: The claimed agent is not the launched one. A
            ``ValidationError`` subclass, so the retry loop catches it.
    """
    claimed = str(payload["agent"]).strip()
    if claimed.casefold() != agent_name.casefold():
        raise AgentIdentityError(
            f"verdict claims agent {claimed!r} but {agent_name!r} was launched"
        )


class TrioExecutor(ABC):
    """Runs the three mages for one payload and returns their validated verdicts."""

    @abstractmethod
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
        """Run all three mages and return their validated verdicts.

        Args:
            prompt: The user prompt payload, identical for the three mages.
            agents_dir: Directory holding the ``{mage}.md`` system prompts.
            output_dir: Directory for raw and per-mage debug artifacts.
            timeout: Per-mage timeout in seconds.
            models: Mapping of agent name to model short name (e.g. ``"opus"``),
                already extracted from any richer spec by the caller.
            max_attempts: Attempts per mage before giving up. A transport that
                retries internally ignores this.
            on_state: Optional callback ``(agent_name, state)`` for a live status
                display, where state is one of ``running``, ``retrying``,
                ``success``, ``timeout``, ``failed``. It must not raise; a
                display glitch must never cost a verdict.

        Returns:
            Mapping of agent name to either the validated verdict dict (the
            seven-key contract) or the exception that killed that mage. There is
            always one entry per mage in *models*.

            Failures are returned rather than raised, and the exact exception
            object is preserved rather than wrapped: the caller branches on its
            subtype -- re-raising ``_EndpointDown``, re-raising ``CancelledError``
            on the rotation path, and letting ``KeyboardInterrupt`` and
            ``SystemExit`` propagate while treating every other exception as a
            degraded agent. Collapsing them into one type would swallow a fatal
            signal, and the caller's warning prints the exception text, so a
            wrapper would also cost the diagnosis.

        Raises:
            Nothing per-mage. A failure in the transport's own setup -- before any
            mage runs -- still propagates.
        """
        ...


class ClaudeTrio(TrioExecutor):
    """Runs the three mages as three concurrent `claude -p` subprocesses.

    The transport is the v5 code moved verbatim from ``ClaudeBackend``; the
    per-mage retry loop is the v5 code moved verbatim from
    ``run_orchestrator._legacy_tracked_launch``. Behaviour must not change --
    R20 demands zero regression on this path.

    Complexity: three concurrent subprocesses, so wall-clock is the slowest
    mage rather than the sum of the three. Each mage may run up to
    ``max_attempts`` times, sequentially within its own slot.

    Side effects: launches subprocesses, and writes FOUR artifacts per mage
    under *output_dir*, split across two methods on purpose:
    ``_run_with_retry`` writes ``{agent}.prompt.txt`` (before each attempt, so it
    holds the prompt actually sent last), ``{agent}.raw.json`` and
    ``{agent}.json`` (after the attempt that succeeds); ``_run_one`` writes
    ``{agent}.stderr.log``, because only it holds the stderr bytes. The split
    follows the data, not a layering rule. The first three are
    moved from ``launch_agent`` (``run_magi.py:552-558``), which wrote them
    around the transport call: dropping them would be a silent artifact
    regression no current test catches, and would contradict this docstring.

    Example:
        >>> verdicts = await ClaudeTrio().run_trio(
        ...     prompt, agents_dir, out_dir, 900,
        ...     {"melchior": "opus", "balthasar": "opus", "caspar": "opus"}, 2,
        ... )
        >>> sorted(verdicts)
        ['balthasar', 'caspar', 'melchior']
    """

    def __init__(
        self,
        build_retry_prompt: (
            Callable[[str, ValidationError | json.JSONDecodeError], str] | None
        ) = None,
    ) -> None:
        """Store the retry-prompt builder.

        Args:
            build_retry_prompt: Builds the corrective prompt for a retry from
                the original prompt and the failure. Injected rather than
                imported because it lives in ``run_magi``, which imports this
                module -- importing it back would create a cycle. Defaults to
                the real builder, resolved lazily on first use.
        """
        self._build_retry_prompt = build_retry_prompt

    def _retry_prompt(self, prompt: str, error: ValidationError | json.JSONDecodeError) -> str:
        """Return the ORIGINAL prompt with cause-specific feedback appended.

        Args:
            prompt: The original user prompt -- never the previous attempt's.
                Feedback does not accumulate across retries; this mirrors the v5
                behaviour at ``run_magi.py:1770``.
            error: The schema failure that triggered the retry; it selects which
                corrective template the sentinel appends.

        Returns:
            The prompt for the next attempt.
        """
        builder = self._build_retry_prompt
        if builder is None:
            # Deferred to call time, never module load: ``run_magi`` imports this
            # module, so a top-level import would close the cycle too early. By
            # the first call the cycle is already resolved.
            from run_magi import _build_retry_prompt

            builder = self._build_retry_prompt = _build_retry_prompt
        return builder(prompt, error)

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
        """Run the three mages concurrently. See :meth:`TrioExecutor.run_trio`.

        Raises:
            ValueError: If *max_attempts* is below 1. Validated ONCE, up front --
                a caller error is not a per-mage failure, and reporting it three
                times as three dead mages would hide the cause.
        """
        if max_attempts < 1:
            raise ValueError(_MAX_ATTEMPTS_ERROR)

        names = list(models)
        outcomes = await asyncio.gather(
            *(
                self._run_with_retry(
                    name,
                    agents_dir,
                    prompt,
                    output_dir,
                    timeout,
                    models[name],
                    max_attempts,
                    on_state,
                )
                for name in names
            ),
            return_exceptions=True,
        )

        # One entry per mage, always: a failure is RETURNED, with its exact type
        # intact, because the caller branches on that subtype.
        verdicts: dict[str, dict[str, Any] | BaseException] = {}
        for name, outcome in zip(names, outcomes, strict=True):
            verdicts[name] = outcome
        return verdicts

    def _emit(
        self,
        on_state: Callable[[str, str], None] | None,
        agent: str,
        state: str,
    ) -> None:
        """Report a state transition, swallowing display failures.

        A live-display glitch must never cost a verdict, so the callback's
        exceptions are caught here and reported on stderr instead of
        propagating. This mirrors ``run_magi._safe_display_update``.
        """
        if on_state is None:
            return
        try:
            on_state(agent, state)
        except Exception as exc:
            # ``Exception``, deliberately NOT ``BaseException``: a
            # ``KeyboardInterrupt`` or ``SystemExit`` raised inside a display
            # callback is a shutdown signal and has to tear the run down, not be
            # swallowed as a cosmetic failure.
            print(
                f"WARNING: status callback failed for {agent} ({state}): {exc}",
                file=sys.stderr,
            )

    async def _run_with_retry(
        self,
        agent_name: str,
        agents_dir: str,
        prompt: str,
        output_dir: str,
        timeout: int,
        model: str,
        max_attempts: int,
        on_state: Callable[[str, str], None] | None,
    ) -> dict[str, Any]:
        """Run one mage, retrying on schema failure, and return its verdict.

        Args:
            agent_name: One of 'melchior', 'balthasar', 'caspar'.
            agents_dir: Directory holding ``{agent_name}.md``.
            prompt: The user prompt payload.
            output_dir: Directory for artifacts.
            timeout: Timeout in seconds, fresh for each attempt.
            model: Model short name.
            max_attempts: Attempts before giving up.
            on_state: Optional state callback; see :meth:`run_trio`.

        Returns:
            The validated verdict dict (seven-key contract).

        Raises:
            ValueError: If *max_attempts* is below 1.
            TimeoutError: If an attempt exceeds *timeout*. **Not retried**:
                waiting again does not fix a hung model, and the v5 path did
                not retry it either.
            RuntimeError: If the subprocess exits non-zero. **Not retried**, for
                the same reason.
            ValidationError, json.JSONDecodeError: If the LAST attempt still
                fails the schema. These two -- and only these two -- trigger a
                retry, because the retry IS the fix: the mage is re-prompted
                with the sentinel's cause-specific feedback.
        """
        if max_attempts < 1:
            raise ValueError(_MAX_ATTEMPTS_ERROR)

        system_prompt_path = os.path.join(agents_dir, f"{agent_name}.md")
        raw_file = os.path.join(output_dir, f"{agent_name}.raw.json")
        parsed_file = os.path.join(output_dir, f"{agent_name}.json")

        self._emit(on_state, agent_name, "running")
        try:
            attempt_prompt = prompt
            for attempt in range(max_attempts):
                # Rewritten on EVERY attempt so the artifact holds the prompt
                # actually sent last, retry feedback included.
                try:
                    _write_text(output_dir, f"{agent_name}.prompt.txt", attempt_prompt)
                except OSError as log_exc:
                    print(
                        f"WARNING: Failed to persist {agent_name}.prompt.txt: {log_exc}",
                        file=sys.stderr,
                    )
                try:
                    stdout = await self._run_one(
                        agent_name,
                        system_prompt_path,
                        attempt_prompt,
                        model,
                        timeout,
                        output_dir,
                    )
                    # Not fail-soft, unlike the artifacts above: the parser and the
                    # validator read these two files back, so a failed write here
                    # genuinely costs the verdict.
                    with open(raw_file, "wb") as handle:
                        handle.write(stdout)
                    parse_agent_output(raw_file, parsed_file)
                    result = load_agent_output(parsed_file)
                    _reject_echoed_example(result)
                    _require_claimed_identity(result, agent_name)
                    break
                except (ValidationError, json.JSONDecodeError) as err:
                    if attempt + 1 >= max_attempts:
                        raise
                    self._emit(on_state, agent_name, "retrying")
                    # Built from the ORIGINAL prompt: feedback does not accumulate.
                    attempt_prompt = self._retry_prompt(prompt, err)
        except (asyncio.TimeoutError, TimeoutError):
            self._emit(on_state, agent_name, "timeout")
            raise
        except BaseException:
            # Catches asyncio.CancelledError (a BaseException since 3.8), every
            # Exception subclass, KeyboardInterrupt and SystemExit. Always
            # re-raised: the state report is a best-effort side effect and must
            # never mask the real cause.
            self._emit(on_state, agent_name, "failed")
            raise
        self._emit(on_state, agent_name, "success")
        return result

    async def _run_one(
        self,
        agent_name: str,
        system_prompt_path: str,
        prompt: str,
        model: str,
        timeout: int,
        output_dir: str,
    ) -> bytes:
        """Run one `claude -p` attempt and return its raw stdout bytes.

        Moved verbatim from ``ClaudeBackend.run``: same argv, same timeout
        handling, same stderr persistence, same error types.

        Args:
            agent_name: One of 'melchior', 'balthasar', 'caspar'.
            system_prompt_path: Path to that mage's system-prompt .md file.
            prompt: The prompt for this attempt (may carry retry feedback).
            model: Model short name, resolved via :func:`models.resolve_model`.
            timeout: Timeout in seconds for this attempt.
            output_dir: Directory for the ``{agent}.stderr.log`` artifact.

        Returns:
            Raw stdout bytes, to be parsed by ``parse_agent_output``.

        Raises:
            TimeoutError: If the attempt exceeds *timeout*. The subprocess is
                killed and reaped, and buffered stderr is persisted and
                included in the message.
            RuntimeError: If the subprocess exits with a non-zero code.
        """
        model_id = resolve_model(model)
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            model_id,
            "--system-prompt-file",
            system_prompt_path,
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

        try:
            _write_stderr_log(output_dir, agent_name, stderr)
        except OSError as log_exc:
            print(
                f"WARNING: Failed to persist {agent_name}.stderr.log: {log_exc}",
                file=sys.stderr,
            )

        if proc.returncode != 0:
            stderr_text = (
                stderr.decode("utf-8", errors="replace").strip() if stderr else "no stderr"
            )
            raise RuntimeError(
                f"Agent '{agent_name}' exited with code {proc.returncode}: {stderr_text}"
            )
        return stdout
