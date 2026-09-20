# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-20
"""Shared orchestrator test harness for the Claude path.

A plain helper module rather than ``conftest`` free functions: pytest does not
expose ``conftest`` module-level functions by name, and a harness copy-pasted
into each test class drifts apart (finding by Melchior, Checkpoint 2). Every
``run_magi`` symbol is imported LAZILY (inside the functions) so importing this
module never breaks collection.

Until 6.1.0 this module also carried the v5 rotation harness (a fake preflight,
a ``RotationContext`` builder); that path was deleted with the plugin's own
Ollama client, and what is left is the adapter the Claude-path tests run through.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from unittest.mock import patch

from models import ModelSpec
from validate import ValidationError


TRIO = {
    "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),
    "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
    "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
}


def _valid(agent: str) -> dict[str, Any]:
    """A schema-valid verdict (same shape as TestSingleShotRetry._valid)."""
    return {
        "agent": agent,
        "verdict": "approve",
        "confidence": 0.85,
        "summary": f"{agent} OK",
        "reasoning": "Fine",
        "findings": [],
        "recommendation": "Merge",
    }


def _emit(on_state, agent_name, state):
    """Report a state the way the executor does: best effort, never fatal.

    Mirrors `ClaudeTrio._emit`. The callback is arbitrary caller-supplied code
    and the ABC promises a raising one cannot cost a verdict, so the double has
    to hold the same promise -- otherwise a test that installs a broken display
    would pass here and fail against the real executor.

    Args:
        on_state: The optional callback, ``(agent_name, state) -> None``.
        agent_name: The mage whose state changed.
        state: The new state.
    """
    if on_state is None:
        return
    try:
        on_state(agent_name, state)
    except Exception as exc:  # noqa: BLE001 - a broken display must not cost a verdict
        # The warning, not just the survival: the executor tells the operator,
        # and a double that stayed silent would let a regression that removed
        # that warning pass every orchestrator test running through here.
        print(
            f"WARNING: status callback failed for {agent_name} ({state}): {exc}",
            file=sys.stderr,
        )


def _trio_from_launch(mock_launch: Any) -> Any:
    """Adapt a per-mage `launch_agent` fake into a `ClaudeTrio` stand-in.

    MS1 moved the Claude path's call site: `run_orchestrator` no longer calls
    `launch_agent`, it calls `ClaudeTrio.run_trio`. Every test that mocked
    `launch_agent` and then ran the orchestrator stopped intercepting anything --
    the mock was still installed, just no longer on the path.

    Rather than rewrite each fake against the trio-level signature, this adapter
    keeps them EXACTLY as they are and turns one call into three. What each test
    asserts is unchanged; only the seam moved.

    Args:
        mock_launch: The per-mage fake, ``async (agent_name, ...) -> verdict``.

    Returns:
        A factory suitable for ``patch("run_magi.ClaudeTrio", ...)``.
    """

    class _Adapted:
        _on_extraction_failure = None

        async def run_trio(
            self,
            prompt,
            agents_dir,
            output_dir,
            timeout,
            models,
            max_attempts,
            on_state=None,
        ):
            if max_attempts < 1:
                # The executor validates ONCE, up front, before a single mage is
                # launched. A double that skipped this would let a test pass a
                # nonsensical value and see three silent no-ops instead of the
                # rejection the real thing raises.
                raise ValueError("max_attempts must be >= 1")
            verdicts = {}
            for name in models:
                attempt_prompt = prompt
                # ONCE per mage, OUTSIDE the attempt loop -- the executor emits
                # "running" before it starts attempting, not before each attempt.
                # Emitting it per attempt would let a display regression that
                # re-announces a retrying mage as fresh slip through every test
                # that goes through this double.
                _emit(on_state, name, "running")
                for attempt in range(max_attempts):
                    try:
                        verdicts[name] = await mock_launch(
                            name,
                            agents_dir,
                            attempt_prompt,
                            output_dir,
                            timeout,
                            models[name],
                        )
                    except (ValidationError, json.JSONDecodeError) as err:
                        if self._on_extraction_failure is not None:
                            self._on_extraction_failure(name, err)
                        # The RETRY lives inside the executor now, so the adapter
                        # has to reproduce it: these fakes fail once and succeed
                        # on the second call, and they are testing the retry, not
                        # the transport.
                        if attempt + 1 >= max_attempts:
                            verdicts[name] = err
                            _emit(on_state, name, "failed")
                            break
                        _emit(on_state, name, "retrying")
                        # Built from the ORIGINAL prompt, exactly as the executor
                        # does: the tests assert the error text reaches the model.
                        from run_magi import _build_retry_prompt

                        attempt_prompt = _build_retry_prompt(prompt, err)
                        continue
                    except BaseException as exc:  # noqa: BLE001
                        # One entry per mage, the exact exception object: the
                        # orchestrator branches on its subtype, exactly as it
                        # does with the real executor. Terminal -- a timeout or a
                        # non-zero exit is not retried.
                        _emit(
                            on_state,
                            name,
                            "timeout"
                            if isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                            else "failed",
                        )
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            # MEASURED against the real executor: asyncio's Task
                            # gives these two special treatment, so they escape
                            # `run_trio` even under `return_exceptions=True` and
                            # tear the run down. Every other BaseException is
                            # returned as that mage's outcome.
                            raise
                        verdicts[name] = exc
                        break
                    else:
                        _emit(on_state, name, "success")
                        break
            return verdicts

    def _factory(*args, **kwargs):
        # The adapter is a test double for the EXECUTOR, so it has to be faithful
        # to the executor's constructor too: `run_orchestrator` injects the R18
        # recorder there, and a stub that drops it makes `extraction_failures`
        # silently empty on the Claude path.
        adapted = _Adapted()
        adapted._on_extraction_failure = kwargs.get("on_extraction_failure")
        return adapted

    return _factory


async def _run(tmp_path: Any, mock_launch: Any, **kw: Any) -> dict[str, Any]:
    """Drive run_orchestrator with a per-mage fake behind the ``ClaudeTrio`` seam."""
    from run_magi import run_orchestrator

    # MS1 moved the Claude path's call site to `ClaudeTrio`: patch that seam, or
    # the mock sits off the path and every mage "dies".
    with patch("run_magi.ClaudeTrio", _trio_from_launch(mock_launch)):
        return await run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
            agent_models=dict(TRIO),
            show_status=False,
            **kw,
        )
