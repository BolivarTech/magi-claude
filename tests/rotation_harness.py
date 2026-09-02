# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-12
"""Shared rotation test harness (T9-T12 reuse it).

A plain helper module rather than ``conftest`` free functions: pytest does not
expose ``conftest`` module-level functions by name, and a harness copy-pasted
into each test class drifts apart (finding by Melchior, Checkpoint 2). Every
``run_magi`` symbol is imported LAZILY (inside the functions) so importing this
module never breaks collection before the Green step lands those symbols.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from fallback_policy import LineageRegistry, ModelCapability, RotationPolicy
from ollama_config import ModelSpec
from validate import ValidationError

TRIO = {
    "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),
    "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),
    "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),
}
FALLBACK = [
    ModelSpec("glm-5.2:cloud", "zhipu"),
    ModelSpec("gpt-oss:120b-cloud", "openai"),
    ModelSpec("minimax-m3:cloud", "minimax"),
]
BIG_WINDOW = 200_000
REQUIRED = 50_000


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


def _preflight_result() -> Any:
    """A minimal PreflightResult for the rotation tests (they do not exercise it)."""
    from ollama_preflight import CONTEXT_GUARD_ENFORCED, PreflightResult

    return PreflightResult(
        capabilities={},
        min_window_tokens=REQUIRED,
        required_tokens=REQUIRED,
        context_guard=CONTEXT_GUARD_ENFORCED,
        lineage_warnings=[],
        fallback=tuple(FALLBACK),
        token_estimate_delta=[],
    )


def _cfg() -> Any:
    """A real OllamaConfig with the built-in defaults (see the ollama_config fixture)."""
    from ollama_config import DEFAULT_FALLBACK, DEFAULT_MODELS, OllamaConfig

    return OllamaConfig(
        base_url="http://localhost:11434/v1",
        api_key=None,
        models=dict(DEFAULT_MODELS),
        fallback=tuple(DEFAULT_FALLBACK),
        max_attempts_per_model=2,
        max_rotations=2,
        max_probe_attempts=3,
        output_headroom_tokens=8192,
        input_margin_pct=40,
        strict_context_guard=False,
        retry_backoff_seconds=2.0,
        preflight_timeout_seconds=30,
        probe_timeout_seconds=120,
    )


def _rotation(
    *,
    windows: dict[str, int] | None = None,
    probe: Any = None,
    max_attempts: int = 2,
    max_rotations: int = 2,
    max_probe_attempts: int = 3,
    strict: bool = False,
    fallback: Any = None,
    digests: dict[str, str] | None = None,
) -> Any:
    """Build a RotationContext wired to fakes: no sockets, no sleeping.

    Args:
        windows: Optional per-model context-window override.
        probe: Optional token-probe stand-in.
        max_attempts: Attempts per model before a mage rotates.
        max_rotations: Rotation budget.
        max_probe_attempts: Probe-retry budget inside a single rotation.
        strict: ``strict_context_guard`` value.
        fallback: Optional override for the fallback list (defaults to the
            module's ``FALLBACK``) -- Task 5b digest tests use this to inject
            NON-cloud candidates the shared trio/fallback never has.
        digests: Optional model -> digest map (Task 5b). Trio entries seed
            ``RotationContext.digest_by_model`` (mirroring
            ``PreflightResult.digest_by_model``); every entry also flows into
            the fake ``ModelCapability.digest`` so ``policy.digest_of`` can
            resolve it with zero I/O, exactly like the real preflight cache.
    """
    from run_magi import RotationContext, RotationRuntimeConfig

    fb = list(FALLBACK if fallback is None else fallback)
    digest_map = digests or {}
    caps = {
        spec.model: ModelCapability(
            window=(windows or {}).get(spec.model, BIG_WINDOW),
            supports_completion=True,
            digest=digest_map.get(spec.model),
        )
        for spec in list(TRIO.values()) + fb
    }
    policy = RotationPolicy(
        fallback=fb,
        max_rotations=max_rotations,
        min_window_tokens=REQUIRED,  # RAW payload: pre-filter only (C2-1)
        capabilities=caps,
        strict_context_guard=strict,
    )

    async def _default_probe(model: str, prompt: str, timeout: int) -> int | None:
        return REQUIRED  # measured and it fits

    trio_digest_seed = {
        spec.model: digest_map[spec.model] for spec in TRIO.values() if spec.model in digest_map
    }

    return RotationContext(
        registry=LineageRegistry(TRIO),
        policy=policy,
        preflight=_preflight_result(),
        config=replace(
            RotationRuntimeConfig.from_config(_cfg()),
            max_attempts_per_model=max_attempts,
            max_probe_attempts=max_probe_attempts,
            strict_context_guard=strict,
            output_headroom_tokens=0,  # the fakes measure exactly; no headroom noise
        ),
        probe=probe or _default_probe,
        digest_by_model=trio_digest_seed,
    )


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

    The tests that call `launch_agent` DIRECTLY are untouched: they test
    `launch_agent`, which is live code until MS5.

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


async def _run(tmp_path: Any, mock_launch: Any, rotation: Any = None, **kw: Any) -> dict[str, Any]:
    """Drive run_orchestrator with a patched launch_agent (existing pattern)."""
    from run_magi import run_orchestrator

    # MS1 moved the Claude path's call site to `ClaudeTrio`; the rotation path
    # still goes through `launch_agent`. Patch the seam that the run under test
    # actually uses, or the mock sits off the path and every mage "dies".
    seam = (
        patch("run_magi.launch_agent", side_effect=mock_launch)
        if rotation is not None
        else patch("run_magi.ClaudeTrio", _trio_from_launch(mock_launch))
    )
    with seam:
        return await run_orchestrator(
            agents_dir=str(tmp_path),
            prompt="test",
            output_dir=str(tmp_path),
            timeout=300,
            agent_models=dict(TRIO),
            rotation=rotation,
            show_status=False,
            **kw,
        )
