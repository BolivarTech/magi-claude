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

from dataclasses import replace
from typing import Any
from unittest.mock import patch

from fallback_policy import LineageRegistry, ModelCapability, RotationPolicy
from ollama_config import ModelSpec

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


async def _run(tmp_path: Any, mock_launch: Any, rotation: Any = None, **kw: Any) -> dict[str, Any]:
    """Drive run_orchestrator with a patched launch_agent (existing pattern)."""
    from run_magi import run_orchestrator

    with patch("run_magi.launch_agent", side_effect=mock_launch):
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
