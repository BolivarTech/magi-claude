#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 5.0.0
# Date: 2026-07-11
"""Measure Ollama model context windows and completion capabilities.

This module performs read-only I/O against Ollama's ``/api/show`` endpoint
and builds a pre-loaded cache of ``ModelCapability`` records.  It does not
decide rotation policy; it only supplies the data that ``RotationPolicy``
needs to stay pure and I/O-free.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import urllib.request
from typing import Any, Awaitable, Callable, Mapping, Sequence

from fallback_policy import ModelCapability
from ollama_config import OllamaConfig
from redaction import redact_secrets

PREFLIGHT_RETRIES = 2
CAPABILITY_COMPLETION = "completion"
_WINDOW_KEYS = ("context_length", "context_window")

ShowFn = Callable[[str, int], Awaitable[dict[str, Any]]]
ProbeFn = Callable[[str, str, int], Awaitable[dict[str, Any]]]


async def _retrying(
    show: ShowFn,
    model: str,
    timeout: int,
    backoff: float,
) -> dict[str, Any]:
    """Call ``show`` for ``model`` with bounded retries for transient failures.

    ``urllib.error.URLError`` is a subclass of ``OSError``, so network-level
    transients are caught by the ``OSError`` branch.

    Args:
        show: Async callable that performs the ``/api/show`` lookup.
        model: Name of the model to inspect.
        timeout: Per-attempt request timeout in seconds.
        backoff: Seconds to sleep between retries.

    Returns:
        Parsed JSON object returned by ``show``.

    Raises:
        OSError: If all retries are exhausted.
    """
    last: Exception | None = None
    for attempt in range(PREFLIGHT_RETRIES + 1):
        try:
            return await show(model, timeout)
        except (OSError, ValueError) as exc:
            last = exc
            if attempt < PREFLIGHT_RETRIES:
                await asyncio.sleep(backoff)
    raise OSError(str(last))


async def fetch_capabilities(
    config: OllamaConfig,
    models: Sequence[str],
    *,
    _show: ShowFn | None = None,
) -> dict[str, ModelCapability]:
    """Pre-load context-window and completion data for ``models``.

    Each model is queried concurrently.  Failures for one model do not
    cancel or discard results for the others.

    Args:
        config: Runtime Ollama configuration (timeouts, base URL, key).
        models: Model names to measure.
        _show: Optional injected show function for testing.

    Returns:
        Mapping from model name to measured ``ModelCapability``.
    """
    show = _show or _default_show(config)
    tasks = (
        _retrying(show, m, config.preflight_timeout_seconds, config.retry_backoff_seconds)
        for m in models
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    caps: dict[str, ModelCapability] = {}
    for model, res in zip(models, results):
        if isinstance(res, BaseException):
            safe = redact_secrets(str(res), config.api_key)
            print(f"WARNING: capability probe failed for {model}: {safe}", file=sys.stderr)
            caps[model] = ModelCapability(window=None, supports_completion=True)
        else:
            caps[model] = ModelCapability(
                window=_read_window(res),
                supports_completion=_read_completion(res),
            )
    return caps


def _read_window(payload: Mapping[str, Any]) -> int | None:
    """Extract a positive context-window value from an ``/api/show`` payload.

    Rejects booleans and non-integers.  Never manufactures a default number.

    Args:
        payload: Untrusted JSON object from Ollama.

    Returns:
        Positive integer context window, or ``None`` if not present/valid.
    """
    for key in _WINDOW_KEYS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value
    return None


def _read_completion(payload: Mapping[str, Any]) -> bool:
    """Determine whether a model declares the ``completion`` capability.

    Older Ollama responses may omit the ``capabilities`` field entirely; in
    that case we allow completion because absence of evidence is not
    evidence of absence.

    Args:
        payload: Untrusted JSON object from Ollama.

    Returns:
        ``True`` if completion is declared or the field is absent.
    """
    declared = payload.get("capabilities")
    if not isinstance(declared, list):
        return True
    return CAPABILITY_COMPLETION in declared


def _api_root(base_url: str) -> str:
    """Return the Ollama root URL for the ``/api/show`` endpoint.

    ``/api/show`` lives at the Ollama root, not under ``/v1``.

    Args:
        base_url: Configured Ollama base URL, optionally ending in ``/v1``.

    Returns:
        Root URL suitable for appending ``/api/show``.
    """
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


def _default_show(config: OllamaConfig) -> ShowFn:
    """Build the production ``/api/show`` caller for ``config``.

    Args:
        config: Runtime Ollama configuration.

    Returns:
        Async function that fetches and parses ``/api/show``.
    """

    async def _show(model: str, timeout: int) -> dict[str, Any]:
        url = _api_root(config.base_url) + "/api/show"
        body = json.dumps({"model": model}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        def _call() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"/api/show returned {type(parsed).__name__}, expected object")
            return parsed

        return await asyncio.to_thread(_call)

    return _show


def _default_probe(config: OllamaConfig) -> ProbeFn:
    """Build the production one-token chat-completion probe for ``config``.

    This exact-token probe is consumed by later preflight tasks.

    Args:
        config: Runtime Ollama configuration.

    Returns:
        Async function that POSTs a single-token chat probe.
    """

    async def _post(model: str, prompt: str, timeout: int) -> dict[str, Any]:
        url = config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 1,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        def _call() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"/chat/completions returned {type(parsed).__name__}")
            return parsed

        return await asyncio.to_thread(_call)

    return _post


MAX_ERROR_CHARS = 400
#: Fixed block ~355 chars (~90 tok) + 400 error chars priced at the TRUE worst ratio.
#: That ratio is 4 TOKENS PER CHARACTER, not 1 and not 3: an EMOJI is 4 UTF-8 bytes,
#: and a byte-level BPE that fails to merge them emits one token per byte. This bound
#: has now been wrong twice (1 tok/char, then 3) -- each time by assuming a comfortable
#: ratio instead of the worst one that actually exists. 400 * 4 + 90 + reserve = 2048.
#: Bounding CHARS does not bound TOKENS.
MAX_RETRY_FEEDBACK_TOKENS = 2048

#: Percent-to-fraction base for the input-margin sizing (no bare ``100`` in the body).
#: ``(_PERCENT + input_margin_pct) / _PERCENT`` == ``1 + margin%``.
_PERCENT = 100

#: The probe adapter's shape -- the callable a later task stores and returns from
#: ``make_probe``: (model, prompt, timeout) -> exact tokens or None.
ProbeTokensFn = Callable[[str, str, int], Awaitable[int | None]]


def compute_required_tokens(
    payload_tokens: int,
    *,
    output_headroom_tokens: int,
    input_margin_pct: int,
    exact: bool = True,
) -> int:
    """Tokens a model's window must hold for the WORST attempt of this run.

    The retry prompt carries a corrective feedback block, so it is BIGGER than the
    first attempt. Sizing the guard against the first attempt would leave a hole
    through which the very failure it prevents (silent truncation) could walk in on
    the retry.

    Takes plain ints, NOT a config object: both ``OllamaConfig`` (preflight) and the
    orchestrator's runtime config call it, and coupling it to either one would force
    the other to fake it (a type error under mypy --strict). Low coupling is what
    lets one function serve both callers.

    Args:
        payload_tokens: Exact count (probe) or heuristic estimate.
        output_headroom_tokens: Room reserved for the verdict AND the model's
            thinking tokens, which never appear in the report but do consume window.
        input_margin_pct: Cushion applied ONLY to an estimate. It is a pre-filter,
            never a guarantee: the exact probe makes the real decision.
        exact: False when *payload_tokens* is an estimate, so the margin applies.

    Returns:
        The minimum window size, in tokens, a model must have to be eligible.
    """
    base = payload_tokens
    if not exact:
        base = math.ceil(payload_tokens * (_PERCENT + input_margin_pct) / _PERCENT)
    return base + MAX_RETRY_FEEDBACK_TOKENS + output_headroom_tokens


async def probe_prompt_tokens(
    config: OllamaConfig,
    model: str,
    prompt: str,
    *,
    timeout: int | None = None,
    _post: ProbeFn | None = None,
) -> int | None:
    """Measure the payload with the MODEL'S OWN tokenizer.

    Sends the real prompt with ``max_tokens=1`` and reads ``usage.prompt_tokens`` --
    part of the OpenAI standard, so this works against ANY compatible endpoint
    (unlike /api/show). The chars/4 heuristic underestimates real tokenizers by
    14.8%-19.8% (measured 2026-07-11), which is exactly the error that produces
    silent truncation.

    Every failure is REPORTED, never swallowed (R18): a degraded guard the reader
    cannot see is worse than no guard, because the report would imply a protection
    that never happened.

    Args:
        config: Resolved config (endpoint, auth, probe timeout).
        model: Model whose tokenizer should count the payload.
        prompt: The exact prompt the agent will receive.
        timeout: Override for the probe timeout, in seconds.
        _post: Injected chat-completions caller (tests).

    Returns:
        The exact prompt token count, or None when it could not be measured (the
        response omitted ``usage``, or the probe itself failed). NEVER a guess: the
        caller decides what to do with "unmeasurable" via ``strict_context_guard``.
    """
    post = _post or _default_probe(config)
    try:
        data = await post(model, prompt, timeout or config.probe_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 -- accuracy optimisation, never fatal
        # Deliberate broad catch: ANY probe failure degrades to the estimator.
        # It is NOT silent -- the warning is the whole point (R18).
        print(
            f"WARNING: token probe failed for {model} "
            f"({redact_secrets(str(exc), config.api_key)}); "
            f"falling back to the estimator (context_guard=estimated)",
            file=sys.stderr,
        )
        return None
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        # ``isinstance(True, int)`` is True in Python -- reject bools explicitly.
        print(
            f"WARNING: endpoint returned no usage.prompt_tokens for {model}; "
            f"falling back to the estimator (context_guard=estimated)",
            file=sys.stderr,
        )
        return None
    return tokens


def make_probe(config: OllamaConfig) -> ProbeTokensFn:
    """Bind *config* into a probe callable the orchestrator can call blind.

    The orchestrator holds a :data:`ProbeTokensFn` -- a ``(model, prompt, timeout)``
    coroutine -- so it never carries the config or knows how the payload is
    measured. That keeps :class:`~fallback_policy.RotationPolicy` pure and the
    rotation path free of the endpoint/auth details.

    Args:
        config: Resolved config (endpoint, auth, probe timeout).

    Returns:
        An async ``(model, prompt, timeout) -> int | None`` -- exact prompt
        tokens, or ``None`` when the payload could not be measured.
    """

    async def _probe(model: str, prompt: str, timeout: int) -> int | None:
        return await probe_prompt_tokens(config, model, prompt, timeout=timeout)

    return _probe
