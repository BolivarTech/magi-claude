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
