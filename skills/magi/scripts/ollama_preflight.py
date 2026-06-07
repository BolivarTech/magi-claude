#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Fail-fast preflight for the Ollama backend (reachability + model presence)."""

from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request

from ollama_config import OllamaConfig
from validate import ValidationError

PREFLIGHT_TIMEOUT = 10


class OllamaPreflightError(ValidationError):
    """Raised when the Ollama host is unreachable or a trio model is missing."""


def _is_cloud_tag(tag: str) -> bool:
    """True for Ollama cloud tags, whose suffix is exactly ':cloud' or '-cloud'.

    Covers ``:cloud`` (e.g. ``glm-5:cloud``) and ``-cloud`` variants
    (e.g. ``gpt-oss:120b-cloud``) that Ollama uses for subscription-gated
    cloud models. Tags whose variant merely contains ``cloud`` as a substring
    (e.g. ``foo:precloud``) are NOT matched.

    Args:
        tag: A full Ollama model tag string (e.g. ``"gpt-oss:120b-cloud"``).

    Returns:
        ``True`` if *tag* ends with ``":cloud"`` or ``"-cloud"``, ``False`` otherwise.
    """
    return tag.endswith((":cloud", "-cloud"))


def _redact(text: str, api_key: str | None) -> str:
    return text.replace(api_key, "***") if api_key else text


def preflight(config: OllamaConfig) -> None:
    """Verify host reachable and trio models available; abort otherwise.

    Raises:
        OllamaPreflightError: host unreachable, auth failure, or a configured
            model is absent from a returned /models list. A 404/501 on /models
            warns and proceeds (reachability OK, listing unsupported).
    """
    url = f"{config.base_url}/models"
    headers = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=PREFLIGHT_TIMEOUT) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise OllamaPreflightError(
                _redact(
                    f"Auth failed ({exc.code}) for {config.base_url}; "
                    "check api_key / `ollama signin`.",
                    config.api_key,
                )
            ) from None
        if exc.code in (404, 501):
            print(
                f"WARNING: {config.base_url}/models unavailable ({exc.code}); "
                "skipping model-existence check.",
                file=sys.stderr,
            )
            return
        raise OllamaPreflightError(
            _redact(f"Preflight HTTP {exc.code} at {url}.", config.api_key)
        ) from None
    except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
        raise OllamaPreflightError(
            f"Cannot reach Ollama at {config.base_url}: {exc}. "
            "Is it running? Try `ollama signin` for cloud."
        ) from None

    available = {m.get("id") for m in payload.get("data", []) if isinstance(m, dict)}
    missing = sorted(set(config.models.values()) - available)
    if missing:
        # Cloud-no-signin diagnostic (BDD-27 / F-B): when the whole trio is
        # :cloud-tagged but the daemon lists NO :cloud model, the likely cause
        # is a missing `ollama signin` — surface that as the primary hint.
        all_cloud = all(_is_cloud_tag(tag) for tag in config.models.values())
        none_cloud_available = not any(_is_cloud_tag(str(m)) for m in available)
        if all_cloud and none_cloud_available:
            raise OllamaPreflightError(
                f"No :cloud models available on {config.base_url} (the trio is all :cloud). "
                "Run `ollama signin` first (cloud models need a cloud session on the local "
                "daemon), or set api_key for the direct cloud API, or switch to local tags."
            )
        raise OllamaPreflightError(
            f"Missing models on {config.base_url}: {missing}. "
            "Fix: `ollama pull <model>` (local) / `ollama signin` or api_key (cloud) / "
            "edit magi-ollama.toml or MAGI_OLLAMA_MODEL_* / run `--ollama-init`."
        )
