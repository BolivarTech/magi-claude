#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Layered configuration for the Ollama (OpenAI-compatible) backend.

Precedence (per key): env > repo TOML > global TOML > built-in defaults,
with OLLAMA_HOST / OLLAMA_API_KEY as generic env fallbacks BELOW files.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from validate import ValidationError

DEFAULT_BASE_URL = "http://localhost:11434/v1"
#: Default trio (tier "Máximo", cloud) — single source for resolver + --ollama-init.
DEFAULT_MODELS: Mapping[str, str] = MappingProxyType(
    {
        "melchior": "qwen3.5:397b-cloud",  # Scientist  (Alibaba)  -- theoretical analysis
        "balthasar": "gpt-oss:120b-cloud",  # Pragmatist (OpenAI)   -- practical trade-offs
        "caspar": "glm-5.2:cloud",  # Critic     (Zhipu)    -- adversarial review
    }
)
_MAGES = ("melchior", "balthasar", "caspar")
_KNOWN_TOP_KEYS = {"base_url", "api_key", "models", "structured"}


class OllamaConfigError(ValidationError):
    """Raised when an Ollama config file is malformed."""


@dataclass(frozen=True)
class OllamaConfig:
    """Resolved configuration for the Ollama backend.

    Attributes:
        base_url: Base URL of the OpenAI-compatible endpoint.
        api_key: Bearer token for authentication, or None for no auth.
        models: Mapping of mage name to model identifier.
        structured: Output structure mode ("schema" | "object" | "off").
    """

    base_url: str
    api_key: str | None
    models: Mapping[str, str]
    structured: str = "schema"  # "schema" | "object" | "off" (R16)


def _load_toml(path: str) -> dict[str, Any]:
    """Load a TOML config file, returning empty dict if not found.

    Args:
        path: Filesystem path to the TOML file.

    Returns:
        Parsed TOML content as a dict, or {} if file does not exist.

    Raises:
        OllamaConfigError: If the file exists but is malformed TOML.
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise OllamaConfigError(f"Malformed TOML: {exc}", path) from exc
    for key in set(data) - _KNOWN_TOP_KEYS:
        print(f"WARNING: unknown key '{key}' in {path} (ignored)", file=sys.stderr)
    return data


def _normalize_base_url(raw: str) -> str:
    """Normalize a raw host/URL string to a clean base URL.

    Rules:
    - Strip trailing slash.
    - If no scheme, prepend ``http://``.
    - If the authority portion has no path component, append ``/v1``.
    - Any explicit path is kept verbatim (proxy prefix, custom mount, etc.).

    Args:
        raw: Raw host string or full URL.

    Returns:
        Normalized base URL string.

    Examples:
        >>> _normalize_base_url("1.2.3.4:11434")
        'http://1.2.3.4:11434/v1'
        >>> _normalize_base_url("http://gw/proxy")
        'http://gw/proxy'
    """
    raw = raw.rstrip("/")
    if "://" not in raw:
        raw = f"http://{raw}"
    # Has an explicit version path? leave it; else append /v1.
    tail = raw.split("://", 1)[1]
    if "/" not in tail:
        raw = f"{raw}/v1"
    return raw


def resolve_config(
    *,
    global_path: str | None = None,
    repo_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> OllamaConfig:
    """Resolve OllamaConfig from defaults + global TOML + repo TOML + env.

    Precedence per key (high → low):
    1. MAGI-specific env vars (``MAGI_OLLAMA_*``).
    2. Repo-level TOML (``.claude/magi-ollama.toml``).
    3. Global TOML (``~/.claude/magi-ollama.toml``).
    4. Generic env fallbacks (``OLLAMA_HOST``, ``OLLAMA_API_KEY``).
    5. Built-in defaults.

    Presence semantics are used throughout (``var in env`` / ``is not None``),
    NOT ``or``-truthiness. This means ``MAGI_OLLAMA_API_KEY=""`` sets
    ``api_key=None`` (explicit no-auth) rather than falling through to a file
    value (BDD-26 / F-C CI leak guard).

    Args:
        global_path: Path to the global TOML config. Defaults to
            ``~/.claude/magi-ollama.toml``.
        repo_path: Path to the repo TOML config. Defaults to
            ``.claude/magi-ollama.toml`` in the current directory.
        env: Environment mapping to use. Defaults to ``os.environ``.

    Returns:
        Fully resolved :class:`OllamaConfig`.

    Raises:
        OllamaConfigError: If any TOML file is malformed.
    """
    if env is None:
        env = os.environ
    if global_path is None:
        global_path = os.path.expanduser("~/.claude/magi-ollama.toml")
    if repo_path is None:
        repo_path = os.path.join(os.getcwd(), ".claude", "magi-ollama.toml")

    g = _load_toml(global_path)
    r = _load_toml(repo_path)

    # base_url (presence-based; R17 — MAGI-specific env present wins; empty host = skip)
    if env.get("MAGI_OLLAMA_HOST"):
        raw_host = env["MAGI_OLLAMA_HOST"]
    elif r.get("base_url"):
        raw_host = r["base_url"]
    elif g.get("base_url"):
        raw_host = g["base_url"]
    elif env.get("OLLAMA_HOST"):
        raw_host = env["OLLAMA_HOST"]
    else:
        raw_host = DEFAULT_BASE_URL
    base_url = _normalize_base_url(raw_host)

    # api_key (presence-based; R17/F-C — empty MAGI env => explicit None, no fall-through)
    if "MAGI_OLLAMA_API_KEY" in env:
        api_key = env["MAGI_OLLAMA_API_KEY"] or None  # "" => None (no auth in CI)
    elif r.get("api_key") is not None:
        api_key = r["api_key"] or None
    elif g.get("api_key") is not None:
        api_key = g["api_key"] or None
    elif env.get("OLLAMA_API_KEY"):
        api_key = env["OLLAMA_API_KEY"]
    else:
        api_key = None

    # structured mode (R16)
    structured = (
        env.get("MAGI_OLLAMA_STRUCTURED") or r.get("structured") or g.get("structured") or "schema"
    )

    # models per mage (presence-based; empty string is not a valid model -> skip)
    g_models = g.get("models", {}) or {}
    r_models = r.get("models", {}) or {}
    models: dict[str, str] = {}
    for mage in _MAGES:
        ekey = f"MAGI_OLLAMA_MODEL_{mage.upper()}"
        if env.get(ekey):
            models[mage] = env[ekey]
        elif r_models.get(mage):
            models[mage] = r_models[mage]
        elif g_models.get(mage):
            models[mage] = g_models[mage]
        else:
            models[mage] = DEFAULT_MODELS[mage]

    return OllamaConfig(base_url=base_url, api_key=api_key, models=models, structured=structured)
