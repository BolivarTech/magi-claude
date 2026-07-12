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
import re
import sys
import tomllib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from validate import ValidationError

DEFAULT_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class ModelSpec:
    """A model tag plus the lineage (originating lab) it belongs to.

    The lineage is DECLARED by the user, never inferred: prefix inference is
    fragile for custom/self-hosted tags, and a silently wrong lineage would
    break the "one lineage, one mage" invariant with no error signal.

    Attributes:
        model: Backend model identifier (e.g. "qwen3.5:397b-cloud").
        lineage: Originating lab (e.g. "alibaba"). Non-empty.
    """

    model: str
    lineage: str


#: Default trio (tier "Máximo", cloud) — single source for resolver + --ollama-init.
DEFAULT_MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "melchior": ModelSpec("qwen3.5:397b-cloud", "alibaba"),  # Scientist  -- theory
        "balthasar": ModelSpec("kimi-k2.6:cloud", "moonshot"),  # Pragmatist -- trade-offs
        "caspar": ModelSpec("deepseek-v4-pro:cloud", "deepseek"),  # Critic  -- adversarial
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
        models: Mapping of mage name to its :class:`ModelSpec` (tag + lineage).
        structured: Output structure mode ("schema" | "object" | "off").
    """

    base_url: str
    api_key: str | None
    models: Mapping[str, ModelSpec]
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


#: NOTE: this is a plain concatenation, NOT an f-string -- single braces are
#: literal here. Doubling them would emit invalid TOML in the migration hint,
#: which is the very artifact that replaces the (rejected) auto-migration shim.
_MIGRATION_HINT = (
    "MAGI v5.0.0 changed the [models] schema: each mage now declares its "
    "lineage explicitly.\n"
    '  OLD:  melchior = "qwen3.5:397b-cloud"\n'
    '  NEW:  melchior = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }\n'
    "Run `python scripts/validate_magi_toml.py` for a per-key report."
)

#: A model tag is user input from a TOML, and it reaches BOTH a JSON body (Ollama)
#: and a SUBPROCESS argv (`claude -p --model <tag>`). Validate it -- the standard
#: says validate all inputs, and an unvalidated string on an argv is how argument
#: injection happens (finding by Caspar, Checkpoint 2).
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _validate_tag(tag: str, mage: str, path: str) -> str:
    """Reject a model tag that could not be a legitimate model identifier.

    Args:
        tag: The tag as written in the TOML.
        mage: Mage name, for the error message.
        path: Config path, for the error message.

    Returns:
        The tag, unchanged, when it is safe.

    Raises:
        OllamaConfigError: If the tag contains anything outside
            ``[A-Za-z0-9._:/-]``, is empty, or exceeds 128 characters. Whitespace,
            quotes, dashes-as-first-char and control characters are all rejected:
            none of them appear in a real tag, and all of them are how a string
            becomes a flag on someone else's command line.
    """
    if not _TAG_RE.match(tag):
        raise OllamaConfigError(
            f"[models].{mage}.model is not a valid model tag: {tag!r} "
            f"(allowed: letters, digits, and . _ : / -)",
            path,
        )
    return tag


def _normalise_lineage(raw: str) -> str:
    """Canonicalise a lineage label so string equality means what it should.

    EVERY guard in this feature -- R22 (trio uniqueness), R11.3 (fallback
    uniqueness), the in-play skip, the failed-lineage sets -- compares lineage
    strings with `==` and `in`. So "DeepSeek" and "deepseek" would be two DIFFERENT
    lineages, two mages would end up with the same lab, and the invariant the whole
    milestone exists to protect would be defeated by a capital letter (finding by
    Balthasar, Checkpoint 2).

    Args:
        raw: The lineage as written in the TOML.

    Returns:
        The lineage lowercased, stripped, and with internal whitespace collapsed.
    """
    return " ".join(raw.strip().lower().split())


def _parse_model_spec(mage: str, raw: Any, path: str) -> ModelSpec:
    """Parse one [models] entry into a ModelSpec.

    Args:
        mage: Mage name the entry belongs to.
        raw: Raw TOML value for that mage.
        path: Config file path, for error messages.

    Returns:
        The parsed ModelSpec.

    Raises:
        OllamaConfigError: If *raw* is a bare string (pre-v5 schema) or if
            `model`/`lineage` are missing or empty.
    """
    if isinstance(raw, str):
        raise OllamaConfigError(
            f"[models].{mage} is a bare string in {path}. {_MIGRATION_HINT}", path
        )
    if not isinstance(raw, dict):
        raise OllamaConfigError(f"[models].{mage} must be a table in {path}", path)
    # .strip() first: a whitespace-only value is truthy in Python, so " " would
    # otherwise sail through as a valid model tag / lineage and only blow up much
    # later -- as a confusing 404 at chat time, or as a lineage that matches
    # nothing and silently defeats the "one lineage, one mage" invariant.
    model = raw.get("model")
    lineage = raw.get("lineage")
    if not isinstance(model, str) or not model.strip():
        raise OllamaConfigError(f"[models].{mage}.model is missing/empty in {path}", path)
    if not isinstance(lineage, str) or not lineage.strip():
        raise OllamaConfigError(f"[models].{mage}.lineage is missing/empty in {path}", path)
    return ModelSpec(
        model=_validate_tag(model.strip(), mage, path),
        lineage=_normalise_lineage(lineage),
    )


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

    # models per mage (presence-based; per-key fallback to DEFAULT_MODELS).
    # Repo/global entries are now tables parsed into ModelSpec (R3, BREAKING);
    # a bare string raises the migration error. An env MODEL_* override forces
    # the tag but keeps the DECLARED lineage of the resolved spec -- it never
    # infers a lineage (which the whole feature forbids).
    g_models = g.get("models", {}) or {}
    r_models = r.get("models", {}) or {}
    models: dict[str, ModelSpec] = {}
    for mage in _MAGES:
        if mage in r_models:
            spec = _parse_model_spec(mage, r_models[mage], repo_path)
        elif mage in g_models:
            spec = _parse_model_spec(mage, g_models[mage], global_path)
        else:
            spec = DEFAULT_MODELS[mage]
        ekey = f"MAGI_OLLAMA_MODEL_{mage.upper()}"
        if env.get(ekey):
            spec = ModelSpec(
                model=_validate_tag(env[ekey].strip(), mage, "<env>"), lineage=spec.lineage
            )
        models[mage] = spec

    return OllamaConfig(base_url=base_url, api_key=api_key, models=models, structured=structured)
