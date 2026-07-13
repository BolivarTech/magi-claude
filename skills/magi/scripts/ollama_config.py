#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Layered configuration for the Ollama (OpenAI-compatible) backend.

Precedence (per key): env > repo TOML > global TOML > built-in defaults,
with OLLAMA_HOST / OLLAMA_API_KEY as generic env fallbacks BELOW files.
"""

from __future__ import annotations

import math
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

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

# Rotation config built-in defaults (R6/R12/R14/R24). Named constants, no magic
# numbers -- each is the single source for both the resolver and --ollama-init.
DEFAULT_MAX_ATTEMPTS_PER_MODEL = 2
DEFAULT_MAX_ROTATIONS = 2
DEFAULT_MAX_PROBE_ATTEMPTS = 3
DEFAULT_OUTPUT_HEADROOM_TOKENS = 8192  # measured: verdicts 811-2189 tok + thinking headroom
DEFAULT_INPUT_MARGIN_PCT = 40  # pre-filter only; the exact probe decides (R24)
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 30  # metadata calls
DEFAULT_PROBE_TIMEOUT_SECONDS = 120  # probe processes the whole prompt
DEFAULT_STRICT_CONTEXT_GUARD = False

#: Ordered strong->weak, ONE model per lineage, none colliding with the trio's
#: lineages. Verified against registry.ollama.ai on 2026-07-11 (the website lists
#: tags the registry does not serve, so the check must hit the registry). The
#: pre-release check itself is maintainer tooling and is not shipped.
DEFAULT_FALLBACK: tuple[ModelSpec, ...] = (
    ModelSpec("glm-5.2:cloud", "zhipu"),
    ModelSpec("gpt-oss:120b-cloud", "openai"),
    ModelSpec("minimax-m3:cloud", "minimax"),
    ModelSpec("nemotron-3-super:cloud", "nvidia"),
    # The Google slot was gemini-3-flash-preview:latest, taken as a known risk: one model
    # per lineage, and Gemini 3 beat gemma4. Ollama is retiring that tag (2026-07-15), so
    # it goes back to gemma4:cloud (verified against registry.ollama.ai, HTTP 200). A
    # retired default is harmless by R11.1 -- a missing fallback warns, never aborts -- but
    # it would put a dead entry, and a warning, in every scaffolded config. Never ship a
    # preview tag as a default: it is a promise the vendor has not made (pinned by test).
    ModelSpec("gemma4:cloud", "google"),
)

_KNOWN_TOP_KEYS = {
    "base_url",
    "api_key",
    "models",
    "fallback",
    "max_attempts_per_model",
    "max_rotations",
    "max_probe_attempts",
    "output_headroom_tokens",
    "input_margin_pct",
    "strict_context_guard",
    "retry_backoff_seconds",
    "preflight_timeout_seconds",
    "probe_timeout_seconds",
}


class OllamaConfigError(ValidationError):
    """Raised when an Ollama config file is malformed."""


@dataclass(frozen=True)
class OllamaConfig:
    """Resolved configuration for the Ollama backend.

    Attributes:
        base_url: Base URL of the OpenAI-compatible endpoint.
        api_key: Bearer token for authentication, or None for no auth.
        models: Mapping of mage name to its :class:`ModelSpec` (tag + lineage).
        fallback: Ordered fallback specs (strong->weak); () disables rotation (R4).
        max_attempts_per_model: Attempts per active model before rotating (R1).
        max_rotations: Max rotations per mage; 0 disables rotation (R6/R17).
        max_probe_attempts: Max propose-verify probe attempts per mage (R24).
        output_headroom_tokens: Output+thinking tokens reserved by the guard (R5b).
        input_margin_pct: Rotation-candidate pre-filter margin percent (R24).
        strict_context_guard: Treat unknown context windows as fail-closed (R18).
        retry_backoff_seconds: Backoff between transport retries; 0 disables (R12).
        preflight_timeout_seconds: Timeout for preflight metadata calls (R18).
        probe_timeout_seconds: Timeout for the context probe call (R24).
    """

    base_url: str
    api_key: str | None
    models: Mapping[str, ModelSpec]
    fallback: Sequence[ModelSpec] = ()
    max_attempts_per_model: int = DEFAULT_MAX_ATTEMPTS_PER_MODEL
    max_rotations: int = DEFAULT_MAX_ROTATIONS
    max_probe_attempts: int = DEFAULT_MAX_PROBE_ATTEMPTS
    output_headroom_tokens: int = DEFAULT_OUTPUT_HEADROOM_TOKENS
    input_margin_pct: int = DEFAULT_INPUT_MARGIN_PCT
    strict_context_guard: bool = DEFAULT_STRICT_CONTEXT_GUARD
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS
    preflight_timeout_seconds: int = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS
    probe_timeout_seconds: int = DEFAULT_PROBE_TIMEOUT_SECONDS


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
    "Run `python skills/magi/scripts/validate_magi_toml.py` for a per-key report."
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


def _require_float(value: Any, *, key: str, minimum: float, path: str) -> float:
    """Coerce *value* to a float >= *minimum* or fail closed.

    Args:
        value: Raw value from TOML (int/float) or env (str).
        key: Key name, for the error message.
        minimum: Inclusive lower bound.
        path: Config path, for the error message.

    Returns:
        The validated float.

    Raises:
        OllamaConfigError: If *value* is a bool, is not numeric, or is below
            *minimum*. Bools are rejected first: ``isinstance(True, int)`` is True,
            so ``retry_backoff_seconds = true`` would otherwise become 1.0.
    """
    if isinstance(value, bool):
        raise OllamaConfigError(f"{key} must be a number, not a boolean", path)
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise OllamaConfigError(f"{key} must be a number (got {value!r})", path) from exc
    else:
        raise OllamaConfigError(f"{key} must be a number (got {value!r})", path)
    # Reject inf/nan BEFORE the range check: ``nan >= minimum`` and ``nan < minimum``
    # are BOTH False, so a NaN slips through, and ``inf`` passes any lower bound --
    # either would reach ``asyncio.sleep`` and hang the orchestrator (MAGI gate, Caspar).
    if not math.isfinite(parsed):
        raise OllamaConfigError(f"{key} must be a finite number (got {parsed})", path)
    if parsed < minimum:
        raise OllamaConfigError(f"{key} must be >= {minimum} (got {parsed})", path)
    return parsed


#: Env vars are strings; TOML gives real booleans. Both must mean the same thing,
#: and anything else must FAIL rather than be guessed at (a silently misread
#: strict_context_guard would silently disable a safety guard).
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _require_bool(value: Any, *, key: str, path: str) -> bool:
    """Coerce *value* to a bool or fail closed.

    Args:
        value: Raw value: a TOML boolean, or an env string.
        key: Key name, for the error message.
        path: Config path, for the error message.

    Returns:
        The validated boolean.

    Raises:
        OllamaConfigError: If the value is neither a boolean nor one of
            {1,true,yes,on} / {0,false,no,off} (case-insensitive). ``bool("false")``
            is True in Python -- guessing here would flip a safety flag silently.
    """
    if isinstance(value, bool):
        return value
    # A TOML integer literal ``key = 1`` / ``key = 0`` is an int, not a bool or str, so
    # it fell through to the raise -- while the error message lists "1/0" as accepted, so
    # a migrating user who followed the message still failed (MAGI gate, Balthasar). Accept
    # the two unambiguous integers; anything else (2, -1, ...) still fails closed.
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise OllamaConfigError(
        f"{key} must be a boolean (true/false, 1/0, yes/no, on/off); got {value!r}", path
    )


def _require_int(value: Any, *, key: str, minimum: int, path: str) -> int:
    """Coerce *value* to an int >= *minimum* or fail closed.

    Args:
        value: Raw value from TOML or env (str from env, int from TOML).
        key: Key name, for the error message.
        minimum: Inclusive lower bound.
        path: Config path, for the error message.

    Returns:
        The validated integer.

    Raises:
        OllamaConfigError: If *value* is a bool, a float, a non-numeric string, or
            is below *minimum*.

    Notes:
        Two coercions are rejected explicitly because both are SILENT failures,
        which the standard forbids:

        * ``bool``: ``isinstance(True, int)`` is True in Python, so
          ``max_rotations = true`` would be accepted as 1.
        * ``float``: ``int(2.7)`` is 2 -- ``max_rotations = 2.7`` would be
          silently truncated instead of telling the user their config is wrong.
    """
    if isinstance(value, bool):
        raise OllamaConfigError(f"{key} must be an integer, not a boolean", path)
    if isinstance(value, float):
        raise OllamaConfigError(f"{key} must be an integer, not a float (got {value!r})", path)
    if isinstance(value, str):
        # A strict regex, not lstrip("+-"): "+-5" and "--5" would survive a strip
        # and then int() would raise a bare ValueError with a useless message
        # (finding by Caspar, Checkpoint 2). Env vars are user input: validate them.
        if not re.fullmatch(r"[+-]?\d+", value.strip()):
            raise OllamaConfigError(f"{key} must be an integer (got {value!r})", path)
        parsed = int(value.strip())
    elif isinstance(value, int):
        parsed = value
    else:
        raise OllamaConfigError(f"{key} must be an integer (got {value!r})", path)
    if parsed < minimum:
        raise OllamaConfigError(f"{key} must be >= {minimum} (got {parsed})", path)
    return parsed


#: Per-scalar resolution table (R14): (config key, env var, validator, default).
#: The env-name mapping and precedence chain live here ONCE (DRY) instead of being
#: copy-pasted per field; ``partial`` binds each numeric guard's ``minimum``.
_SCALAR_SPECS: tuple[tuple[str, str, Callable[..., Any], Any], ...] = (
    (
        "max_attempts_per_model",
        "MAGI_OLLAMA_MAX_ATTEMPTS",
        partial(_require_int, minimum=1),
        DEFAULT_MAX_ATTEMPTS_PER_MODEL,
    ),
    (
        "max_rotations",
        "MAGI_OLLAMA_MAX_ROTATIONS",
        partial(_require_int, minimum=0),
        DEFAULT_MAX_ROTATIONS,
    ),
    (
        "max_probe_attempts",
        "MAGI_OLLAMA_MAX_PROBE_ATTEMPTS",
        partial(_require_int, minimum=1),
        DEFAULT_MAX_PROBE_ATTEMPTS,
    ),
    (
        "output_headroom_tokens",
        "MAGI_OLLAMA_OUTPUT_HEADROOM_TOKENS",
        partial(_require_int, minimum=0),
        DEFAULT_OUTPUT_HEADROOM_TOKENS,
    ),
    (
        "input_margin_pct",
        "MAGI_OLLAMA_INPUT_MARGIN_PCT",
        partial(_require_int, minimum=0),
        DEFAULT_INPUT_MARGIN_PCT,
    ),
    (
        "retry_backoff_seconds",
        "MAGI_OLLAMA_RETRY_BACKOFF_SECONDS",
        partial(_require_float, minimum=0.0),
        DEFAULT_RETRY_BACKOFF_SECONDS,
    ),
    (
        "preflight_timeout_seconds",
        "MAGI_OLLAMA_PREFLIGHT_TIMEOUT_SECONDS",
        partial(_require_int, minimum=1),
        DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    ),
    (
        "probe_timeout_seconds",
        "MAGI_OLLAMA_PROBE_TIMEOUT_SECONDS",
        partial(_require_int, minimum=1),
        DEFAULT_PROBE_TIMEOUT_SECONDS,
    ),
    (
        "strict_context_guard",
        "MAGI_OLLAMA_STRICT_CONTEXT",
        _require_bool,
        DEFAULT_STRICT_CONTEXT_GUARD,
    ),
)


def _resolve_scalar(
    global_cfg: dict[str, Any],
    repo_cfg: dict[str, Any],
    env: Mapping[str, str],
    key: str,
    env_name: str,
    validate: Callable[..., Any],
    default: Any,
    repo_path: str,
    global_path: str,
) -> Any:
    """Resolve one scalar with env > repo > global > default precedence (R14).

    Args:
        global_cfg: Global TOML config mapping.
        repo_cfg: Repository TOML config mapping.
        env: Environment variable mapping.
        key: TOML config key.
        env_name: Environment variable name for this key.
        validate: Validator called as ``(value, key=key, path=path)``.
        default: Built-in default used when the key is absent everywhere.
        repo_path: Path to the repo config file, for error messages.
        global_path: Path to the global config file, for error messages.

    Returns:
        The resolved and validated scalar value.

    Raises:
        OllamaConfigError: If the resolved env or TOML value is invalid.
    """
    if env_name in env:
        return validate(env[env_name], key=key, path="<env>")
    if key in repo_cfg:
        return validate(repo_cfg[key], key=key, path=repo_path)
    if key in global_cfg:
        return validate(global_cfg[key], key=key, path=global_path)
    return default


def _parse_fallback_entries(raw: Any, path: str) -> tuple[ModelSpec, ...]:
    """Parse a ``[[fallback]]`` array-of-tables into validated model specs (R4).

    Args:
        raw: Raw ``fallback`` value from a TOML config file.
        path: Path to the config file, for error messages.

    Returns:
        Tuple of validated fallback :class:`ModelSpec` entries, in declared order.

    Raises:
        OllamaConfigError: If ``raw`` is not a list, or any entry is not a valid
            ``{ model, lineage }`` table (delegated to :func:`_parse_model_spec`).
    """
    if not isinstance(raw, list):
        raise OllamaConfigError(
            f"fallback must be an array of tables; got {type(raw).__name__}", path
        )
    return tuple(_parse_model_spec(f"fallback[{i}]", entry, path) for i, entry in enumerate(raw))


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

    # fallback list (R4): repo wins if present, else global, else empty (rotation
    # OFF). NEVER falls through to DEFAULT_FALLBACK -- the built-in list reaches
    # the user only via the --ollama-init template (decision #65).
    fallback: tuple[ModelSpec, ...] = ()
    if "fallback" in r:
        fallback = _parse_fallback_entries(r["fallback"], repo_path)
    elif "fallback" in g:
        fallback = _parse_fallback_entries(g["fallback"], global_path)

    # rotation scalars (R14 precedence, one declaration per key via _SCALAR_SPECS).
    scalars: dict[str, Any] = {}
    for key, env_name, validate, default in _SCALAR_SPECS:
        scalars[key] = _resolve_scalar(
            g, r, env, key, env_name, validate, default, repo_path, global_path
        )

    return OllamaConfig(
        base_url=base_url,
        api_key=api_key,
        models=models,
        fallback=fallback,
        **scalars,
    )
