#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 2.0.0
# Date: 2026-09-18
"""``--ollama-init``: one call to ``magi-rs init`` and nothing else (v6.0.0).

:func:`run_ollama_init` is the whole command. It bootstraps and verifies the
pinned binary first -- R4 has no carve-out for the init command, which
executes the binary too -- then invokes ``magi-rs init -w <root>`` with the
allowlisted environment and relays whatever it said. It writes no file of its
own, reads nothing under ``.claude/``, and passes no ``-p``: there is no
passphrase, no key and no plugin config any more (R5, R5a, R5b).

The template functions below (``render_template``/``write_template``) belong
to the v5 ``magi-ollama.toml`` and are no longer called by the plugin; they
stay in place, inert, until the v5 open-weight subsystem is deleted in one
movement (MS5), together with the tests that still pin their shape.
"""

from __future__ import annotations

import os
import subprocess

from backoff import DEFAULT_RETRY_AFTER_MAX_SECONDS, DEFAULT_RETRY_BACKOFF_MAX_SECONDS
from ollama_config import (
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACK,
    DEFAULT_INPUT_MARGIN_PCT,
    DEFAULT_MAX_ATTEMPTS_PER_MODEL,
    DEFAULT_MAX_PROBE_ATTEMPTS,
    DEFAULT_MAX_ROTATIONS,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_HEADROOM_TOKENS,
    DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    DEFAULT_STRICT_CONTEXT_GUARD,
    DEFAULT_STRICT_LINEAGE,
    DEFAULT_TIMEOUT_SECONDS,
)

from magi_rs_bootstrap import BinaryBootstrap
from magi_rs_env import EnvironmentPolicy
from magi_rs_paths import bin_dir, pin_path, plugin_root
from magi_rs_pin import Pin

REPO_CONFIG_RELPATH = os.path.join(".claude", "magi-ollama.toml")

_INIT_SUBCOMMAND = "init"
_WORKDIR_FLAG = "-w"
_INIT_TIMEOUT_SECONDS = 120
"""``magi-rs init`` scaffolds a directory and a small database; two minutes is
far beyond any healthy run and only bounds a hung child."""


def run_ollama_init(workdir: str) -> str:
    """Initialise ``workdir`` for the magi-rs backend, by delegating entirely.

    Args:
        workdir: The project root, passed to ``magi-rs`` as ``-w``.

    Returns:
        What ``magi-rs init`` printed (stdout, then stderr), relayed verbatim.
        A refusal (non-zero exit) is relayed the same way: the requirement is
        on this side -- finish cleanly whatever magi-rs decided -- and never
        force anything (BDD-7b).

    Raises:
        UnsafeBinDirectoryError, DigestMismatchError, VersionMismatchError,
        DownloadError, UnsupportedPlatformError, PinFormatError: The binary
            could not be obtained or did not verify. It is NOT executed (R4).
        PluginRootError: ``CLAUDE_PLUGIN_ROOT`` is set but wrong.
        subprocess.TimeoutExpired: ``magi-rs init`` hung past the bound.

    Side effects:
        May download and install the binary; launches ``magi-rs init``, which
        creates ``<workdir>/.magi/``. Writes nothing itself and never touches
        ``.claude/``.
    """
    root = plugin_root()
    bootstrap = BinaryBootstrap(Pin.load(pin_path(root)), bin_dir(root))
    bootstrap.ensure()
    executable = bootstrap.verify()
    completed = subprocess.run(
        [executable, _INIT_SUBCOMMAND, _WORKDIR_FLAG, workdir],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=EnvironmentPolicy.build(os.environ),
        timeout=_INIT_TIMEOUT_SECONDS,
    )
    return "".join(part for part in (completed.stdout, completed.stderr) if part)


def render_template() -> str:
    """Return the TOML template text (base_url active, api_key commented).

    Returns:
        A TOML-formatted string with a two-mode header, the local base_url
        active, api_key commented out, the rotation/context-window scalars
        emitted as active top-level keys at their built-in defaults, and the
        default trio plus fallback list populated as ``{ model, lineage }``
        tables (v5.0.0 schema).
    """
    # v5.0.0 (R3): each mage is a table declaring its lineage explicitly. Built
    # from DEFAULT_MODELS so template and resolver stay a single source of truth.
    model_lines = "".join(
        f'{mage:<9} = {{ model = "{spec.model}", lineage = "{spec.lineage}" }}\n'
        for mage, spec in DEFAULT_MODELS.items()
    )
    # v5.0.0 (R4): the built-in fallback list reaches the user through this template
    # (decision #65) -- the resolver never injects it. Ordered strong->weak, one
    # lineage each, none colliding with the trio.
    fallback_lines = "".join(
        f'\n[[fallback]]\nmodel = "{spec.model}"\nlineage = "{spec.lineage}"\n'
        for spec in DEFAULT_FALLBACK
    )
    # v5.0.0 (R12/R17): the rotation/context-window tunables are top-level scalars
    # (they apply to ALL mages), so TOML requires them BEFORE any [table] header.
    # (key, value, comment) for every scalar the resolver accepts; values come from the
    # DEFAULT_* constants -- one source of truth -- so an untouched scaffold round-trips
    # to the defaults while every knob (and the kill-switch) stays visible and editable.
    strict_literal = "true" if DEFAULT_STRICT_CONTEXT_GUARD else "false"
    strict_lineage_literal = "true" if DEFAULT_STRICT_LINEAGE else "false"
    tunable_specs = (
        (
            "max_attempts_per_model",
            DEFAULT_MAX_ATTEMPTS_PER_MODEL,
            "tries per model before rotating to a fallback (>= 1)",
        ),
        (
            "max_rotations",
            DEFAULT_MAX_ROTATIONS,
            "fallback models a mage may rotate through (0 disables rotation)",
        ),
        (
            "max_probe_attempts",
            DEFAULT_MAX_PROBE_ATTEMPTS,
            "fallback candidates to size-check before a mage gives up (>= 1)",
        ),
        (
            "output_headroom_tokens",
            DEFAULT_OUTPUT_HEADROOM_TOKENS,
            "context tokens reserved for the model's answer plus its thinking",
        ),
        (
            "input_margin_pct",
            DEFAULT_INPUT_MARGIN_PCT,
            "extra margin when checking the input fits a model's window, percent",
        ),
        (
            "strict_context_guard",
            strict_literal,
            "refuse a model whose context window cannot be measured (default: true)",
        ),
        (
            "retry_backoff_seconds",
            DEFAULT_RETRY_BACKOFF_SECONDS,
            "seconds to wait between transport retries (0 = no wait)",
        ),
        (
            "retry_backoff_max_seconds",
            DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
            "ceiling for OUR backoff formula: min(base * 2^n, this)",
        ),
        (
            "retry_after_max_seconds",
            DEFAULT_RETRY_AFTER_MAX_SECONDS,
            "ceiling for a SERVER Retry-After header (defense, not a model cap)",
        ),
        (
            "preflight_timeout_seconds",
            DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
            "timeout for preflight metadata calls, seconds",
        ),
        (
            "probe_timeout_seconds",
            DEFAULT_PROBE_TIMEOUT_SECONDS,
            "timeout for the context-probe call, seconds",
        ),
        (
            "timeout",
            DEFAULT_TIMEOUT_SECONDS,
            "per-agent request timeout, seconds (overridden by --timeout)",
        ),
        (
            "strict_lineage",
            strict_lineage_literal,
            "abort (not just warn) when a probed architecture contradicts a "
            "declared lineage (default: false)",
        ),
    )
    key_width = max(len(key) for key, _, _ in tunable_specs)
    tunables_lines = (
        "# Rotation and context-window settings (apply to ALL mages; see docs/ollama-backend.md).\n"
        "# Kill-switch: max_rotations = 0 (or env MAGI_OLLAMA_MAX_ROTATIONS=0) disables rotation.\n"
        + "".join(
            f"{key:<{key_width}} = {value}  # {comment}\n" for key, value, comment in tunable_specs
        )
        + "\n"
    )
    return (
        "# MAGI Ollama backend - repo tier (./.claude/magi-ollama.toml)\n"
        "# Precedence (per key): env > this file (repo) > ~/.claude/magi-ollama.toml > built-in\n"
        "#\n"
        "# TWO MODES:\n"
        "#  A) Cloud (DEFAULT): the [models] trio below uses ':cloud' tags. Run\n"
        "#     `ollama signin` once on your local daemon -- cloud models then run\n"
        "#     WITHOUT downloading weights (only a tiny manifest).\n"
        "#  B) Local: replace the ':cloud' tags with local tags you have pulled\n"
        "#     (e.g. deepseek-r1:32b / gpt-oss:20b / qwen3:30b-thinking), OR point\n"
        "#     base_url at a remote/cloud /v1 and set api_key for the direct cloud API.\n\n"
        "# OpenAI-compatible base URL (Ollama or any OpenAI-compatible server).\n"
        "# Active local default; for Ollama Cloud point at the cloud /v1 and set api_key.\n"
        f'base_url = "{DEFAULT_BASE_URL}"\n\n'
        "# API key for cloud/authenticated endpoints. LOCAL Ollama needs none.\n"
        "# SECURITY: do not commit a real key.\n"
        '# api_key = "sk-..."\n\n' + tunables_lines + "[models]\n"
        "# Default trio = tier 'Maximo' (cloud, 3 distinct lineages). Needs `ollama signin` (mode A).\n"
        "# Each mage declares its lineage explicitly (v5.0.0); it is never inferred.\n"
        + model_lines
        + "\n# Fallback models, ordered strongest->weakest, one lineage each.\n"
        "# Each fallback tag needs `ollama pull <tag>` first (manifest only, no weights).\n"
        + fallback_lines
    )


def write_template(repo_root: str | None = None) -> str:
    """Write the template to ``<repo_root>/.claude/magi-ollama.toml``.

    Args:
        repo_root: Root directory of the repository. Defaults to ``os.getcwd()``.

    Returns:
        The absolute path of the written file.

    Raises:
        FileExistsError: if the target already exists (never clobbers).
    """
    if repo_root is None:
        repo_root = os.getcwd()
    path = os.path.join(repo_root, REPO_CONFIG_RELPATH)
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_template())
    return path
