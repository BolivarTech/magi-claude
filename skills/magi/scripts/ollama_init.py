#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Scaffold ./.claude/magi-ollama.toml from canonical defaults (refuse-if-exists)."""

from __future__ import annotations

import os

from ollama_config import (
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACK,
    DEFAULT_INPUT_MARGIN_PCT,
    DEFAULT_MAX_ATTEMPTS_PER_MODEL,
    DEFAULT_MAX_PROBE_ATTEMPTS,
    DEFAULT_MAX_ROTATIONS,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_HEADROOM_TOKENS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    DEFAULT_STRICT_CONTEXT_GUARD,
)

REPO_CONFIG_RELPATH = os.path.join(".claude", "magi-ollama.toml")


def render_template() -> str:
    """Return the TOML template text (base_url active, api_key commented).

    Returns:
        A TOML-formatted string with a two-mode header, the local base_url
        active, api_key commented out, and the default trio models populated
        as ``{ model, lineage }`` tables (v5.0.0 schema).
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
    # v5.0.0 (R12/R17): the rotation/context-guard tunables are top-level scalars
    # (they apply to ALL mages), so TOML requires them BEFORE any [table] header.
    # Emitted as active keys at their built-in defaults from the DEFAULT_* constants
    # -- one source of truth with the resolver -- so an untouched scaffold round-trips
    # to the defaults while every knob (and the kill-switch) is visible and editable.
    strict_literal = "true" if DEFAULT_STRICT_CONTEXT_GUARD else "false"
    tunables_lines = (
        "# Rotation and context-window settings (apply to ALL mages; see docs/ollama-backend.md).\n"
        "# Kill-switch: max_rotations = 0 (or env MAGI_OLLAMA_MAX_ROTATIONS=0) disables rotation.\n"
        f"max_attempts_per_model = {DEFAULT_MAX_ATTEMPTS_PER_MODEL}  # tries per model before rotating to a fallback (>= 1)\n"
        f"max_rotations          = {DEFAULT_MAX_ROTATIONS}  # fallback models a mage may rotate through (0 disables rotation)\n"
        f"max_probe_attempts     = {DEFAULT_MAX_PROBE_ATTEMPTS}  # fallback candidates to size-check before a mage gives up (>= 1)\n"
        f"output_headroom_tokens = {DEFAULT_OUTPUT_HEADROOM_TOKENS}  # context tokens reserved for the model's answer plus its thinking\n"
        f"input_margin_pct       = {DEFAULT_INPUT_MARGIN_PCT}  # extra margin when checking the input fits a model's window, percent\n"
        f"strict_context_guard   = {strict_literal}  # if true, refuse a model whose context window cannot be measured\n"
        f"retry_backoff_seconds  = {DEFAULT_RETRY_BACKOFF_SECONDS}  # seconds to wait between transport retries (0 = no wait)\n\n"
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
