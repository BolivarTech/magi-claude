#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-06-06
"""Scaffold ./.claude/magi-ollama.toml from canonical defaults (refuse-if-exists)."""

from __future__ import annotations

import os

from ollama_config import DEFAULT_BASE_URL, DEFAULT_MODELS

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
        '# api_key = "sk-..."\n\n'
        "[models]\n"
        "# Default trio = tier 'Maximo' (cloud, 3 distinct lineages). Needs `ollama signin` (mode A).\n"
        "# Each mage declares its lineage explicitly (v5.0.0); it is never inferred.\n"
        + model_lines
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
