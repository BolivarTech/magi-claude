#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 5.0.3
# Date: 2026-07-12
"""Check a magi-ollama.toml against the v5 schema. Reports; never rewrites.

This is the **assisted migration path** for the v5.0.0 breaking change, and it ships
with the plugin: the fail-closed error a v4 config raises, the README, the skill and
``docs/ollama-backend.md`` all tell the user to run it, so it must be here rather than
in the developer-local ``scripts/`` tree.

It reports and never rewrites, by design (spec decision #14): auto-converting a v4
string entry would require *inferring* the lineage from the tag, and a wrong guess
assigns a silently incorrect lineage -- two mages of one lineage produce a consensus
that only LOOKS like three independent perspectives. Failing with an actionable message
is strictly better than guessing.

Usage:
    python skills/magi/scripts/validate_magi_toml.py [path]   # default: .claude/magi-ollama.toml

Exit codes:
    0: the config is valid for v5.
    1: the config is invalid (the offending key is named).
    2: CLI misuse (no such file).
"""

import argparse
import sys
from pathlib import Path

from ollama_config import OllamaConfigError, resolve_config

DEFAULT_CONFIG_PATH = ".claude/magi-ollama.toml"

#: Shown whenever a config is rejected: the whole point of refusing to guess a lineage
#: is that the user is told exactly what to write instead.
V5_SHAPE_HINT = (
    "\nThe v5 schema declares a lineage per mage:\n"
    '  melchior = { model = "qwen3.5:397b-cloud", lineage = "alibaba" }\n'
    "\nThe lineage is NOT inferred: two mages sharing a lineage would give you a\n"
    "consensus that only LOOKS like three independent perspectives."
)


def main() -> int:
    """Validate the TOML at the given path and report the first problem found.

    Returns:
        ``0`` if the config is a valid v5 config, ``1`` if it is not.

    Raises:
        SystemExit: with code ``2`` on CLI misuse -- including a path that does not
            exist. Resolving a missing file would silently fall through to the built-in
            defaults and report ``OK``, telling the user their config is fine when it
            was never read: a fail-open in the one tool whose job is to say otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate a magi-ollama.toml against v5.")
    parser.add_argument("path", nargs="?", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    if not Path(args.path).is_file():
        parser.error(f"no such config file: {args.path}")

    try:
        config = resolve_config(repo_path=args.path, global_path=None, env={})
    except OllamaConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        print(V5_SHAPE_HINT, file=sys.stderr)
        return 1

    lineages = [spec.lineage for spec in config.models.values()]
    if len(set(lineages)) != len(lineages):
        print(f"INVALID: the trio shares a lineage: {lineages}", file=sys.stderr)
        print(V5_SHAPE_HINT, file=sys.stderr)
        return 1

    print(f"OK: {args.path} is a valid v5 config ({len(config.fallback)} fallback(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
