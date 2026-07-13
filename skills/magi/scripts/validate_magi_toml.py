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

The verdict is about the FILE you pass -- nothing else. Neither the global
``~/.claude/magi-ollama.toml`` nor the ``MAGI_OLLAMA_*`` environment overrides are
applied, so the answer does not change with the shell you happen to be in. (Those
overrides do apply to a real run, and they fail loudly there if invalid.)

Usage:
    python skills/magi/scripts/validate_magi_toml.py [path]   # default: .claude/magi-ollama.toml

Exit codes:
    0: the config is valid for v5 (the resolved trio is echoed, so you can see it was read).
    1: the config is rejected -- unparseable TOML, a v4 schema entry, or a lineage that
       breaks the one-lineage-one-mage invariant. The offending key is named.
    2: CLI misuse -- the path is missing, is not a file, or cannot be read.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from ollama_config import OllamaConfigError, resolve_config
from ollama_preflight import OllamaPreflightError, check_config_offline

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
        ``0`` if the config is a valid v5 config, ``1`` if it is not (bad TOML syntax,
        or a schema/lineage rejection).

    Raises:
        SystemExit: with code ``2`` on CLI misuse -- a path that is missing, is not a
            file, or cannot be read. A missing path must never be a silent ``OK``:
            resolving it would fall through to the built-in defaults and validate
            THOSE, reporting that a config it never read is fine.
    """
    parser = argparse.ArgumentParser(description="Validate a magi-ollama.toml against v5.")
    parser.add_argument("path", nargs="?", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        parser.error(f"no such config file: {args.path}")
    if not path.is_file():
        parser.error(f"not a file: {args.path}")

    # Parse the syntax ourselves so a broken bracket is not answered with a lecture about
    # lineages, and so an unreadable file leaves as a message rather than a stack trace.
    try:
        with open(path, "rb") as handle:
            tomllib.load(handle)
    except OSError as exc:
        parser.error(f"cannot read {args.path}: {exc.strerror or exc}")
    except tomllib.TOMLDecodeError as exc:
        print(f"INVALID: {args.path} is not valid TOML: {exc}", file=sys.stderr)
        return 1

    try:
        # global_path="" (falsy), NOT None: None is resolve_config's sentinel for "use
        # ~/.claude/magi-ollama.toml", so a broken file in the user's HOME would make us
        # reject a config they never asked us about. The verdict is about THIS file only.
        config = resolve_config(repo_path=args.path, global_path="", env={})
    except OllamaConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        print(V5_SHAPE_HINT, file=sys.stderr)
        return 1

    # The product's own offline checks -- NOT a hand-rolled copy of them. A pre-run tool
    # that green-lights a config the preflight refuses to run is worse than no tool.
    try:
        for warning in check_config_offline(config):
            print(f"WARNING: {warning}", file=sys.stderr)
    except OllamaPreflightError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        print(V5_SHAPE_HINT, file=sys.stderr)
        return 1

    # Echo what was accepted: an empty file also resolves to a valid config (the built-in
    # defaults), so a bare "OK" cannot be told apart from an endorsement of YOUR trio.
    print(f"OK: {args.path} is a valid v5 config")
    for agent, spec in config.models.items():
        print(f"  {agent:<9} = {spec.model} [{spec.lineage}]")
    print(f"  fallback  = {len(config.fallback)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
