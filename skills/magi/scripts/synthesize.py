#!/usr/bin/env python3
# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-04-01
"""MAGI Synthesis Engine — facade module.

Re-exports from validate, consensus, and reporting sub-modules for
backward compatibility.  Run directly for CLI usage:

    python synthesize.py <agent1.json> <agent2.json> [agent3.json] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bootstrap: see CLAUDE.md "Open technical debt / synthesize import gap [LOCKED]".
# Direct invocation and pytest already cover this; ``python -m
# skills.magi.scripts.synthesize`` does not.
_SCRIPT_DIR = str(Path(__file__).parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Re-export public API so existing ``from synthesize import ...`` still works.
from validate import (  # noqa: E402
    VALID_AGENTS,
    VALID_SEVERITIES,
    VALID_VERDICTS,
    ValidationError,
    clean_title,
    load_agent_output,
)
from consensus import (  # noqa: E402
    VERDICT_WEIGHT,
    determine_consensus,
)
from reporting import (  # noqa: E402
    AGENT_TITLES,
    format_banner,
    format_report,
)

__all__ = [
    "AGENT_TITLES",
    "VALID_AGENTS",
    "VALID_SEVERITIES",
    "VALID_VERDICTS",
    "VERDICT_WEIGHT",
    "ValidationError",
    "clean_title",
    "determine_consensus",
    "format_banner",
    "format_report",
    "load_agent_output",
]


def main() -> None:
    """Run MAGI synthesis from command line."""
    parser = argparse.ArgumentParser(description="MAGI Synthesis Engine")
    parser.add_argument(
        "agent_files", nargs="+", help="Paths to agent JSON output files (2-3 required)"
    )
    parser.add_argument("--output", "-o", help="Save JSON report to file")
    parser.add_argument(
        "--format",
        choices=["text", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    if len(args.agent_files) < 2 or len(args.agent_files) > 3:
        parser.error("Expected 2-3 agent files")

    agents = []
    for filepath in args.agent_files:
        try:
            agents.append(load_agent_output(filepath))
        except ValidationError as e:
            print(f"WARNING: Skipping {filepath}: {e}", file=sys.stderr)

    if len(agents) < 2:
        print("ERROR: Need at least 2 valid agent outputs", file=sys.stderr)
        sys.exit(1)

    consensus = determine_consensus(agents)

    if args.format in ("text", "both"):
        print(format_report(agents, consensus))

    if args.format in ("json", "both"):
        report = {
            "agents": agents,
            "consensus": consensus,
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
