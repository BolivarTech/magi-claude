# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Per-run cost aggregation for MAGI.

Sums the authoritative ``total_cost_usd`` that ``claude -p --output-format
json`` reports in each agent's raw envelope (``{agent}.raw.json``). Unlike
panóptico (which estimates from token counts because its other backends do
not return cost), MAGI has ground-truth cost and only needs to sum it.
Total: any read/parse error degrades to 0 for that agent — never raises.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _agent_cost(output_dir: str, agent: str) -> float:
    """Return *agent*'s ``total_cost_usd`` from its raw envelope, or 0.0."""
    path = os.path.join(output_dir, f"{agent}.raw.json")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        value = data.get("total_cost_usd")
        return (
            float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return 0.0


def aggregate_cost(output_dir: str, agents: list[str]) -> dict[str, Any]:
    """Sum per-agent ``total_cost_usd`` into ``{per_agent, total_usd}``.

    Fail-safe: a missing/corrupt envelope contributes 0 for that agent.

    Args:
        output_dir: Directory containing ``{agent}.raw.json`` files.
        agents: List of agent names to aggregate costs for.

    Returns:
        Dict with ``per_agent`` mapping agent names to individual costs,
        and ``total_usd`` with the rounded sum of all agent costs.
    """
    per_agent = {agent: _agent_cost(output_dir, agent) for agent in agents}
    return {"per_agent": per_agent, "total_usd": round(sum(per_agent.values()), 6)}
