# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-05-23
"""Tests for cost.py — per-run cost aggregation from agent envelopes."""

from __future__ import annotations

import json
from typing import Any


class TestAggregateCost:
    def _raw(self, tmp_path, agent: str, cost: float | None) -> None:
        p = tmp_path / f"{agent}.raw.json"
        body: dict[str, Any] = {"type": "result", "result": "{}"}
        if cost is not None:
            body["total_cost_usd"] = cost
        p.write_text(json.dumps(body), encoding="utf-8")

    def test_sums_total_cost_usd(self, tmp_path):
        from cost import aggregate_cost

        self._raw(tmp_path, "melchior", 1.0)
        self._raw(tmp_path, "balthasar", 0.5)
        self._raw(tmp_path, "caspar", 0.25)
        out = aggregate_cost(str(tmp_path), ["melchior", "balthasar", "caspar"])
        assert out["total_usd"] == 1.75
        assert out["per_agent"]["melchior"] == 1.0

    def test_missing_cost_field_treated_as_zero(self, tmp_path):
        from cost import aggregate_cost

        self._raw(tmp_path, "melchior", 1.0)
        self._raw(tmp_path, "balthasar", None)  # no total_cost_usd
        out = aggregate_cost(str(tmp_path), ["melchior", "balthasar"])
        assert out["total_usd"] == 1.0
        assert out["per_agent"]["balthasar"] == 0.0

    def test_missing_or_bad_file_is_fail_safe(self, tmp_path):
        from cost import aggregate_cost

        out = aggregate_cost(str(tmp_path), ["melchior"])  # no raw file at all
        assert out["total_usd"] == 0.0 and out["per_agent"]["melchior"] == 0.0
