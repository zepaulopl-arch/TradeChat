from __future__ import annotations

from typing import Any

from .types import ExecutionCostConfig


def execution_cost_config(cfg: dict[str, Any]) -> ExecutionCostConfig:
    sim_cfg = cfg.get("simulation", {}) or {}
    costs = dict(sim_cfg.get("costs", {}) or {})
    return ExecutionCostConfig(
        fee_mode=costs.get("fee_mode", sim_cfg.get("fee_mode")),
        fee_amount=float(costs.get("fee_amount", sim_cfg.get("fee_amount", 0.0)) or 0.0),
        slippage_pct=float(costs.get("slippage_pct", sim_cfg.get("slippage_pct", 0.0)) or 0.0),
    )
