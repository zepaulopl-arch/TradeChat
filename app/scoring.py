from __future__ import annotations

import math
from typing import Any

HORIZON_DAYS = {"d1": 1, "d5": 5, "d20": 20}
SIGNAL_PRIORITY_MAP = {
    "STRONG BUY": 100,
    "BUY": 80,
    "STRONG SELL": 70,
    "SELL": 60,
    "NEUTRAL": 10,
}


def signal_label(signal: dict[str, Any]) -> str:
    policy = signal.get("policy", {}) or {}
    return str(policy.get("label", "NEUTRAL")).upper()


def trigger_horizon(signal: dict[str, Any]) -> str:
    policy = signal.get("policy", {}) or {}
    return str(policy.get("horizon", "d1")).lower()


def trigger_result(signal: dict[str, Any]) -> dict[str, Any]:
    horizons = signal.get("horizons", {}) or {}
    horizon = trigger_horizon(signal)
    return horizons.get(horizon, horizons.get("d1", {})) or {}


def signal_priority(signal: dict[str, Any]) -> int:
    return SIGNAL_PRIORITY_MAP.get(signal_label(signal), 0)


def signal_score(signal: dict[str, Any]) -> float:
    policy = signal.get("policy", {}) or {}
    horizon = trigger_horizon(signal)
    horizon_days = HORIZON_DAYS.get(horizon, 1)
    triggered = trigger_result(signal)
    quality_pct = float(policy.get("quality_pct", policy.get("confidence_pct", 0.0)) or 0.0)
    triggered_ret_pct = abs(float(triggered.get("prediction_return", 0.0) or 0.0) * 100.0)
    return quality_pct * (triggered_ret_pct / math.sqrt(horizon_days)) if horizon_days > 0 else 0.0


def signal_side(label_or_signal: str | dict[str, Any]) -> str | None:
    label = (
        signal_label(label_or_signal)
        if isinstance(label_or_signal, dict)
        else str(label_or_signal).upper()
    )
    if "BUY" in label:
        return "LONG"
    if "SELL" in label:
        return "SHORT"
    return None


def is_actionable_signal(signal: dict[str, Any]) -> bool:
    policy = signal.get("policy", {}) or {}
    if policy.get("actionable") is False:
        return False
    plan = signal.get("trade_plan", {}) or {}
    if plan and str(plan.get("action", "ENTER")).upper() != "ENTER":
        return False
    return signal_side(signal) is not None
