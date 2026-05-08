from __future__ import annotations

from typing import Any

from .utils import normalize_ticker


def _pct(value: Any, default: float) -> float:
    try:
        return float(default if value is None else value)
    except Exception:
        return float(default)


def _label_side(label: str, *, allow_short: bool, actionable: bool) -> tuple[str, str]:
    label = str(label or "NEUTRAL").upper()
    if "BUY" in label and actionable:
        return "LONG", "ENTER"
    if "SELL" in label:
        if allow_short and actionable:
            return "SHORT", "ENTER"
        return "FLAT", "EXIT_ONLY"
    return "FLAT", "WAIT"


def _max_hold_days(cfg: dict[str, Any], horizon: str) -> int:
    tm = cfg.get("trading", {}).get("trade_management", {}) or {}
    defaults = {"d1": 3, "d5": 10, "d20": 30}
    raw = tm.get("max_hold_days", defaults)
    if isinstance(raw, dict):
        return int(raw.get(str(horizon).lower(), defaults.get(str(horizon).lower(), 10)))
    return int(raw or defaults.get(str(horizon).lower(), 10))


def _target_midpoint(entry: float, target_final: float) -> float:
    return entry + ((target_final - entry) * 0.5)


def _risk_reward(entry: float, target: float, stop: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return float(reward / risk) if risk > 0 else 0.0


def build_trade_plan(
    cfg: dict[str, Any],
    *,
    ticker: str,
    policy: dict[str, Any],
    latest_price: float,
    latest_risk_pct: float = 0.0,
) -> dict[str, Any]:
    """Create the operational plan from a signal policy.

    Policy decides whether there is an edge. The trade plan decides how that edge
    is executed: sizing, partial, stop, breakeven and trailing rules.
    """
    policy = dict(policy or {})
    tcfg = cfg.get("trading", {}) or {}
    tm = tcfg.get("trade_management", {}) or {}
    allow_short = bool(
        policy.get(
            "allow_short",
            tcfg.get("allow_short", cfg.get("simulation", {}).get("allow_short", False)),
        )
    )
    actionable = bool(policy.get("actionable", False))
    label = str(policy.get("label", "NEUTRAL")).upper()
    side, action = _label_side(label, allow_short=allow_short, actionable=actionable)

    entry = float(latest_price or 0.0)
    horizon = str(policy.get("horizon", "d1")).lower()
    target_final = _pct(policy.get("target_price"), entry)
    stop_initial = _pct(policy.get("stop_loss_price"), entry)
    target_1 = _pct(policy.get("target_partial"), _target_midpoint(entry, target_final))
    breakeven_trigger = _pct(policy.get("breakeven_trigger"), target_1)
    partial_pct = max(
        0.0,
        min(
            100.0,
            _pct(tm.get("partial_take_profit_pct", tcfg.get("partial_take_profit_pct")), 50.0),
        ),
    )
    stop_distance_pct = abs((entry - stop_initial) / entry) * 100.0 if entry > 0 else 0.0
    risk_pct = max(0.0, _pct(latest_risk_pct, 0.0))
    trailing_multiple = max(0.0, _pct(tm.get("trailing_distance_risk_multiple"), 0.75))
    trailing_distance_pct = (
        stop_distance_pct * trailing_multiple
        if stop_distance_pct > 0
        else risk_pct * trailing_multiple
    )

    rr = _pct(policy.get("risk_reward_ratio"), _risk_reward(entry, target_final, stop_initial))
    plan = {
        "ticker": normalize_ticker(ticker),
        "label": label,
        "action": action,
        "side": side,
        "horizon": horizon,
        "entry_price": entry,
        "target_1": float(target_1),
        "target_final": float(target_final),
        "stop_initial": float(stop_initial),
        "stop_current": float(stop_initial),
        "breakeven_trigger": float(breakeven_trigger),
        "partial_take_profit_pct": float(partial_pct),
        "partial_executed": False,
        "breakeven_after_partial": bool(tm.get("breakeven_after_partial", True)),
        "trailing_enabled": bool(tm.get("trailing_stop_enabled", True)),
        "trailing_active": False,
        "trailing_distance_pct": float(trailing_distance_pct),
        "max_hold_days": _max_hold_days(cfg, horizon),
        "position_size": int(policy.get("position_size", 0) or 0),
        "risk_reward_ratio": float(rr),
        "actionable": bool(action == "ENTER"),
        "notes": list(policy.get("reasons", []) or []),
    }
    if action == "EXIT_ONLY":
        plan["notes"].append("Signal is informational/exit-only; no new short entry.")
    return plan


def trade_plan_from_signal(cfg: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    existing = signal.get("trade_plan")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    return build_trade_plan(
        cfg,
        ticker=str(signal.get("ticker", "N/A")),
        policy=signal.get("policy", {}) or {},
        latest_price=float(signal.get("latest_price", 0.0) or 0.0),
        latest_risk_pct=float(
            ((signal.get("dataset_meta", {}) or {}).get("latest_risk_pct", 0.0)) or 0.0
        ),
    )


def is_long_plan(plan: dict[str, Any], shares: int | None = None) -> bool:
    if shares is not None and int(shares) != 0:
        return int(shares) > 0
    return str(plan.get("side", "")).upper() == "LONG"


def hit_target(side: str, price: float, target: float) -> bool:
    if target <= 0:
        return False
    return (
        float(price) >= float(target)
        if str(side).upper() == "LONG"
        else float(price) <= float(target)
    )


def hit_stop(side: str, price: float, stop: float) -> bool:
    if stop <= 0:
        return False
    return (
        float(price) <= float(stop) if str(side).upper() == "LONG" else float(price) >= float(stop)
    )


def partial_signed_shares(shares: int, partial_pct: float) -> int:
    shares = int(shares)
    if shares == 0:
        return 0
    qty = int(abs(shares) * max(0.0, min(100.0, float(partial_pct))) / 100.0)
    qty = max(1, min(abs(shares), qty))
    return qty if shares > 0 else -qty


def next_trailing_stop(
    plan: dict[str, Any], *, side: str, price: float, current_stop: float
) -> float:
    if not bool(plan.get("trailing_enabled", True)):
        return float(current_stop)
    distance_pct = float(plan.get("trailing_distance_pct", 0.0) or 0.0)
    if distance_pct <= 0:
        return float(current_stop)
    price = float(price)
    if str(side).upper() == "LONG":
        candidate = price * (1.0 - distance_pct / 100.0)
        return max(float(current_stop), candidate)
    candidate = price * (1.0 + distance_pct / 100.0)
    return min(float(current_stop), candidate) if float(current_stop) > 0 else candidate
