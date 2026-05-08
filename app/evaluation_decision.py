from __future__ import annotations

from typing import Any

DEFAULT_DECISION_CONFIG: dict[str, Any] = {
    "required_baselines": [
        "zero_return_no_trade",
        "buy_and_hold_equal_weight",
        "mean_return_long_flat",
        "last_return_long_flat",
    ],
    "min_trade_count": 5,
    "min_profit_factor": 1.0,
    "max_drawdown_pct_floor": -25.0,
    "min_active_exposure_pct": 5.0,
    "max_active_exposure_pct": 95.0,
    "min_hit_rate_pct": 0.0,
    "min_avg_return_pct": 0.0,
    "require_positive_total_return": True,
    "require_baseline_beat_rate_pct": 60.0,
}


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_DECISION_CONFIG)
    out.update((config or {}).get("validation_decision", {}) or {})
    return out


def _float(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(mapping.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _check(passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "details": details}


def _baseline_rows(baseline_comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("baseline")): row
        for row in baseline_comparison.get("rows", []) or []
        if row.get("baseline")
    }


def make_validation_decision(
    model_metrics: dict[str, Any],
    baseline_comparison: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn validation metrics into a conservative operational decision."""
    params = _cfg(config)
    rows_by_baseline = _baseline_rows(baseline_comparison or {})
    required = [str(name) for name in params["required_baselines"]]
    available_required = [name for name in required if name in rows_by_baseline]
    missing_required = [name for name in required if name not in rows_by_baseline]
    beaten_required = [
        name
        for name in available_required
        if bool(rows_by_baseline[name].get("beat_return", False))
    ]

    total_return = _float(model_metrics, "total_return_pct")
    max_drawdown = _float(model_metrics, "max_drawdown_pct")
    profit_factor = _float(model_metrics, "profit_factor")
    trade_count = _int(model_metrics, "trade_count")
    exposure = _float(
        model_metrics, "active_exposure_pct", _float(model_metrics, "avg_exposure_pct")
    )
    hit_rate = _float(model_metrics, "hit_rate_pct", _float(model_metrics, "win_rate"))
    avg_return = _float(
        model_metrics, "avg_return_pct", _float(model_metrics, "avg_trade_return_pct")
    )
    beat_rate = _float(baseline_comparison or {}, "beat_rate_pct")

    required_beat_rate = _float(params, "require_baseline_beat_rate_pct")
    baseline_passed = (
        bool(available_required)
        and not missing_required
        and beat_rate >= required_beat_rate
        and len(beaten_required) >= max(1, int(len(required) * required_beat_rate / 100.0))
    )
    positive_return_ok = total_return > 0 if bool(params["require_positive_total_return"]) else True

    checks = {
        "beats_required_baselines": _check(
            baseline_passed,
            {
                "required": required,
                "missing": missing_required,
                "beaten_required": beaten_required,
                "beat_rate_pct": beat_rate,
                "required_beat_rate_pct": required_beat_rate,
            },
        ),
        "positive_total_return": _check(
            positive_return_ok,
            {
                "total_return_pct": total_return,
                "required": bool(params["require_positive_total_return"]),
            },
        ),
        "max_drawdown_ok": _check(
            max_drawdown >= _float(params, "max_drawdown_pct_floor"),
            {
                "max_drawdown_pct": max_drawdown,
                "floor_pct": _float(params, "max_drawdown_pct_floor"),
            },
        ),
        "profit_factor_ok": _check(
            profit_factor >= _float(params, "min_profit_factor"),
            {"profit_factor": profit_factor, "minimum": _float(params, "min_profit_factor")},
        ),
        "enough_trades": _check(
            trade_count >= _int(params, "min_trade_count"),
            {"trade_count": trade_count, "minimum": _int(params, "min_trade_count")},
        ),
        "exposure_ok": _check(
            _float(params, "min_active_exposure_pct")
            <= exposure
            <= _float(params, "max_active_exposure_pct"),
            {
                "active_exposure_pct": exposure,
                "minimum": _float(params, "min_active_exposure_pct"),
                "maximum": _float(params, "max_active_exposure_pct"),
            },
        ),
        "hit_rate_ok": _check(
            hit_rate >= _float(params, "min_hit_rate_pct"),
            {"hit_rate_pct": hit_rate, "minimum": _float(params, "min_hit_rate_pct")},
        ),
        "avg_return_ok": _check(
            avg_return >= _float(params, "min_avg_return_pct"),
            {"avg_return_pct": avg_return, "minimum": _float(params, "min_avg_return_pct")},
        ),
    }

    critical = [
        "beats_required_baselines",
        "positive_total_return",
        "max_drawdown_ok",
        "profit_factor_ok",
        "enough_trades",
        "exposure_ok",
    ]
    passed_critical = sum(1 for name in critical if checks[name]["passed"])
    score = float(passed_critical / len(critical) * 100.0)

    if missing_required or not model_metrics:
        final_decision = "inconclusive"
    elif checks["enough_trades"]["passed"] is False or checks["exposure_ok"]["passed"] is False:
        final_decision = "inconclusive" if positive_return_ok else "reject"
    elif passed_critical == len(critical):
        final_decision = "promote"
    elif not positive_return_ok or (
        not baseline_passed and profit_factor < _float(params, "min_profit_factor")
    ):
        final_decision = "reject"
    else:
        final_decision = "observe"

    explanation = [
        f"Model beat {len(beaten_required)}/{len(required)} required baselines.",
        f"Trade count is {trade_count}; minimum is {_int(params, 'min_trade_count')}.",
        f"Profit factor is {profit_factor:.2f}; minimum is {_float(params, 'min_profit_factor'):.2f}.",
        f"Active exposure is {exposure:.1f}%.",
    ]
    if final_decision == "promote":
        explanation.append("Walk-forward validation is strong enough for promotion review.")
    elif final_decision == "observe":
        explanation.append(
            "Result is promising, but one or more critical checks needs observation."
        )
    elif final_decision == "reject":
        explanation.append("Economic evidence is not strong enough against required baselines.")
    else:
        explanation.append(
            "Sample or baseline evidence is insufficient for a methodological conclusion."
        )

    return {
        "final_decision": final_decision,
        "score": score,
        "checks": checks,
        "explanation": explanation,
    }
