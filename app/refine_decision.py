from __future__ import annotations

from typing import Any

PROFILE_TO_REMOVED_FAMILY = {
    "technical_only": "context+fundamentals+sentiment",
    "no_context": "context",
    "no_fundamentals": "fundamentals",
    "no_sentiment": "sentiment",
}


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _decision_from_economic_delta(
    *,
    profile: str,
    return_delta: float,
    drawdown_delta: float,
    profit_factor_delta: float,
    trade_count: int,
    min_trade_count: int,
) -> tuple[str, list[str]]:
    rationale: list[str] = []
    if profile == "full":
        return "reference", ["Full profile is the reference configuration."]
    if trade_count < min_trade_count:
        return "inconclusive", [f"Only {trade_count} trades; minimum is {min_trade_count}."]
    if return_delta > 0 and drawdown_delta >= 0 and profit_factor_delta >= 0:
        rationale.append("Removal improved return without worsening drawdown or profit factor.")
        rationale.append("Treat as a remove candidate, not as an automatic change.")
        return "remove_candidate", rationale
    if return_delta < 0 or profit_factor_delta < 0:
        rationale.append("Removal worsened return or profit factor versus full.")
        return "keep_family", rationale
    rationale.append("Removal impact is small or mixed.")
    return "observe", rationale


def _decision_from_holdout_delta(
    *,
    profile: str,
    mae_delta: float,
    quality_delta: float,
    selected_feature_count: int,
) -> tuple[str, list[str]]:
    if profile == "full":
        return "reference", ["Full profile is the reference configuration."]
    if selected_feature_count <= 0:
        return "inconclusive", ["Removal produced no selected features."]
    if mae_delta > 0:
        return "keep_family", ["Removal increased validation MAE versus full."]
    if mae_delta < 0 and quality_delta >= 0:
        return "observe", [
            "Removal improved holdout MAE, but economic validation is still required."
        ]
    return "observe", [
        "Holdout result is mixed; use walk-forward removal before changing defaults."
    ]


def build_refine_decision_matrix(
    rows: list[dict[str, Any]],
    *,
    min_trade_count: int = 5,
) -> list[dict[str, Any]]:
    """Compare controlled-removal profiles against the full profile."""
    if not rows:
        return []

    has_economic_metrics = any("total_return_pct" in row for row in rows)
    decisions: list[dict[str, Any]] = []

    if has_economic_metrics:
        base = next((row for row in rows if row.get("profile") == "full"), rows[0])
        base_return = _float(base, "total_return_pct")
        base_drawdown = _float(base, "max_drawdown_pct")
        base_pf = _float(base, "profit_factor")
        base_trades = _int(base, "trade_count")
        base_exposure = _float(base, "active_exposure_pct")
        for row in rows:
            profile = str(row.get("profile", "n/a"))
            return_delta = _float(row, "total_return_pct") - base_return
            drawdown_delta = _float(row, "max_drawdown_pct") - base_drawdown
            pf_delta = _float(row, "profit_factor") - base_pf
            trade_delta = _int(row, "trade_count") - base_trades
            exposure_delta = _float(row, "active_exposure_pct") - base_exposure
            decision, rationale = _decision_from_economic_delta(
                profile=profile,
                return_delta=return_delta,
                drawdown_delta=drawdown_delta,
                profit_factor_delta=pf_delta,
                trade_count=_int(row, "trade_count"),
                min_trade_count=min_trade_count,
            )
            decisions.append(
                {
                    "profile": profile,
                    "removed_family": PROFILE_TO_REMOVED_FAMILY.get(profile, "none"),
                    "return_delta_pct": float(return_delta),
                    "drawdown_delta_pct": float(drawdown_delta),
                    "profit_factor_delta": float(pf_delta),
                    "trade_count_delta": int(trade_delta),
                    "exposure_delta_pct": float(exposure_delta),
                    "decision": decision,
                    "rationale": rationale,
                }
            )
        return decisions

    grouped: dict[tuple[str, str], dict[str, Any]] = {
        (str(row.get("ticker", "")), str(row.get("horizon", "")), str(row.get("profile", ""))): row
        for row in rows
    }
    for row in rows:
        ticker = str(row.get("ticker", ""))
        horizon = str(row.get("horizon", ""))
        profile = str(row.get("profile", "n/a"))
        base = grouped.get((ticker, horizon, "full"), row)
        mae_delta = _float(row, "mae_return") - _float(base, "mae_return")
        quality_delta = _float(row, "quality") - _float(base, "quality")
        decision, rationale = _decision_from_holdout_delta(
            profile=profile,
            mae_delta=mae_delta,
            quality_delta=quality_delta,
            selected_feature_count=_int(row, "selected_feature_count"),
        )
        decisions.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "profile": profile,
                "removed_family": PROFILE_TO_REMOVED_FAMILY.get(profile, "none"),
                "mae_delta": float(mae_delta),
                "quality_delta": float(quality_delta),
                "return_delta_pct": "",
                "drawdown_delta_pct": "",
                "profit_factor_delta": "",
                "trade_count_delta": "",
                "exposure_delta_pct": "",
                "decision": decision,
                "rationale": rationale,
            }
        )
    return decisions
