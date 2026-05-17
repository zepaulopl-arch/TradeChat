from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.runtime_policy_config import (
    load_runtime_policy_config,
)

OPERATIONAL_PROFILE = "active"
POLICY_TYPE = "asset_specific_active"


def _safe_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

    if value is pd.NA:
        return None

    return value


def _safe_float(value: Any, default: float | None = None) -> float | None:
    value = _safe_value(value)

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    if text.lower() in {"inf", "+inf", "infinity"}:
        return math.inf

    try:
        number = float(text)
    except Exception:
        return default

    if math.isnan(number):
        return default

    return number


def _safe_int(value: Any, default: int | None = None) -> int | None:
    number = _safe_float(value, None)

    if number is None or math.isinf(number):
        return default

    return int(number)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = dict(base or {})

    for key, value in (override or {}).items():
        current = result.get(key)

        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value

    return result


def _extract_evidence(row: pd.Series) -> dict[str, Any]:
    preferred_columns = [
        "phase",
        "ticker",
        "scope",
        "profile",
        "policy",
        "decision",
        "score",
        "return_pct",
        "trades",
        "drawdown_pct",
        "hit_pct",
        "avg_trade_pct",
        "profit_factor",
        "profit_factor_display",
        "turnover_pct",
        "exposure_pct",
        "cost",
        "gross_plus",
        "gross_minus",
        "net",
        "before_cost_pct",
        "after_cost_pct",
        "beat_rate_pct",
        "sample",
        "n_assets",
        "log_path",
        "sharpe",
        "avg_return_pct",
        "win_rate",
        "threshold",
        "horizon",
        "risk_reward_ratio",
        "matrix_rr",
        "rr",
        "eligibility_status",
        "ineligible_reason",
        "data_rows",
    ]

    evidence: dict[str, Any] = {}

    for column in preferred_columns:
        if column not in row.index:
            continue

        evidence[column] = _safe_value(row[column])

    return evidence


def _text_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row.index:
        return default

    value = _safe_value(row[column])

    if value is None:
        return default

    return str(value).strip()


def _matrix_decision(row: pd.Series) -> str:
    return _text_value(row, "decision", "n/a").upper()


def _is_ineligible_row(row: pd.Series) -> bool:
    status = _text_value(row, "eligibility_status").lower()
    decision = _matrix_decision(row)

    return status in {"ineligible_data", "skip_data"} or decision in {
        "INELIGIBLE_DATA",
        "SKIP_DATA",
    }


def _ineligible_reason(row: pd.Series) -> str:
    reason = _text_value(row, "ineligible_reason")

    if reason:
        return reason

    return "insufficient history"


def _actionable_decisions(promotion_cfg: dict[str, Any]) -> set[str]:
    guard_cfg = promotion_cfg.get("runtime_decision_guard", {}) or {}
    decisions = guard_cfg.get("decisions", {}) or {}
    actionable = {
        str(decision).upper()
        for decision, rule in decisions.items()
        if str((rule or {}).get("max_signal", "")).upper() == "ACTIONABLE"
    }

    return actionable or {"APPROVE"}


def _profile_overrides_from_config(
    promotion_cfg: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    runtime_overrides = promotion_cfg.get("runtime_overrides", {}) or {}

    if not bool(runtime_overrides.get("enabled", False)):
        return {}

    profiles = runtime_overrides.get("profiles", {}) or {}

    return profiles.get(str(profile), {}) or {}


def _preferred_horizon(row: pd.Series) -> str:
    horizon = _text_value(row, "horizon").lower()

    if horizon in {"d1", "d5", "d20"}:
        return horizon

    trades = _safe_int(row["trades"], 0) if "trades" in row.index else 0

    if trades is not None and trades >= 30:
        return "d5"

    return "d20"


def _profit_factor(row: pd.Series) -> float | None:
    value = None

    if "profit_factor" in row.index:
        value = _safe_float(row["profit_factor"], None)

    if value is None and "profit_factor_display" in row.index:
        value = _safe_float(row["profit_factor_display"], None)

    return value


def _matrix_rr(row: pd.Series) -> float | None:
    for column in (
        "matrix_rr",
        "risk_reward_ratio",
        "rr",
        "reward_risk_ratio",
    ):
        if column in row.index:
            value = _safe_float(row[column], None)
            if value is not None:
                return value

    return None


def _asset_specific_active_overrides(
    row: pd.Series,
    *,
    promotion_cfg: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    adaptive_cfg = promotion_cfg.get("asset_specific_active", {}) or {}
    bounds = adaptive_cfg.get("bounds", {}) or {}

    base_threshold = _safe_float(row["threshold"], None) if "threshold" in row.index else None
    avg_trade = _safe_float(row["avg_trade_pct"], None) if "avg_trade_pct" in row.index else None

    if base_threshold is None and avg_trade is not None:
        base_threshold = abs(avg_trade) * 0.35

    if base_threshold is None:
        base_threshold = float(bounds.get("default_buy_return_pct", 0.08))

    buy_threshold = round(
        _clamp(
            abs(float(base_threshold)),
            float(bounds.get("min_buy_return_pct", 0.04)),
            float(bounds.get("max_buy_return_pct", 0.30)),
        ),
        4,
    )

    score = _safe_float(row["score"], 70.0) or 70.0
    hit_pct = _safe_float(row["hit_pct"], 55.0) or 55.0
    confidence = 0.46 - (float(score) / 500.0) - max(0.0, float(hit_pct) - 55.0) / 1000.0
    min_confidence = round(
        _clamp(
            confidence,
            float(bounds.get("min_confidence_pct_floor", 0.32)),
            float(bounds.get("min_confidence_pct_cap", 0.45)),
        ),
        4,
    )

    rr_value = _matrix_rr(row)
    pf = _profit_factor(row)

    if rr_value is not None and not math.isinf(rr_value):
        min_rr = float(rr_value) * 0.50
    elif pf is not None and math.isinf(pf):
        min_rr = 0.12
    elif pf is not None:
        min_rr = 0.42 - min(max(float(pf), 1.0), 3.5) * 0.08
    else:
        min_rr = 0.25

    min_rr = round(
        _clamp(
            min_rr,
            float(bounds.get("min_rr_floor", 0.10)),
            float(bounds.get("min_rr_cap", 0.60)),
        ),
        4,
    )

    drawdown = abs(_safe_float(row["drawdown_pct"], 0.0) or 0.0)
    exposure = _safe_float(row["exposure_pct"], 0.0) or 0.0
    risk_budget = 1.0 - (float(drawdown) / 25.0) - (float(exposure) / 250.0)
    risk_budget = round(
        _clamp(
            risk_budget,
            float(bounds.get("risk_per_trade_pct_floor", 0.25)),
            float(bounds.get("risk_per_trade_pct_cap", 1.0)),
        ),
        4,
    )

    min_trades = int(constraints.get("min_trades", 15))
    max_drawdown = float(constraints.get("max_drawdown_pct", 15.0))
    max_exposure = float(constraints.get("max_exposure_pct", 80.0))
    max_position_pct = round(
        _clamp(
            40.0 - (float(exposure) * 0.25) - (float(drawdown) * 0.50),
            float(bounds.get("max_position_pct_floor", 5.0)),
            float(bounds.get("max_position_pct_cap", 40.0)),
        ),
        4,
    )
    matrix_decision = _matrix_decision(row)

    return {
        "buy_return_pct": buy_threshold,
        "sell_return_pct": -buy_threshold,
        "min_confidence_pct": min_confidence,
        "preferred_horizon": _preferred_horizon(row),
        "risk_management": {
            "min_rr_threshold": min_rr,
            "risk_per_trade_pct": risk_budget,
            "max_position_pct": max_position_pct,
        },
        "validation_constraints": {
            "min_trades": min_trades,
            "max_drawdown_pct": max_drawdown,
            "max_exposure_pct": max_exposure,
            "observed_trades": _safe_int(row["trades"], None)
            if "trades" in row.index
            else None,
            "observed_drawdown_pct": _safe_float(row["drawdown_pct"], None)
            if "drawdown_pct" in row.index
            else None,
            "observed_exposure_pct": _safe_float(row["exposure_pct"], None)
            if "exposure_pct" in row.index
            else None,
            "observed_profit_factor": _safe_value(row["profit_factor"])
            if "profit_factor" in row.index
            else None,
        },
        "asset_specific_active": {
            "enabled": True,
            "source": "policy_matrix",
            "policy_type": POLICY_TYPE,
            "matrix_decision": matrix_decision,
            "matrix_profile": _text_value(row, "profile", OPERATIONAL_PROFILE),
            "score": _safe_value(row["score"]) if "score" in row.index else None,
            "profit_factor": _safe_value(row["profit_factor"])
            if "profit_factor" in row.index
            else None,
            "trades": _safe_value(row["trades"]) if "trades" in row.index else None,
        },
    }


def _runtime_overrides_for_asset(
    row: pd.Series,
    *,
    promotion_cfg: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    base = _profile_overrides_from_config(
        promotion_cfg,
        OPERATIONAL_PROFILE,
    )
    asset_specific = _asset_specific_active_overrides(
        row,
        promotion_cfg=promotion_cfg,
        constraints=constraints,
    )

    return _deep_merge(
        base,
        asset_specific,
    )


def _passes_constraints(
    row: pd.Series,
    *,
    max_drawdown: float,
    min_trades: int,
    min_sharpe: float,
    max_exposure: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if "drawdown_pct" in row.index:
        drawdown = _safe_value(row["drawdown_pct"])

        if drawdown is not None and float(drawdown) > max_drawdown:
            reasons.append(f"drawdown_pct {float(drawdown):.2f} > {max_drawdown:.2f}")

    if "trades" in row.index:
        trades = _safe_value(row["trades"])

        if trades is not None and int(trades) < min_trades:
            reasons.append(f"trades {int(trades)} < {min_trades}")

    if "sharpe" in row.index:
        sharpe = _safe_value(row["sharpe"])

        if sharpe is not None and float(sharpe) < min_sharpe:
            reasons.append(f"sharpe {float(sharpe):.2f} < {min_sharpe:.2f}")

    if "exposure_pct" in row.index:
        exposure = _safe_value(row["exposure_pct"])

        if exposure is not None and float(exposure) > max_exposure:
            reasons.append(f"exposure_pct {float(exposure):.2f} > {max_exposure:.2f}")

    return len(reasons) == 0, reasons


def _sort_candidates(
    df: pd.DataFrame,
    *,
    metric: str,
    tie_break: list[str],
) -> pd.DataFrame:
    sort_columns = [metric]

    for column in tie_break:
        if column in df.columns and column not in sort_columns:
            sort_columns.append(column)

    ascending: list[bool] = []

    for column in sort_columns:
        if column in {
            "drawdown_pct",
            "max_drawdown",
        }:
            ascending.append(True)
        else:
            ascending.append(False)

    return df.sort_values(
        sort_columns,
        ascending=ascending,
    )


def _sort_columns(
    df: pd.DataFrame,
    *,
    metric: str,
    tie_break: list[str],
) -> list[str]:
    columns = [metric]

    for column in tie_break:
        if column in df.columns and column not in columns:
            columns.append(column)

    return columns


def promote_policy(matrix_dir: str) -> None:
    runtime_cfg = load_runtime_policy_config()

    promotion_cfg = runtime_cfg.get("promotion", {}) or {}

    constraints = promotion_cfg.get("constraints", {}) or {}

    tie_break = list(promotion_cfg.get("tie_break", []) or [])

    metric = str(promotion_cfg.get("metric", "sharpe"))

    mode = str(promotion_cfg.get("mode", "by-asset"))

    max_drawdown = float(constraints.get("max_drawdown_pct", 15.0))

    min_trades = int(constraints.get("min_trades", 20))

    min_sharpe = float(constraints.get("min_sharpe", 0.80))

    max_exposure = float(constraints.get("max_exposure_pct", 100.0))

    decisions_actionable = _actionable_decisions(promotion_cfg)

    matrix_path = Path(matrix_dir)

    summary_path = matrix_path / "validation_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"validation_summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    if mode != "by-asset":
        raise NotImplementedError("only by-asset mode implemented")

    if "ticker" not in df.columns:
        raise ValueError("validation_summary.csv must contain ticker column")

    if "profile" not in df.columns:
        raise ValueError("validation_summary.csv must contain profile column")

    if metric not in df.columns:
        raise ValueError(f"metric column not found: {metric}")

    assets: dict[str, dict[str, Any]] = {}

    grouped = df.groupby("ticker")

    for ticker, group in grouped:
        local = group.copy()

        local_sorted = _sort_candidates(
            local,
            metric=metric,
            tie_break=tie_break,
        )

        ineligible_rows = [
            (index, row)
            for index, row in local_sorted.iterrows()
            if _is_ineligible_row(row)
        ]

        eligible_sorted = local_sorted[
            ~local_sorted.apply(
                _is_ineligible_row,
                axis=1,
            )
        ]

        sort_columns = _sort_columns(
            local,
            metric=metric,
            tie_break=tie_break,
        )

        valid_rows: list[tuple[int, pd.Series, list[str]]] = []
        rejected_rows: list[tuple[int, pd.Series, list[str]]] = []

        for index, row in eligible_sorted.iterrows():
            passed, rejection_reasons = _passes_constraints(
                row,
                max_drawdown=max_drawdown,
                min_trades=min_trades,
                min_sharpe=min_sharpe,
                max_exposure=max_exposure,
            )

            if passed:
                valid_rows.append((index, row, rejection_reasons))
            else:
                rejected_rows.append((index, row, rejection_reasons))

        if valid_rows:
            _, best, rejection_reasons = valid_rows[0]
            promoted = True
            promotion_status = "promoted"
            source = "policy_matrix"
            ineligible_data = False
        elif rejected_rows:
            _, best, rejection_reasons = rejected_rows[0]
            promoted = False
            promotion_status = "rejected_by_constraints"
            source = "policy_matrix"
            ineligible_data = False
        elif ineligible_rows:
            _, best = ineligible_rows[0]
            promoted = False
            promotion_status = "ineligible_data"
            rejection_reasons = [_ineligible_reason(best)]
            source = "data_eligibility"
            ineligible_data = True
        else:
            _, best = next(iter(local_sorted.iterrows()))
            promoted = False
            promotion_status = "ineligible_data"
            rejection_reasons = ["insufficient history"]
            source = "data_eligibility"
            ineligible_data = True

        matrix_profile = str(best["profile"])
        profile = OPERATIONAL_PROFILE
        decision = _matrix_decision(best)
        actionable_candidate = (
            promoted
            and not ineligible_data
            and decision in decisions_actionable
        )
        blocker = (
            str(rejection_reasons[0])
            if rejection_reasons
            else (
                f"Matrix decision is {decision}"
                if promoted and not actionable_candidate and decision != "N/A"
                else None
            )
        )

        assets[str(ticker)] = {
            "profile": profile,
            "policy_type": POLICY_TYPE,
            "source": source,
            "evaluated": True,
            "ineligible_data": ineligible_data,
            "promoted": promoted,
            "actionable_candidate": actionable_candidate,
            "promotion_status": promotion_status,
            "rejection_reasons": rejection_reasons,
            "blocker": blocker,
            "overrides": (
                _runtime_overrides_for_asset(
                    best,
                    promotion_cfg=promotion_cfg,
                    constraints=constraints,
                )
                if promoted and not ineligible_data
                else {}
            ),
            "selection": {
                "metric": metric,
                "mode": mode,
                "sort_columns": sort_columns,
                "matrix_profile": matrix_profile,
                "policy_type": POLICY_TYPE,
                "asset_specific_parameters": (
                    [
                        "buy_return_pct",
                        "sell_return_pct",
                        "min_confidence_pct",
                        "preferred_horizon",
                        "risk_management.min_rr_threshold",
                        "risk_management.risk_per_trade_pct",
                        "risk_management.max_position_pct",
                        "validation_constraints.min_trades",
                        "validation_constraints.max_drawdown_pct",
                        "validation_constraints.max_exposure_pct",
                    ]
                    if promoted and not ineligible_data
                    else []
                ),
            },
            "evidence": _extract_evidence(best),
        }

    promoted_count = sum(1 for item in assets.values() if bool(item.get("promoted", False)))

    rejected_count = sum(1 for item in assets.values() if not bool(item.get("promoted", False)))

    ineligible_count = sum(
        1 for item in assets.values() if bool(item.get("ineligible_data", False))
    )

    actionable_candidate_count = sum(
        1 for item in assets.values() if bool(item.get("actionable_candidate", False))
    )

    output = {
        "selection_mode": mode,
        "policy_type": POLICY_TYPE,
        "operational_profile": OPERATIONAL_PROFILE,
        "metric": metric,
        "constraints": constraints,
        "tie_break": tie_break,
        "runtime_overrides_enabled": bool(
            (promotion_cfg.get("runtime_overrides", {}) or {}).get("enabled", False)
        ),
        "assets_total": len(assets),
        "assets_promoted": promoted_count,
        "assets_rejected": rejected_count,
        "assets_ineligible": ineligible_count,
        "assets_actionable_candidates": actionable_candidate_count,
        "assets": assets,
    }

    runtime_dir = Path("runtime")
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = runtime_dir / "runtime_policy.json"

    out_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Runtime policy generated:")
    print(out_path)
    print(f"Assets total: {len(assets)}")
    print(f"Assets promoted: {promoted_count}")
    print(f"Assets rejected: {rejected_count}")
    print(f"Assets ineligible: {ineligible_count}")
    print(f"Actionable candidates: {actionable_candidate_count}")
    print()
