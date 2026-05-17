from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.runtime_policy_config import (
    load_runtime_policy_config,
)


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
    ]

    evidence: dict[str, Any] = {}

    for column in preferred_columns:
        if column not in row.index:
            continue

        evidence[column] = _safe_value(row[column])

    return evidence


def _profile_overrides_from_config(
    promotion_cfg: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    runtime_overrides = promotion_cfg.get("runtime_overrides", {}) or {}

    if not bool(runtime_overrides.get("enabled", False)):
        return {}

    profiles = runtime_overrides.get("profiles", {}) or {}

    return profiles.get(str(profile), {}) or {}


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

        sort_columns = _sort_columns(
            local,
            metric=metric,
            tie_break=tie_break,
        )

        valid_rows: list[tuple[int, pd.Series, list[str]]] = []
        rejected_rows: list[tuple[int, pd.Series, list[str]]] = []

        for index, row in local_sorted.iterrows():
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
        else:
            _, best, rejection_reasons = rejected_rows[0]
            promoted = False
            promotion_status = "rejected_by_constraints"

        profile = str(best["profile"])

        assets[str(ticker)] = {
            "profile": profile,
            "source": "policy_matrix",
            "promoted": promoted,
            "promotion_status": promotion_status,
            "rejection_reasons": rejection_reasons,
            "overrides": (
                _profile_overrides_from_config(
                    promotion_cfg,
                    profile,
                )
                if promoted
                else {}
            ),
            "selection": {
                "metric": metric,
                "mode": mode,
                "sort_columns": sort_columns,
            },
            "evidence": _extract_evidence(best),
        }

    promoted_count = sum(1 for item in assets.values() if bool(item.get("promoted", False)))

    rejected_count = sum(1 for item in assets.values() if not bool(item.get("promoted", False)))

    output = {
        "selection_mode": mode,
        "metric": metric,
        "constraints": constraints,
        "tie_break": tie_break,
        "runtime_overrides_enabled": bool(
            (promotion_cfg.get("runtime_overrides", {}) or {}).get("enabled", False)
        ),
        "assets_total": len(assets),
        "assets_promoted": promoted_count,
        "assets_rejected": rejected_count,
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
    print()
