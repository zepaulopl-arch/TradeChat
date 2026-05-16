from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.runtime_policy_config import (
    load_runtime_policy_config,
)


def _safe_value(
    value: Any,
) -> Any:

    if pd.isna(value):
        return None

    if hasattr(
        value,
        "item",
    ):
        return value.item()

    return value


def _extract_evidence(
    row: pd.Series,
) -> dict[str, Any]:

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
    runtime_overrides = (
        promotion_cfg.get(
            "runtime_overrides",
            {},
        )
        or {}
    )

    if not bool(
        runtime_overrides.get(
            "enabled",
            False,
        )
    ):
        return {}

    profiles = (
        runtime_overrides.get(
            "profiles",
            {},
        )
        or {}
    )

    return (
        profiles.get(
            str(profile),
            {},
        )
        or {}
    )


def promote_policy(
    matrix_dir: str,
):

    runtime_cfg = load_runtime_policy_config()

    promotion_cfg = (
        runtime_cfg.get(
            "promotion",
            {},
        )
        or {}
    )

    constraints = (
        promotion_cfg.get(
            "constraints",
            {},
        )
        or {}
    )

    tie_break = list(
        promotion_cfg.get(
            "tie_break",
            [],
        )
        or []
    )

    metric = str(
        promotion_cfg.get(
            "metric",
            "sharpe",
        )
    )

    mode = str(
        promotion_cfg.get(
            "mode",
            "by-asset",
        )
    )

    max_drawdown = float(
        constraints.get(
            "max_drawdown_pct",
            15.0,
        )
    )

    min_trades = int(
        constraints.get(
            "min_trades",
            20,
        )
    )

    min_sharpe = float(
        constraints.get(
            "min_sharpe",
            0.80,
        )
    )

    max_exposure = float(
        constraints.get(
            "max_exposure_pct",
            100.0,
        )
    )

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

        if "drawdown_pct" in local.columns:

            local = local[local["drawdown_pct"] <= max_drawdown]

        if "trades" in local.columns:

            local = local[local["trades"] >= min_trades]

        if "sharpe" in local.columns:

            local = local[local["sharpe"] >= min_sharpe]

        if "exposure_pct" in local.columns:

            local = local[local["exposure_pct"] <= max_exposure]

        if local.empty:
            continue

        sort_columns = [metric]

        for column in tie_break:
            if column in local.columns and column not in sort_columns:
                sort_columns.append(column)

        ascending = []

        for column in sort_columns:
            if column in {
                "drawdown_pct",
                "max_drawdown",
            }:
                ascending.append(True)
            else:
                ascending.append(False)

        local = local.sort_values(
            sort_columns,
            ascending=ascending,
        )

        best = local.iloc[0]

        profile = str(best["profile"])

        assets[str(ticker)] = {
            "profile": profile,
            "source": "policy_matrix",
            "overrides": _profile_overrides_from_config(
                promotion_cfg,
                profile,
            ),
            "selection": {
                "metric": metric,
                "mode": mode,
                "sort_columns": sort_columns,
            },
            "evidence": _extract_evidence(best),
        }

    output = {
        "selection_mode": mode,
        "metric": metric,
        "constraints": constraints,
        "tie_break": tie_break,
        "runtime_overrides_enabled": bool(
            (
                promotion_cfg.get(
                    "runtime_overrides",
                    {},
                )
                or {}
            ).get(
                "enabled",
                False,
            )
        ),
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
        ),
        encoding="utf-8",
    )

    print()
    print("Runtime policy generated:")
    print(out_path)
    print(f"Assets promoted: {len(assets)}")
    print()
