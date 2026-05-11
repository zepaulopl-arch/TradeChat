from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.runtime_policy_config import (
    load_runtime_policy_config,
)


def promote_policy(
    matrix_dir: str,
):

    runtime_cfg = (
        load_runtime_policy_config()
    )

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

    matrix_path = Path(
        matrix_dir
    )

    summary_path = (
        matrix_path
        / "validation_summary.csv"
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            f"validation_summary.csv not found: {summary_path}"
        )

    df = pd.read_csv(
        summary_path
    )

    if mode != "by-asset":

        raise NotImplementedError(
            "only by-asset mode implemented"
        )

    assets = {}

    grouped = df.groupby(
        "ticker"
    )

    for ticker, group in grouped:

        local = group.copy()

        if (
            "drawdown_pct"
            in local.columns
        ):

            local = local[
                local["drawdown_pct"]
                <= max_drawdown
            ]

        if (
            "trades"
            in local.columns
        ):

            local = local[
                local["trades"]
                >= min_trades
            ]

        if (
            "sharpe"
            in local.columns
        ):

            local = local[
                local["sharpe"]
                >= min_sharpe
            ]

        if local.empty:
            continue

        if metric not in local.columns:

            raise ValueError(
                f"metric column not found: {metric}"
            )

        local = local.sort_values(
            metric,
            ascending=False,
        )

        best = local.iloc[0]

        profile = str(
            best["profile"]
        )

        assets[
            str(ticker)
        ] = profile

    output = {
        "selection_mode": mode,
        "metric": metric,
        "constraints": constraints,
        "assets": assets,
    }

    runtime_dir = Path(
        "runtime"
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        runtime_dir
        / "runtime_policy.json"
    )

    out_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Runtime policy generated:"
    )
    print(out_path)
    print(
        f"Assets promoted: {len(assets)}"
    )
    print()
