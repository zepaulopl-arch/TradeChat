from __future__ import annotations

import pandas as pd

from app.context_policy import ContextPolicy, context_coverage_decisions, filter_context_columns


def test_context_policy_drops_low_coverage_context() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, 10.1, 10.2, 10.3, 10.4],
            "GOOD": [100.0, 101.0, 102.0, 103.0, 104.0],
            "BAD": [None, None, None, 1.0, None],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    filtered, decisions = filter_context_columns(
        df,
        asset_column="PETR4.SA",
        context_columns=["GOOD", "BAD"],
        policy=ContextPolicy(min_valid_count=2, use_min_coverage_pct=70),
    )

    by_ticker = {item.ticker: item for item in decisions}
    assert "GOOD" in filtered.columns
    assert "BAD" not in filtered.columns
    assert by_ticker["GOOD"].action == "use"
    assert by_ticker["BAD"].action == "drop"


def test_context_policy_uses_effective_asset_range_not_pre_asset_padding() -> None:
    df = pd.DataFrame(
        {
            "RENT3.SA": [None, None, 10.0, 10.1, 10.2],
            "CTX": [None, None, 100.0, 101.0, 102.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )

    decisions = context_coverage_decisions(
        df,
        asset_column="RENT3.SA",
        context_columns=["CTX"],
        policy=ContextPolicy(min_valid_count=3, use_min_coverage_pct=70),
    )

    assert decisions[0].valid_count == 3
    assert decisions[0].total_count == 3
    assert decisions[0].coverage_pct == 100.0
    assert decisions[0].action == "use"


def test_context_policy_drops_context_with_too_few_valid_values() -> None:
    df = pd.DataFrame(
        {
            "ITUB4.SA": [10.0, 10.1, 10.2, 10.3],
            "IFNC.SA": [None, None, 1.0, None],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )

    filtered, decisions = filter_context_columns(
        df,
        asset_column="ITUB4.SA",
        context_columns=["IFNC.SA"],
        policy=ContextPolicy(min_valid_count=2, use_min_coverage_pct=10),
    )

    assert "IFNC.SA" not in filtered.columns
    assert decisions[0].action == "drop"
    assert decisions[0].reason == "valid_count 1 < 2"
