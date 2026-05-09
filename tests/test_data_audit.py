from __future__ import annotations

import pandas as pd

from app.data_audit import audit_dataframe


def test_data_audit_detects_quality_issues() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, None, 11.0, 11.5],
            "^BVSP": [100.0, 101.0, None, 103.0],
            "EMPTY": [None, None, None, None],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-10"]),
    )

    audit = audit_dataframe(
        df,
        ticker="PETR4.SA",
        requested_context_tickers=["^BVSP", "USDBRL=X"],
        min_rows=10,
        stale_days=3,
        today="2026-01-12",
    )

    assert audit["status"] == "warning"
    assert audit["rows"] == 4
    assert audit["effective_rows"] == 4
    assert audit["has_min_rows"] is False
    assert audit["close_missing_count"] == 1
    assert audit["internal_missing_close_count"] == 1
    assert audit["duplicate_date_count"] == 1
    assert audit["effective_largest_gap_days"] == 8
    assert audit["present_context_count"] == 1
    assert audit["missing_context_tickers"] == ["USDBRL=X"]
    assert "EMPTY" in audit["all_missing_columns"]


def test_data_audit_passes_clean_small_frame() -> None:
    df = pd.DataFrame(
        {
            "VALE3.SA": [10.0, 10.2, 10.4],
            "^BVSP": [100.0, 101.0, 102.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
    )

    audit = audit_dataframe(
        df,
        ticker="VALE3.SA",
        requested_context_tickers=["^BVSP"],
        min_rows=3,
        stale_days=10,
        today="2026-01-06",
    )

    assert audit["status"] == "ok"
    assert audit["has_min_rows"] is True
    assert audit["close_missing_count"] == 0
    assert audit["internal_missing_close_count"] == 0
    assert audit["duplicate_date_count"] == 0
    assert audit["context_coverage_pct"] == 100.0
    assert audit["issues"] == []


def test_pre_asset_padding_is_not_internal_missing_close() -> None:
    df = pd.DataFrame(
        {
            "RENT3.SA": [None, None, 10.0, 10.2, 10.4],
            "^BVSP": [90.0, 91.0, 92.0, 93.0, 94.0],
        },
        index=pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        ),
    )

    audit = audit_dataframe(
        df,
        ticker="RENT3.SA",
        requested_context_tickers=["^BVSP"],
        min_rows=3,
        stale_days=10,
        today="2026-01-08",
    )

    assert audit["status"] == "ok"
    assert audit["close_missing_count"] == 2
    assert audit["pre_asset_padding_count"] == 2
    assert audit["internal_missing_close_count"] == 0
    assert audit["effective_rows"] == 3
    assert audit["effective_first_date"] == "2026-01-05"
    assert audit["effective_last_date"] == "2026-01-07"
    assert not any(issue["check"] == "missing_close" for issue in audit["issues"])
    assert not any(issue["check"] == "internal_missing_close" for issue in audit["issues"])


def test_internal_missing_close_is_warned_inside_effective_range() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, None, 10.5, 10.8],
            "^BVSP": [100.0, 101.0, 102.0, 103.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    audit = audit_dataframe(
        df,
        ticker="PETR4.SA",
        requested_context_tickers=["^BVSP"],
        min_rows=3,
        stale_days=10,
        today="2026-01-07",
    )

    assert audit["status"] == "warning"
    assert audit["pre_asset_padding_count"] == 0
    assert audit["internal_missing_close_count"] == 1
    assert any(issue["check"] == "internal_missing_close" for issue in audit["issues"])


def test_trailing_missing_price_uses_last_valid_price_for_freshness() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, 10.2, None, None],
            "^BVSP": [100.0, 101.0, 102.0, 103.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    audit = audit_dataframe(
        df,
        ticker="PETR4.SA",
        requested_context_tickers=["^BVSP"],
        min_rows=2,
        stale_days=2,
        today="2026-01-06",
    )

    assert audit["status"] == "warning"
    assert audit["effective_last_date"] == "2026-01-02"
    assert audit["post_asset_missing_count"] == 2
    assert audit["is_stale"] is True
    assert any(issue["check"] == "post_asset_missing" for issue in audit["issues"])
    assert any(issue["check"] == "freshness" for issue in audit["issues"])


def test_context_missing_inside_effective_range_is_separate_from_price_missing() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, 10.2, 10.4],
            "^BVSP": [100.0, None, 102.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
    )

    audit = audit_dataframe(
        df,
        ticker="PETR4.SA",
        requested_context_tickers=["^BVSP"],
        min_rows=3,
        stale_days=10,
        today="2026-01-06",
    )

    assert audit["status"] == "warning"
    assert audit["internal_missing_close_count"] == 0
    assert audit["context_missing_inside_count"] == 1
    assert audit["context_complete_rows_count"] == 2
    assert audit["context_missing_top_tickers"] == ["^BVSP"]
    context_detail = audit["context_missing_by_ticker"][0]
    assert context_detail["ticker"] == "^BVSP"
    assert context_detail["missing_count"] == 1
    assert round(context_detail["missing_pct"], 2) == 33.33
    assert context_detail["valid_count"] == 2
    assert context_detail["all_missing"] is False
    assert any(issue["check"] == "context_missing_inside" for issue in audit["issues"])


def test_context_missing_by_ticker_orders_worst_context_first() -> None:
    df = pd.DataFrame(
        {
            "PETR4.SA": [10.0, 10.2, 10.4, 10.6],
            "CTX_A": [None, None, None, None],
            "CTX_B": [100.0, None, 101.0, 102.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    audit = audit_dataframe(
        df,
        ticker="PETR4.SA",
        requested_context_tickers=["CTX_A", "CTX_B"],
        min_rows=3,
        stale_days=10,
        today="2026-01-07",
    )

    assert audit["status"] == "warning"
    assert audit["context_missing_inside_count"] == 4
    assert audit["context_complete_rows_count"] == 0
    assert audit["context_missing_top_tickers"] == ["CTX_A", "CTX_B"]
    assert audit["context_missing_by_ticker"][0]["ticker"] == "CTX_A"
    assert audit["context_missing_by_ticker"][0]["missing_count"] == 4
    assert audit["context_missing_by_ticker"][0]["all_missing"] is True
    assert audit["context_missing_by_ticker"][1]["ticker"] == "CTX_B"
    assert audit["context_missing_by_ticker"][1]["missing_count"] == 1
    assert audit["context_missing_by_ticker"][1]["all_missing"] is False
