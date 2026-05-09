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
    assert audit["has_min_rows"] is False
    assert audit["close_missing_count"] == 1
    assert audit["duplicate_date_count"] == 1
    assert audit["largest_gap_days"] == 8
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
    assert audit["duplicate_date_count"] == 0
    assert audit["context_coverage_pct"] == 100.0
    assert audit["issues"] == []
