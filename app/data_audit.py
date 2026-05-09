from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    check: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "check": self.check, "message": self.message}


def _safe_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()
    except Exception:
        return None


def _format_date(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return str(value.date())


def _largest_index_gap(index: pd.Index) -> dict[str, Any]:
    if len(index) < 2:
        return {"days": 0, "start": None, "end": None}
    dates = pd.to_datetime(pd.Index(index).dropna()).sort_values().unique()
    if len(dates) < 2:
        return {"days": 0, "start": None, "end": None}
    gaps = pd.Series(dates[1:]) - pd.Series(dates[:-1])
    max_idx = int(gaps.dt.days.argmax())
    return {
        "days": int(gaps.dt.days.iloc[max_idx]),
        "start": _format_date(pd.Timestamp(dates[max_idx])),
        "end": _format_date(pd.Timestamp(dates[max_idx + 1])),
    }


def _pct(part: int | float, whole: int | float) -> float:
    if not whole:
        return 0.0
    return float(part) / float(whole) * 100.0


def audit_dataframe(
    df: pd.DataFrame,
    *,
    ticker: str,
    requested_context_tickers: list[str] | None = None,
    min_rows: int | None = None,
    stale_days: int = 3,
    today: Any = None,
) -> dict[str, Any]:
    """Return a compact, deterministic quality audit for cached price data."""
    requested_context = [str(item) for item in (requested_context_tickers or []) if str(item).strip()]
    min_rows = int(min_rows or 0)
    today_ts = _safe_timestamp(today) or pd.Timestamp.today().normalize()
    issues: list[AuditIssue] = []

    rows = int(len(df)) if df is not None else 0
    columns = [str(col) for col in getattr(df, "columns", [])]
    first_date = _safe_timestamp(df.index.min()) if rows else None
    last_date = _safe_timestamp(df.index.max()) if rows else None
    age_days = int((today_ts - last_date).days) if last_date is not None else None
    duplicate_dates = int(pd.Index(df.index).duplicated().sum()) if rows else 0
    largest_gap = _largest_index_gap(df.index) if rows else {"days": 0, "start": None, "end": None}

    price_column_present = str(ticker) in columns
    price_col = str(ticker) if price_column_present else (columns[0] if columns else None)
    close_missing_count = int(df[price_col].isna().sum()) if rows and price_col in df.columns else rows
    rows_with_any_missing = int(df.isna().any(axis=1).sum()) if rows else 0
    all_missing_columns = [str(col) for col in df.columns if bool(df[col].isna().all())] if rows else []

    available = set(columns)
    present_context = [item for item in requested_context if item in available]
    missing_context = [item for item in requested_context if item not in available]
    context_coverage_pct = _pct(len(present_context), len(requested_context))

    if not rows:
        issues.append(AuditIssue("error", "rows", "cache exists but contains no rows"))
    elif min_rows and rows < min_rows:
        issues.append(AuditIssue("warning", "rows", f"rows below configured minimum ({rows} < {min_rows})"))
    if not price_column_present:
        issues.append(AuditIssue("error", "price_column", f"main ticker column not found: {ticker}"))
    if close_missing_count:
        issues.append(AuditIssue("warning", "missing_close", f"main price column has {close_missing_count} missing values"))
    if duplicate_dates:
        issues.append(AuditIssue("warning", "duplicate_dates", f"{duplicate_dates} duplicate index dates found"))
    if age_days is None:
        issues.append(AuditIssue("warning", "freshness", "last price date is unavailable"))
    elif age_days > int(stale_days):
        issues.append(AuditIssue("warning", "freshness", f"last price date is {age_days} days old"))
    if missing_context:
        issues.append(AuditIssue("warning", "context", f"missing context tickers: {', '.join(missing_context)}"))
    if all_missing_columns:
        issues.append(AuditIssue("warning", "all_missing_columns", f"all-missing columns: {', '.join(all_missing_columns[:5])}"))

    severity_rank = {"ok": 0, "warning": 1, "error": 2}
    worst = "ok"
    for issue in issues:
        if severity_rank[issue.severity] > severity_rank[worst]:
            worst = issue.severity

    return {
        "status": worst,
        "rows": rows,
        "columns": len(columns),
        "first_date": _format_date(first_date),
        "last_date": _format_date(last_date),
        "age_days": age_days,
        "stale_days": int(stale_days),
        "is_stale": bool(age_days is not None and age_days > int(stale_days)),
        "min_rows": min_rows,
        "has_min_rows": bool(rows >= min_rows) if min_rows else True,
        "price_column_present": price_column_present,
        "close_missing_count": close_missing_count,
        "close_missing_pct": _pct(close_missing_count, rows),
        "rows_with_any_missing": rows_with_any_missing,
        "rows_with_any_missing_pct": _pct(rows_with_any_missing, rows),
        "duplicate_date_count": duplicate_dates,
        "largest_gap_days": largest_gap["days"],
        "largest_gap_start": largest_gap["start"],
        "largest_gap_end": largest_gap["end"],
        "requested_context_count": len(requested_context),
        "present_context_count": len(present_context),
        "missing_context_count": len(missing_context),
        "context_coverage_pct": context_coverage_pct,
        "present_context_tickers": present_context,
        "missing_context_tickers": missing_context,
        "all_missing_columns": all_missing_columns,
        "issues": [issue.as_dict() for issue in issues],
    }


def audit_cached_prices(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Audit the cached parquet file for one asset without downloading data."""
    from .data import price_cache_path, resolve_asset, resolve_context_tickers

    resolved = resolve_asset(cfg, ticker)
    canonical = resolved.get("canonical", ticker)
    path = price_cache_path(cfg, canonical)
    requested_context = resolve_context_tickers(cfg, canonical)
    min_rows = cfg.get("data", {}).get("min_rows", 0)
    stale_days = int(cfg.get("data", {}).get("stale_days", 3))

    if not Path(path).exists():
        return {
            "status": "error",
            "rows": 0,
            "columns": 0,
            "first_date": None,
            "last_date": None,
            "age_days": None,
            "stale_days": stale_days,
            "is_stale": True,
            "min_rows": int(min_rows or 0),
            "has_min_rows": False,
            "price_column_present": False,
            "close_missing_count": 0,
            "close_missing_pct": 0.0,
            "rows_with_any_missing": 0,
            "rows_with_any_missing_pct": 0.0,
            "duplicate_date_count": 0,
            "largest_gap_days": 0,
            "largest_gap_start": None,
            "largest_gap_end": None,
            "requested_context_count": len(requested_context),
            "present_context_count": 0,
            "missing_context_count": len(requested_context),
            "context_coverage_pct": 0.0,
            "present_context_tickers": [],
            "missing_context_tickers": requested_context,
            "all_missing_columns": [],
            "issues": [
                {
                    "severity": "error",
                    "check": "cache",
                    "message": f"price cache not found: {Path(path).name}",
                }
            ],
        }

    df = pd.read_parquet(path)
    return audit_dataframe(
        df,
        ticker=canonical,
        requested_context_tickers=requested_context,
        min_rows=int(min_rows or 0),
        stale_days=stale_days,
    )
