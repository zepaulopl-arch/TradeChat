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


def _find_price_column(df: pd.DataFrame, ticker: str) -> str | None:
    """Resolve the main price column in a deterministic, explicit order."""
    if df is None or not len(getattr(df, "columns", [])):
        return None
    columns = [str(col) for col in df.columns]
    if ticker in columns:
        return ticker
    for candidate in ("Close", "close", "Adj Close", "adj_close", "close_adj"):
        if candidate in columns:
            return candidate
    return columns[0] if columns else None


def _empty_audit(
    *,
    requested_context: list[str],
    min_rows: int,
    stale_days: int,
    issues: list[AuditIssue],
) -> dict[str, Any]:
    return {
        "status": "error",
        "rows": 0,
        "columns": 0,
        "raw_rows": 0,
        "raw_first_date": None,
        "raw_last_date": None,
        "first_date": None,
        "last_date": None,
        "age_days": None,
        "stale_days": stale_days,
        "is_stale": True,
        "min_rows": min_rows,
        "has_min_rows": False,
        "price_column": None,
        "price_column_present": False,
        "price_valid_count": 0,
        "effective_rows": 0,
        "effective_first_date": None,
        "effective_last_date": None,
        "pre_asset_padding_count": 0,
        "post_asset_missing_count": 0,
        "close_missing_count": 0,
        "close_missing_pct": 0.0,
        "internal_missing_close_count": 0,
        "internal_missing_close_pct": 0.0,
        "rows_with_any_missing": 0,
        "rows_with_any_missing_pct": 0.0,
        "effective_rows_with_any_missing": 0,
        "effective_rows_with_any_missing_pct": 0.0,
        "context_missing_inside_count": 0,
        "context_missing_inside_pct": 0.0,
        "context_complete_rows_count": 0,
        "context_complete_rows_pct": 0.0,
        "context_missing_by_ticker": [],
        "context_missing_top_tickers": [],
        "duplicate_date_count": 0,
        "largest_gap_days": 0,
        "largest_gap_start": None,
        "largest_gap_end": None,
        "effective_largest_gap_days": 0,
        "effective_largest_gap_start": None,
        "effective_largest_gap_end": None,
        "requested_context_count": len(requested_context),
        "present_context_count": 0,
        "missing_context_count": len(requested_context),
        "context_coverage_pct": 0.0,
        "present_context_tickers": [],
        "missing_context_tickers": requested_context,
        "all_missing_columns": [],
        "issues": [issue.as_dict() for issue in issues],
    }


def audit_dataframe(
    df: pd.DataFrame,
    *,
    ticker: str,
    requested_context_tickers: list[str] | None = None,
    min_rows: int | None = None,
    stale_days: int = 3,
    today: Any = None,
) -> dict[str, Any]:
    """Return a compact, deterministic quality audit for cached price data.

    The audit deliberately separates the raw aligned table range from the effective
    asset range. Padding before the first valid asset price is informational and should
    not be treated as an internal missing-price failure.
    """
    requested_context = [
        str(item) for item in (requested_context_tickers or []) if str(item).strip()
    ]
    min_rows = int(min_rows or 0)
    stale_days = int(stale_days)
    today_ts = _safe_timestamp(today) or pd.Timestamp.today().normalize()
    issues: list[AuditIssue] = []

    rows = int(len(df)) if df is not None else 0
    columns = [str(col) for col in getattr(df, "columns", [])]
    if not rows:
        issues.append(AuditIssue("error", "rows", "cache exists but contains no rows"))
        return _empty_audit(
            requested_context=requested_context,
            min_rows=min_rows,
            stale_days=stale_days,
            issues=issues,
        )

    raw_first_date = _safe_timestamp(df.index.min())
    raw_last_date = _safe_timestamp(df.index.max())
    duplicate_dates = int(pd.Index(df.index).duplicated().sum())
    largest_gap = _largest_index_gap(df.index)

    price_col = _find_price_column(df, ticker)
    price_column_present = bool(price_col and price_col in df.columns)
    if not price_column_present:
        issues.append(
            AuditIssue("error", "price_column", f"main ticker column not found: {ticker}")
        )
        return _empty_audit(
            requested_context=requested_context,
            min_rows=min_rows,
            stale_days=stale_days,
            issues=issues,
        ) | {
            "rows": rows,
            "raw_rows": rows,
            "columns": len(columns),
            "raw_first_date": _format_date(raw_first_date),
            "raw_last_date": _format_date(raw_last_date),
            "first_date": _format_date(raw_first_date),
            "last_date": _format_date(raw_last_date),
        }

    price_series = df[price_col]
    valid_price = price_series.notna()
    price_valid_count = int(valid_price.sum())
    close_missing_count = int(price_series.isna().sum())
    rows_with_any_missing = int(df.isna().any(axis=1).sum())
    all_missing_columns = [str(col) for col in df.columns if bool(df[col].isna().all())]

    if price_valid_count <= 0:
        issues.append(AuditIssue("error", "price", "main price column has no valid values"))
        return _empty_audit(
            requested_context=requested_context,
            min_rows=min_rows,
            stale_days=stale_days,
            issues=issues,
        ) | {
            "rows": rows,
            "raw_rows": rows,
            "columns": len(columns),
            "raw_first_date": _format_date(raw_first_date),
            "raw_last_date": _format_date(raw_last_date),
            "first_date": _format_date(raw_first_date),
            "last_date": _format_date(raw_last_date),
            "price_column": str(price_col),
            "price_column_present": True,
            "close_missing_count": close_missing_count,
            "close_missing_pct": _pct(close_missing_count, rows),
            "rows_with_any_missing": rows_with_any_missing,
            "rows_with_any_missing_pct": _pct(rows_with_any_missing, rows),
            "all_missing_columns": all_missing_columns,
        }

    valid_index = df.index[valid_price]
    effective_first = _safe_timestamp(pd.Index(valid_index).min())
    effective_last = _safe_timestamp(pd.Index(valid_index).max())
    effective_mask = valid_price.copy()
    if effective_first is not None and effective_last is not None:
        date_index = pd.to_datetime(df.index)
        effective_mask = (date_index >= effective_first) & (date_index <= effective_last)

    effective_df = df.loc[effective_mask]
    effective_price = price_series.loc[effective_mask]
    effective_rows = int(len(effective_df))
    internal_missing_close_count = int(effective_price.isna().sum()) if effective_rows else 0
    pre_asset_padding_count = (
        int((pd.to_datetime(df.index) < effective_first).sum())
        if effective_first is not None
        else 0
    )
    post_asset_missing_count = (
        int((pd.to_datetime(df.index) > effective_last).sum()) if effective_last is not None else 0
    )
    effective_rows_with_any_missing = (
        int(effective_df.isna().any(axis=1).sum()) if effective_rows else 0
    )
    effective_largest_gap = (
        _largest_index_gap(effective_df.index)
        if effective_rows
        else {"days": 0, "start": None, "end": None}
    )

    available = set(columns)
    present_context = [item for item in requested_context if item in available]
    missing_context = [item for item in requested_context if item not in available]
    context_coverage_pct = _pct(len(present_context), len(requested_context))
    context_missing_by_ticker: list[dict[str, Any]] = []
    if present_context and effective_rows:
        context_frame = effective_df[present_context]
        context_missing_inside_count = int(context_frame.isna().any(axis=1).sum())
        for context_ticker in present_context:
            missing_count = int(context_frame[context_ticker].isna().sum())
            valid_count = int(context_frame[context_ticker].notna().sum())
            context_missing_by_ticker.append(
                {
                    "ticker": context_ticker,
                    "missing_count": missing_count,
                    "missing_pct": _pct(missing_count, effective_rows),
                    "valid_count": valid_count,
                    "all_missing": bool(missing_count == effective_rows),
                }
            )
        context_missing_by_ticker.sort(
            key=lambda item: (float(item["missing_pct"]), int(item["missing_count"])), reverse=True
        )
    else:
        context_missing_inside_count = 0
    context_complete_rows_count = max(int(effective_rows) - int(context_missing_inside_count), 0)
    context_missing_top_tickers = [
        str(item["ticker"])
        for item in context_missing_by_ticker
        if int(item.get("missing_count", 0)) > 0
    ][:5]

    age_days = int((today_ts - effective_last).days) if effective_last is not None else None

    if min_rows and effective_rows < min_rows:
        issues.append(
            AuditIssue(
                "warning",
                "effective_rows",
                f"effective rows below configured minimum ({effective_rows} < {min_rows})",
            )
        )
    if internal_missing_close_count:
        issues.append(
            AuditIssue(
                "warning",
                "internal_missing_close",
                f"main price column has {internal_missing_close_count} missing values inside effective range",
            )
        )
    if post_asset_missing_count:
        issues.append(
            AuditIssue(
                "warning",
                "post_asset_missing",
                f"{post_asset_missing_count} rows exist after the last valid asset price",
            )
        )
    if duplicate_dates:
        issues.append(
            AuditIssue(
                "warning", "duplicate_dates", f"{duplicate_dates} duplicate index dates found"
            )
        )
    if age_days is None:
        issues.append(AuditIssue("warning", "freshness", "last valid price date is unavailable"))
    elif age_days > stale_days:
        issues.append(
            AuditIssue("warning", "freshness", f"last valid price date is {age_days} days old")
        )
    if missing_context:
        issues.append(
            AuditIssue(
                "warning", "context", f"missing context tickers: {', '.join(missing_context)}"
            )
        )
    if context_missing_inside_count:
        top_context = (
            ", ".join(context_missing_top_tickers) if context_missing_top_tickers else "n/a"
        )
        issues.append(
            AuditIssue(
                "warning",
                "context_missing_inside",
                (
                    f"context has missing values in {context_missing_inside_count} effective rows; "
                    f"top affected: {top_context}"
                ),
            )
        )
    if all_missing_columns:
        issues.append(
            AuditIssue(
                "warning",
                "all_missing_columns",
                f"all-missing columns: {', '.join(all_missing_columns[:5])}",
            )
        )

    severity_rank = {"ok": 0, "warning": 1, "error": 2}
    worst = "ok"
    for issue in issues:
        if severity_rank[issue.severity] > severity_rank[worst]:
            worst = issue.severity

    return {
        "status": worst,
        "rows": rows,
        "columns": len(columns),
        "raw_rows": rows,
        "raw_first_date": _format_date(raw_first_date),
        "raw_last_date": _format_date(raw_last_date),
        "first_date": _format_date(raw_first_date),
        "last_date": _format_date(effective_last),
        "age_days": age_days,
        "stale_days": stale_days,
        "is_stale": bool(age_days is not None and age_days > stale_days),
        "min_rows": min_rows,
        "has_min_rows": bool(effective_rows >= min_rows) if min_rows else True,
        "price_column": str(price_col),
        "price_column_present": price_column_present,
        "price_valid_count": price_valid_count,
        "effective_rows": effective_rows,
        "effective_first_date": _format_date(effective_first),
        "effective_last_date": _format_date(effective_last),
        "pre_asset_padding_count": pre_asset_padding_count,
        "post_asset_missing_count": post_asset_missing_count,
        "close_missing_count": close_missing_count,
        "close_missing_pct": _pct(close_missing_count, rows),
        "internal_missing_close_count": internal_missing_close_count,
        "internal_missing_close_pct": _pct(internal_missing_close_count, effective_rows),
        "rows_with_any_missing": rows_with_any_missing,
        "rows_with_any_missing_pct": _pct(rows_with_any_missing, rows),
        "effective_rows_with_any_missing": effective_rows_with_any_missing,
        "effective_rows_with_any_missing_pct": _pct(
            effective_rows_with_any_missing, effective_rows
        ),
        "context_missing_inside_count": context_missing_inside_count,
        "context_missing_inside_pct": _pct(context_missing_inside_count, effective_rows),
        "context_complete_rows_count": context_complete_rows_count,
        "context_complete_rows_pct": _pct(context_complete_rows_count, effective_rows),
        "context_missing_by_ticker": context_missing_by_ticker,
        "context_missing_top_tickers": context_missing_top_tickers,
        "duplicate_date_count": duplicate_dates,
        "largest_gap_days": largest_gap["days"],
        "largest_gap_start": largest_gap["start"],
        "largest_gap_end": largest_gap["end"],
        "effective_largest_gap_days": effective_largest_gap["days"],
        "effective_largest_gap_start": effective_largest_gap["start"],
        "effective_largest_gap_end": effective_largest_gap["end"],
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
        return _empty_audit(
            requested_context=requested_context,
            min_rows=int(min_rows or 0),
            stale_days=stale_days,
            issues=[
                AuditIssue(
                    "error",
                    "cache",
                    f"price cache not found: {Path(path).name}",
                )
            ],
        )

    df = pd.read_parquet(path)
    return audit_dataframe(
        df,
        ticker=canonical,
        requested_context_tickers=requested_context,
        min_rows=int(min_rows or 0),
        stale_days=stale_days,
    )
