from __future__ import annotations

import contextlib
import io
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from .asset_registry import canonical_asset_ticker
from .config import historical_dir, load_data_registry
from .context_policy import filter_context_columns, load_context_policy
from .utils import normalize_ticker, safe_ticker


def _period_fallbacks(period: str, is_context: bool) -> list[str]:
    """Return safe yfinance period attempts."""
    period = str(period or "max").strip()
    attempts = [period]
    if is_context and period == "max":
        attempts.extend(["10y", "5y", "2y", "1y"])
    elif is_context and period not in {"5y", "2y", "1y"}:
        attempts.extend(["5y", "2y", "1y"])
    return _unique(attempts)


def _download_one_yahoo(ticker: str, period: str, *, is_context: bool) -> pd.Series | None:
    for attempt in _period_fallbacks(period, is_context=is_context):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    data = yf.download(
                        ticker, period=attempt, progress=False, auto_adjust=False, threads=False
                    )
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0].copy()
            else:
                close = data["Close"].copy()
            close.name = ticker
            close = close.ffill().dropna()
            if not close.empty:
                return close
        except Exception:
            continue
    return None


def _download_yahoo(
    tickers: list[str], period: str, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Download asset and context prices with provider validation."""
    series: list[pd.Series] = []
    dcfg = (cfg or {}).get("data", {})
    delay = float(dcfg.get("download_delay_seconds", 1.2))

    for idx, ticker in enumerate(tickers):
        if idx > 0 and delay > 0:
            time.sleep(delay)

        item = _download_one_yahoo(ticker, period, is_context=idx > 0)

        if item is not None and not item.empty:
            series.append(item)

    if not series:
        return pd.DataFrame()
    close = pd.concat(series, axis=1).sort_index().ffill().dropna(how="all")
    return close


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        if item and item not in out:
            out.append(item)
    return out


def _registry(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("assets_registry"):
        return cfg.get("assets_registry", {}) or {}
    try:
        return load_data_registry(cfg)
    except FileNotFoundError:
        return {}


def resolve_asset(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    requested = normalize_ticker(ticker)
    registry = _registry(cfg)
    assets = registry.get("assets", {}) or {}

    def lookup_key(value: str) -> str | None:
        value = normalize_ticker(value)
        if value in assets:
            return value
        bare = value.split(".")[0]
        if bare in assets:
            return bare
        return None

    key = lookup_key(requested)
    if str(registry.get("_registry_source", "")).lower() == "split":
        canonical = normalize_ticker(key or requested)
    else:
        canonical = normalize_ticker(
            canonical_asset_ticker({"assets_registry": registry}, key or requested)
        )
    profile = dict(assets.get(canonical, {}) or {})
    if not profile and canonical.split(".")[0] in assets:
        profile = dict(assets.get(canonical.split(".")[0], {}) or {})
    return {
        "requested": requested,
        "canonical": canonical,
        "changed": canonical != requested,
        "profile": profile,
    }


def get_asset_profile(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    return resolve_asset(cfg, ticker).get("profile", {}) or {}


def _index_to_yahoo(registry: dict[str, Any], item: str) -> str | None:
    if not item:
        return None
    item = str(item).strip()
    if item.startswith("^") or item.endswith(".SA") or "=" in item:
        return item
    indices = registry.get("indices", {}) or {}
    catalog = indices.get("catalog", {}) or indices.get("available_now", {}) or {}
    entry = catalog.get(item) or catalog.get(item.upper())
    if isinstance(entry, dict):
        if entry.get("enabled", True) is False:
            return None
        return entry.get("yahoo_ticker") or entry.get("ticker")
    return None


def resolve_context_tickers(cfg: dict[str, Any], ticker: str) -> list[str]:
    resolved = resolve_asset(cfg, ticker)
    profile = resolved.get("profile", {}) or {}
    registry = _registry(cfg)
    context_cfg = registry.get("context", {}) or {}
    out: list[str] = []

    def add_many(values: list[str] | tuple[str, ...] | None) -> None:
        for value in values or []:
            yahoo = _index_to_yahoo(registry, str(value))
            if yahoo:
                out.append(yahoo)

    add_many(
        context_cfg.get("global", []) or registry.get("indices", {}).get("default_context", [])
    )
    group = profile.get("group")
    subgroup = profile.get("subgroup")
    add_many((context_cfg.get("group_defaults", {}) or {}).get(group, []))
    add_many((context_cfg.get("subgroup_defaults", {}) or {}).get(subgroup, []))
    add_many(profile.get("linked_indices", []))
    add_many(profile.get("context_tickers", []))

    if not out:
        defaults = registry.get("defaults", {}) or {}
        add_many(defaults.get("context_tickers", []))
    if not out:
        out = [str(x) for x in cfg.get("data", {}).get("macro_tickers", [])]

    canonical = resolved.get("canonical")
    return _unique([x for x in out if x and x != canonical])


def price_cache_path(cfg: dict[str, Any], ticker: str) -> Path:
    canonical = resolve_asset(cfg, ticker).get("canonical", normalize_ticker(ticker))
    return historical_dir(cfg) / f"prices_{safe_ticker(canonical)}.parquet"


def load_prices(cfg: dict[str, Any], ticker: str, update: bool = False) -> pd.DataFrame:
    resolved = resolve_asset(cfg, ticker)
    canonical = resolved["canonical"]
    path = price_cache_path(cfg, canonical)

    if path.exists() and not update:
        return pd.read_parquet(path)

    period = cfg.get("data", {}).get("period", "5y")
    macros = resolve_context_tickers(cfg, canonical)
    tickers = _unique([canonical] + macros)
    close = _download_yahoo(tickers, period=period, cfg=cfg)

    close = close.dropna(axis=1, how="all")
    if canonical not in close.columns:
        requested = resolved.get("requested")
        raise RuntimeError(
            f"Yahoo did not return close prices for {canonical} requested as {requested}."
        )

    close, _context_decisions = filter_context_columns(
        close,
        asset_column=canonical,
        context_columns=macros,
        policy=load_context_policy(cfg),
    )

    close.to_parquet(path)
    return close


def update_cvm_database(years: list[int] | None = None) -> bool:
    from .cvm_conn import CVMConnector

    return CVMConnector().update_cvm_database(years=years)


def data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    resolved = resolve_asset(cfg, ticker)
    canonical = resolved["canonical"]
    path = price_cache_path(cfg, canonical)
    exists = path.exists()
    rows = 0
    start = end = None
    requested_context_tickers = resolve_context_tickers(cfg, canonical)
    context_tickers = requested_context_tickers
    unavailable_context: list[str] = []
    if exists:
        df = pd.read_parquet(path)
        rows = int(len(df))
        if rows:
            start = str(df.index.min().date())
            end = str(df.index.max().date())
        available_cols = set(map(str, df.columns))
        context_tickers = [c for c in requested_context_tickers if c in available_cols]
        unavailable_context = [c for c in requested_context_tickers if c not in available_cols]
    profile = resolved.get("profile", {}) or {}
    return {
        "ticker": canonical,
        "requested_ticker": resolved.get("requested"),
        "resolved_ticker": canonical,
        "ticker_changed": bool(resolved.get("changed")),
        "cache_exists": exists,
        "rows": rows,
        "start": start,
        "end": end,
        "path": str(path),
        "context_tickers": context_tickers,
        "requested_context_tickers": requested_context_tickers,
        "unavailable_context_tickers": unavailable_context,
        "asset_profile": {
            "name": profile.get("name"),
            "group": profile.get("group"),
            "subgroup": profile.get("subgroup"),
            "financial_class": profile.get("financial_class"),
            "cnpj": profile.get("cnpj"),
            "linked_indices": profile.get("linked_indices", []),
            "registry_status": profile.get("registry_status", "active"),
        },
    }
