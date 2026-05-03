from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from .config import cache_dir, load_data_registry
from .utils import normalize_ticker, safe_ticker


def _download_yahoo(tickers: list[str], period: str) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.ffill().dropna(how="all")
    return close


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def get_asset_profile(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Return cadastral metadata for a ticker from data.yaml.

    data.yaml is intentionally separated from config.yaml: config controls behavior;
    data.yaml controls asset identity, groups, subgroups, CNPJ and context baskets.
    """
    normalized = normalize_ticker(ticker)
    try:
        registry = load_data_registry(cfg)
    except FileNotFoundError:
        registry = {}
    assets = registry.get("assets", {}) or {}
    return assets.get(normalized) or assets.get(normalized.split(".")[0]) or {}


def resolve_context_tickers(cfg: dict[str, Any], ticker: str) -> list[str]:
    """Return macro/index columns associated with an asset through data.yaml."""
    ticker = normalize_ticker(ticker)
    profile = get_asset_profile(cfg, ticker)
    if profile.get("context_tickers"):
        return _unique([str(x) for x in profile.get("context_tickers", [])])
    try:
        registry = load_data_registry(cfg)
        defaults = registry.get("defaults", {}) or {}
        if defaults.get("context_tickers"):
            return _unique([str(x) for x in defaults.get("context_tickers", [])])
    except FileNotFoundError:
        pass
    return _unique([str(x) for x in cfg.get("data", {}).get("macro_tickers", [])])


def price_cache_path(cfg: dict[str, Any], ticker: str) -> Path:
    return cache_dir(cfg) / f"prices_{safe_ticker(ticker)}.parquet"


def load_prices(cfg: dict[str, Any], ticker: str, update: bool = False) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    path = price_cache_path(cfg, ticker)
    if path.exists() and not update:
        return pd.read_parquet(path)

    period = cfg.get("data", {}).get("period", "5y")
    macros = resolve_context_tickers(cfg, ticker)
    close = _download_yahoo([ticker] + macros, period=period)
    if ticker not in close.columns:
        raise RuntimeError(f"Yahoo did not return close prices for {ticker}")
    close.to_parquet(path)
    return close


def update_cvm_database(years: list[int] | None = None) -> bool:
    from src.connectors.cvm_conn import CVMConnector
    return CVMConnector().update_cvm_database(years=years)


def data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    path = price_cache_path(cfg, ticker)
    exists = path.exists()
    rows = 0
    start = end = None
    context_tickers = resolve_context_tickers(cfg, ticker)
    if exists:
        df = pd.read_parquet(path)
        rows = int(len(df))
        if rows:
            start = str(df.index.min().date())
            end = str(df.index.max().date())
    profile = get_asset_profile(cfg, ticker)
    return {
        "ticker": ticker,
        "cache_exists": exists,
        "rows": rows,
        "start": start,
        "end": end,
        "path": str(path),
        "context_tickers": context_tickers,
        "asset_profile": {
            "name": profile.get("name"),
            "group": profile.get("group"),
            "subgroup": profile.get("subgroup"),
            "financial_class": profile.get("financial_class"),
            "cnpj": profile.get("cnpj"),
            "linked_indices": profile.get("linked_indices", []),
        },
    }
