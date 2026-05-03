from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from .config import cache_dir, load_data_registry
from .utils import normalize_ticker, safe_ticker


def _period_fallbacks(period: str, is_context: bool) -> list[str]:
    """Return safe yfinance period attempts.

    Some B3/index symbols accepted by Yahoo do not support ``max`` even when
    equities do. Context indices must not break the asset update; they fall
    back to shorter periods and, if still unavailable, are skipped.
    """
    period = str(period or "max").strip()
    attempts = [period]
    if is_context and period == "max":
        attempts.extend(["10y", "5y", "2y", "1y"])
    elif is_context and period not in {"5y", "2y", "1y"}:
        attempts.extend(["5y", "2y", "1y"])
    return _unique(attempts)


def _download_one_yahoo(ticker: str, period: str, *, is_context: bool) -> pd.Series | None:
    import contextlib
    import io
    import warnings
    import yfinance as yf

    for attempt in _period_fallbacks(period, is_context=is_context):
        try:
            # yfinance prints failed-download diagnostics directly; keep data screen clean
            # and surface only effective context columns after validation.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    data = yf.download(ticker, period=attempt, progress=False, auto_adjust=False, threads=False)
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
    if is_context:
        return None
    return None


def _download_yahoo(tickers: list[str], period: str) -> pd.DataFrame:
    """Download asset and context prices with provider validation.

    The first ticker is the asset and is mandatory. Remaining tickers are context
    indices/macros: they are useful, but must degrade gracefully when a provider
    rejects a period such as ``max`` or temporarily lacks data.
    """
    series: list[pd.Series] = []
    for idx, ticker in enumerate(tickers):
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
    try:
        return load_data_registry(cfg)
    except FileNotFoundError:
        return {}


def resolve_asset(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Resolve user ticker to the canonical active ticker defined in data.yaml.

    This handles ticker migrations without lateral CLI growth. Example: ELET3.SA
    can resolve to AXIA3.SA through data.yaml aliases, so old user habits do not
    break data loading, training or prediction.
    """
    requested = normalize_ticker(ticker)
    registry = _registry(cfg)
    assets = registry.get("assets", {}) or {}
    aliases = registry.get("aliases", {}) or {}

    def lookup_key(value: str) -> str | None:
        value = normalize_ticker(value)
        if value in assets:
            return value
        bare = value.split(".")[0]
        if bare in assets:
            return bare
        return None

    key = lookup_key(requested)
    alias_info: dict[str, Any] = {}
    if key and isinstance(assets.get(key), dict) and assets[key].get("registry_status") == "inactive_alias":
        alias_target = assets[key].get("canonical_ticker")
        alias_info = {"canonical": alias_target, "reason": "ticker_migration"}
        key = lookup_key(str(alias_target)) or normalize_ticker(str(alias_target))
    elif not key:
        alias_target = aliases.get(requested) or aliases.get(requested.split(".")[0])
        if isinstance(alias_target, dict):
            alias_info = dict(alias_target)
            alias_target = alias_info.get("canonical") or alias_info.get("to")
        if alias_target:
            key = lookup_key(str(alias_target)) or normalize_ticker(str(alias_target))
            alias_info.setdefault("canonical", key)
            alias_info.setdefault("reason", "ticker_alias")

    canonical = normalize_ticker(key or requested)
    profile = dict(assets.get(canonical, {}) or {})
    if not profile and canonical.split(".")[0] in assets:
        profile = dict(assets.get(canonical.split(".")[0], {}) or {})
    return {
        "requested": requested,
        "canonical": canonical,
        "changed": canonical != requested,
        "alias": alias_info,
        "profile": profile,
    }


def get_asset_profile(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Return cadastral metadata for a ticker from data.yaml."""
    return resolve_asset(cfg, ticker).get("profile", {}) or {}


def _index_to_yahoo(registry: dict[str, Any], item: str) -> str | None:
    """Convert an internal index code from data.yaml to a fetchable Yahoo ticker."""
    if not item:
        return None
    item = str(item).strip()
    # Direct yfinance symbols should pass through unchanged.
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
    """Return context tickers associated with an asset through data.yaml.

    Resolution order is deliberately conservative and deterministic:
    global context -> group defaults -> subgroup defaults -> linked indices ->
    asset-level context tickers. Codes such as IFNC or IEEX are translated to
    Yahoo-compatible symbols only when data.yaml marks them as available.
    """
    resolved = resolve_asset(cfg, ticker)
    profile = resolved.get("profile", {}) or {}
    registry = _registry(cfg)
    context_cfg = registry.get("context", {}) or {}
    out: list[str] = []

    def add_many(values: list[str] | tuple[str, ...] | None) -> None:
        for value in values or []:
            yahoo = _index_to_yahoo(registry, str(value)) or str(value)
            if yahoo:
                out.append(yahoo)

    # New structured registry.
    add_many(context_cfg.get("global", []) or registry.get("indices", {}).get("default_context", []))
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
    return cache_dir(cfg) / f"prices_{safe_ticker(canonical)}.parquet"


def load_prices(cfg: dict[str, Any], ticker: str, update: bool = False) -> pd.DataFrame:
    resolved = resolve_asset(cfg, ticker)
    canonical = resolved["canonical"]
    path = price_cache_path(cfg, canonical)
    if path.exists() and not update:
        return pd.read_parquet(path)

    period = cfg.get("data", {}).get("period", "5y")
    macros = resolve_context_tickers(cfg, canonical)
    tickers = _unique([canonical] + macros)
    close = _download_yahoo(tickers, period=period)

    # Remove empty/failed context downloads but fail clearly if the asset itself is absent.
    close = close.dropna(axis=1, how="all")
    if canonical not in close.columns:
        requested = resolved.get("requested")
        alias_msg = f" (resolved from {requested})" if requested and requested != canonical else ""
        raise RuntimeError(
            f"Yahoo did not return close prices for {canonical}{alias_msg}. "
            "Check data.yaml aliases or the current B3/Yahoo ticker."
        )
    close.to_parquet(path)
    return close


def update_cvm_database(years: list[int] | None = None) -> bool:
    from src.connectors.cvm_conn import CVMConnector
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
        "alias": resolved.get("alias", {}),
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
