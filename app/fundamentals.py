from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .utils import normalize_ticker
from .data import get_asset_profile


def yahoo_snapshot(ticker: str) -> dict[str, float | str]:
    import yfinance as yf
    ticker = normalize_ticker(ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    # Robust price discovery
    price = info.get("currentPrice") or info.get("previousClose") or info.get("regularMarketPrice") or info.get("navPrice") or 0.0
    if not price:
        # Last resort: try to get the last closing price from fast_info or history
        try:
            price = t.fast_info.get("lastPrice", 0.0)
        except Exception:
            pass
            
    dy = info.get("dividendYield") or 0.0
    # yfinance sanity check: sometimes it's 0.05 (decimal), sometimes 5.0 (percent)
    if dy > 1.0:
        dy = dy / 100.0
    # Last resort: if it's still huge, it's likely an error in the data provider
    if dy > 2.0: # More than 200% yield is likely a bug
        dy = 0.0

    market_cap = float(info.get("marketCap") or 0.0)
    shares = float(info.get("sharesOutstanding") or 0.0)
    if not shares and market_cap and price:
        shares = market_cap / price

    return {
        "source": "yfinance",
        "pl": float(info.get("trailingPE") or info.get("forwardPE") or 0.0),
        "dy": float(dy or 0.0),
        "pvp": float(info.get("priceToBook") or 0.0),
        "roe": float(info.get("returnOnEquity") or 0.0),
        "market_cap": market_cap,
        "current_price": float(price or 0.0),
        "shares": shares,
    }


def _clip_feature(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).clip(lower=min_value, upper=max_value)


def add_fundamental_features(df: pd.DataFrame, ticker: str, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add fundamental features only when they are temporally valid.

    Snapshot fundamentals are still collected for report/policy context, but they no
    longer become constant training columns unless explicitly allowed in config.yaml.
    """
    cfg = cfg or {}
    fcfg = cfg.get("features", {}).get("fundamentals", {})

    ticker = normalize_ticker(ticker)
    out = df.copy()
    snap = yahoo_snapshot(ticker)
    asset_profile = get_asset_profile(cfg, ticker)
    meta = dict(snap)
    meta["asset_group"] = asset_profile.get("group")
    meta["asset_subgroup"] = asset_profile.get("subgroup")
    meta["financial_class"] = asset_profile.get("financial_class")
    meta["cnpj"] = asset_profile.get("cnpj")
    meta["cnpj_status"] = asset_profile.get("cnpj_status")

    # Selection controls whether fundamentals enter the model. It must not erase
    # operational/report information from predict/report. Therefore disabled
    # fundamentals still return a snapshot/meta block, but add no training column.
    if not bool(fcfg.get("enabled", True)):
        meta.update({
            "enabled": False,
            "source": "snapshot_report_only_disabled_as_feature",
            "cvm_rows": 0,
            "features_added": False,
            "snapshot_source": "yfinance",
            "snapshot_only_in_report": True,
        })
        return out, meta

    from .cvm_conn import CVMConnector
    meta["asset_group"] = asset_profile.get("group")
    meta["asset_subgroup"] = asset_profile.get("subgroup")
    meta["financial_class"] = asset_profile.get("financial_class")
    meta["cnpj"] = asset_profile.get("cnpj")
    meta["cnpj_status"] = asset_profile.get("cnpj_status")
    meta.update({"enabled": True, "snapshot_source": "yfinance", "snapshot_only_in_report": bool(fcfg.get("snapshot_only_in_report", True))})
    try:
        hist = CVMConnector().fetch_historical_fundamentals(ticker)
    except Exception as exc:
        hist = pd.DataFrame()
        meta["cvm_error"] = str(exc)

    has_hist = (not hist.empty) and bool(snap.get("shares", 0)) and ticker in out.columns
    use_snapshot_as_features = bool(fcfg.get("use_snapshot_as_features", False))
    require_historical = bool(fcfg.get("require_historical", True))

    if has_hist:
        idx = out.index.tz_localize(None) if getattr(out.index, "tz", None) is not None else out.index
        hist = hist.copy()
        hist.index = hist.index.tz_localize(None) if getattr(hist.index, "tz", None) is not None else hist.index
        aligned = hist.reindex(idx, method="ffill")
        market_cap = out[ticker].to_numpy() * float(snap["shares"])
        lucro = aligned["LUCRO_LIQUIDO"].replace(0, np.nan).to_numpy() * 1000
        patrimonio = aligned["PATRIMONIO_LIQUIDO"].replace(0, np.nan).to_numpy() * 1000
        out["pl"] = pd.Series(market_cap / lucro, index=out.index).replace([np.inf, -np.inf], np.nan)
        out["pvp"] = pd.Series(market_cap / patrimonio, index=out.index).replace([np.inf, -np.inf], np.nan)
        out["roe"] = pd.Series(lucro / patrimonio, index=out.index).replace([np.inf, -np.inf], np.nan)
        if use_snapshot_as_features:
            out["pl"] = out["pl"].fillna(float(snap["pl"]))
            out["pvp"] = out["pvp"].fillna(float(snap["pvp"]))
            out["roe"] = out["roe"].fillna(float(snap["roe"]))
            out["dy"] = float(snap["dy"])
        meta.update({"source": "cvm_cache_temporal", "cvm_rows": int(len(hist)), "features_added": True})
    elif use_snapshot_as_features and not require_historical:
        out["pl"] = float(snap["pl"])
        out["pvp"] = float(snap["pvp"])
        out["roe"] = float(snap["roe"])
        out["dy"] = float(snap["dy"])
        meta.update({"source": "yfinance_snapshot_as_features", "cvm_rows": 0, "features_added": True})
    else:
        meta.update({"source": "snapshot_report_only_no_temporal_features", "cvm_rows": 0, "features_added": False})
        return out, meta

    if bool(fcfg.get("add_regime_features", True)) and {"pl", "roe"}.issubset(set(out.columns)):
        cheap_pl = float(fcfg.get("cheap_pl", 10.0))
        expensive_pl = float(fcfg.get("expensive_pl", 22.0))
        good_roe = float(fcfg.get("good_roe", 0.12))
        weak_roe = float(fcfg.get("weak_roe", 0.04))
        out["fund_value_score"] = _clip_feature((expensive_pl - out["pl"]) / max(expensive_pl - cheap_pl, 1.0), -1.0, 1.0)
        out["fund_quality_score"] = _clip_feature((out["roe"] - weak_roe) / max(good_roe - weak_roe, 0.01), -1.0, 1.5)
        if "dy" in out.columns:
            out["fund_yield_score"] = _clip_feature(out["dy"] / max(float(fcfg.get("good_dy", 0.06)), 0.001), 0.0, 2.0)
        else:
            out["fund_yield_score"] = 0.0
        out["fund_regime_score"] = (
            out["fund_value_score"] * float(fcfg.get("value_weight", 0.35))
            + out["fund_quality_score"] * float(fcfg.get("quality_weight", 0.45))
            + out["fund_yield_score"] * float(fcfg.get("yield_weight", 0.20))
        )

    return out, meta
