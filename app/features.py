from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .context import add_market_context_features
from .data import get_asset_profile
from .fundamentals import add_fundamental_features
from .sentiment import get_sentiment, load_sentiment_daily_series
from .utils import normalize_ticker

_DEFAULT_LEVEL_FEATURES = {
    "frac_mem",
    "sma_10",
    "sma_50",
    "ema_20",
    "ema_100",
    "bb_mid",
    "bb_std",
}


def _stationarity_exclusions(cfg: dict[str, Any], ticker: str, columns: list[str]) -> list[str]:
    prep = cfg.get("features", {}).get("preparation", {}) or {}
    stationarity = prep.get("stationarity", {}) or {}
    blocked: set[str] = set()
    if bool(stationarity.get("drop_raw_price", True)):
        blocked.add(normalize_ticker(ticker))
    if bool(stationarity.get("drop_level_features", True)):
        configured = stationarity.get("level_feature_names")
        blocked.update(str(item) for item in (configured or sorted(_DEFAULT_LEVEL_FEATURES)))
    return [col for col in columns if col in blocked]


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _frac_diff(series: pd.Series, d: float) -> pd.Series:
    return series.diff() * d + series.shift(1) * (1 - d)


def _hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Simplified rescaled range (R/S) analysis for Hurst exponent.
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """

    def calc_hurst(x):
        if len(x) < 20:
            return 0.5
        lags = np.arange(2, 20, dtype=float)
        tau = np.array(
            [
                np.sqrt(np.std(np.subtract(x[int(lag) :], x[: -int(lag)])))
                for lag in lags
            ],
            dtype=float,
        )
        valid = np.isfinite(tau) & (tau > 0)
        if valid.sum() < 2:
            return 0.5
        log_lags = np.log(lags[valid])
        log_tau = np.log(tau[valid])
        finite = np.isfinite(log_lags) & np.isfinite(log_tau)
        if finite.sum() < 2:
            return 0.5
        x_log = log_lags[finite]
        y_log = log_tau[finite]
        denom = float(np.sum((x_log - x_log.mean()) ** 2))
        if denom <= 0:
            return 0.5
        slope = float(np.sum((x_log - x_log.mean()) * (y_log - y_log.mean())) / denom)
        return float(np.clip(slope * 2.0, 0.0, 1.5))

    return series.rolling(window).apply(calc_hurst, raw=True)


def build_dataset(
    cfg: dict[str, Any], prices: pd.DataFrame, ticker: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    if ticker not in prices.columns:
        raise ValueError(f"missing ticker column: {ticker}")

    fcfg = cfg.get("features", {})
    tech = fcfg.get("technical", {})
    tech_features = tech.get("features", {}) or {}

    def tech_enabled(name: str, default: bool = True) -> bool:
        return bool(tech.get("enabled", True)) and bool(tech_features.get(name, default))

    df = prices.copy().ffill()
    px = df[ticker]

    dataset = pd.DataFrame(index=df.index)
    dataset[ticker] = px

    # Target: returns for different horizons
    # d1: Next day return
    # d5: Return after 5 trading days (approx. 1 week)
    # d20: Return after 20 trading days (approx. 1 month)
    dataset["target_return_d1"] = px.shift(-1) / px - 1
    dataset["target_return_d5"] = px.shift(-5) / px - 1
    dataset["target_return_d20"] = px.shift(-20) / px - 1

    if tech_enabled("fractional_memory"):
        dataset["frac_mem"] = _frac_diff(px, float(tech.get("frac_diff_d", 0.5)))
    if tech_enabled("returns"):
        dataset["ret_1"] = px.pct_change(1)
        dataset["ret_5"] = px.pct_change(5)
        dataset["ret_20"] = px.pct_change(20)
    if tech_enabled("volatility"):
        ret_1 = dataset["ret_1"] if "ret_1" in dataset.columns else px.pct_change(1)
        vol_windows = [
            int(w) for w in tech.get("vol_windows", [tech.get("vol_window", 20)]) if int(w) > 1
        ]
        for w in sorted(set(vol_windows or [20])):
            dataset[f"vol_{w}"] = ret_1.rolling(w).std()
    if tech_enabled("rsi"):
        dataset["rsi"] = _rsi(px, int(tech.get("rsi_window", 14)))
    sma_short_w = int(tech.get("sma_short", 10))
    sma_long_w = int(tech.get("sma_long", 50))
    ema_short_w = int(tech.get("ema_short", 20))
    ema_long_w = int(tech.get("ema_long", 100))
    if tech_enabled("moving_averages"):
        dataset["sma_10"] = px.rolling(sma_short_w).mean()
        dataset["sma_50"] = px.rolling(sma_long_w).mean()
        dataset["sma_ratio"] = dataset["sma_10"] / dataset["sma_50"] - 1
        dataset["price_to_sma_10"] = px / dataset["sma_10"] - 1
        dataset["price_to_sma_50"] = px / dataset["sma_50"] - 1
    if tech_enabled("ema"):
        dataset["ema_20"] = px.ewm(span=ema_short_w, adjust=False).mean()
        dataset["ema_100"] = px.ewm(span=ema_long_w, adjust=False).mean()
        dataset["ema_ratio"] = dataset["ema_20"] / dataset["ema_100"] - 1
        dataset["price_to_ema_20"] = px / dataset["ema_20"] - 1
        dataset["price_to_ema_100"] = px / dataset["ema_100"] - 1
    if tech_enabled("roc"):
        dataset["roc_10"] = px.pct_change(int(tech.get("roc_window", 10)))
    if tech_enabled("bollinger"):
        dataset["bb_mid"] = px.rolling(20).mean()
        dataset["bb_std"] = px.rolling(20).std()
        dataset["bb_width"] = (4 * dataset["bb_std"]) / dataset["bb_mid"]
        dataset["bb_pos"] = (px - (dataset["bb_mid"] - 2 * dataset["bb_std"])) / (
            4 * dataset["bb_std"]
        )
        roll_max_20 = px.rolling(20).max()
        roll_min_20 = px.rolling(20).min()
        dataset["drawdown_20"] = px / roll_max_20 - 1
        dataset["range_pos_20"] = (px - roll_min_20) / (roll_max_20 - roll_min_20)
    if "vol_5" in dataset.columns and "vol_20" in dataset.columns:
        dataset["vol_ratio_5_20"] = dataset["vol_5"] / dataset["vol_20"] - 1
    if "vol_20" in dataset.columns:
        ret_1 = dataset["ret_1"] if "ret_1" in dataset.columns else px.pct_change(1)
        ret_mean_20 = ret_1.rolling(20).mean()
        dataset["ret_z_20"] = (ret_1 - ret_mean_20) / dataset["vol_20"].replace(0, np.nan)

    # Advanced / Complex Features
    if tech_enabled("volatility"):
        dataset["hurst"] = _hurst_exponent(px, 100)

    # RSI Divergence proxy: Slope of Price vs Slope of RSI
    if "rsi" in dataset.columns:
        px_slope = px.pct_change(5).rolling(5).mean()
        rsi_slope = dataset["rsi"].diff(5).rolling(5).mean()
        dataset["rsi_div"] = rsi_slope - px_slope

    # Relative ATR (Proxy)
    if tech_enabled("volatility"):
        tr = pd.concat(
            [px - px.shift(1), (px - px.shift(1)).abs(), px.rolling(2).std()], axis=1
        ).max(axis=1)
        dataset["atr_rel"] = tr.rolling(14).mean() / px

    dataset, context_meta = add_market_context_features(dataset, df, ticker, cfg)
    dataset, fundamental_meta = add_fundamental_features(dataset, ticker, cfg)

    sent_cfg = fcfg.get("sentiment", {})
    use_sentiment = bool(sent_cfg.get("enabled", False))
    if use_sentiment:
        sentiment_value, current_sentiment_meta = get_sentiment(ticker, cfg)
        if str(sent_cfg.get("mode", "temporal_feature")) == "temporal_feature":
            sent_features, sentiment_meta = load_sentiment_daily_series(ticker, cfg, dataset.index)
            for col in sent_features.columns:
                dataset[col] = sent_features[col]
            sentiment_meta["current_value"] = float(sentiment_value)
            sentiment_meta["current"] = current_sentiment_meta
        else:
            sentiment_meta = current_sentiment_meta
            sentiment_meta["mode"] = "report_only"
    else:
        sentiment_value, sentiment_meta = 0.0, {"enabled": False, "source": "disabled", "items": 0}

    # External/contextual families often start later than the local price series.
    # They should not shrink the training sample just because a macro/fundamental/
    # sentiment window is initially unavailable. Technical rolling NaNs still drop
    # naturally below; external NaNs become neutral values after temporal fill.
    external_prefixes = ("ctx_", "sent_", "fund_")
    external_exact = {"pl", "pvp", "roe", "dy"}
    external_cols = [
        c for c in dataset.columns if c.startswith(external_prefixes) or c in external_exact
    ]
    if external_cols:
        dataset[external_cols] = dataset[external_cols].replace([np.inf, -np.inf], np.nan)
        dataset[external_cols] = dataset[external_cols].ffill().fillna(0.0)

    # We want to keep all targets but not as features
    target_cols = ["target_return_d1", "target_return_d5", "target_return_d20"]
    excluded_cols = _stationarity_exclusions(cfg, ticker, list(dataset.columns))
    drop_cols = [col for col in target_cols + excluded_cols if col in dataset.columns]
    raw_X = dataset.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).dropna()

    # We return the raw X and a dataframe of all targets
    all_y = dataset.loc[raw_X.index, target_cols]

    asset_profile = get_asset_profile(cfg, ticker)

    meta = {
        "ticker": ticker,
        "asset_profile": asset_profile,
        "rows": int(len(raw_X)),
        "generated_features": list(raw_X.columns),
        "generated_feature_count": len(raw_X.columns),
        "excluded_training_features": excluded_cols,
        "feature_safety": {
            "stationarity": "raw_price_and_level_features_excluded",
        },
        "features": list(raw_X.columns),
        "selected_features": list(raw_X.columns),
        "context": context_meta,
        "fundamentals": fundamental_meta,
        "sentiment": sentiment_meta,
        "latest_price": float(px.dropna().iloc[-1]),
        "latest_date": str(px.dropna().index[-1].date()),
        "latest_risk_pct": (
            float(dataset["ret_1"].tail(20).std() * 100) if len(dataset) >= 20 else 0.0
        ),
        "sentiment_value": float(sentiment_value),
    }
    return raw_X, all_y, meta
