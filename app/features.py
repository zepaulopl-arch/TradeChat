from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .context import add_market_context_features
from .fundamentals import add_fundamental_features
from .sentiment import get_sentiment, load_sentiment_daily_series
from .utils import normalize_ticker
from .data import get_asset_profile
from .preparation import prepare_training_matrix


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _frac_diff(series: pd.Series, d: float) -> pd.Series:
    return series.diff() * d + series.shift(1) * (1 - d)


def build_dataset(cfg: dict[str, Any], prices: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    if ticker not in prices.columns:
        raise ValueError(f"missing ticker column: {ticker}")

    fcfg = cfg.get("features", {})
    tech = fcfg.get("technical", {})
    df = prices.copy().ffill()
    px = df[ticker]

    dataset = pd.DataFrame(index=df.index)
    dataset[ticker] = px
    dataset["target_return_d1"] = px.pct_change().shift(-1)
    dataset["frac_mem"] = _frac_diff(px, float(tech.get("frac_diff_d", 0.5)))
    dataset["ret_1"] = px.pct_change(1)
    dataset["ret_5"] = px.pct_change(5)
    dataset["ret_20"] = px.pct_change(20)
    dataset["vol_20"] = dataset["ret_1"].rolling(int(tech.get("vol_window", 20))).std()
    dataset["rsi"] = _rsi(px, int(tech.get("rsi_window", 14)))
    dataset["sma_10"] = px.rolling(int(tech.get("sma_short", 10))).mean()
    dataset["sma_50"] = px.rolling(int(tech.get("sma_long", 50))).mean()
    dataset["ema_20"] = px.ewm(span=int(tech.get("ema_short", 20)), adjust=False).mean()
    dataset["ema_100"] = px.ewm(span=int(tech.get("ema_long", 100)), adjust=False).mean()
    dataset["sma_ratio"] = dataset["sma_10"] / dataset["sma_50"] - 1
    dataset["ema_ratio"] = dataset["ema_20"] / dataset["ema_100"] - 1
    dataset["roc_10"] = px.pct_change(int(tech.get("roc_window", 10)))
    dataset["bb_mid"] = px.rolling(20).mean()
    dataset["bb_std"] = px.rolling(20).std()
    dataset["bb_width"] = (4 * dataset["bb_std"]) / dataset["bb_mid"]
    dataset["bb_pos"] = (px - (dataset["bb_mid"] - 2 * dataset["bb_std"])) / (4 * dataset["bb_std"])


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
        c for c in dataset.columns
        if c.startswith(external_prefixes) or c in external_exact
    ]
    if external_cols:
        dataset[external_cols] = (
            dataset[external_cols]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
            .fillna(0.0)
        )

    raw_X = dataset.drop(columns=["target_return_d1"]).replace([np.inf, -np.inf], np.nan).dropna()
    raw_y = dataset.loc[raw_X.index, "target_return_d1"].dropna()
    raw_X = raw_X.loc[raw_y.index]

    X, y, preparation_meta = prepare_training_matrix(raw_X, raw_y, cfg)

    asset_profile = get_asset_profile(cfg, ticker)

    meta = {
        "ticker": ticker,
        "asset_profile": asset_profile,
        "rows": int(len(X)),
        "generated_features": list(raw_X.columns),
        "generated_feature_count": len(raw_X.columns),
        "features": list(X.columns),
        "selected_features": list(X.columns),
        "preparation": preparation_meta,
        "dropped_correlated_features": preparation_meta.get("selection", {}).get("rejected_sample", {}),
        "context": context_meta,
        "fundamentals": fundamental_meta,
        "sentiment": sentiment_meta,
        "latest_price": float(px.dropna().iloc[-1]),
        "latest_date": str(px.dropna().index[-1].date()),
        "latest_risk_pct": float(dataset["ret_1"].tail(20).std() * 100) if len(dataset) >= 20 else 0.0,
        "sentiment_value": float(sentiment_value),
    }
    return X, y, meta
