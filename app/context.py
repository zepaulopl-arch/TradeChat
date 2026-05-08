from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _clean(name: str) -> str:
    return str(name).replace("^", "").replace("=", "_").replace("-", "_").replace(".", "_")


def _rolling_beta(asset_ret: pd.Series, macro_ret: pd.Series, window: int) -> pd.Series:
    cov = asset_ret.rolling(window).cov(macro_ret)
    var = macro_ret.rolling(window).var()
    return cov / var.replace(0, np.nan)


def add_market_context_features(dataset: pd.DataFrame, prices: pd.DataFrame, ticker: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add temporal market-context features from the asset's configured index basket.

    Context is no longer a generic snapshot. Each macro/index column is transformed by
    rolling windows: return, volatility, correlation and beta. Benchmark-specific
    features also measure relative strength and directional alignment.
    """
    ccfg = cfg.get("features", {}).get("context", {})
    if not bool(ccfg.get("enabled", True)):
        return dataset, {"enabled": False, "source": "disabled", "columns": []}

    out = dataset.copy()
    windows = [int(w) for w in ccfg.get("windows", [5, 20, 60]) if int(w) > 1]
    return_windows = [int(w) for w in ccfg.get("return_windows", windows) if int(w) > 1]
    volatility_windows = [int(w) for w in ccfg.get("volatility_windows", windows) if int(w) > 1]
    correlation_windows = [int(w) for w in ccfg.get("correlation_windows", windows) if int(w) > 1]
    beta_windows = [int(w) for w in ccfg.get("beta_windows", windows) if int(w) > 1]
    columns_added: list[str] = []
    macro_cols = [c for c in prices.columns if c != ticker]
    if not macro_cols:
        return out, {"enabled": True, "source": "no_context_columns", "columns": []}

    px_asset = prices[ticker].ffill()
    asset_ret_1 = px_asset.pct_change(1)

    for macro in macro_cols:
        clean = _clean(str(macro))
        macro_px = prices[macro].ffill()
        if macro_px.dropna().empty:
            continue
        macro_ret_1 = macro_px.pct_change(1)
        if bool(ccfg.get("use_returns", True)):
            for w in return_windows:
                col = f"ctx_{clean}_ret_{w}"
                out[col] = macro_px.pct_change(w)
                columns_added.append(col)
        if bool(ccfg.get("use_volatility", True)):
            for w in volatility_windows:
                col = f"ctx_{clean}_vol_{w}"
                out[col] = macro_ret_1.rolling(w).std()
                columns_added.append(col)
        if bool(ccfg.get("use_correlation", True)):
            for w in correlation_windows:
                col = f"ctx_{clean}_corr_{w}"
                out[col] = asset_ret_1.rolling(w).corr(macro_ret_1)
                columns_added.append(col)
        if bool(ccfg.get("use_beta", True)):
            for w in beta_windows:
                col = f"ctx_{clean}_beta_{w}"
                out[col] = _rolling_beta(asset_ret_1, macro_ret_1, w)
                columns_added.append(col)

    benchmark = ccfg.get("benchmark", "^BVSP")
    bench = prices[benchmark].ffill() if benchmark in prices.columns else prices[macro_cols[0]].ffill()
    short_w = int(ccfg.get("alignment_short_window", 5))
    long_w = int(ccfg.get("alignment_long_window", 20))
    rel_short_w = int(ccfg.get("relative_strength_short_window", short_w))
    rel_long_w = int(ccfg.get("relative_strength_long_window", long_w))

    if bool(ccfg.get("use_alignment", ccfg.get("add_alignment_features", True))):
        asset_short = px_asset.pct_change(short_w)
        bench_short = bench.pct_change(short_w)
        asset_long = px_asset.pct_change(long_w)
        bench_long = bench.pct_change(long_w)
        out["ctx_benchmark_alignment_short"] = np.sign(asset_short) * np.sign(bench_short)
        out["ctx_benchmark_alignment_long"] = np.sign(asset_long) * np.sign(bench_long)
        columns_added.extend(["ctx_benchmark_alignment_short", "ctx_benchmark_alignment_long"])

    if bool(ccfg.get("use_relative_strength", True)):
        asset_short = px_asset.pct_change(rel_short_w)
        bench_short = bench.pct_change(rel_short_w)
        asset_long = px_asset.pct_change(rel_long_w)
        bench_long = bench.pct_change(rel_long_w)
        out["ctx_relative_strength_short"] = asset_short - bench_short
        out["ctx_relative_strength_long"] = asset_long - bench_long
        columns_added.extend(["ctx_relative_strength_short", "ctx_relative_strength_long"])

    return out, {
        "enabled": True,
        "source": "asset_context_registry_from_data_cache",
        "context_columns": list(map(str, macro_cols)),
        "windows": {
            "returns": return_windows,
            "volatility": volatility_windows,
            "correlation": correlation_windows,
            "beta": beta_windows,
        },
        "columns": columns_added,
    }
