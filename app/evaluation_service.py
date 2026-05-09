from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def close_matrix_from_bars(bars: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    required = {"date", "symbol", "close"}
    missing = required.difference(set(bars.columns))
    if missing:
        raise ValueError(f"bars is missing required columns: {', '.join(sorted(missing))}")

    frame = bars.loc[:, ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    matrix = frame.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    if tickers:
        wanted = [str(ticker) for ticker in tickers if str(ticker) in matrix.columns]
        matrix = matrix.loc[:, wanted]
    matrix = matrix.ffill().dropna(how="all")
    return matrix


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).fillna(0.0)
    return float(drawdown.min() * 100.0)


def _metrics_from_returns(
    returns: pd.Series,
    *,
    initial_cash: float,
    trade_count: int,
    exposure: pd.Series | None = None,
) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = float(initial_cash) * (1.0 + clean).cumprod()
    final_equity = float(equity.iloc[-1]) if not equity.empty else float(initial_cash)
    total_return = (final_equity / float(initial_cash) - 1.0) * 100.0 if initial_cash else 0.0
    positive = clean[clean > 0]
    negative = clean[clean < 0]
    downside = negative.std(ddof=0)
    volatility = clean.std(ddof=0)
    sharpe = (clean.mean() / volatility * np.sqrt(252.0)) if volatility and volatility > 0 else 0.0
    sortino = (clean.mean() / downside * np.sqrt(252.0)) if downside and downside > 0 else 0.0
    active_exposure = (
        float(exposure.mean() * 100.0) if exposure is not None and not exposure.empty else 0.0
    )
    return {
        "total_return_pct": float(total_return),
        "max_drawdown_pct": _max_drawdown_pct(equity),
        "volatility_pct": float(volatility * np.sqrt(252.0) * 100.0) if volatility else 0.0,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "hit_rate_pct": float((clean > 0).mean() * 100.0) if len(clean) else 0.0,
        "avg_return_pct": float(clean.mean() * 100.0) if len(clean) else 0.0,
        "profit_factor": (
            float(positive.sum() / abs(negative.sum())) if abs(negative.sum()) > 0 else 0.0
        ),
        "trade_count": int(trade_count),
        "active_exposure_pct": active_exposure,
        "final_equity": final_equity,
    }


def _equal_weight_returns(
    close_matrix: pd.DataFrame, weights: pd.DataFrame | None = None
) -> pd.Series:
    asset_returns = close_matrix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if weights is None:
        return asset_returns.mean(axis=1).fillna(0.0)
    aligned = weights.reindex(asset_returns.index).fillna(0.0)
    active = aligned.sum(axis=1).replace(0.0, np.nan)
    return (asset_returns * aligned).sum(axis=1).div(active).fillna(0.0)


def _transition_count(weights: pd.DataFrame) -> int:
    changed = weights.fillna(0.0).diff().abs().sum(axis=1) > 0
    return int(changed.sum())


def _long_flat_weights_from_signal(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.shift(1).gt(0).astype(float).fillna(0.0)


def evaluate_baselines(
    bars: pd.DataFrame,
    tickers: list[str],
    *,
    initial_cash: float = 10000.0,
    random_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    close_matrix = close_matrix_from_bars(bars, tickers)
    if close_matrix.empty:
        return {}

    asset_returns = close_matrix.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    zero_returns = pd.Series(0.0, index=close_matrix.index)
    buy_hold_returns = _equal_weight_returns(close_matrix)
    mean_signal = asset_returns.expanding(min_periods=2).mean()
    last_signal = asset_returns
    rng = np.random.default_rng(random_seed)
    random_signal = pd.DataFrame(
        rng.choice([-1.0, 1.0], size=asset_returns.shape),
        index=asset_returns.index,
        columns=asset_returns.columns,
    )

    mean_weights = _long_flat_weights_from_signal(mean_signal)
    last_weights = _long_flat_weights_from_signal(last_signal)
    random_weights = _long_flat_weights_from_signal(random_signal)
    full_exposure = pd.Series(1.0, index=close_matrix.index)
    flat_exposure = pd.Series(0.0, index=close_matrix.index)

    return {
        "zero_return_no_trade": {
            "description": "No position; verifies the model beats doing nothing.",
            "metrics": _metrics_from_returns(
                zero_returns, initial_cash=initial_cash, trade_count=0, exposure=flat_exposure
            ),
        },
        "buy_and_hold_equal_weight": {
            "description": "Equal-weight long basket held through the full window.",
            "metrics": _metrics_from_returns(
                buy_hold_returns,
                initial_cash=initial_cash,
                trade_count=len(close_matrix.columns),
                exposure=full_exposure,
            ),
        },
        "mean_return_long_flat": {
            "description": (
                "Long only when the expanding mean return known before the bar is positive."
            ),
            "metrics": _metrics_from_returns(
                _equal_weight_returns(close_matrix, mean_weights),
                initial_cash=initial_cash,
                trade_count=_transition_count(mean_weights),
                exposure=mean_weights.mean(axis=1),
            ),
        },
        "last_return_long_flat": {
            "description": "Long only when the previous bar return was positive.",
            "metrics": _metrics_from_returns(
                _equal_weight_returns(close_matrix, last_weights),
                initial_cash=initial_cash,
                trade_count=_transition_count(last_weights),
                exposure=last_weights.mean(axis=1),
            ),
        },
        "random_long_flat": {
            "description": (
                "Deterministic random long/flat baseline using only prior random states."
            ),
            "metrics": _metrics_from_returns(
                _equal_weight_returns(close_matrix, random_weights),
                initial_cash=initial_cash,
                trade_count=_transition_count(random_weights),
                exposure=random_weights.mean(axis=1),
            ),
        },
    }


def compare_model_to_baselines(
    model_metrics: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model_return = float(model_metrics.get("total_return_pct", 0.0) or 0.0)
    model_drawdown = float(model_metrics.get("max_drawdown_pct", 0.0) or 0.0)
    model_trades = int(float(model_metrics.get("trade_count", 0) or 0))
    model_hit_rate = float(
        model_metrics.get("hit_rate_pct", model_metrics.get("win_rate", 0.0)) or 0.0
    )
    model_profit_factor = float(model_metrics.get("profit_factor", 0.0) or 0.0)
    model_avg_return = float(
        model_metrics.get("avg_return_pct", model_metrics.get("avg_trade_return_pct", 0.0)) or 0.0
    )
    model_exposure = float(
        model_metrics.get("active_exposure_pct", model_metrics.get("avg_exposure_pct", 0.0)) or 0.0
    )
    rows: list[dict[str, Any]] = []
    beat_count = 0
    comparable_count = 0
    for name, payload in (baselines or {}).items():
        metrics = payload.get("metrics", {}) or {}
        base_return = float(metrics.get("total_return_pct", 0.0) or 0.0)
        base_drawdown = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
        base_trades = int(float(metrics.get("trade_count", 0) or 0))
        base_hit_rate = float(metrics.get("hit_rate_pct", metrics.get("win_rate", 0.0)) or 0.0)
        base_profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        base_avg_return = float(
            metrics.get("avg_return_pct", metrics.get("avg_trade_return_pct", 0.0)) or 0.0
        )
        base_exposure = float(
            metrics.get("active_exposure_pct", metrics.get("avg_exposure_pct", 0.0)) or 0.0
        )
        return_delta = model_return - base_return
        drawdown_delta = model_drawdown - base_drawdown
        beat = return_delta > 0
        comparable_count += 1
        beat_count += 1 if beat else 0
        rows.append(
            {
                "baseline": name,
                "model_return_pct": model_return,
                "baseline_return_pct": base_return,
                "return_delta_pct": float(return_delta),
                "model_max_drawdown_pct": model_drawdown,
                "baseline_max_drawdown_pct": base_drawdown,
                "drawdown_delta_pct": float(drawdown_delta),
                "model_trade_count": model_trades,
                "baseline_trade_count": base_trades,
                "hit_rate_delta_pct": float(model_hit_rate - base_hit_rate),
                "profit_factor_delta": float(model_profit_factor - base_profit_factor),
                "avg_return_delta_pct": float(model_avg_return - base_avg_return),
                "exposure_delta_pct": float(model_exposure - base_exposure),
                "beat_return": bool(beat),
            }
        )
    return {
        "rows": rows,
        "beat_count": int(beat_count),
        "baseline_count": int(comparable_count),
        "beat_rate_pct": float(beat_count / comparable_count * 100.0) if comparable_count else 0.0,
        "decision": (
            "passes_baselines"
            if comparable_count and beat_count == comparable_count
            else "mixed_or_fails_baselines"
        ),
    }


def enrich_model_metrics_from_execution(
    metrics: dict[str, Any],
    *,
    trades: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
    initial_cash: float = 10000.0,
) -> dict[str, Any]:
    out = dict(metrics or {})
    trades_df = trades if trades is not None else pd.DataFrame()
    orders_df = orders if orders is not None else pd.DataFrame()

    trade_pnl = _numeric_column(
        trades_df, ["pnl", "profit", "profit_loss", "pl", "realized_pnl", "return"]
    )
    trade_returns = _numeric_column(
        trades_df, ["return_pct", "ret_pct", "pct_return", "return_percent"]
    )
    order_values = _order_notional(orders_df)
    order_costs = _numeric_column(
        orders_df, ["fees", "fee", "commission", "commissions", "cost", "costs"]
    )

    if "trade_count" not in out:
        out["trade_count"] = int(len(trades_df)) if not trades_df.empty else 0
    if trade_pnl is not None and len(trade_pnl):
        positive = trade_pnl[trade_pnl > 0]
        negative = trade_pnl[trade_pnl < 0]
        out["total_pnl"] = float(trade_pnl.sum())
        out["avg_trade_pnl"] = float(trade_pnl.mean())
        out["hit_rate_pct"] = float((trade_pnl > 0).mean() * 100.0)
        out["profit_factor"] = (
            float(positive.sum() / abs(negative.sum())) if abs(negative.sum()) > 0 else 0.0
        )
    else:
        out.setdefault("hit_rate_pct", float(out.get("win_rate", 0.0) or 0.0))
        out.setdefault("profit_factor", 0.0)
        out.setdefault("avg_trade_pnl", 0.0)

    if trade_returns is not None and len(trade_returns):
        out["avg_trade_return_pct"] = float(trade_returns.mean())
        out["avg_return_pct"] = float(trade_returns.mean())
    else:
        out.setdefault("avg_trade_return_pct", float(out.get("avg_return_pct", 0.0) or 0.0))

    if order_values is not None and len(order_values):
        gross_notional = float(order_values.abs().sum())
        out["gross_order_value"] = gross_notional
        out["turnover_pct"] = (
            (gross_notional / float(initial_cash) * 100.0) if initial_cash else 0.0
        )
        out["avg_order_value"] = float(order_values.abs().mean())
    else:
        out.setdefault("gross_order_value", 0.0)
        out.setdefault("turnover_pct", 0.0)
        out.setdefault("avg_order_value", 0.0)

    if order_costs is not None and len(order_costs):
        out["total_cost"] = float(order_costs.abs().sum())
        out["cost_pct_initial_cash"] = (
            (float(order_costs.abs().sum()) / float(initial_cash) * 100.0) if initial_cash else 0.0
        )
    else:
        out.setdefault("total_cost", 0.0)
        out.setdefault("cost_pct_initial_cash", 0.0)

    out.setdefault("active_exposure_pct", float(out.get("avg_exposure_pct", 0.0) or 0.0))
    out["economic_metrics_source"] = {
        "trades_columns": list(trades_df.columns) if not trades_df.empty else [],
        "orders_columns": list(orders_df.columns) if not orders_df.empty else [],
        "trade_pnl_detected": trade_pnl is not None,
        "order_notional_detected": order_values is not None,
    }
    return out


def _numeric_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            values = pd.to_numeric(frame[lower_map[candidate.lower()]], errors="coerce").dropna()
            return values.astype(float)
    return None


def _order_notional(orders: pd.DataFrame) -> pd.Series | None:
    if orders is None or orders.empty:
        return None
    value = _numeric_column(orders, ["value", "amount", "notional", "order_value"])
    if value is not None:
        return value
    shares = _numeric_column(orders, ["shares", "qty", "quantity", "fill_shares"])
    price = _numeric_column(orders, ["fill_price", "price", "avg_price", "close"])
    if shares is None or price is None:
        return None
    aligned = pd.concat(
        [shares.reset_index(drop=True), price.reset_index(drop=True)], axis=1
    ).dropna()
    if aligned.empty:
        return None
    return aligned.iloc[:, 0].astype(float) * aligned.iloc[:, 1].astype(float)
