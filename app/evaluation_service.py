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
                "profit_factor_delta": (
                    None
                    if bool(model_metrics.get("profit_factor_infinite"))
                    or bool(metrics.get("profit_factor_infinite"))
                    else float(model_profit_factor - base_profit_factor)
                ),
                "profit_factor_delta_display": (
                    "n/a"
                    if bool(model_metrics.get("profit_factor_infinite"))
                    or bool(metrics.get("profit_factor_infinite"))
                    else f"{float(model_profit_factor - base_profit_factor):+.2f}"
                ),
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


_PROFIT_FACTOR_INF_SENTINEL = 999.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _profit_factor_from_pnl(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return {
            "profit_factor": 0.0,
            "profit_factor_display": "n/a",
            "profit_factor_infinite": False,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
        }
    gross_profit = float(clean[clean > 0].sum())
    gross_loss = float(abs(clean[clean < 0].sum()))
    net_profit = float(clean.sum())
    if gross_loss > 0:
        pf = gross_profit / gross_loss
        display = f"{pf:.2f}"
        infinite = False
    elif gross_profit > 0:
        pf = _PROFIT_FACTOR_INF_SENTINEL
        display = "inf"
        infinite = True
    else:
        pf = 0.0
        display = "0.00"
        infinite = False
    return {
        "profit_factor": float(pf),
        "profit_factor_display": display,
        "profit_factor_infinite": bool(infinite),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
    }


def _date_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            values = pd.to_datetime(frame[lower_map[candidate.lower()]], errors="coerce").dropna()
            if not values.empty:
                return (
                    values.dt.normalize()
                    if hasattr(values, "dt")
                    else pd.Series(values).dt.normalize()
                )
    return None


def _exposure_from_trade_dates(
    trades: pd.DataFrame,
    *,
    start_date: Any = None,
    end_date: Any = None,
) -> float | None:
    if trades is None or trades.empty:
        return None
    entry = _date_column(trades, ["entry_date", "entry_dt", "open_date", "open_dt", "entry_time"])
    exit_ = _date_column(trades, ["exit_date", "exit_dt", "close_date", "close_dt", "exit_time"])
    if entry is None or exit_ is None:
        return None
    aligned = pd.concat(
        [entry.reset_index(drop=True), exit_.reset_index(drop=True)], axis=1
    ).dropna()
    if aligned.empty:
        return None
    aligned.columns = ["entry", "exit"]
    start = (
        pd.Timestamp(start_date).normalize()
        if start_date is not None
        else pd.Timestamp(aligned["entry"].min()).normalize()
    )
    end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp(aligned["exit"].max()).normalize()
    )
    if end < start:
        return None
    total_days = max(1, int((end - start).days) + 1)
    active_days: set[pd.Timestamp] = set()
    for row in aligned.to_dict(orient="records"):
        entry_date = max(pd.Timestamp(row["entry"]).normalize(), start)
        exit_date = min(pd.Timestamp(row["exit"]).normalize(), end)
        if exit_date < entry_date:
            continue
        for day in pd.date_range(entry_date, exit_date, freq="D"):
            active_days.add(pd.Timestamp(day).normalize())
    return float(len(active_days) / total_days * 100.0) if active_days else 0.0


def enrich_model_metrics_from_execution(
    metrics: dict[str, Any],
    *,
    trades: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
    initial_cash: float = 10000.0,
    start_date: Any = None,
    end_date: Any = None,
) -> dict[str, Any]:
    """Add execution-derived economic metrics to PyBroker's native metrics.

    PyBroker output schemas can vary across versions. This function keeps the
    native metrics, derives what it can from trades/orders, and records explicit
    warnings when an important metric could not be computed reliably.
    """
    out = dict(metrics or {})
    warnings: list[str] = list(out.get("metric_warnings", []) or [])
    trades_df = trades if trades is not None else pd.DataFrame()
    orders_df = orders if orders is not None else pd.DataFrame()

    trade_pnl = _numeric_column(trades_df, ["pnl", "profit", "profit_loss", "pl", "realized_pnl"])
    trade_returns = _numeric_column(
        trades_df, ["return_pct", "ret_pct", "pct_return", "return_percent", "return"]
    )
    order_values = _order_notional(orders_df)
    order_costs = _numeric_column(
        orders_df, ["fees", "fee", "commission", "commissions", "cost", "costs"]
    )

    inferred_trade_count = int(len(trades_df)) if not trades_df.empty else 0
    if "trade_count" not in out:
        out["trade_count"] = inferred_trade_count
    trade_count = int(float(out.get("trade_count", inferred_trade_count) or 0))

    if trade_pnl is not None and len(trade_pnl):
        clean_pnl = pd.to_numeric(trade_pnl, errors="coerce").dropna().astype(float)
        pf_payload = _profit_factor_from_pnl(clean_pnl)
        out.update(pf_payload)
        out["total_pnl"] = float(clean_pnl.sum())
        out["avg_trade_pnl"] = float(clean_pnl.mean())
        out["hit_rate_pct"] = float((clean_pnl > 0).mean() * 100.0)
    else:
        total_return_pct = _safe_float(out.get("total_return_pct", 0.0), 0.0)
        estimated_total_pnl = float(initial_cash) * total_return_pct / 100.0
        out.setdefault("total_pnl", estimated_total_pnl)
        out.setdefault("net_profit", estimated_total_pnl)
        out.setdefault("gross_profit", max(0.0, estimated_total_pnl))
        out.setdefault("gross_loss", abs(min(0.0, estimated_total_pnl)))
        out.setdefault("hit_rate_pct", float(out.get("win_rate", 0.0) or 0.0))
        out.setdefault("avg_trade_pnl", (estimated_total_pnl / trade_count) if trade_count else 0.0)
        native_pf = out.get("profit_factor")
        if native_pf is None or _safe_float(native_pf, 0.0) == 0.0:
            if (
                trade_count > 0
                and estimated_total_pnl > 0
                and _safe_float(out.get("hit_rate_pct"), 0.0) >= 100.0
            ):
                out["profit_factor"] = _PROFIT_FACTOR_INF_SENTINEL
                out["profit_factor_display"] = "inf"
                out["profit_factor_infinite"] = True
                warnings.append(
                    "profit factor treated as infinite: positive result with no detected losing trades"
                )
            elif trade_count > 0:
                out["profit_factor"] = 0.0
                out["profit_factor_display"] = "n/a"
                out["profit_factor_infinite"] = False
                warnings.append("profit factor unavailable: no trade PnL column detected")
            else:
                out.setdefault("profit_factor", 0.0)
                out.setdefault("profit_factor_display", "0.00")
                out.setdefault("profit_factor_infinite", False)
        else:
            out["profit_factor"] = float(native_pf)
            out.setdefault("profit_factor_display", f"{float(native_pf):.2f}")
            out.setdefault("profit_factor_infinite", False)

    if trade_returns is not None and len(trade_returns):
        clean_returns = pd.to_numeric(trade_returns, errors="coerce").dropna().astype(float)
        if not clean_returns.empty:
            out["avg_trade_return_pct"] = float(clean_returns.mean())
            out["avg_return_pct"] = float(clean_returns.mean())
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
        total_cost = float(order_costs.abs().sum())
        out["total_cost"] = total_cost
        out["cost_pct_initial_cash"] = (
            (total_cost / float(initial_cash) * 100.0) if initial_cash else 0.0
        )
    else:
        out.setdefault("total_cost", 0.0)
        out.setdefault("cost_pct_initial_cash", 0.0)

    total_return_pct = _safe_float(out.get("total_return_pct", 0.0), 0.0)
    cost_pct = _safe_float(out.get("cost_pct_initial_cash", 0.0), 0.0)
    out["return_after_costs_pct"] = total_return_pct
    out["return_before_costs_pct"] = total_return_pct + cost_pct

    exposure_value = out.get("active_exposure_pct", out.get("avg_exposure_pct"))
    exposure_available = exposure_value is not None
    exposure_pct = _safe_float(exposure_value, 0.0) if exposure_available else 0.0
    if trade_count > 0 and exposure_pct <= 0.0:
        estimated_exposure = _exposure_from_trade_dates(
            trades_df,
            start_date=start_date,
            end_date=end_date,
        )
        if estimated_exposure is not None and estimated_exposure > 0.0:
            exposure_pct = float(estimated_exposure)
            exposure_available = True
            out["active_exposure_source"] = "trade_dates_estimate"
        else:
            warnings.append("active exposure unavailable or zero despite executed trades")
            out["active_exposure_source"] = "unavailable"
    else:
        out.setdefault("active_exposure_source", "native")
    out["active_exposure_pct"] = float(exposure_pct)
    out["active_exposure_available"] = bool(
        exposure_available or exposure_pct > 0.0 or trade_count == 0
    )

    if (
        trade_count > 0
        and _safe_float(out.get("hit_rate_pct"), 0.0) >= 100.0
        and _safe_float(out.get("total_return_pct"), 0.0) > 0
        and _safe_float(out.get("profit_factor"), 0.0) == 0.0
    ):
        warnings.append("inconsistent metrics: positive 100% hit-rate run has zero profit factor")

    out["metric_warnings"] = sorted(set(str(item) for item in warnings if item))
    out["economic_metrics_source"] = {
        "trades_columns": list(trades_df.columns) if not trades_df.empty else [],
        "orders_columns": list(orders_df.columns) if not orders_df.empty else [],
        "trade_pnl_detected": trade_pnl is not None,
        "trade_return_detected": trade_returns is not None,
        "order_notional_detected": order_values is not None,
        "order_cost_detected": order_costs is not None,
        "exposure_source": out.get("active_exposure_source", "native"),
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
