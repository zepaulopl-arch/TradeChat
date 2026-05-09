from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import artifact_dir
from ..data import load_prices
from ..evaluation_decision import make_validation_decision
from ..evaluation_service import (
    compare_model_to_baselines,
    enrich_model_metrics_from_execution,
    evaluate_baselines,
)
from ..features import build_dataset
from ..models import predict_multi_horizon, train_models
from ..pipeline_service import canonical_ticker
from ..policy import active_policy_profile, classify_signal
from ..scoring import is_actionable_signal, signal_score
from ..trade_plan_service import build_trade_plan, trade_plan_from_signal
from ..utils import safe_ticker, write_json
from .execution_costs import execution_cost_config
from .metrics_bridge import metrics_frame_to_dict
from .pybroker_adapter import (
    FeeMode,
    PositionMode,
    RandomSlippageModel,
    Strategy,
    StrategyConfig,
    YFinance,
    quiet_pybroker,
    require_pybroker,
)
from .replay import normalize_validation_mode


def simulation_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "simulations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shadow_config(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    shadow = copy.deepcopy(cfg)
    app_cfg = dict(shadow.get("app", {}) or {})
    base_artifacts = Path(str(app_cfg.get("artifact_dir", "artifacts")))
    app_cfg["artifact_dir"] = str(base_artifacts / "simulations" / run_id / "shadow_artifacts")
    shadow["app"] = app_cfg
    return shadow


def _require_pybroker() -> None:
    require_pybroker()


def _quiet_pybroker():
    return quiet_pybroker()


def _coerce_timestamp(value: str | datetime | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if value is None:
        return fallback
    return pd.Timestamp(value).normalize()


def _default_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365)
    return start, end


def _schedule_rebalance_dates(
    bars: pd.DataFrame,
    *,
    rebalance_days: int,
    warmup_bars: int,
) -> list[pd.Timestamp]:
    unique_dates = pd.Index(pd.to_datetime(bars["date"]).dt.normalize().unique()).sort_values()
    effective_warmup = min(max(1, int(warmup_bars)), max(1, len(unique_dates) - 1))
    if len(unique_dates) <= effective_warmup:
        return []
    step = max(1, int(rebalance_days))
    return [
        pd.Timestamp(unique_dates[idx]) for idx in range(effective_warmup, len(unique_dates), step)
    ]


def _build_signal_as_of(
    cfg: dict[str, Any],
    ticker: str,
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    *,
    mode: str = "replay",
    train_cfg: dict[str, Any] | None = None,
    autotune: bool = False,
    inner_threads: int | None = 1,
) -> dict[str, Any] | None:
    sliced = prices.loc[:as_of_date]
    min_rows = int(cfg.get("data", {}).get("min_rows", 150))
    if len(sliced) < min_rows:
        return None

    raw_X, _all_y, meta = build_dataset(cfg, sliced, ticker)
    if len(raw_X) < min_rows:
        return None

    if mode == "walkforward":
        wf_cfg = train_cfg or cfg
        for target_col in ["target_return_d1", "target_return_d5", "target_return_d20"]:
            horizon = target_col.split("_")[-1]
            y_series = _all_y[target_col].dropna()
            if len(y_series) < min_rows:
                return None
            h_meta = meta.copy()
            h_meta["horizon"] = horizon
            train_models(
                wf_cfg,
                ticker,
                raw_X,
                _all_y[target_col],
                h_meta,
                autotune=bool(autotune),
                horizon=horizon,
                inner_threads=inner_threads,
            )
        results = predict_multi_horizon(wf_cfg, ticker, raw_X)
    else:
        results = predict_multi_horizon(cfg, ticker, raw_X)
    policy = classify_signal(cfg, results, meta)
    latest_price = float(meta.get("latest_price", 0.0) or 0.0)
    trade_plan = build_trade_plan(
        cfg,
        ticker=ticker,
        policy=policy,
        latest_price=latest_price,
        latest_risk_pct=float(meta.get("latest_risk_pct", 0.0) or 0.0),
    )
    return {
        "ticker": ticker,
        "as_of_date": str(as_of_date.date()),
        "latest_date": meta.get("latest_date"),
        "latest_price": latest_price,
        "policy": policy,
        "trade_plan": trade_plan,
        "horizons": results,
    }


def _normalize_signal_plan(
    by_date: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    by_symbol: dict[str, dict[str, dict[str, Any]]] = {}
    for date_key, symbols in by_date.items():
        actionable = []
        total_score = 0.0
        for ticker, signal in symbols.items():
            if not is_actionable_signal(signal):
                continue
            score = signal_score(signal)
            if score <= 0:
                continue
            actionable.append((ticker, signal, score))
            total_score += score

        for ticker, signal in symbols.items():
            signal["score"] = 0.0
            signal["target_weight"] = 0.0
            by_symbol.setdefault(ticker, {})[date_key] = signal

        if total_score <= 0:
            continue

        for ticker, signal, score in actionable:
            signal["score"] = float(score)
            signal["target_weight"] = float(score / total_score)
            by_symbol.setdefault(ticker, {})[date_key] = signal
    return by_date, by_symbol


def _build_signal_plan(
    cfg: dict[str, Any],
    tickers: list[str],
    rebalance_dates: list[pd.Timestamp],
    *,
    mode: str = "replay",
    run_id: str | None = None,
    autotune: bool = False,
    inner_threads: int | None = 1,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    prices_by_ticker = {ticker: load_prices(cfg, ticker, update=False) for ticker in tickers}
    train_cfg = _shadow_config(cfg, run_id or "simulation_shadow") if mode == "walkforward" else cfg
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for rebalance_date in rebalance_dates:
        date_key = str(rebalance_date.date())
        daily: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            signal = _build_signal_as_of(
                cfg,
                ticker,
                prices_by_ticker[ticker],
                rebalance_date,
                mode=mode,
                train_cfg=train_cfg,
                autotune=autotune,
                inner_threads=inner_threads,
            )
            if signal is None:
                continue
            daily[ticker] = signal
        by_date[date_key] = daily
    return _normalize_signal_plan(by_date)


def _pct_distance(reference_price: float, target_price: float) -> float:
    if reference_price <= 0 or target_price <= 0:
        return 0.0
    return abs((target_price / reference_price) - 1.0) * 100.0


def _simulation_costs(cfg: dict[str, Any]) -> dict[str, Any]:
    costs = execution_cost_config(cfg)
    return {
        "fee_mode": costs.fee_mode,
        "fee_amount": costs.fee_amount,
        "slippage_pct": costs.slippage_pct,
    }


def _simulation_execution(cfg: dict[str, Any]) -> dict[str, Any]:
    sim_cfg = cfg.get("simulation", {}) or {}
    execution = dict(sim_cfg.get("execution", {}) or {})
    return {
        "native_stop_loss": bool(execution.get("native_stop_loss", True)),
        "native_take_profit": bool(execution.get("native_take_profit", True)),
        "native_trailing": bool(execution.get("native_trailing", True)),
        "native_hold_bars": bool(execution.get("native_hold_bars", True)),
        "max_position_pct": float(execution.get("max_position_pct", 100.0) or 100.0),
    }


def _fee_mode_from_config(value: Any):
    if value is None:
        return None
    raw = str(value).strip().lower().replace("-", "_")
    if raw in {"", "none", "off", "disabled", "false"}:
        return None
    if FeeMode is None:
        return None
    mapping = {
        "order_percent": FeeMode.ORDER_PERCENT,
        "percent": FeeMode.ORDER_PERCENT,
        "per_order": FeeMode.PER_ORDER,
        "order": FeeMode.PER_ORDER,
        "per_share": FeeMode.PER_SHARE,
        "share": FeeMode.PER_SHARE,
    }
    if raw not in mapping:
        raise ValueError(
            "simulation.costs.fee_mode must be order_percent, per_order, per_share or none"
        )
    return mapping[raw]


def _position_mode(allow_short: bool):
    if PositionMode is None:
        return None
    return PositionMode.DEFAULT if allow_short else PositionMode.LONG_ONLY


def _build_strategy_config(
    cfg: dict[str, Any],
    *,
    initial_cash: float | None,
    max_positions: int | None,
    allow_short: bool,
    symbol_count: int,
):
    costs = _simulation_costs(cfg)
    slots = max_positions or max(1, int(symbol_count))
    return StrategyConfig(
        initial_cash=float(initial_cash or cfg.get("trading", {}).get("capital", 10000.0)),
        fee_mode=_fee_mode_from_config(costs.get("fee_mode")),
        fee_amount=float(costs.get("fee_amount", 0.0) or 0.0),
        position_mode=_position_mode(allow_short),
        max_long_positions=slots,
        max_short_positions=slots if allow_short else None,
        exit_on_last_bar=True,
        return_signals=True,
        return_stops=True,
    )


def _slippage_model_from_config(cfg: dict[str, Any]):
    if RandomSlippageModel is None:
        return None
    slippage_pct = max(0.0, float(_simulation_costs(cfg).get("slippage_pct", 0.0) or 0.0))
    if slippage_pct <= 0:
        return None
    return RandomSlippageModel(0.0, slippage_pct)


def _apply_native_trade_management(
    ctx,
    trade_plan: dict[str, Any],
    policy: dict[str, Any],
    *,
    latest_price: float,
    execution_cfg: dict[str, Any],
) -> None:
    if execution_cfg.get("native_stop_loss", True):
        stop_loss_pct = _pct_distance(
            latest_price,
            float(
                trade_plan.get("stop_initial", policy.get("stop_loss_price", latest_price))
                or latest_price
            ),
        )
        if stop_loss_pct > 0:
            ctx.stop_loss_pct = stop_loss_pct
    if execution_cfg.get("native_take_profit", True):
        stop_profit_pct = _pct_distance(
            latest_price,
            float(
                trade_plan.get("target_final", policy.get("target_price", latest_price))
                or latest_price
            ),
        )
        if stop_profit_pct > 0:
            ctx.stop_profit_pct = stop_profit_pct
    if execution_cfg.get("native_trailing", True) and bool(
        trade_plan.get("trailing_enabled", True)
    ):
        trailing_pct = float(trade_plan.get("trailing_distance_pct", 0.0) or 0.0)
        if trailing_pct > 0:
            ctx.stop_trailing_pct = trailing_pct
    if execution_cfg.get("native_hold_bars", True):
        hold_bars = int(trade_plan.get("max_hold_days", 0) or 0)
        if hold_bars > 0:
            ctx.hold_bars = hold_bars


def _signal_date_key(signal: Any) -> str:
    return str(pd.Timestamp(signal.bar_data.date[-1]).date())


def _lookup_signal_data(
    plan_by_symbol: dict[str, dict[str, dict[str, Any]]],
    symbol: str,
    date_key: str,
) -> dict[str, Any] | None:
    symbol_plan = plan_by_symbol.get(str(symbol), {})
    if date_key in symbol_plan:
        return symbol_plan[date_key]
    try:
        current = pd.Timestamp(date_key)
        candidates = [key for key in symbol_plan if pd.Timestamp(key) <= current]
    except Exception:
        candidates = []
    if not candidates:
        return None
    return symbol_plan[max(candidates, key=lambda key: pd.Timestamp(key))]


def _pending_signal_data(ctx: Any, signal: Any) -> dict[str, Any] | None:
    sessions = getattr(ctx, "sessions", {}) or {}
    session = sessions.get(str(signal.symbol), {}) or {}
    key = f"pending_{signal.type}_signal"
    data = session.get(key)
    return dict(data) if isinstance(data, dict) else None


def _clear_pending_signal(ctx: Any, signal: Any) -> None:
    sessions = getattr(ctx, "sessions", {}) or {}
    session = sessions.get(str(signal.symbol), {}) or {}
    session.pop(f"pending_{signal.type}_signal", None)


def _planned_entry_shares(
    cfg: dict[str, Any],
    signal: dict[str, Any],
    *,
    price: float,
    equity: float,
) -> int:
    if price <= 0 or equity <= 0:
        return 0
    trade_plan = trade_plan_from_signal(cfg, signal)
    trading_cfg = cfg.get("trading", {}) or {}
    execution_cfg = _simulation_execution(cfg)
    stop_price = float(trade_plan.get("stop_initial", 0.0) or 0.0)
    risk_per_share = abs(float(price) - stop_price) if stop_price > 0 else 0.0
    risk_cash = (
        float(equity) * max(0.0, float(trading_cfg.get("risk_per_trade_pct", 1.0) or 0.0)) / 100.0
    )
    risk_shares = int(risk_cash / risk_per_share) if risk_per_share > 0 and risk_cash > 0 else 0

    weight = min(1.0, max(0.0, float(signal.get("target_weight", 0.0) or 0.0)))
    weight_shares = int((float(equity) * weight) / float(price)) if weight > 0 else 0
    explicit_shares = int(trade_plan.get("position_size", 0) or 0)
    candidates = [shares for shares in (risk_shares, weight_shares) if shares > 0]
    shares = min(candidates) if candidates else explicit_shares

    max_position_pct = max(
        0.0, min(100.0, float(execution_cfg.get("max_position_pct", 100.0) or 100.0))
    )
    if max_position_pct > 0:
        max_shares = int((float(equity) * max_position_pct / 100.0) / float(price))
        shares = min(shares, max_shares) if shares > 0 else 0
    return max(0, int(shares))


def _position_size_handler_factory(
    cfg: dict[str, Any],
    plan_by_symbol: dict[str, dict[str, dict[str, Any]]],
):
    def pos_size_handler(ctx) -> None:
        equity = float(ctx.total_equity)
        for signal in ctx.signals():
            if signal.score is None or float(signal.score) <= 0:
                continue
            signal_data = _pending_signal_data(ctx, signal) or _lookup_signal_data(
                plan_by_symbol,
                str(signal.symbol),
                _signal_date_key(signal),
            )
            if not signal_data:
                ctx.set_shares(signal, 0)
                continue
            price = float(signal.bar_data.close[-1])
            shares = _planned_entry_shares(cfg, signal_data, price=price, equity=equity)
            ctx.set_shares(signal, shares)
            _clear_pending_signal(ctx, signal)

    return pos_size_handler


def _execution_fn_factory(
    plan_by_symbol: dict[str, dict[str, dict[str, Any]]],
    *,
    allow_short: bool,
    execution_cfg: dict[str, Any] | None = None,
):
    execution_cfg = execution_cfg or _simulation_execution({})

    def exec_fn(ctx) -> None:
        date_key = str(pd.Timestamp(ctx.dt).date())
        symbol_plan = plan_by_symbol.get(str(ctx.symbol), {})
        signal = symbol_plan.get(date_key)
        if not signal:
            return

        policy = signal.get("policy", {}) or {}
        trade_plan = trade_plan_from_signal({}, signal)
        label = str(policy.get("label", "NEUTRAL")).upper()
        score = float(signal.get("score", 0.0) or 0.0)
        latest_price = float(signal.get("latest_price", float(ctx.close[-1])) or 0.0)
        action = str(trade_plan.get("action", "WAIT")).upper()
        side = str(trade_plan.get("side", "FLAT")).upper()
        can_enter = bool(action == "ENTER" and score > 0)

        if "BUY" in label and side == "LONG":
            if ctx.short_pos() is not None:
                ctx.cover_all_shares()
                return
            if ctx.long_pos() is None and can_enter:
                ctx.buy_shares = 1
                ctx.score = score
                ctx.session["pending_buy_signal"] = signal
                _apply_native_trade_management(
                    ctx,
                    trade_plan,
                    policy,
                    latest_price=latest_price,
                    execution_cfg=execution_cfg,
                )
            return

        if "SELL" in label:
            if ctx.long_pos() is not None:
                ctx.sell_all_shares()
                return
            if allow_short and side == "SHORT" and ctx.short_pos() is None and can_enter:
                ctx.sell_shares = 1
                ctx.score = score
                ctx.session["pending_sell_signal"] = signal
                _apply_native_trade_management(
                    ctx,
                    trade_plan,
                    policy,
                    latest_price=latest_price,
                    execution_cfg=execution_cfg,
                )
            return

        if ctx.long_pos() is not None:
            ctx.sell_all_shares()
        elif ctx.short_pos() is not None:
            ctx.cover_all_shares()

    return exec_fn


def _metrics_to_dict(metrics_df: pd.DataFrame) -> dict[str, Any]:
    return metrics_frame_to_dict(metrics_df)


def _write_simulation_artifacts(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    *,
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    stops: pd.DataFrame | None = None,
    signal_plan: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, str]:
    out_dir = simulation_dir(cfg) / str(summary["run_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"
    trades_csv = out_dir / "trades.csv"
    orders_csv = out_dir / "orders.csv"
    stops_csv = out_dir / "stops.csv"
    signals_json = out_dir / "signal_plan.json"

    trades.to_csv(trades_csv, index=True)
    orders.to_csv(orders_csv, index=True)
    if stops is not None:
        stops.to_csv(stops_csv, index=True)
    artifacts = {
        "dir": str(out_dir),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "trades_csv": str(trades_csv),
        "orders_csv": str(orders_csv),
        "signals_json": str(signals_json),
    }
    if stops is not None:
        artifacts["stops_csv"] = str(stops_csv)
    write_json(summary_json, {**summary, "artifacts": artifacts})
    write_json(signals_json, signal_plan)

    metrics = summary.get("metrics", {}) or {}
    baselines = summary.get("baselines", {}) or {}
    baseline_comparison = summary.get("baseline_comparison", {}) or {}
    validation_decision = summary.get("validation_decision", {}) or {}
    execution = summary.get("pybroker_execution", {}) or {}
    costs = execution.get("costs", {}) or {}
    note = (
        "NOTE: This simulation retrains models inside shadow artifacts at each rebalance date."
        if summary.get("mode") == "pybroker_walkforward_shadow"
        else "NOTE: This simulation replays saved operational models."
    )
    hit_rate = float(metrics.get("hit_rate_pct", metrics.get("win_rate", 0.0)) or 0.0)
    avg_trade_return = float(
        metrics.get("avg_trade_return_pct", metrics.get("avg_return_pct", 0.0)) or 0.0
    )
    total_cost = float(metrics.get("total_cost", 0.0) or 0.0)
    cost_pct = float(metrics.get("cost_pct_initial_cash", 0.0) or 0.0)
    native_stops = (
        f"loss={execution.get('native_stop_loss', True)} "
        f"profit={execution.get('native_take_profit', True)} "
        f"trailing={execution.get('native_trailing', True)} "
        f"hold={execution.get('native_hold_bars', True)}"
    )
    costs_text = (
        f"fee_mode={costs.get('fee_mode', 'none')} "
        f"fee_amount={float(costs.get('fee_amount', 0.0) or 0.0):.4f} "
        f"slippage_pct={float(costs.get('slippage_pct', 0.0) or 0.0):.4f}"
    )
    profit_factor_display = metrics.get("profit_factor_display")
    if profit_factor_display is None:
        profit_factor_display = f"{float(metrics.get('profit_factor', 0.0) or 0.0):.2f}"

    lines = [
        f"RUN ID: {summary['run_id']}",
        f"MODE: {summary['mode']}",
        f"WINDOW: {summary['start_date']} -> {summary['end_date']}",
        f"TICKERS: {', '.join(summary.get('tickers', []) or [])}",
        f"REBALANCE DAYS: {summary['rebalance_days']}",
        f"POLICY PROFILE: {summary.get('policy_profile', 'strict')}",
        f"WARMUP BARS: {summary['warmup_bars']}",
        f"TRADES: {int(float(metrics.get('trade_count', 0) or 0))}",
        f"RETURN: {float(metrics.get('total_return_pct', 0.0) or 0.0):+.2f}%",
        f"WIN RATE: {float(metrics.get('win_rate', 0.0) or 0.0):.1f}%",
        f"HIT RATE: {hit_rate:.1f}%",
        f"AVG TRADE RETURN: {avg_trade_return:+.2f}%",
        f"PROFIT FACTOR: {profit_factor_display}",
        f"TURNOVER: {float(metrics.get('turnover_pct', 0.0) or 0.0):.2f}%",
        f"AVG EXPOSURE: {float(metrics.get('active_exposure_pct', 0.0) or 0.0):.2f}%",
        f"COST: {total_cost:+.2f} ({cost_pct:.2f}%)",
        f"TOTAL PNL: {float(metrics.get('total_pnl', 0.0) or 0.0):+.2f}",
        f"PYBROKER SIZING: {execution.get('position_sizing', 'n/a')}",
        f"NATIVE STOPS: {native_stops}",
        f"COSTS: {costs_text}",
        "",
        "BASELINES:",
    ]
    for name, payload in baselines.items():
        base_metrics = payload.get("metrics", {}) or {}
        lines.append(
            f"- {name}: return={float(base_metrics.get('total_return_pct', 0.0) or 0.0):+.2f}% "
            f"drawdown={float(base_metrics.get('max_drawdown_pct', 0.0) or 0.0):+.2f}% "
            f"trades={int(float(base_metrics.get('trade_count', 0) or 0))}"
        )
    if baseline_comparison:
        lines.extend(
            [
                "",
                "MODEL VS BASELINES:",
                f"DECISION: {baseline_comparison.get('decision', 'n/a')}",
                f"BEAT RATE: {float(baseline_comparison.get('beat_rate_pct', 0.0) or 0.0):.1f}%",
            ]
        )
        for row in baseline_comparison.get("rows", []) or []:
            delta_return = float(row.get("return_delta_pct", 0.0) or 0.0)
            delta_drawdown = float(row.get("drawdown_delta_pct", 0.0) or 0.0)
            delta_hit = float(row.get("hit_rate_delta_pct", 0.0) or 0.0)
            delta_pf = float(row.get("profit_factor_delta", 0.0) or 0.0)
            lines.append(
                f"- {row.get('baseline')}: delta_return={delta_return:+.2f}% "
                f"delta_drawdown={delta_drawdown:+.2f}% "
                f"delta_hit={delta_hit:+.2f}% "
                f"delta_pf={delta_pf:+.2f}"
            )
    if validation_decision:
        lines.extend(
            [
                "",
                "VALIDATION DECISION:",
                f"FINAL: {str(validation_decision.get('final_decision', 'inconclusive')).upper()}",
                f"SCORE: {float(validation_decision.get('score', 0.0) or 0.0):.1f}",
            ]
        )
        for item in validation_decision.get("explanation", []) or []:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            note,
            "Use it as a research sanity check before promoting operational rules.",
        ]
    )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifacts


def run_pybroker_replay(
    cfg: dict[str, Any],
    tickers: list[str],
    *,
    mode: str = "replay",
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    rebalance_days: int = 5,
    warmup_bars: int = 150,
    initial_cash: float | None = None,
    max_positions: int | None = None,
    allow_short: bool = False,
    walkforward_autotune: bool = False,
    inner_threads: int | None = 1,
) -> dict[str, Any]:
    _require_pybroker()
    mode = normalize_validation_mode(mode)
    canonical = [canonical_ticker(cfg, ticker) for ticker in tickers]
    run_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_ticker(canonical[0])}"
    default_start, default_end = _default_window()
    start_ts = _coerce_timestamp(start_date, default_start)
    end_ts = _coerce_timestamp(end_date, default_end)
    if end_ts <= start_ts:
        raise ValueError("simulation end date must be after start date")

    with _quiet_pybroker():
        bars = YFinance(auto_adjust=False).query(
            canonical, start_ts.to_pydatetime(), end_ts.to_pydatetime()
        )
    if bars is None or bars.empty:
        raise RuntimeError("PyBroker/YFinance returned no bars for the requested simulation window")
    baseline_cash = float(initial_cash or cfg.get("trading", {}).get("capital", 10000.0))
    baselines = evaluate_baselines(bars, canonical, initial_cash=baseline_cash)

    market_date_count = len(pd.Index(pd.to_datetime(bars["date"]).dt.normalize().unique()))
    effective_warmup = min(max(1, int(warmup_bars)), max(1, market_date_count - 1))
    rebalance_dates = _schedule_rebalance_dates(
        bars,
        rebalance_days=max(1, int(rebalance_days)),
        warmup_bars=effective_warmup,
    )
    if not rebalance_dates:
        raise RuntimeError(
            "simulation window is too short for the configured warmup/rebalance schedule"
        )

    plan_by_date, plan_by_symbol = _build_signal_plan(
        cfg,
        canonical,
        rebalance_dates,
        mode=mode,
        run_id=run_id,
        autotune=bool(walkforward_autotune),
        inner_threads=inner_threads,
    )
    config = _build_strategy_config(
        cfg,
        initial_cash=initial_cash,
        max_positions=max_positions,
        allow_short=allow_short,
        symbol_count=len(canonical),
    )
    strategy = Strategy(bars, start_ts.to_pydatetime(), end_ts.to_pydatetime(), config=config)
    slippage_model = _slippage_model_from_config(cfg)
    if slippage_model is not None:
        strategy.set_slippage_model(slippage_model)
    strategy.set_pos_size_handler(_position_size_handler_factory(cfg, plan_by_symbol))
    strategy.add_execution(
        _execution_fn_factory(
            plan_by_symbol,
            allow_short=allow_short,
            execution_cfg=_simulation_execution(cfg),
        ),
        canonical,
    )
    with _quiet_pybroker():
        result = strategy.backtest(
            start_date=start_ts.to_pydatetime(),
            end_date=end_ts.to_pydatetime(),
            warmup=max(1, int(effective_warmup)),
            disable_parallel=True,
        )

    metrics = enrich_model_metrics_from_execution(
        _metrics_to_dict(result.metrics_df),
        trades=result.trades,
        orders=result.orders,
        initial_cash=baseline_cash,
        start_date=start_ts,
        end_date=end_ts,
    )
    baseline_comparison = compare_model_to_baselines(metrics, baselines)
    validation_decision = make_validation_decision(metrics, baseline_comparison, cfg)
    summary = {
        "run_id": run_id,
        "mode": (
            "pybroker_walkforward_shadow" if mode == "walkforward" else "pybroker_artifact_replay"
        ),
        "tickers": canonical,
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "rebalance_days": int(rebalance_days),
        "warmup_bars": int(effective_warmup),
        "allow_short": bool(allow_short),
        "max_positions": int(max_positions or max(1, len(canonical))),
        "policy_profile": active_policy_profile(cfg),
        "metrics": metrics,
        "baselines": baselines,
        "baseline_comparison": baseline_comparison,
        "validation_decision": validation_decision,
        "rebalance_points": len(rebalance_dates),
        "signal_points": sum(len(items) for items in plan_by_date.values()),
        "pybroker_execution": {
            "position_sizing": "score_ranked_risk_handler",
            "native_stop_loss": bool(_simulation_execution(cfg).get("native_stop_loss", True)),
            "native_take_profit": bool(_simulation_execution(cfg).get("native_take_profit", True)),
            "native_trailing": bool(_simulation_execution(cfg).get("native_trailing", True)),
            "native_hold_bars": bool(_simulation_execution(cfg).get("native_hold_bars", True)),
            "costs": _simulation_costs(cfg),
        },
        "walkforward_autotune": bool(walkforward_autotune) if mode == "walkforward" else False,
    }
    summary["artifacts"] = _write_simulation_artifacts(
        cfg,
        summary,
        orders=result.orders,
        trades=result.trades,
        stops=result.stops,
        signal_plan=plan_by_date,
    )
    return summary
