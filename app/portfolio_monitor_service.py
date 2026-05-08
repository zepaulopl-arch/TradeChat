from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any

import yfinance as yf

from .portfolio_service import (
    load_latest_signal,
    load_portfolio_state,
    position_side,
    save_portfolio_state,
)
from .presentation import (
    C,
    banner,
    divider,
    money_br,
    paint,
    render_facts,
    render_table,
    screen_width,
    tone_delta,
    tone_signal,
)
from .trade_plan_service import (
    build_trade_plan,
    hit_stop,
    hit_target,
    is_long_plan,
    next_trailing_stop,
    partial_signed_shares,
)

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_live_price(ticker: str) -> float | None:
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        return None
    return None


def _position_plan(
    cfg: dict[str, Any], ticker: str, pos: dict[str, Any], signal_data: dict[str, Any]
) -> dict[str, Any]:
    plan = dict(pos.get("trade_plan") or signal_data.get("trade_plan") or {})
    if not plan:
        policy = signal_data.get("policy", {}) or {}
        plan = build_trade_plan(
            cfg,
            ticker=ticker,
            policy=policy,
            latest_price=float(signal_data.get("latest_price", pos.get("entry_price", 0.0)) or 0.0),
        )
    plan.setdefault(
        "stop_current", pos.get("stop_current", pos.get("stop_loss", plan.get("stop_initial", 0.0)))
    )
    plan.setdefault("target_1", pos.get("target_partial", plan.get("target_final", 0.0)))
    plan.setdefault("target_final", pos.get("target_final", plan.get("target_price", 0.0)))
    plan["partial_executed"] = bool(
        pos.get("partial_executed", plan.get("partial_executed", False))
    )
    plan["trailing_active"] = bool(pos.get("trailing_active", plan.get("trailing_active", False)))
    return plan


def _apply_signed_close(
    *,
    account: dict[str, Any],
    history: list[dict[str, Any]],
    ticker: str,
    pos: dict[str, Any],
    price: float,
    signed_shares: int,
    action: str,
) -> dict[str, Any]:
    entry_price = float(pos.get("entry_price", 0.0) or 0.0)
    signed_shares = int(signed_shares)
    pl_cash = (float(price) - entry_price) * signed_shares
    account["cash"] += signed_shares * float(price)
    record = {
        "ticker": ticker,
        "action": action,
        "price": float(price),
        "shares": signed_shares,
        "pl_cash": pl_cash,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    history.append(record)
    return record


def _close_positions_on_live_triggers(
    cfg: dict[str, Any],
    portfolio: dict[str, Any],
) -> list[dict[str, Any]]:
    positions = portfolio.get("positions", {}) or {}
    history = portfolio.get("history", []) or []
    account = portfolio.get("account", {"cash": 10000.0, "initial_capital": 10000.0})
    events_this_run: list[dict[str, Any]] = []
    changed = False

    for ticker in list(positions.keys()):
        pos = positions[ticker]
        signal_data = load_latest_signal(cfg, ticker)
        if not signal_data:
            continue

        live_price = get_live_price(ticker)
        if live_price is None:
            continue

        shares = int(pos.get("shares", 0))
        if shares == 0:
            continue

        plan = _position_plan(cfg, ticker, pos, signal_data)
        side = "LONG" if is_long_plan(plan, shares) else "SHORT"
        target_final = float(plan.get("target_final", 0.0) or 0.0)
        target_1 = float(plan.get("target_1", 0.0) or 0.0)
        stop_current = float(
            pos.get("stop_current", plan.get("stop_current", plan.get("stop_initial", 0.0))) or 0.0
        )

        if hit_stop(side, live_price, stop_current):
            action = (
                "TRAILING STOP"
                if bool(pos.get("trailing_active", plan.get("trailing_active", False)))
                else "STOP"
            )
            trade_record = _apply_signed_close(
                account=account,
                history=history,
                ticker=ticker,
                pos=pos,
                price=live_price,
                signed_shares=shares,
                action=action,
            )
            events_this_run.append(trade_record)
            del positions[ticker]
            changed = True
            continue

        if hit_target(side, live_price, target_final):
            trade_record = _apply_signed_close(
                account=account,
                history=history,
                ticker=ticker,
                pos=pos,
                price=live_price,
                signed_shares=shares,
                action="TARGET",
            )
            events_this_run.append(trade_record)
            del positions[ticker]
            changed = True
            continue

        partial_executed = bool(pos.get("partial_executed", plan.get("partial_executed", False)))
        if (
            not partial_executed
            and abs(shares) > 1
            and float(plan.get("partial_take_profit_pct", 0.0) or 0.0) > 0
            and hit_target(side, live_price, target_1)
        ):
            signed_partial = partial_signed_shares(
                shares, float(plan.get("partial_take_profit_pct", 50.0) or 50.0)
            )
            trade_record = _apply_signed_close(
                account=account,
                history=history,
                ticker=ticker,
                pos=pos,
                price=live_price,
                signed_shares=signed_partial,
                action="PARTIAL",
            )
            remaining = shares - signed_partial
            pos["shares"] = remaining
            pos["partial_executed"] = True
            if bool(plan.get("breakeven_after_partial", True)):
                stop_current = float(pos.get("entry_price", stop_current) or stop_current)
            pos["stop_current"] = stop_current
            plan["stop_current"] = stop_current
            plan["partial_executed"] = True
            pos["trade_plan"] = plan
            events_this_run.append(trade_record)
            changed = True
            continue

        if bool(plan.get("trailing_enabled", True)) and hit_target(
            side, live_price, float(plan.get("breakeven_trigger", target_1) or target_1)
        ):
            next_stop = next_trailing_stop(
                plan, side=side, price=live_price, current_stop=stop_current
            )
            if abs(next_stop - stop_current) > 1e-9:
                pos["stop_current"] = next_stop
                pos["trailing_active"] = True
                plan["stop_current"] = next_stop
                plan["trailing_active"] = True
                pos["trade_plan"] = plan
                changed = True

    if changed:
        portfolio["history"] = history
        save_portfolio_state(portfolio)
    return events_this_run


def render_live_portfolio(cfg: dict[str, Any]) -> list[str]:
    portfolio = load_portfolio_state(capital=float(cfg.get("trading", {}).get("capital", 10000.0)))
    _close_positions_on_live_triggers(cfg, portfolio)

    positions = portfolio.get("positions", {}) or {}
    history = portfolio.get("history", []) or []
    account = portfolio.get("account", {"cash": 10000.0, "initial_capital": 10000.0})
    width = screen_width()
    compact = width < 110

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_activity = [item for item in history if str(item.get("date", "")).startswith(today_str)]

    total_market_value = 0.0
    gross_market_value = 0.0
    live_prices: dict[str, float | None] = {}
    latest_signals: dict[str, dict[str, Any] | None] = {}
    for ticker, pos in positions.items():
        signal_data = load_latest_signal(cfg, ticker)
        live_price = get_live_price(ticker)
        live_prices[ticker] = live_price
        latest_signals[ticker] = signal_data
        current_price = (
            live_price
            if live_price is not None
            else (signal_data.get("latest_price", 0.0) if signal_data else 0.0)
        )
        shares = int(pos.get("shares", 0))
        exposure = shares * float(current_price or 0.0)
        total_market_value += exposure
        gross_market_value += abs(exposure)

    total_nav = float(account.get("cash", 0.0) or 0.0) + total_market_value
    initial_capital = float(account.get("initial_capital", 0.0) or 0.0)
    perf_pct = (total_nav / initial_capital - 1) * 100 if initial_capital else 0.0
    lines: list[str] = []

    lines.append("")
    lines.extend(
        banner(
            "TACTICAL PORTFOLIO",
            paint(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C.BLUE),
            width=width,
        )
    )
    lines.extend(
        render_facts(
            [
                ("Cash", money_br(float(account.get("cash", 0.0) or 0.0))),
                ("Net Exp", money_br(total_market_value), tone_delta(total_market_value)),
                ("Gross Exp", money_br(gross_market_value)),
                ("Equity", money_br(total_nav), tone_delta(perf_pct)),
                ("Perf", f"{perf_pct:+.2f}%", tone_delta(perf_pct)),
                ("Positions", len(positions)),
            ],
            width=width,
            max_columns=3,
        )
    )

    if today_activity:
        lines.append(paint("TODAY", C.DIM))
        if compact:
            activity_headers = ["ACTION", "TICKER", "P/L"]
            activity_rows = []
            for item in today_activity:
                has_pl = "pl_cash" in item
                pl_cash = float(item.get("pl_cash", 0.0) or 0.0)
                activity_rows.append(
                    [
                        item.get("action", item.get("exit_reason", "N/A")),
                        paint(item.get("ticker", "N/A"), C.BOLD),
                        (
                            paint(money_br(pl_cash), tone_delta(pl_cash))
                            if has_pl
                            else paint("n/a", C.DIM)
                        ),
                    ]
                )
            aligns = ["left", "left", "right"]
            min_widths = [6, 8, 12]
        else:
            activity_headers = ["ACTION", "TICKER", "PRICE", "SHARES", "P/L CASH"]
            activity_rows = []
            for item in today_activity:
                price = float(item.get("price", item.get("exit_price", 0.0)) or 0.0)
                shares = int(item.get("shares", 0) or 0)
                has_pl = "pl_cash" in item
                pl_cash = float(item.get("pl_cash", 0.0) or 0.0)
                activity_rows.append(
                    [
                        item.get("action", item.get("exit_reason", "N/A")),
                        paint(item.get("ticker", "N/A"), C.BOLD),
                        f"{price:.2f}",
                        str(shares),
                        (
                            paint(money_br(pl_cash), tone_delta(pl_cash))
                            if has_pl
                            else paint("n/a", C.DIM)
                        ),
                    ]
                )
            aligns = ["left", "left", "right", "right", "right"]
            min_widths = [6, 8, 8, 6, 12]
        lines.extend(
            render_table(
                activity_headers, activity_rows, width=width, aligns=aligns, min_widths=min_widths
            )
        )

    if not positions:
        lines.append(paint("Portfolio in cash. No active positions.", C.DIM))
        lines.append(divider(width))
        return lines

    if compact:
        headers = ["TICKER", "SIDE", "SHARES", "CURR", "P/L %", "SIGNAL"]
        aligns = ["left", "left", "right", "right", "right", "left"]
        min_widths = [8, 5, 6, 8, 7, 7]
    else:
        headers = [
            "TICKER",
            "SIDE",
            "SHARES",
            "ENTRY",
            "CURRENT",
            "P/L %",
            "SIGNAL",
            "TARGET",
            "STOP",
        ]
        aligns = ["left", "left", "right", "right", "right", "right", "left", "right", "right"]
        min_widths = [8, 5, 6, 8, 8, 7, 7, 8, 8]

    rows = []
    for ticker, pos in positions.items():
        signal_data = latest_signals.get(ticker)
        current_price = live_prices.get(ticker)
        if current_price is None:
            current_price = signal_data.get("latest_price", 0.0) if signal_data else 0.0
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)
        shares = int(pos.get("shares", 0))
        basis = abs(entry_price * shares)
        pl_cash = (float(current_price or 0.0) - entry_price) * shares
        pl_pct = (pl_cash / basis) * 100 if basis > 0 else 0.0

        policy = signal_data.get("policy", {}) if signal_data else {}
        plan = (
            _position_plan(cfg, ticker, pos, signal_data or {})
            if signal_data
            else dict(pos.get("trade_plan") or {})
        )
        target = float(plan.get("target_final", policy.get("target_price", 0.0)) or 0.0)
        stop = float(
            pos.get("stop_current", plan.get("stop_current", policy.get("stop_loss_price", 0.0)))
            or 0.0
        )
        current_signal = str(policy.get("label", plan.get("label", "N/A")))
        side = position_side(shares)
        side_tone = C.RED if side == "SHORT" else C.GREEN

        base_row = [
            paint(ticker, C.BOLD),
            paint(side, side_tone),
            str(shares),
            f"{float(current_price or 0.0):.2f}" if compact else f"{entry_price:.2f}",
        ]
        if compact:
            base_row.extend(
                [
                    paint(f"{pl_pct:+.2f}%", tone_delta(pl_pct)),
                    paint(current_signal, tone_signal(current_signal)),
                ]
            )
        else:
            base_row.extend(
                [
                    f"{float(current_price or 0.0):.2f}",
                    paint(f"{pl_pct:+.2f}%", tone_delta(pl_pct)),
                    paint(current_signal, tone_signal(current_signal)),
                    f"{target:.2f}",
                    f"{stop:.2f}",
                ]
            )
        rows.append(base_row)

    lines.extend(render_table(headers, rows, width=width, aligns=aligns, min_widths=min_widths))
    lines.append(paint("Monitoring active positions with auto exit on target or stop.", C.DIM))
    lines.append(divider(width))
    return lines
