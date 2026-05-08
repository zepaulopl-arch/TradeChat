from __future__ import annotations

from datetime import datetime
from typing import Any

from .portfolio_service import iter_latest_signals, load_portfolio_state, save_portfolio_state
from .presentation import C, banner, divider, money_br, paint, render_facts, render_table, screen_width, tone_signal
from .scoring import is_actionable_signal, signal_score
from .trade_plan_service import trade_plan_from_signal


def rebalance_portfolio(cfg: dict[str, Any]) -> dict[str, Any]:
    portfolio = load_portfolio_state(capital=float(cfg.get("trading", {}).get("capital", 10000.0)))
    current_positions = portfolio.get("positions", {})
    account = portfolio.get("account", {"cash": 10000.0, "initial_capital": 10000.0})
    history = portfolio.get("history", [])
    signals = iter_latest_signals(cfg)

    active_signals_map = {
        signal["ticker"]: signal
        for signal in signals
        if is_actionable_signal(signal)
    }

    closed_events: list[dict[str, Any]] = []
    for ticker, pos in list(current_positions.items()):
        sig = active_signals_map.get(ticker)
        shares = int(pos.get("shares", 0))
        is_long = shares > 0
        should_close = False
        reason = ""

        if not sig:
            should_close = True
            reason = "neutral / no signal"
        else:
            new_label = str((sig.get("policy", {}) or {}).get("label", ""))
            if is_long and "SELL" in new_label:
                should_close = True
                reason = "signal flipped to sell"
            elif (not is_long) and "BUY" in new_label:
                should_close = True
                reason = "signal flipped to buy"

        if should_close:
            exit_price = float(sig.get("latest_price") if sig else pos.get("entry_price", 0.0))
            pl_cash = (exit_price - float(pos.get("entry_price", 0.0))) * shares
            account["cash"] += shares * exit_price
            closed_record = {
                "ticker": ticker,
                "action": "CLOSE",
                "price": exit_price,
                "shares": shares,
                "pl_cash": pl_cash,
                "reason": reason,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            history.append(closed_record)
            closed_events.append(closed_record)
            del current_positions[ticker]

    kept_value = 0.0
    for ticker, pos in current_positions.items():
        sig = active_signals_map.get(ticker)
        price = float(sig.get("latest_price") if sig else pos.get("entry_price", 0.0))
        kept_value += int(pos.get("shares", 0)) * price

    total_equity = float(account.get("cash", 0.0) or 0.0) + kept_value
    scored_signals: list[tuple[str, dict[str, Any]]] = []
    total_score = 0.0
    for ticker, signal in active_signals_map.items():
        score = signal_score(signal)
        if score <= 0:
            continue
        signal["_temp_score"] = score
        scored_signals.append((ticker, signal))
        total_score += score

    final_positions: dict[str, dict[str, Any]] = {}
    plan_rows: list[dict[str, Any]] = []
    if total_score > 0:
        for ticker, signal in scored_signals:
            policy = signal.get("policy", {}) or {}
            trade_plan = trade_plan_from_signal(cfg, signal)
            label = str(policy.get("label", ""))
            side = str(trade_plan.get("side", "LONG")).upper()
            is_short = side == "SHORT"
            weight = signal["_temp_score"] / total_score
            amount_to_invest = total_equity * weight
            price = float(signal.get("latest_price", 0.0) or 0.0)
            if price <= 0:
                continue
            shares = int(amount_to_invest / price)
            plan_size = int(trade_plan.get("position_size", 0) or 0)
            if plan_size > 0:
                shares = min(shares, plan_size)
            if shares <= 0:
                continue

            signed_shares = -shares if is_short else shares
            final_positions[ticker] = {
                "shares": signed_shares,
                "entry_price": price,
                "date": signal.get("latest_date"),
                "side": side,
                "target_partial": float(trade_plan.get("target_1", price) or price),
                "target_final": float(trade_plan.get("target_final", price) or price),
                "stop_loss": float(trade_plan.get("stop_initial", price) or price),
                "stop_current": float(trade_plan.get("stop_current", trade_plan.get("stop_initial", price)) or price),
                "partial_executed": False,
                "trailing_active": False,
                "trade_plan": trade_plan,
            }

            old_shares = int(current_positions.get(ticker, {}).get("shares", 0))
            if old_shares == 0:
                status = "ENTER"
            elif (old_shares > 0 and signed_shares < 0) or (old_shares < 0 and signed_shares > 0):
                status = "REVERSE"
            elif old_shares != signed_shares:
                status = "ADJUST"
            else:
                status = "KEEP"

            plan_rows.append(
                {
                    "status": status,
                    "ticker": ticker,
                    "side": side,
                    "weight": weight,
                    "shares": signed_shares,
                    "price": price,
                    "target_1": float(trade_plan.get("target_1", price) or price),
                    "target_final": float(trade_plan.get("target_final", price) or price),
                    "stop_current": float(trade_plan.get("stop_current", trade_plan.get("stop_initial", price)) or price),
                    "signal": label,
                }
            )

    portfolio["positions"] = final_positions
    portfolio["account"]["cash"] = total_equity - sum(
        int(position["shares"]) * float(position["entry_price"])
        for position in final_positions.values()
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    for ticker, pos in final_positions.items():
        old_pos = current_positions.get(ticker)
        action = None
        if not old_pos:
            action = "ENTER"
        elif (int(old_pos["shares"]) > 0 and int(pos["shares"]) < 0) or (int(old_pos["shares"]) < 0 and int(pos["shares"]) > 0):
            action = "REVERSE"
        if action:
            history.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "price": pos["entry_price"],
                    "shares": pos["shares"],
                    "date": now_str,
                }
            )

    portfolio["history"] = history
    save_portfolio_state(portfolio)
    return {
        "portfolio": portfolio,
        "active_signals": len(active_signals_map),
        "scored_signals": len(scored_signals),
        "closed_events": closed_events,
        "equity": total_equity,
        "plan_rows": plan_rows,
        "final_positions": final_positions,
    }


def render_rebalance_summary(summary: dict[str, Any]) -> list[str]:
    width = screen_width()
    compact = width < 110
    portfolio = summary.get("portfolio", {}) or {}
    account = portfolio.get("account", {}) or {}
    lines: list[str] = []

    lines.append("")
    lines.extend(banner("TACTICAL REBALANCE", paint(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C.BLUE), width=width))
    lines.extend(
        render_facts(
            [
                ("Signals", int(summary.get("active_signals", 0))),
                ("Scored", int(summary.get("scored_signals", 0))),
                ("Closed", len(summary.get("closed_events", []) or [])),
                ("Equity", money_br(float(summary.get("equity", 0.0) or 0.0))),
                ("Cash After", money_br(float(account.get("cash", 0.0) or 0.0))),
                ("Active", len(summary.get("final_positions", {}) or {})),
            ],
            width=width,
            max_columns=3,
        )
    )

    closed_events = summary.get("closed_events", []) or []
    if closed_events:
        lines.append(paint("CLOSED", C.DIM))
        closed_rows = [
            [
                item["action"],
                paint(item["ticker"], C.BOLD),
                paint("SHORT" if int(item["shares"]) < 0 else "LONG", C.RED if int(item["shares"]) < 0 else C.GREEN),
                paint(money_br(float(item.get("pl_cash", 0.0) or 0.0)), C.GREEN if float(item.get("pl_cash", 0.0) or 0.0) >= 0 else C.RED),
                item.get("reason", "n/a"),
            ]
            for item in closed_events
        ]
        lines.extend(
            render_table(
                ["ACTION", "TICKER", "SIDE", "P/L CASH", "REASON"],
                closed_rows,
                width=width,
                aligns=["left", "left", "left", "right", "left"],
                min_widths=[6, 8, 5, 12, 16],
            )
        )

    lines.append(paint("TARGET ALLOCATION", C.DIM))
    plan_rows = summary.get("plan_rows", []) or []
    if not plan_rows:
        lines.append(paint("No actionable signals with positive score. Portfolio remains in cash.", C.DIM))
    else:
        headers = ["STATUS", "TICKER", "SIDE", "WEIGHT", "SHARES", "PRICE"]
        aligns = ["left", "left", "left", "right", "right", "right"]
        min_widths = [6, 8, 5, 7, 6, 8]
        rows = []
        for row in plan_rows:
            status = str(row["status"])
            status_tone = C.GREEN if status == "ENTER" else C.YELLOW if status == "REVERSE" else C.CYAN if status == "ADJUST" else C.DIM
            rendered = [
                paint(status, status_tone),
                paint(row["ticker"], C.BOLD),
                paint(row["side"], C.RED if row["side"] == "SHORT" else C.GREEN),
                f"{float(row['weight']) * 100:.1f}%",
                str(row["shares"]),
                f"{float(row['price']):.2f}",
            ]
            if not compact:
                rendered.extend(
                    [
                        f"{float(row.get('target_1', row['price'])):.2f}",
                        f"{float(row.get('target_final', row['price'])):.2f}",
                        f"{float(row.get('stop_current', row['price'])):.2f}",
                        paint(row["signal"], tone_signal(row["signal"])),
                    ]
                )
            rows.append(rendered)
        if not compact:
            headers.extend(["T1", "TARGET", "STOP", "SIGNAL"])
            aligns.extend(["right", "right", "right", "left"])
            min_widths.extend([8, 8, 8, 7])
        lines.extend(render_table(headers, rows, width=width, aligns=aligns, min_widths=min_widths))
    lines.append(divider(width))
    return lines
