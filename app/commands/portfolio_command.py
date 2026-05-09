from __future__ import annotations

import argparse

from ..config import load_config
from ..pipeline_service import load_latest_signal
from ..portfolio_monitor_service import render_live_portfolio
from ..portfolio_service import load_portfolio_state, position_side
from ..presentation import (
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
from ..rebalance_service import rebalance_portfolio, render_rebalance_summary
from ..report import C


def _portfolio_action(args: argparse.Namespace) -> str:
    return str(getattr(args, "portfolio_action", None) or "status")


def _display_position_side(pos: dict) -> str:
    """Return the real position side, ignoring stale stored labels such as FLAT."""
    return position_side(int(pos.get("shares", 0) or 0))

def _render_status(cfg: dict) -> None:
    portfolio = load_portfolio_state(capital=float(cfg.get("trading", {}).get("capital", 10000.0)))
    account = portfolio["account"]
    positions = portfolio["positions"]
    width = screen_width()

    total_exposure = 0.0
    gross_exposure = 0.0
    rows: list[list[str]] = []
    for ticker, pos in positions.items():
        shares = int(pos.get("shares", 0) or 0)
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)
        signal = load_latest_signal(cfg, ticker) or {}
        policy = signal.get("policy", {}) or {}
        position = _display_position_side(pos)
        if position == "NONE":
            continue
        label = str(policy.get("label", "n/a") or "n/a").upper()
        target_value = pos.get("target_final", policy.get("target_price"))
        stop_value = pos.get("stop_loss", policy.get("stop_loss_price"))
        exposure = shares * entry_price
        total_exposure += exposure
        gross_exposure += abs(exposure)
        rows.append(
            [
                paint(ticker, C.BOLD),
                paint(position, C.RED if position == "SHORT" else C.GREEN),
                str(shares),
                f"{entry_price:.2f}",
                f"{float(target_value):.2f}" if target_value is not None else "n/a",
                f"{float(stop_value):.2f}" if stop_value is not None else "n/a",
                paint(label, tone_signal(label)),
            ]
        )

    equity = float(account.get("cash", 0.0) or 0.0) + total_exposure
    initial_capital = float(account.get("initial_capital", 0.0) or 0.0)
    perf_pct = (equity / initial_capital - 1) * 100 if initial_capital else 0.0

    print()
    for line in banner("VIRTUAL PORTFOLIO", width=width):
        print(line)
    for line in render_facts(
        [
            ("Cash", money_br(float(account.get("cash", 0.0) or 0.0))),
            ("Net Exp", money_br(total_exposure), tone_delta(total_exposure)),
            ("Gross Exp", money_br(gross_exposure)),
            ("Equity", money_br(equity), tone_delta(perf_pct)),
            ("Perf", f"{perf_pct:+.2f}%", tone_delta(perf_pct)),
            ("Positions", len(positions)),
        ],
        width=width,
        max_columns=3,
    ):
        print(line)

    if not rows:
        print(paint("No active positions.", C.DIM))
    else:
        for line in render_table(
            ["TICKER", "POSITION", "SHARES", "ENTRY", "TARGET", "STOP", "SIGNAL"],
            rows,
            width=width,
            aligns=["left", "left", "right", "right", "right", "right", "left"],
            min_widths=[8, 8, 6, 8, 8, 8, 7],
        ):
            print(line)
    print(divider(width))


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    action = _portfolio_action(args)
    if action == "live":
        for line in render_live_portfolio(cfg):
            print(line)
        return
    if action == "rebalance":
        summary = rebalance_portfolio(cfg, persist=True)
        for line in render_rebalance_summary(summary):
            print(line)
        return
    if action in {"plan", "simulate"}:
        summary = rebalance_portfolio(cfg, persist=False)
        for line in render_rebalance_summary(summary):
            print(line)
        return
    if action == "status":
        _render_status(cfg)
        return
    raise SystemExit(f"unknown portfolio action: {action}")
