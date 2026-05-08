from __future__ import annotations

import argparse

from ..config import load_config
from ..simulator_service import run_pybroker_replay
from ..validation_view import render_validation_summary
from ._shared import resolve_cli_tickers


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = resolve_cli_tickers(cfg, args)
    sim_cfg = cfg.get("simulation", {}) or {}
    mode = str(args.mode or sim_cfg.get("mode", "replay") or "replay").lower()
    summary = run_pybroker_replay(
        cfg,
        tickers,
        mode=mode,
        start_date=args.start,
        end_date=args.end,
        rebalance_days=(
            args.rebalance_days
            if args.rebalance_days > 0
            else int(sim_cfg.get("rebalance_days", 5) or 5)
        ),
        warmup_bars=(
            args.warmup_bars
            if args.warmup_bars > 0
            else int(sim_cfg.get("warmup_bars", 150) or 150)
        ),
        initial_cash=args.cash,
        max_positions=args.max_positions,
        allow_short=bool(args.allow_short or sim_cfg.get("allow_short", False)),
        walkforward_autotune=bool(args.walkforward_autotune),
        inner_threads=1,
    )
    for line in render_validation_summary(
        summary,
        mode=mode,
        screen_title=str(getattr(args, "screen_title", "VALIDATE")),
        verbose=bool(args.verbose),
    ):
        print(line)
