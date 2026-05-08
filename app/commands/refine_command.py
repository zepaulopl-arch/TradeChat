from __future__ import annotations

import argparse

from ..config import load_config
from ..refine_service import (
    collect_refine_summary,
    render_refine_summary,
    render_removal_summary,
    render_removal_walkforward_summary,
    run_feature_removal,
    run_feature_removal_walkforward,
)
from ._shared import resolve_cli_tickers


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = resolve_cli_tickers(cfg, args)
    if bool(getattr(args, "walkforward", False)) and not bool(getattr(args, "removal", False)):
        raise SystemExit("refine --walkforward requires --removal")
    if bool(getattr(args, "removal", False)):
        if bool(getattr(args, "walkforward", False)):
            sim_cfg = cfg.get("simulation", {}) or {}
            summary = run_feature_removal_walkforward(
                cfg,
                tickers,
                profiles=args.profiles,
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
                autotune=bool(args.autotune),
                inner_threads=1,
            )
            for line in render_removal_walkforward_summary(summary):
                print(line)
            return
        summary = run_feature_removal(
            cfg,
            tickers,
            horizons=args.horizons,
            profiles=args.profiles,
            update=bool(args.update),
            autotune=bool(args.autotune),
            inner_threads=1,
        )
        for line in render_removal_summary(summary):
            print(line)
        return
    summary = collect_refine_summary(cfg, tickers)
    for line in render_refine_summary(summary):
        print(line)
