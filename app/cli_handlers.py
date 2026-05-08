from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from .batch_service import safe_worker_count, train_one_asset
from .config import load_config, models_dir
from .data import data_status, load_prices
from .pipeline_service import (
    fundamentals_data_status as _fundamentals_data_status,
    latest_signal_path as _latest_signal_path,
    load_latest_signal as _latest_signal_for,
    make_signal as _make_signal,
    resolve_tickers as _resolve_tickers,
    sentiment_data_status as _sentiment_data_status,
)
from .portfolio_monitor_service import render_live_portfolio
from .portfolio_service import load_portfolio_state, position_side
from .presentation import banner, divider, money_br, paint, render_facts, render_table, screen_width, tone_delta
from .ranking_service import render_ranking
from .rebalance_service import rebalance_portfolio, render_rebalance_summary
from .refine_service import collect_refine_summary, render_ablation_summary, render_refine_summary, run_feature_ablation
from .report import C, print_data_summary, print_multi_horizon_train_summary, print_signal, write_txt_report
from .simulator_service import run_pybroker_replay
from .utils import safe_ticker, read_json
from .validation_view import render_validation_summary


def cmd_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in _resolve_tickers(cfg, args.tickers):
        load_prices(cfg, ticker, update=True)
        st = data_status(cfg, ticker)
        st["status"] = "updated" if st.get("cache_exists") else "not_found"
        st["period"] = cfg.get("data", {}).get("period", "n/a")
        st["min_rows"] = cfg.get("data", {}).get("min_rows")
        canonical = st.get("ticker", ticker)
        st["fundamentals"] = _fundamentals_data_status(cfg, canonical)
        st["sentiment"] = _sentiment_data_status(cfg, canonical)
        
        # Check horizons trained
        h_status = []
        for h in ["d1", "d5", "d20"]:
            p = models_dir(cfg) / safe_ticker(canonical) / f"latest_train_{h}.json"
            h_status.append(f"{h.upper()}: {'Ok' if p.exists() else 'None'}")
        st["models"] = " | ".join(h_status)
        
        print_data_summary(st)


def _print_train_result(result: dict[str, object], *, width: int) -> None:
    ticker = str(result.get("ticker", "n/a"))
    manifests = list(result.get("manifests", []) or [])
    print()
    for line in banner("TRAINING", ticker, "multi-horizon", width=width):
        print(line)
    for line in render_facts(
        [
            ("Rows", result.get("rows", 0)),
            ("Autotune", bool(result.get("autotune", False))),
            ("Update", bool(result.get("update", False))),
        ],
        width=width,
        max_columns=3,
    ):
        print(line)
    print_multi_horizon_train_summary(manifests)
    print(paint(f"Training complete for {ticker}.", C.DIM))
    print(divider(width))


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = _resolve_tickers(cfg, args.tickers)
    width = screen_width()
    autotune_enabled = bool(args.autotune or cfg.get("model", {}).get("autotune", {}).get("enabled_by_default", False))
    requested_workers = args.workers if args.workers > 0 else int(cfg.get("batch", {}).get("train_workers", 1) or 1)
    workers = safe_worker_count(len(tickers), requested=requested_workers, default=1)

    if len(tickers) > 1:
        print()
        for line in banner("BATCH TRAINING", f"assets={len(tickers)}", f"workers={workers}", width=width):
            print(line)
        for line in render_facts(
            [
                ("Assets", len(tickers)),
                ("Workers", workers),
                ("Autotune", autotune_enabled),
                ("Update", bool(args.update)),
            ],
            width=width,
            max_columns=4,
        ):
            print(line)

    if workers == 1:
        for ticker in tickers:
            try:
                result = train_one_asset(
                    cfg,
                    ticker,
                    update=args.update,
                    autotune=autotune_enabled,
                    inner_threads=None,
                )
                _print_train_result(result, width=width)
            except Exception as exc:
                print(paint(f"ERROR: Failed to train {ticker}: {exc}", C.RED))
        return

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for ticker in tickers:
            futures[
                executor.submit(
                    train_one_asset,
                    cfg,
                    ticker,
                    update=args.update,
                    autotune=autotune_enabled,
                    inner_threads=1,
                )
            ] = ticker
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                _print_train_result(result, width=width)
            except Exception as exc:
                print(paint(f"ERROR: Failed to train {ticker}: {exc}", C.RED))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.tickers:
        for ticker in _resolve_tickers(cfg, args.tickers):
            signal = _make_signal(cfg, ticker, update=args.update)
            if not args.rank:
                print_signal(signal)
    elif not args.rank:
        raise SystemExit("predict requires at least one ticker unless --rank is used")
    if args.rank:
        for line in render_ranking(cfg, limit=args.rank_limit):
            print(line)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in _resolve_tickers(cfg, args.tickers):
        path = _latest_signal_path(cfg, ticker)
        if not path.exists() or args.refresh:
            signal = _make_signal(cfg, ticker, update=args.update)
        else:
            signal = read_json(path)
        report_path = write_txt_report(cfg, signal)
        print(f"report written: {report_path}")


def cmd_portfolio(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if bool(getattr(args, "live", False)):
        for line in render_live_portfolio(cfg):
            print(line)
        return
    if bool(getattr(args, "rebalance", False)):
        summary = rebalance_portfolio(cfg)
        for line in render_rebalance_summary(summary):
            print(line)
        return

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
        signal = _latest_signal_for(cfg, ticker) or {}
        policy = signal.get("policy", {}) or {}
        side = str(pos.get("side") or position_side(shares)).upper()
        target_value = pos.get("target_final", policy.get("target_price"))
        stop_value = pos.get("stop_loss", policy.get("stop_loss_price"))
        exposure = shares * entry_price
        total_exposure += exposure
        gross_exposure += abs(exposure)
        rows.append(
            [
                paint(ticker, C.BOLD),
                paint(side, C.RED if side == "SHORT" else C.GREEN),
                str(shares),
                f"{entry_price:.2f}",
                f"{float(target_value):.2f}" if target_value is not None else "n/a",
                f"{float(stop_value):.2f}" if stop_value is not None else "n/a",
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
            ["TICKER", "SIDE", "SHARES", "ENTRY", "TARGET", "STOP"],
            rows,
            width=width,
            aligns=["left", "left", "right", "right", "right", "right"],
            min_widths=[8, 5, 6, 8, 8, 8],
        ):
            print(line)
    print(divider(width))


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = _resolve_tickers(cfg, args.tickers)
    sim_cfg = cfg.get("simulation", {}) or {}
    mode = str(args.mode or sim_cfg.get("mode", "replay") or "replay").lower()
    summary = run_pybroker_replay(
        cfg,
        tickers,
        mode=mode,
        start_date=args.start,
        end_date=args.end,
        rebalance_days=args.rebalance_days if args.rebalance_days > 0 else int(sim_cfg.get("rebalance_days", 5) or 5),
        warmup_bars=args.warmup_bars if args.warmup_bars > 0 else int(sim_cfg.get("warmup_bars", 150) or 150),
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


def cmd_refine(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = _resolve_tickers(cfg, args.tickers)
    if bool(getattr(args, "ablation", False)):
        summary = run_feature_ablation(
            cfg,
            tickers,
            horizons=args.horizons,
            profiles=args.profiles,
            update=bool(args.update),
            autotune=bool(args.autotune),
            inner_threads=1,
        )
        for line in render_ablation_summary(summary):
            print(line)
        return
    summary = collect_refine_summary(cfg, tickers)
    for line in render_refine_summary(summary):
        print(line)
