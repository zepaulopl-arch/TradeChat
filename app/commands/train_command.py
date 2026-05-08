from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..batch_service import safe_worker_count, train_one_asset
from ..config import load_config
from ..presentation import banner, divider, paint, render_facts, screen_width
from ..report import C, print_multi_horizon_train_summary
from ._shared import resolve_cli_tickers


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


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    tickers = resolve_cli_tickers(cfg, args)
    width = screen_width()
    autotune_enabled = bool(
        args.autotune or cfg.get("model", {}).get("autotune", {}).get("enabled_by_default", False)
    )
    requested_workers = (
        args.workers if args.workers > 0 else int(cfg.get("batch", {}).get("train_workers", 1) or 1)
    )
    workers = safe_worker_count(len(tickers), requested=requested_workers, default=1)

    if len(tickers) > 1:
        print()
        for line in banner(
            "BATCH TRAINING", f"assets={len(tickers)}", f"workers={workers}", width=width
        ):
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
