from __future__ import annotations

import argparse

from ..config import load_config
from ..pipeline_service import latest_signal_path, make_signal
from ..ranking_service import render_ranking
from ..report import print_signal, write_txt_report
from ..utils import read_json
from ._shared import resolve_cli_tickers


def _generate(cfg: dict, args: argparse.Namespace, *, print_output: bool = True) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=True)
    for ticker in tickers:
        signal = make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
        if print_output:
            print_signal(signal, verbose=bool(getattr(args, "verbose", False)))


def _rank(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=False)
    if tickers:
        for ticker in tickers:
            make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
    for line in render_ranking(
        cfg,
        limit=int(getattr(args, "rank_limit", 40) or 40),
        tickers=tickers or None,
    ):
        print(line)


def _report(cfg: dict, args: argparse.Namespace) -> None:
    for ticker in resolve_cli_tickers(cfg, args, required=True):
        path = latest_signal_path(cfg, ticker)
        if not path.exists() or bool(getattr(args, "refresh", False)):
            signal = make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
        else:
            signal = read_json(path)
        write_txt_report(cfg, signal)
        print("report written")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    action = str(getattr(args, "signal_action", "generate"))
    if action == "generate":
        _generate(cfg, args, print_output=True)
    elif action == "rank":
        _rank(cfg, args)
    elif action == "report":
        _report(cfg, args)
    else:
        raise SystemExit(f"unknown signal action: {action}")
