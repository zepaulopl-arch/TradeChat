from __future__ import annotations

import argparse

from ..config import load_config
from ..pipeline_service import latest_signal_path, make_signal
from ..ranking_service import render_ranking
from ..report import print_signal, write_txt_report
from ..utils import read_json
from ._shared import print_deprecated, resolve_cli_tickers

SIGNAL_ACTIONS = {"generate", "rank", "report"}


def _normalize_signal_args(args: argparse.Namespace) -> argparse.Namespace:
    first = getattr(args, "signal_action_or_ticker", None)
    rest = list(getattr(args, "tickers", []) or [])
    if getattr(args, "deprecated_alias", None) == "predict":
        print_deprecated("use 'signal generate' instead of 'predict'.")
        args.signal_action = "rank" if bool(getattr(args, "rank", False)) else "generate"
        return args
    if getattr(args, "deprecated_alias", None) == "report":
        print_deprecated("use 'signal report' instead of 'report'.")
        args.signal_action = "report"
        return args
    if first in SIGNAL_ACTIONS:
        args.signal_action = first
        args.tickers = rest
    else:
        args.signal_action = "generate"
        args.tickers = ([first] if first else []) + rest
    return args


def _generate(cfg: dict, args: argparse.Namespace, *, print_output: bool = True) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=True)
    for ticker in tickers:
        signal = make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
        if print_output:
            print_signal(signal)


def _rank(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=False)
    if tickers:
        for ticker in tickers:
            make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
    for line in render_ranking(cfg, limit=int(getattr(args, "rank_limit", 40) or 40)):
        print(line)


def _report(cfg: dict, args: argparse.Namespace) -> None:
    for ticker in resolve_cli_tickers(cfg, args, required=True):
        path = latest_signal_path(cfg, ticker)
        if not path.exists() or bool(getattr(args, "refresh", False)):
            signal = make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
        else:
            signal = read_json(path)
        report_path = write_txt_report(cfg, signal)
        print(f"report written: {report_path}")


def run(args: argparse.Namespace) -> None:
    args = _normalize_signal_args(args)
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
