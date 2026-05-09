from __future__ import annotations

import argparse

from ..config import load_config
from ..pipeline_service import latest_signal_path, make_signal
from ..policy import apply_policy_profile
from ..ranking_service import render_ranking
from ..report import print_signal, write_txt_report
from ..utils import read_json
from ._shared import resolve_cli_tickers


def _policy_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    profile = getattr(args, "policy_profile", None)
    return apply_policy_profile(cfg, profile) if profile else cfg


def _generate(cfg: dict, args: argparse.Namespace, *, print_output: bool = True) -> None:
    cfg = _policy_cfg(cfg, args)
    tickers = resolve_cli_tickers(cfg, args, required=True)
    for ticker in tickers:
        signal = make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
        if print_output:
            print_signal(
                signal,
                verbose=bool(getattr(args, "verbose", False)),
                diagnostic=bool(getattr(args, "diagnostic", False)),
                cfg=cfg,
            )


def _rank(cfg: dict, args: argparse.Namespace) -> None:
    cfg = _policy_cfg(cfg, args)
    tickers = resolve_cli_tickers(cfg, args, required=False)
    if tickers:
        for ticker in tickers:
            make_signal(cfg, ticker, update=bool(getattr(args, "update", False)))
    for line in render_ranking(
        cfg,
        limit=int(getattr(args, "rank_limit", 40) or 40),
        tickers=tickers or None,
        diagnostic=bool(getattr(args, "diagnostic", False)),
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
