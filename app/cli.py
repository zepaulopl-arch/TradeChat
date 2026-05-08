from __future__ import annotations

import argparse

from . import cli_handlers as handlers


def cmd_data(args: argparse.Namespace) -> None:
    return handlers.cmd_data(args)


def cmd_train(args: argparse.Namespace) -> None:
    return handlers.cmd_train(args)


def cmd_predict(args: argparse.Namespace) -> None:
    return handlers.cmd_predict(args)


def cmd_report(args: argparse.Namespace) -> None:
    return handlers.cmd_report(args)


def cmd_portfolio(args: argparse.Namespace) -> None:
    return handlers.cmd_portfolio(args)


def cmd_validate(args: argparse.Namespace) -> None:
    return handlers.cmd_validate(args)


def cmd_refine(args: argparse.Namespace) -> None:
    return handlers.cmd_refine(args)


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--mode", choices=["replay", "walkforward"], default=None, help="validation mode; default comes from config")
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    parser.add_argument("--rebalance-days", type=int, default=0, help="bars between signal rebalances; 0 uses config default")
    parser.add_argument("--warmup-bars", type=int, default=0, help="minimum bars before the first rebalance; 0 uses config default")
    parser.add_argument("--cash", type=float, default=None, help="initial cash for the validation")
    parser.add_argument("--max-positions", type=int, default=None, help="max long and short slots")
    parser.add_argument("--allow-short", action="store_true", help="allow short entries on sell signals")
    parser.add_argument("--walkforward-autotune", action="store_true", help="autotune shadow models during walk-forward validation")
    parser.add_argument("--verbose", action="store_true", help="show technical artifact paths")


def _add_refine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--removal", action="store_true", help="train shadow feature-family removals")
    parser.add_argument("--walkforward", action="store_true", help="run removal profiles with walk-forward validation")
    parser.add_argument("--horizons", default="d1,d5,d20", help="comma-separated horizons for removal: d1,d5,d20")
    parser.add_argument("--profiles", default=None, help="comma-separated removal profiles; default runs all")
    parser.add_argument("--update", action="store_true", help="refresh price cache before removal")
    parser.add_argument("--autotune", action="store_true", help="run autotune inside shadow removal artifacts")
    parser.add_argument("--start", default=None, help="walk-forward start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="walk-forward end date YYYY-MM-DD")
    parser.add_argument("--rebalance-days", type=int, default=0, help="bars between walk-forward rebalances; 0 uses config default")
    parser.add_argument("--warmup-bars", type=int, default=0, help="minimum walk-forward bars before first rebalance; 0 uses config default")
    parser.add_argument("--cash", type=float, default=None, help="initial cash for walk-forward removal")
    parser.add_argument("--max-positions", type=int, default=None, help="max long and short slots for walk-forward removal")
    parser.add_argument("--allow-short", action="store_true", help="allow short entries during walk-forward removal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradechat", description="Simple, practical CLI for B3 signal generation.")
    parser.add_argument("--config", default=None, help="optional config.yaml path")
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{data,train,predict,validate,refine,report,portfolio}",
    )

    data = sub.add_parser("data", help="update and validate local data cache")
    data.add_argument("tickers", nargs="+", help="comma/space separated tickers")
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="train and persist models")
    train.add_argument("tickers", nargs="+")
    train.add_argument("--update", action="store_true", help="refresh price cache before training")
    train.add_argument("--autotune", action="store_true", help="tune XGB, CatBoost and ExtraTrees with BayesSearchCV before Ridge arbitration")
    train.add_argument("--workers", type=int, default=0, help="parallel workers for multi-asset training; 0 uses config default")
    train.set_defaults(func=cmd_train)

    pred = sub.add_parser("predict", help="generate signal using saved model")
    pred.add_argument("tickers", nargs="*")
    pred.add_argument("--update", action="store_true", help="refresh price cache before prediction")
    pred.add_argument("--rank", action="store_true", help="show ranked table after generating signals")
    pred.add_argument("--rank-limit", type=int, default=40, help="max rows shown when using --rank")
    pred.set_defaults(func=cmd_predict)

    rep = sub.add_parser("report", help="write detailed TXT audit report")
    rep.add_argument("tickers", nargs="+")
    rep.add_argument("--refresh", action="store_true", help="regenerate signal before reporting")
    rep.add_argument("--update", action="store_true", help="refresh price cache when regenerating")
    rep.set_defaults(func=cmd_report)

    port = sub.add_parser("portfolio", help="show virtual portfolio")
    port_mode = port.add_mutually_exclusive_group()
    port_mode.add_argument("--live", action="store_true", help="monitor portfolio with live prices and target/stop exits")
    port_mode.add_argument("--rebalance", action="store_true", help="rebalance portfolio from latest actionable signals")
    port.set_defaults(func=cmd_portfolio)

    val = sub.add_parser("validate", help="validate signals with PyBroker replay or walk-forward")
    _add_validate_args(val)
    val.set_defaults(func=cmd_validate, screen_title="VALIDATE")

    refine = sub.add_parser("refine", help="audit trained feature families and model manifests")
    _add_refine_args(refine)
    refine.set_defaults(func=cmd_refine)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
