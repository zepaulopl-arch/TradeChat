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


def cmd_signal(args: argparse.Namespace) -> None:
    return handlers.cmd_signal(args)


def cmd_portfolio(args: argparse.Namespace) -> None:
    return handlers.cmd_portfolio(args)


def cmd_validate(args: argparse.Namespace) -> None:
    return handlers.cmd_validate(args)


def cmd_refine(args: argparse.Namespace) -> None:
    return handlers.cmd_refine(args)


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--list", dest="asset_list", default=None, help="asset list from registry, e.g. validacao"
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "walkforward"],
        default=None,
        help="validation mode; default comes from config",
    )
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=0,
        help="bars between signal rebalances; 0 uses config default",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=0,
        help="minimum bars before the first rebalance; 0 uses config default",
    )
    parser.add_argument("--cash", type=float, default=None, help="initial cash for the validation")
    parser.add_argument("--max-positions", type=int, default=None, help="max long and short slots")
    parser.add_argument(
        "--allow-short", action="store_true", help="allow short entries on sell signals"
    )
    parser.add_argument(
        "--walkforward-autotune",
        action="store_true",
        help="autotune shadow models during walk-forward validation",
    )
    parser.add_argument("--verbose", action="store_true", help="show technical artifact paths")


def _add_refine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tickers", nargs="*")
    parser.add_argument(
        "--list", dest="asset_list", default=None, help="asset list from registry, e.g. validacao"
    )
    parser.add_argument(
        "--removal", action="store_true", help="train shadow feature-family removals"
    )
    parser.add_argument(
        "--walkforward",
        action="store_true",
        help="run removal profiles with walk-forward validation",
    )
    parser.add_argument(
        "--horizons", default="d1,d5,d20", help="comma-separated horizons for removal: d1,d5,d20"
    )
    parser.add_argument(
        "--profiles", default=None, help="comma-separated removal profiles; default runs all"
    )
    parser.add_argument("--update", action="store_true", help="refresh price cache before removal")
    parser.add_argument(
        "--autotune", action="store_true", help="run autotune inside shadow removal artifacts"
    )
    parser.add_argument("--start", default=None, help="walk-forward start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="walk-forward end date YYYY-MM-DD")
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=0,
        help="bars between walk-forward rebalances; 0 uses config default",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=0,
        help="minimum walk-forward bars before first rebalance; 0 uses config default",
    )
    parser.add_argument(
        "--cash", type=float, default=None, help="initial cash for walk-forward removal"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="max long and short slots for walk-forward removal",
    )
    parser.add_argument(
        "--allow-short", action="store_true", help="allow short entries during walk-forward removal"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradechat",
        description="TradeChat - Quantitative CLI for B3 assets",
        usage="python trade.py <command> [options]",
        epilog=(
            "Examples:\n"
            "  python trade.py data load PETR4.SA\n"
            "  python trade.py train PETR4.SA\n"
            "  python trade.py signal generate PETR4.SA\n"
            "  python trade.py signal rank --list validacao\n"
            "  python trade.py validate --list validacao --mode walkforward\n"
            "  python trade.py refine --list validacao --removal --walkforward\n"
            "  python trade.py portfolio status"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="optional config.yaml path")
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{data,train,signal,validate,refine,portfolio}",
    )

    data = sub.add_parser("data", help="Load, inspect and audit market data")
    data.add_argument("data_action_or_ticker", nargs="?", help="load/status/audit or ticker")
    data.add_argument("tickers", nargs="*", help="comma/space separated tickers")
    data.add_argument(
        "--list", dest="asset_list", default=None, help="asset list from registry, e.g. validacao"
    )
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="Train operational models")
    train.add_argument("tickers", nargs="*")
    train.add_argument(
        "--list", dest="asset_list", default=None, help="asset list from registry, e.g. validacao"
    )
    train.add_argument("--update", action="store_true", help="refresh price cache before training")
    train.add_argument(
        "--autotune",
        action="store_true",
        help="tune XGB, CatBoost and ExtraTrees with BayesSearchCV before Ridge arbitration",
    )
    train.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel workers for multi-asset training; 0 uses config default",
    )
    train.set_defaults(func=cmd_train)

    signal = sub.add_parser("signal", help="Generate signals, rankings and audit files")
    signal.add_argument("signal_action_or_ticker", nargs="?", help="generate/rank/report or ticker")
    signal.add_argument("tickers", nargs="*", help="comma/space separated tickers")
    signal.add_argument(
        "--list", dest="asset_list", default=None, help="asset list from registry, e.g. validacao"
    )
    signal.add_argument(
        "--update", action="store_true", help="refresh price cache before signal generation"
    )
    signal.add_argument("--refresh", action="store_true", help="regenerate signal before report")
    signal.add_argument("--rank-limit", type=int, default=40, help="max rows shown by signal rank")
    signal.set_defaults(func=cmd_signal)

    val = sub.add_parser("validate", help="Run replay/walk-forward validation and baselines")
    _add_validate_args(val)
    val.set_defaults(func=cmd_validate, screen_title="VALIDATE")

    refine = sub.add_parser("refine", help="Run controlled removal and contribution analysis")
    _add_refine_args(refine)
    refine.set_defaults(func=cmd_refine)

    port = sub.add_parser("portfolio", help="Inspect and manage portfolio actions")
    port.add_argument(
        "portfolio_action", nargs="?", choices=["status", "rebalance", "live"], default="status"
    )
    port_mode = port.add_mutually_exclusive_group()
    port_mode.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    port_mode.add_argument("--rebalance", action="store_true", help=argparse.SUPPRESS)
    port.set_defaults(func=cmd_portfolio)

    pred = sub.add_parser("predict", help=argparse.SUPPRESS)
    pred.add_argument("tickers", nargs="*")
    pred.add_argument("--list", dest="asset_list", default=None, help=argparse.SUPPRESS)
    pred.add_argument("--update", action="store_true", help=argparse.SUPPRESS)
    pred.add_argument("--rank", action="store_true", help=argparse.SUPPRESS)
    pred.add_argument("--rank-limit", type=int, default=40, help=argparse.SUPPRESS)
    pred.set_defaults(func=cmd_predict, deprecated_alias="predict")

    rep = sub.add_parser("report", help=argparse.SUPPRESS)
    rep.add_argument("tickers", nargs="*")
    rep.add_argument("--list", dest="asset_list", default=None, help=argparse.SUPPRESS)
    rep.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)
    rep.add_argument("--update", action="store_true", help=argparse.SUPPRESS)
    rep.set_defaults(func=cmd_report, deprecated_alias="report")

    sub._choices_actions = [
        action
        for action in sub._choices_actions
        if getattr(action, "dest", None) not in {"predict", "report"}
    ]

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
