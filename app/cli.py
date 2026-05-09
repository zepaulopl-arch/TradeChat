from __future__ import annotations

import argparse

from .commands import (
    data_command,
    portfolio_command,
    refine_command,
    signal_command,
    train_command,
    validate_command,
)

ROOT_COMMANDS = ("data", "train", "signal", "validate", "refine", "portfolio")


def _add_ticker_list_args(parser: argparse.ArgumentParser, *, nargs: str = "*") -> None:
    parser.add_argument("tickers", nargs=nargs)
    parser.add_argument("--list", dest="asset_list", default=None, help="asset list from registry")


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    _add_ticker_list_args(parser)
    parser.add_argument("--mode", choices=["replay", "walkforward"], required=True)
    parser.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    parser.add_argument("--rebalance-days", type=int, default=0)
    parser.add_argument("--warmup-bars", type=int, default=0)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--walkforward-autotune", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--policy-profile",
        default=None,
        help="policy profile for calibration: strict, balanced, active or relaxed",
    )


def _add_refine_args(parser: argparse.ArgumentParser) -> None:
    _add_ticker_list_args(parser)
    parser.add_argument("--removal", action="store_true")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--horizons", default="d1,d5,d20")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance-days", type=int, default=0)
    parser.add_argument("--warmup-bars", type=int, default=0)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--allow-short", action="store_true")


def _add_signal_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="signal_action", required=True)
    generate = sub.add_parser("generate", help="generate operational signals")
    _add_ticker_list_args(generate)
    generate.add_argument("--update", action="store_true")
    generate.add_argument("--verbose", action="store_true")
    generate.add_argument(
        "--diagnostic", action="store_true", help="show policy threshold diagnostics"
    )
    generate.add_argument(
        "--policy-profile",
        default=None,
        help="policy profile for calibration: strict, balanced, active or relaxed",
    )

    rank = sub.add_parser("rank", help="rank latest or freshly generated signals")
    _add_ticker_list_args(rank, nargs="*")
    rank.add_argument("--update", action="store_true")
    rank.add_argument("--rank-limit", type=int, default=40)
    rank.add_argument("--diagnostic", action="store_true", help="show policy blocker summary")
    rank.add_argument(
        "--policy-profile",
        default=None,
        help="policy profile for calibration: strict, balanced, active or relaxed",
    )

    report = sub.add_parser("report", help="write signal audit reports")
    _add_ticker_list_args(report)
    report.add_argument("--refresh", action="store_true")
    report.add_argument("--update", action="store_true")


def _add_data_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="data_action", required=True)
    for action in ("load", "status", "audit"):
        item = sub.add_parser(action)
        _add_ticker_list_args(item)


def _add_portfolio_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="portfolio_action", required=True)
    for action in ("status", "plan", "rebalance", "simulate", "live"):
        sub.add_parser(action)


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
        dest="command", required=True, metavar="{" + ",".join(ROOT_COMMANDS) + "}"
    )

    data = sub.add_parser("data", help="Load, inspect and audit market data")
    _add_data_subcommands(data)
    data.set_defaults(func=data_command.run)

    train = sub.add_parser("train", help="Train operational models")
    _add_ticker_list_args(train)
    train.add_argument("--update", action="store_true")
    train.add_argument("--autotune", action="store_true")
    train.add_argument("--workers", type=int, default=0)
    train.set_defaults(func=train_command.run)

    signal = sub.add_parser("signal", help="Generate signals, rankings and audit files")
    _add_signal_subcommands(signal)
    signal.set_defaults(func=signal_command.run)

    validate = sub.add_parser("validate", help="Run replay/walk-forward validation and baselines")
    _add_validate_args(validate)
    validate.set_defaults(func=validate_command.run, screen_title="VALIDATE")

    refine = sub.add_parser("refine", help="Run controlled removal and contribution analysis")
    _add_refine_args(refine)
    refine.set_defaults(func=refine_command.run)

    portfolio = sub.add_parser("portfolio", help="Inspect and manage portfolio actions")
    _add_portfolio_subcommands(portfolio)
    portfolio.set_defaults(func=portfolio_command.run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
