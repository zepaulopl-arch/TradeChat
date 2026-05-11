from __future__ import annotations

import argparse

from .cli_args import (
    ROOT_COMMANDS,
    add_data_subcommands,
    add_portfolio_subcommands,
    add_refine_args,
    add_signal_subcommands,
    add_ticker_list_args,
    add_validate_args,
    validate_normal_mode,
)
from .commands import (
    data_command,
    portfolio_command,
    refine_command,
    signal_command,
    train_command,
    validate_command,
)


class TradeChatParser(argparse.ArgumentParser):
    def parse_args(self, args: list[str] | None = None, namespace=None):
        parsed = super().parse_args(args, namespace)
        validate_normal_mode(parsed, self)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = TradeChatParser(
        prog="tradechat",
        description="TradeChat - Quantitative CLI for B3 assets",
        usage="python trade.py <command> [options]",
        epilog=(
            "Examples:\n"
            "  python trade.py data load PETR4.SA\n"
            "  python trade.py train PETR4.SA\n"
            "  python trade.py signal generate PETR4.SA --verbose\n"
            "  python trade.py signal rank --list validacao\n"
            "  python trade.py validate --list validacao --mode walkforward\n"
            "  python trade.py validate matrix --universe ibov --jobs 4\n"
            "  python trade.py validate report --latest\n"
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
    add_data_subcommands(data)
    data.set_defaults(func=data_command.run)

    train = sub.add_parser("train", help="Train operational models")
    add_ticker_list_args(train)
    train.add_argument("--update", action="store_true")
    train.add_argument("--autotune", action="store_true")
    train.add_argument("--workers", type=int, default=0)
    train.set_defaults(func=train_command.run)

    signal = sub.add_parser("signal", help="Generate signals, rankings and audit files")
    add_signal_subcommands(signal)
    signal.set_defaults(func=signal_command.run)

    validate = sub.add_parser("validate", help="Run replay/walk-forward validation and baselines")
    add_validate_args(validate)
    validate.set_defaults(func=validate_command.run, screen_title="VALIDATE")

    refine = sub.add_parser("refine", help="Run controlled removal and contribution analysis")
    add_refine_args(refine)
    refine.set_defaults(func=refine_command.run)

    portfolio = sub.add_parser("portfolio", help="Inspect and manage portfolio actions")
    add_portfolio_subcommands(portfolio)
    portfolio.set_defaults(func=portfolio_command.run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
