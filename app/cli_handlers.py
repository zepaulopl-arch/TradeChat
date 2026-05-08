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


def cmd_data(args: argparse.Namespace) -> None:
    return data_command.run(args)


def cmd_train(args: argparse.Namespace) -> None:
    return train_command.run(args)


def cmd_signal(args: argparse.Namespace) -> None:
    return signal_command.run(args)


def cmd_predict(args: argparse.Namespace) -> None:
    return signal_command.run(args)


def cmd_report(args: argparse.Namespace) -> None:
    return signal_command.run(args)


def cmd_portfolio(args: argparse.Namespace) -> None:
    return portfolio_command.run(args)


def cmd_validate(args: argparse.Namespace) -> None:
    return validate_command.run(args)


def cmd_refine(args: argparse.Namespace) -> None:
    return refine_command.run(args)
