from __future__ import annotations

import argparse


ROOT_COMMANDS = ("data", "train", "signal", "validate", "refine", "portfolio")


def add_ticker_list_args(parser: argparse.ArgumentParser, *, nargs: str = "*") -> None:
    parser.add_argument("tickers", nargs=nargs)
    parser.add_argument("--list", dest="asset_list", default=None, help="asset list from registry")


def add_validate_args(parser: argparse.ArgumentParser) -> None:
    add_ticker_list_args(parser)
    parser.add_argument(
        "--mode",
        choices=["replay", "walkforward"],
        default=None,
        help="validation mode; required for normal validate, defaults to replay for validate matrix",
    )
    for flag in ("--start", "--end"):
        parser.add_argument(flag, default=None)
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

    matrix = parser.add_argument_group("validate matrix/report")
    matrix.add_argument("--universe", default=None)
    matrix.add_argument("--profiles", nargs="+", default=None)
    matrix.add_argument("--jobs", type=int, default=1)
    matrix.add_argument("--log-dir", default=None)
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--max-assets", type=int, default=0)
    for flag in (
        "--skip-pytest",
        "--skip-data-audit",
        "--skip-signal-rank",
        "--skip-preflight",
        "--allow-untrained",
        "--skip-full-universe",
        "--include-full-universe",
        "--skip-per-asset",
        "--stop-on-error",
        "--serial-data-audit",
        "--latest",
    ):
        matrix.add_argument(flag, action="store_true")
    matrix.add_argument("--preflight-sample-size", type=int, default=10)
    matrix.add_argument("--out-dir", default=None)
    matrix.add_argument("--min-trades", type=int, default=5)
    matrix.add_argument("--min-pf", type=float, default=1.0)
    matrix.add_argument("--min-return-pct", type=float, default=0.0)


def add_refine_args(parser: argparse.ArgumentParser) -> None:
    add_ticker_list_args(parser)
    for flag in ("--removal", "--walkforward", "--update", "--autotune", "--allow-short"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--horizons", default="d1,d5,d20")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance-days", type=int, default=0)
    parser.add_argument("--warmup-bars", type=int, default=0)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)


def add_signal_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="signal_action", required=True)
    for action in ("generate", "rank", "report"):
        item = sub.add_parser(action)
        add_ticker_list_args(item, nargs="*" if action == "rank" else "*")
        if action in {"generate", "rank"}:
            item.add_argument("--update", action="store_true")
            item.add_argument("--diagnostic", action="store_true")
            item.add_argument("--policy-profile", default=None)
        if action == "generate":
            item.add_argument("--verbose", action="store_true")
        if action == "rank":
            item.add_argument("--rank-limit", type=int, default=40)
        if action == "report":
            item.add_argument("--refresh", action="store_true")
            item.add_argument("--update", action="store_true")


def add_data_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="data_action", required=True)
    for action in ("load", "status", "audit"):
        add_ticker_list_args(sub.add_parser(action))


def add_portfolio_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="portfolio_action", required=True)
    for action in ("status", "plan", "rebalance", "simulate", "live"):
        sub.add_parser(action)


def validate_normal_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if getattr(args, "command", None) != "validate":
        return
    tickers = list(getattr(args, "tickers", []) or [])
    if tickers and str(tickers[0]).lower() in {"matrix", "report"}:
        return
    if not getattr(args, "mode", None):
        parser.error("validate requires --mode replay|walkforward")
