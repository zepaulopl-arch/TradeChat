from __future__ import annotations

import argparse

from ..config import load_config, models_dir
from ..data import data_status, load_prices
from ..pipeline_service import fundamentals_data_status, sentiment_data_status
from ..report import print_data_summary
from ..utils import safe_ticker
from ._shared import resolve_cli_tickers

DATA_ACTIONS = {"load", "status", "audit"}


def _normalize_data_args(args: argparse.Namespace) -> argparse.Namespace:
    first = getattr(args, "data_action_or_ticker", None)
    rest = list(getattr(args, "tickers", []) or [])
    if first in DATA_ACTIONS:
        args.data_action = first
        args.tickers = rest
    else:
        args.data_action = "load"
        args.tickers = ([first] if first else []) + rest
    return args


def _status_payload(cfg: dict, ticker: str, *, status: str) -> dict:
    st = data_status(cfg, ticker)
    st["status"] = status
    st["period"] = cfg.get("data", {}).get("period", "n/a")
    st["min_rows"] = cfg.get("data", {}).get("min_rows")
    canonical = st.get("ticker", ticker)
    st["fundamentals"] = fundamentals_data_status(cfg, canonical)
    st["sentiment"] = sentiment_data_status(cfg, canonical)
    horizon_status = []
    for horizon in ["d1", "d5", "d20"]:
        path = models_dir(cfg) / safe_ticker(canonical) / f"latest_train_{horizon}.json"
        horizon_status.append(f"{horizon.upper()}: {'Ok' if path.exists() else 'None'}")
    st["models"] = " | ".join(horizon_status)
    return st


def run(args: argparse.Namespace) -> None:
    args = _normalize_data_args(args)
    cfg = load_config(args.config)
    tickers = resolve_cli_tickers(cfg, args)
    action = str(getattr(args, "data_action", "load"))

    for ticker in tickers:
        if action == "load":
            load_prices(cfg, ticker, update=True)
            print_data_summary(_status_payload(cfg, ticker, status="updated"))
        elif action in {"status", "audit"}:
            status = "cached" if data_status(cfg, ticker).get("cache_exists") else "not_found"
            print_data_summary(_status_payload(cfg, ticker, status=status))
        else:
            raise SystemExit(f"unknown data action: {action}")
