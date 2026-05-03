from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, artifact_dir
from .data import load_prices, data_status, resolve_asset
from .features import build_dataset
from .report import print_data_summary, print_train_summary, print_signal, write_txt_report
from .utils import normalize_ticker, parse_tickers, safe_ticker, write_json, read_json


def _canonical_ticker(cfg: dict[str, Any], ticker: str) -> str:
    return resolve_asset(cfg, ticker)["canonical"]


def _build_current_dataset(cfg: dict[str, Any], ticker: str, update: bool = False):
    ticker = _canonical_ticker(cfg, ticker)
    prices = load_prices(cfg, ticker, update=update)
    return build_dataset(cfg, prices, ticker)


def _fundamentals_data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    fcfg = cfg.get("features", {}).get("fundamentals", {}) or {}
    if not bool(fcfg.get("enabled", True)):
        return {"status": "disabled", "source": "features.yaml"}
    try:
        from .fundamentals import yahoo_snapshot
        snap = yahoo_snapshot(ticker)
        has_values = any(float(snap.get(key, 0) or 0) != 0 for key in ("pl", "dy", "pvp", "roe"))
        return {
            "status": "available" if has_values else "metadata_only",
            "source": str(snap.get("source", "yfinance")),
        }
    except Exception as exc:
        return {"status": "unavailable", "source": str(exc)[:48]}


def _sentiment_data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    scfg = cfg.get("features", {}).get("sentiment", {}) or {}
    if not bool(scfg.get("enabled", False)):
        return {"status": "disabled", "cache": "off"}
    try:
        from .sentiment import update_sentiment_cache
        meta = update_sentiment_cache(ticker, cfg)
        return {
            "status": "updated",
            "cache": "ok",
            "new_items": int(meta.get("new_items", 0) or 0),
            "cache_rows": int(meta.get("cache_rows", 0) or 0),
        }
    except Exception as exc:
        return {"status": "unavailable", "cache": str(exc)[:48]}


def cmd_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in parse_tickers(args.tickers):
        load_prices(cfg, ticker, update=True)
        st = data_status(cfg, ticker)
        st["status"] = "updated" if st.get("cache_exists") else "not_found"
        st["period"] = cfg.get("data", {}).get("period", "n/a")
        st["min_rows"] = cfg.get("data", {}).get("min_rows")
        canonical = st.get("ticker", ticker)
        st["fundamentals"] = _fundamentals_data_status(cfg, canonical)
        st["sentiment"] = _sentiment_data_status(cfg, canonical)
        print_data_summary(st)

def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in parse_tickers(args.tickers):
        ticker = _canonical_ticker(cfg, ticker)
        X, y, meta = _build_current_dataset(cfg, ticker, update=args.update)
        from .models import train_models
        manifest = train_models(cfg, ticker, X, y, meta, autotune=bool(args.autotune or cfg.get("model", {}).get("autotune", {}).get("enabled_by_default", False)))
        print_train_summary(manifest)


def _make_signal(cfg: dict[str, Any], ticker: str, update: bool = False) -> dict[str, Any]:
    ticker = _canonical_ticker(cfg, ticker)
    X, y, meta = _build_current_dataset(cfg, ticker, update=update)
    from .models import predict_with_model
    from .policy import classify_signal
    pred = predict_with_model(cfg, ticker, X)
    policy = classify_signal(cfg, pred, meta)
    latest_price = float(meta["latest_price"])
    target_price = latest_price * (1 + float(pred["prediction_return"]))
    fundamentals = meta.get("fundamentals", {})
    signal = {
        "ticker": normalize_ticker(ticker),
        "latest_date": meta.get("latest_date"),
        "latest_price": latest_price,
        "target_price": float(target_price),
        "prediction": pred,
        "policy": policy,
        "fundamentals": fundamentals,
        "features_used": meta.get("features", []),
        "train_run_id": pred.get("train_manifest", {}).get("run_id"),
    }
    out_dir = artifact_dir(cfg) / safe_ticker(ticker)
    write_json(out_dir / "latest_signal.json", signal)
    return signal


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in parse_tickers(args.tickers):
        signal = _make_signal(cfg, ticker, update=args.update)
        print_signal(signal)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in parse_tickers(args.tickers):
        path = artifact_dir(cfg) / safe_ticker(ticker) / "latest_signal.json"
        if not path.exists() or args.refresh:
            signal = _make_signal(cfg, ticker, update=args.update)
        else:
            signal = read_json(path)
        report_path = write_txt_report(cfg, signal)
        print(f"report written: {report_path}")


def cmd_daily(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = []
    for ticker in parse_tickers(args.tickers):
        try:
            # Daily intentionally does not train. It refreshes data, predicts with the saved model,
            # saves latest_signal.json and prints only the compact operational prediction table.
            signal = _make_signal(cfg, ticker, update=True)
            if cfg.get("daily", {}).get("generate_report", False):
                write_txt_report(cfg, signal)
            rows.append((signal["ticker"], signal["policy"]["label"], signal["policy"]["score_pct"], signal["policy"]["confidence_pct"], signal["target_price"]))
        except Exception as exc:
            rows.append((normalize_ticker(ticker), "ERROR", 0.0, 0.0, str(exc)))
    print("\nTRADEGEM DAILY")
    print("-" * 80)
    print(f"{'ticker':<12} {'signal':<14} {'return':>9} {'conf':>8} {'target/error':>24}")
    print("-" * 80)
    for t, label, ret, conf, target in rows:
        if isinstance(target, float):
            target_s = f"R$ {target:.2f}"
        else:
            target_s = str(target)[:24]
        print(f"{t:<12} {label:<14} {ret:>+8.2f}% {conf:>7.0f}% {target_s:>24}")
    print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradegem", description="Simple, practical CLI for B3 signal generation.")
    parser.add_argument("--config", default=None, help="optional config.yaml path")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="update and validate local data cache")
    data.add_argument("tickers", nargs="+", help="comma/space separated tickers")
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="train and persist models")
    train.add_argument("tickers", nargs="+")
    train.add_argument("--update", action="store_true", help="refresh price cache before training")
    train.add_argument("--autotune", action="store_true", help="tune XGB, RandomForest and MLP with BayesSearchCV before Ridge arbitration")
    train.set_defaults(func=cmd_train)

    pred = sub.add_parser("predict", help="generate signal using saved model")
    pred.add_argument("tickers", nargs="+")
    pred.add_argument("--update", action="store_true", help="refresh price cache before prediction")
    pred.set_defaults(func=cmd_predict)

    rep = sub.add_parser("report", help="write detailed TXT audit report")
    rep.add_argument("tickers", nargs="+")
    rep.add_argument("--refresh", action="store_true", help="regenerate signal before reporting")
    rep.add_argument("--update", action="store_true", help="refresh price cache when regenerating")
    rep.set_defaults(func=cmd_report)

    daily = sub.add_parser("daily", help="run data -> predict, without training")
    daily.add_argument("tickers", nargs="+")
    daily.set_defaults(func=cmd_daily)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
