from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from typing import Any

from .config import load_config, models_dir, reports_dir
from .data import load_prices, data_status, resolve_asset, get_asset_profile
from .features import build_dataset
from .report import print_data_summary, print_train_summary, print_signal, write_txt_report
from .utils import normalize_ticker, parse_tickers, safe_ticker, write_json, read_json
from .sentiment import update_sentiment_cache
from .fundamentals import yahoo_snapshot, add_fundamental_features
from .models import train_models, predict_multi_horizon
from .preparation import prepare_training_matrix
from .policy import classify_signal
from .cvm_conn import CVMConnector
from .execution import virtual_buy, load_portfolio, update_portfolio_prices


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
        profile = get_asset_profile(cfg, ticker)
        cnpj = profile.get("cnpj")
        
        # Direct check for CVM data without requiring a full DataFrame
        conn = CVMConnector()
        has_cvm = False
        if cnpj and conn.db is not None and not conn.db.empty:
            import re
            clean_cnpj = re.sub(r"\D", "", str(cnpj)).strip()
            db_cnpjs = conn.db['CNPJ_CLEAN'].astype(str).str.strip()
            has_cvm = not conn.db[db_cnpjs == clean_cnpj].empty
            
        snap = yahoo_snapshot(ticker)
        return {
            "status": "available" if (has_cvm or snap.get("pl")) else "metadata_only",
            "source": "cvm" if has_cvm else "yfinance",
        }
    except Exception as exc:
        return {"status": "unavailable", "source": str(exc)[:48]}


def _sentiment_data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    scfg = cfg.get("features", {}).get("sentiment", {}) or {}
    if not bool(scfg.get("enabled", False)):
        return {"status": "disabled", "cache": "off"}
    try:
        meta = update_sentiment_cache(ticker, cfg)
        return {
            "status": "updated",
            "cache": "ok",
            "new_items": int(meta.get("new_items", 0) or 0),
            "cache_rows": int(meta.get("cache_rows", 0) or 0),
            "is_fresh": meta.get("status") == "fresh"
        }
    except Exception as exc:
        return {"status": "unavailable", "cache": str(exc)[:48]}


def _resolve_tickers(cfg: dict[str, Any], raw_tickers: list[str]) -> list[str]:
    if "ALL" in [t.upper() for t in raw_tickers]:
        from .config import load_data_registry
        reg = load_data_registry(cfg)
        assets = reg.get("assets", {})
        return [t for t, meta in assets.items() if isinstance(meta, dict) and meta.get("registry_status") == "active"]
    return parse_tickers(raw_tickers)


def cmd_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in _resolve_tickers(cfg, args.tickers):
        load_prices(cfg, ticker, update=True)
        st = data_status(cfg, ticker)
        st["status"] = "updated" if st.get("cache_exists") else "not_found"
        st["period"] = cfg.get("data", {}).get("period", "n/a")
        st["min_rows"] = cfg.get("data", {}).get("min_rows")
        canonical = st.get("ticker", ticker)
        st["fundamentals"] = _fundamentals_data_status(cfg, canonical)
        st["sentiment"] = _sentiment_data_status(cfg, canonical)
        
        # Check horizons trained
        h_status = []
        for h in ["d1", "d5", "d20"]:
            p = models_dir(cfg) / safe_ticker(canonical) / f"latest_train_{h}.json"
            h_status.append(f"{h.upper()}: {'Ok' if p.exists() else 'None'}")
        st["models"] = " | ".join(h_status)
        
        print_data_summary(st)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    targets = ["target_return_d1", "target_return_d5", "target_return_d20"]
    
    for ticker in _resolve_tickers(cfg, args.tickers):
        try:
            ticker = _canonical_ticker(cfg, ticker)
            raw_X, all_y, meta = _build_current_dataset(cfg, ticker, update=args.update)
            
            print(f"\nTraining multi-horizon models for {ticker}...")
            for t_col in targets:
                horizon = t_col.split("_")[-1]
                y_series = all_y[t_col].dropna()
                X_prepared, y_prepared, prep_meta = prepare_training_matrix(raw_X.loc[y_series.index], y_series, cfg)
                
                h_meta = meta.copy()
                h_meta["preparation"] = prep_meta
                h_meta["horizon"] = horizon
                
                manifest = train_models(cfg, ticker, X_prepared, y_prepared, h_meta, 
                                       autotune=bool(args.autotune or cfg.get("model", {}).get("autotune", {}).get("enabled_by_default", False)),
                                       horizon=horizon)
                print_train_summary(manifest)
            
            print(f"All horizons trained for {ticker}.")
        except Exception as exc:
            print(f"\nERROR: Failed to train {ticker}: {exc}")


def _make_signal(cfg: dict[str, Any], ticker: str, update: bool = False) -> dict[str, Any]:
    ticker = _canonical_ticker(cfg, ticker)
    raw_X, all_y, meta = _build_current_dataset(cfg, ticker, update=update)
    
    # We predict for all horizons
    results = predict_multi_horizon(cfg, ticker, raw_X)
    
    # Operational signal is based on d1 (next day)
    pred_d1 = results.get("d1", {})
    if "error" in pred_d1:
        # Fallback to empty prediction if d1 is missing
        pred_d1 = {"prediction_return": 0.0, "confidence": 0.0, "error": pred_d1["error"]}
        
    policy = classify_signal(cfg, results, meta)
    
    # Use the predicted return of the horizon that triggered the signal for the target price
    trigger_h = policy.get("horizon", "d1")
    pred_trigger = results.get(trigger_h, results.get("d1", {}))
    
    latest_price = float(meta["latest_price"])
    target_price = latest_price * (1 + float(pred_trigger.get("prediction_return", 0.0)))
    
    fundamentals = meta.get("fundamentals", {})
    signal = {
        "ticker": normalize_ticker(ticker),
        "latest_date": meta.get("latest_date"),
        "latest_price": latest_price,
        "target_price": float(target_price),
        "prediction": pred_d1, # Primary prediction
        "horizons": results,   # All predictions (d1, d5, d20)
        "policy": policy,
        "fundamentals": fundamentals,
        "sentiment_value": meta.get("sentiment_value", 0.0),
        "features_used": meta.get("features", []),
        "train_run_id": pred_d1.get("train_manifest", {}).get("run_id"),
    }
    out_dir = models_dir(cfg) / safe_ticker(ticker)
    write_json(out_dir / "latest_signal.json", signal)
    return signal


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in _resolve_tickers(cfg, args.tickers):
        signal = _make_signal(cfg, ticker, update=args.update)
        print_signal(signal)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    for ticker in _resolve_tickers(cfg, args.tickers):
        path = models_dir(cfg) / safe_ticker(ticker) / "latest_signal.json"
        if not path.exists() or args.refresh:
            signal = _make_signal(cfg, ticker, update=args.update)
        else:
            signal = read_json(path)
        report_path = write_txt_report(cfg, signal)
        print(f"report written: {report_path}")


def cmd_daily(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    rows = []
    prices_map = {}
    for ticker in _resolve_tickers(cfg, args.tickers):
        try:
            # Daily intentionally does not train. It refreshes data, predicts with the saved model,
            # saves latest_signal.json and prints only the compact operational prediction table.
            signal = _make_signal(cfg, ticker, update=True)
            if cfg.get("daily", {}).get("generate_report", False):
                write_txt_report(cfg, signal)
            
            ticker_norm = signal["ticker"]
            prices_map[ticker_norm] = float(signal["latest_price"])
            
            rows.append((ticker_norm, signal["policy"]["label"], signal["policy"]["score_pct"], signal["policy"]["confidence_pct"], signal["target_price"]))
        except Exception as exc:
            rows.append((normalize_ticker(ticker), "ERROR", 0.0, 0.0, str(exc)))
            
    print("\nTRADECHAT DAILY")
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
    
    # Portfolio monitoring
    events = update_portfolio_prices(cfg, prices_map)
    if events:
        print("\nPORTFOLIO EVENTS")
        for e in events:
            print(f"-> {e}")
        print("-" * 80)


def cmd_buy(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ticker = normalize_ticker(args.ticker)
    # Generate a fresh signal to have latest prices and policy
    signal = _make_signal(cfg, ticker, update=True)
    try:
        pos = virtual_buy(cfg, ticker, signal)
        print(f"\n{ticker} added to virtual portfolio!")
        print(f"Entry: R$ {pos['entry_price']:.2f} | Shares: {pos['shares']} | Stop: R$ {pos['stop_loss']:.2f}")
    except Exception as exc:
        print(f"\nERROR: Could not execute virtual buy: {exc}")


def cmd_portfolio(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    p = load_portfolio(cfg)
    acc = p["account"]
    
    print("\nVIRTUAL PORTFOLIO")
    print("=" * 60)
    print(f"CASH: R$ {acc['cash']:,.2f} | INITIAL: R$ {acc['initial_capital']:,.2f}")
    print("-" * 60)
    
    positions = p["positions"]
    if not positions:
        print("No active positions.")
    else:
        print(f"{'TICKER':<10} {'SHARES':<8} {'ENTRY':<10} {'TARGET':<10} {'STOP':<10}")
        for t, pos in positions.items():
            print(f"{t:<10} {pos['shares']:<8} {pos['entry_price']:<10.2f} {pos['target_final']:<10.2f} {pos['stop_loss']:<10.2f}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradechat", description="Simple, practical CLI for B3 signal generation.")
    parser.add_argument("--config", default=None, help="optional config.yaml path")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="update and validate local data cache")
    data.add_argument("tickers", nargs="+", help="comma/space separated tickers")
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="train and persist models")
    train.add_argument("tickers", nargs="+")
    train.add_argument("--update", action="store_true", help="refresh price cache before training")
    train.add_argument("--autotune", action="store_true", help="tune XGB, CatBoost and ExtraTrees with BayesSearchCV before Ridge arbitration")
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

    buy = sub.add_parser("buy", help="simulated buy order")
    buy.add_argument("ticker")
    buy.set_defaults(func=cmd_buy)

    port = sub.add_parser("portfolio", help="show virtual portfolio")
    port.set_defaults(func=cmd_portfolio)
    
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
