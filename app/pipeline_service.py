from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_data_registry, models_dir
from .cvm_conn import CVMConnector
from .data import get_asset_profile, load_prices, resolve_asset
from .features import build_dataset
from .fundamentals import yahoo_snapshot
from .models import predict_multi_horizon
from .policy import classify_signal
from .sentiment import update_sentiment_cache
from .trade_plan_service import build_trade_plan
from .utils import normalize_ticker, parse_tickers, read_json, safe_ticker, write_json


def canonical_ticker(cfg: dict[str, Any], ticker: str) -> str:
    return resolve_asset(cfg, ticker)["canonical"]


def is_active_asset(cfg: dict[str, Any], ticker: str) -> bool:
    profile = resolve_asset(cfg, ticker).get("profile", {}) or {}
    return str(profile.get("registry_status", "active")).lower() == "active"


def resolve_tickers(cfg: dict[str, Any], raw_tickers: list[str]) -> list[str]:
    if "ALL" in [ticker.upper() for ticker in raw_tickers]:
        registry = load_data_registry(cfg)
        assets = registry.get("assets", {})
        return [
            ticker
            for ticker, meta in assets.items()
            if isinstance(meta, dict) and meta.get("registry_status") == "active"
        ]
    return [ticker for ticker in parse_tickers(raw_tickers) if is_active_asset(cfg, ticker)]


def build_current_dataset(cfg: dict[str, Any], ticker: str, update: bool = False):
    ticker = canonical_ticker(cfg, ticker)
    prices = load_prices(cfg, ticker, update=update)
    return build_dataset(cfg, prices, ticker)


def fundamentals_data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    fcfg = cfg.get("features", {}).get("fundamentals", {}) or {}
    if not bool(fcfg.get("enabled", True)):
        return {"status": "disabled", "source": "features.yaml"}
    try:
        profile = get_asset_profile(cfg, ticker)
        cnpj = profile.get("cnpj")
        conn = CVMConnector()
        has_cvm = False
        if cnpj and conn.db is not None and not conn.db.empty:
            import re

            clean_cnpj = re.sub(r"\D", "", str(cnpj)).strip()
            db_cnpjs = conn.db["CNPJ_CLEAN"].astype(str).str.strip()
            has_cvm = not conn.db[db_cnpjs == clean_cnpj].empty

        snap = yahoo_snapshot(ticker)
        return {
            "status": "available" if (has_cvm or snap.get("pl")) else "metadata_only",
            "source": "cvm" if has_cvm else "yfinance",
        }
    except Exception as exc:
        return {"status": "unavailable", "source": str(exc)[:48]}


def sentiment_data_status(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
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
            "is_fresh": meta.get("status") == "fresh",
        }
    except Exception as exc:
        return {"status": "unavailable", "cache": str(exc)[:48]}


def latest_signal_path(cfg: dict[str, Any], ticker: str) -> Path:
    return models_dir(cfg) / safe_ticker(canonical_ticker(cfg, ticker)) / "latest_signal.json"


def load_latest_signal(cfg: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    path = latest_signal_path(cfg, ticker)
    return read_json(path) if path.exists() else None


def make_signal(cfg: dict[str, Any], ticker: str, update: bool = False) -> dict[str, Any]:
    ticker = canonical_ticker(cfg, ticker)
    raw_X, _all_y, meta = build_current_dataset(cfg, ticker, update=update)
    results = predict_multi_horizon(cfg, ticker, raw_X)

    pred_d1 = results.get("d1", {})
    if "error" in pred_d1:
        pred_d1 = {"prediction_return": 0.0, "confidence": 0.0, "error": pred_d1["error"]}

    policy = classify_signal(cfg, results, meta)
    trigger_h = policy.get("horizon", "d1")
    pred_trigger = results.get(trigger_h, results.get("d1", {}))
    latest_price = float(meta["latest_price"])
    target_price = latest_price * (1 + float(pred_trigger.get("prediction_return", 0.0)))
    trade_plan = build_trade_plan(
        cfg,
        ticker=ticker,
        policy=policy,
        latest_price=latest_price,
        latest_risk_pct=float(meta.get("latest_risk_pct", 0.0) or 0.0),
    )

    signal = {
        "ticker": normalize_ticker(ticker),
        "latest_date": meta.get("latest_date"),
        "latest_price": latest_price,
        "target_price": float(trade_plan.get("target_final", target_price)),
        "prediction": pred_d1,
        "horizons": results,
        "policy": policy,
        "trade_plan": trade_plan,
        "fundamentals": meta.get("fundamentals", {}),
        "sentiment_value": meta.get("sentiment_value", 0.0),
        "features_used": meta.get("features", []),
        "train_run_id": pred_d1.get("train_manifest", {}).get("run_id"),
    }
    out_dir = models_dir(cfg) / safe_ticker(ticker)
    write_json(out_dir / "latest_signal.json", signal)
    return signal
