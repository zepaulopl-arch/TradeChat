from __future__ import annotations

"""Run a full TradeGem diagnostic batch over registered assets.

This script intentionally does not add a new TradeGem CLI command. It is a
standalone maintenance/diagnostic tool for deciding whether engine, context,
feature-selection or registry changes are really needed.

Usage examples:
    python scripts/diagnose_assets.py
    python scripts/diagnose_assets.py --limit 10
    python scripts/diagnose_assets.py --assets PETR4,VALE3,AXIA3
    python scripts/diagnose_assets.py --no-data
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cli import _build_current_dataset, _fundamentals_data_status, _make_signal, _sentiment_data_status
from app.config import artifact_dir, load_config, load_data_registry
from app.data import data_status, load_prices, resolve_asset
from app.feature_audit import abbreviate_feature_name, feature_family_profile
from app.models import train_models
from app.utils import normalize_ticker, parse_tickers


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _registered_assets(cfg: dict[str, Any]) -> list[str]:
    registry = load_data_registry(cfg)
    assets = registry.get("assets", {}) or {}
    out: list[str] = []
    for ticker, meta in assets.items():
        if not isinstance(meta, dict):
            continue
        status = str(meta.get("registry_status", "active"))
        if status == "inactive_alias":
            continue
        if bool(meta.get("use_in_reference_sample", True)):
            out.append(normalize_ticker(ticker))
    return sorted(dict.fromkeys(out))


def _asset_list(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if args.assets:
        tickers = parse_tickers([args.assets])
    else:
        tickers = _registered_assets(cfg)
    if args.limit:
        tickers = tickers[: int(args.limit)]
    return tickers


def _pct(value: Any) -> float | None:
    try:
        return float(value) * 100.0
    except Exception:
        return None


def _fmt_pct(value: Any) -> str:
    pct = _pct(value)
    return "" if pct is None else f"{pct:+.2f}"


def _join_dict_pct(values: dict[str, Any] | None) -> str:
    if not values:
        return ""
    return " | ".join(f"{k} {_fmt_pct(v)}%" for k, v in values.items())


def _top_feature_line(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return ""
    return ", ".join(str(item.get("short") or abbreviate_feature_name(str(item.get("name", "")))) for item in items[:5])


def _safe_stage(row: dict[str, Any], stage: str, fn) -> Any:
    row["stage"] = stage
    try:
        return fn()
    except Exception as exc:
        row["status"] = "error"
        row["failed_stage"] = stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=6)
        return None


def _diagnose_one(cfg: dict[str, Any], ticker: str, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input_ticker": ticker,
        "status": "ok",
        "failed_stage": "",
        "error": "",
        "traceback": "",
    }

    resolved = resolve_asset(cfg, ticker)
    canonical = resolved.get("canonical", normalize_ticker(ticker))
    profile = resolved.get("profile", {}) or {}
    row.update({
        "ticker": canonical,
        "ticker_changed": bool(resolved.get("changed")),
        "alias_reason": (resolved.get("alias", {}) or {}).get("reason", ""),
        "name": profile.get("name", ""),
        "group": profile.get("group", ""),
        "subgroup": profile.get("subgroup", ""),
        "financial_class": profile.get("financial_class", ""),
        "cnpj": profile.get("cnpj") or "",
        "registry_status": profile.get("registry_status", ""),
    })

    if not args.no_data:
        if _safe_stage(row, "data", lambda: load_prices(cfg, canonical, update=True)) is None:
            return row

    st = _safe_stage(row, "data_status", lambda: data_status(cfg, canonical))
    if st is None:
        return row
    row.update({
        "data_rows": st.get("rows", 0),
        "data_start": st.get("start") or "",
        "data_end": st.get("end") or "",
        "context_available": ",".join(st.get("context_tickers", []) or []),
        "context_requested": ",".join(st.get("requested_context_tickers", []) or []),
        "context_unavailable": ",".join(st.get("unavailable_context_tickers", []) or []),
        "linked_indices": ",".join((st.get("asset_profile", {}) or {}).get("linked_indices", []) or []),
    })

    # These statuses are diagnostic only. They should not stop the batch.
    try:
        fstat = _fundamentals_data_status(cfg, canonical)
        row["fundamentals_status"] = fstat.get("status", "")
        row["fundamentals_source"] = fstat.get("source", "")
    except Exception as exc:
        row["fundamentals_status"] = "error"
        row["fundamentals_source"] = str(exc)[:80]
    try:
        sstat = _sentiment_data_status(cfg, canonical)
        row["sentiment_status"] = sstat.get("status", "")
        row["sentiment_cache_rows"] = sstat.get("cache_rows", 0)
        row["sentiment_new_items"] = sstat.get("new_items", 0)
    except Exception as exc:
        row["sentiment_status"] = "error"
        row["sentiment_cache_rows"] = 0
        row["sentiment_new_items"] = 0
        row["sentiment_error"] = str(exc)[:80]

    built = _safe_stage(row, "dataset", lambda: _build_current_dataset(cfg, canonical, update=False))
    if built is None:
        return row
    X, y, meta = built
    row.update({
        "dataset_rows": len(X),
        "generated_feature_count": ((meta.get("preparation", {}) or {}).get("input_feature_count", "")),
        "selected_feature_count": len(list(X.columns)),
        "feature_mix": json.dumps(feature_family_profile(list(X.columns)), ensure_ascii=False),
    })

    manifest = _safe_stage(row, "train", lambda: train_models(cfg, canonical, X, y, meta, autotune=bool(args.autotune)))
    if manifest is None:
        return row
    metrics = manifest.get("metrics", {}) or {}
    ridge_metrics = metrics.get("ridge_arbiter", {}) or {}
    guard = manifest.get("latest_engine_guard", {}) or {}
    row.update({
        "run_id": manifest.get("run_id", ""),
        "train_rows": manifest.get("train_rows", ""),
        "test_rows": manifest.get("test_rows", ""),
        "features": len(manifest.get("features", []) or []),
        "top_features": _top_feature_line(manifest.get("top_features", []) or []),
        "feature_family_profile": json.dumps(manifest.get("feature_family_profile", {}) or {}, ensure_ascii=False),
        "mae_arbiter": ridge_metrics.get("mae_return", ""),
        "train_prediction_pct": _fmt_pct(manifest.get("latest_prediction_return", 0.0)),
        "train_confidence_pct": float(manifest.get("confidence", 0.0) or 0.0) * 100.0,
        "train_raw_engines": _join_dict_pct(manifest.get("latest_prediction_by_engine_raw", {}) or {}),
        "train_guarded_engines": _join_dict_pct(manifest.get("latest_prediction_by_engine", {}) or {}),
        "train_discarded_engines": ",".join(guard.get("discarded", []) or []),
        "train_used_engines": ",".join(guard.get("used", []) or []),
        "model_path": manifest.get("model_path", ""),
    })

    signal = _safe_stage(row, "predict", lambda: _make_signal(cfg, canonical, update=False))
    if signal is None:
        return row
    pred = signal.get("prediction", {}) or {}
    policy = signal.get("policy", {}) or {}
    fundamentals = signal.get("fundamentals", {}) or {}
    row.update({
        "signal": policy.get("label", ""),
        "posture": policy.get("posture", ""),
        "prediction_pct": float(policy.get("score_pct", 0.0) or 0.0),
        "confidence_pct": float(policy.get("confidence_pct", 0.0) or 0.0),
        "predict_raw_engines": _join_dict_pct(pred.get("raw_by_engine", {}) or {}),
        "predict_guarded_engines": _join_dict_pct(pred.get("by_engine", {}) or {}),
        "predict_discarded_engines": ",".join(pred.get("discarded_engines", []) or []),
        "predict_used_engines": ",".join(pred.get("used_engines", []) or []),
        "reasons": "; ".join(policy.get("reasons", []) or []),
        "pl": fundamentals.get("pl", ""),
        "dy": fundamentals.get("dy", ""),
        "pvp": fundamentals.get("pvp", ""),
        "roe": fundamentals.get("roe", ""),
    })
    row["stage"] = "done"
    return row


def _write_outputs(cfg: dict[str, Any], rows: list[dict[str, Any]], run_id: str) -> dict[str, Path]:
    out_dir = artifact_dir(cfg) / "diagnostics" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "assets_diagnostic.csv"
    json_path = out_dir / "assets_diagnostic.json"
    txt_path = out_dir / "assets_diagnostic_summary.txt"

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key != "traceback":
                fieldnames.append(key)
    if "traceback" not in fieldnames:
        fieldnames.append("traceback")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    ok = [r for r in rows if r.get("status") == "ok"]
    err = [r for r in rows if r.get("status") != "ok"]
    discarded = [r for r in ok if r.get("predict_discarded_engines") or r.get("train_discarded_engines")]
    low_conf = [r for r in ok if float(r.get("confidence_pct", 0) or 0) < 55]
    context_heavy = []
    for r in ok:
        try:
            mix = json.loads(r.get("feature_family_profile") or "{}")
            total = max(1, int(r.get("features", 0) or 0))
            if float(mix.get("context", 0)) / total >= 0.50:
                context_heavy.append(r)
        except Exception:
            pass

    lines: list[str] = []
    lines.append("TRADEGEM ASSET DIAGNOSTIC")
    lines.append("=" * 80)
    lines.append(f"run_id        : {run_id}")
    lines.append(f"assets        : {len(rows)}")
    lines.append(f"ok            : {len(ok)}")
    lines.append(f"errors        : {len(err)}")
    lines.append(f"engine guards : {len(discarded)}")
    lines.append(f"low confidence: {len(low_conf)}")
    lines.append(f"context heavy : {len(context_heavy)}")
    lines.append("")
    if err:
        lines.append("ERRORS")
        lines.append("-" * 80)
        for r in err[:30]:
            lines.append(f"{r.get('ticker') or r.get('input_ticker')}: stage={r.get('failed_stage')} | {r.get('error')}")
        lines.append("")
    if discarded:
        lines.append("ENGINE GUARDS")
        lines.append("-" * 80)
        for r in discarded[:50]:
            lines.append(f"{r.get('ticker')}: discarded={r.get('predict_discarded_engines') or r.get('train_discarded_engines')} | raw={r.get('predict_raw_engines')}")
        lines.append("")
    if context_heavy:
        lines.append("CONTEXT-HEAVY FEATURE SELECTION")
        lines.append("-" * 80)
        for r in context_heavy[:50]:
            lines.append(f"{r.get('ticker')}: mix={r.get('feature_family_profile')} | top={r.get('top_features')}")
        lines.append("")
    lines.append("OUTPUTS")
    lines.append("-" * 80)
    lines.append(f"csv : {csv_path}")
    lines.append(f"json: {json_path}")
    lines.append(f"txt : {txt_path}")
    lines.append("=" * 80)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "txt": txt_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data -> train -> predict diagnostics for registered assets.")
    parser.add_argument("--config", default=None, help="optional config/config.yaml path")
    parser.add_argument("--assets", default="", help="comma/space separated ticker subset; default = all registered reference assets")
    parser.add_argument("--limit", type=int, default=0, help="limit number of assets for smoke testing")
    parser.add_argument("--no-data", action="store_true", help="skip data refresh and use existing cache")
    parser.add_argument("--autotune", action="store_true", help="use autotune during training; slow")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tickers = _asset_list(cfg, args)
    run_id = f"diag_{_now_id()}"
    print("=" * 80)
    print(f"TRADEGEM DIAGNOSTIC | assets={len(tickers)} | run_id={run_id}")
    print("=" * 80)
    rows: list[dict[str, Any]] = []
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx:>3}/{len(tickers):<3}] {ticker:<12} data -> train -> predict", flush=True)
        row = _diagnose_one(cfg, ticker, args)
        rows.append(row)
        status = row.get("status", "ok")
        if status == "ok":
            print(
                f"          ok | pred={row.get('prediction_pct', 0):+6.2f}% | "
                f"conf={row.get('confidence_pct', 0):5.0f}% | mae={row.get('mae_arbiter')} | "
                f"discarded={row.get('predict_discarded_engines') or '-'}",
                flush=True,
            )
        else:
            print(f"          ERROR at {row.get('failed_stage')}: {row.get('error')}", flush=True)
    paths = _write_outputs(cfg, rows, run_id)
    print("=" * 80)
    print("diagnostic written:")
    print(f"csv : {paths['csv']}")
    print(f"json: {paths['json']}")
    print(f"txt : {paths['txt']}")
    print("=" * 80)
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
