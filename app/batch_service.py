from __future__ import annotations

import json
import os
import traceback
from typing import Any

from .data import data_status, load_prices, resolve_asset
from .feature_audit import abbreviate_feature_name
from .models import train_models
from .pipeline_service import (
    build_current_dataset,
    canonical_ticker,
    fundamentals_data_status,
    make_signal,
    sentiment_data_status,
)
from .utils import normalize_ticker


def safe_worker_count(total_items: int, requested: int | None = None, default: int = 1) -> int:
    total = max(1, int(total_items or 1))
    desired = default if requested is None or int(requested) <= 0 else int(requested)
    cpu_total = os.cpu_count() or 1
    cpu_cap = max(1, cpu_total - 1) if cpu_total > 2 else 1
    return max(1, min(total, desired, cpu_cap))


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:+.2f}"
    except Exception:
        return ""


def _top_feature_line(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return ""
    return ", ".join(
        str(item.get("short") or abbreviate_feature_name(str(item.get("name", ""))))
        for item in items[:5]
    )


def train_one_asset(
    cfg: dict[str, Any],
    ticker: str,
    *,
    update: bool = False,
    autotune: bool = False,
    inner_threads: int | None = None,
) -> dict[str, Any]:
    ticker = canonical_ticker(cfg, ticker)
    targets = ["target_return_d1", "target_return_d5", "target_return_d20"]
    raw_X, all_y, meta = build_current_dataset(cfg, ticker, update=update)
    manifests: list[dict[str, Any]] = []

    for t_col in targets:
        horizon = t_col.split("_")[-1]
        y_series = all_y[t_col]
        h_meta = meta.copy()
        h_meta["horizon"] = horizon
        manifests.append(
            train_models(
                cfg,
                ticker,
                raw_X,
                y_series,
                h_meta,
                autotune=bool(autotune),
                horizon=horizon,
                inner_threads=inner_threads,
            )
        )

    return {
        "ticker": ticker,
        "ok": True,
        "rows": len(raw_X),
        "autotune": bool(autotune),
        "update": bool(update),
        "manifests": manifests,
    }


def _safe_stage(row: dict[str, Any], stage: str, fn) -> Any:
    row["stage"] = stage
    try:
        return fn()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        is_insufficient_rows = "insufficient rows" in str(exc).lower()
        row["status"] = "skipped" if is_insufficient_rows else "error"
        row["failed_stage"] = stage
        row["error"] = message
        row["traceback"] = traceback.format_exc(limit=6)
        return None


def _skip_insufficient_rows(row: dict[str, Any], *, stage: str, rows: int, min_rows: int) -> dict[str, Any]:
    row["status"] = "skipped"
    row["stage"] = stage
    row["failed_stage"] = stage
    row["error"] = f"insufficient rows: {rows} < {min_rows}; wait for more history or remove from reference sample"
    row["traceback"] = ""
    return row


def diagnose_one_asset(
    cfg: dict[str, Any],
    ticker: str,
    *,
    no_data: bool = False,
    autotune: bool = False,
    inner_threads: int | None = None,
) -> dict[str, Any]:
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
    row.update(
        {
            "ticker": canonical,
            "ticker_changed": bool(resolved.get("changed")),
            "name": profile.get("name", ""),
            "group": profile.get("group", ""),
            "subgroup": profile.get("subgroup", ""),
            "financial_class": profile.get("financial_class", ""),
            "cnpj": profile.get("cnpj") or "",
            "registry_status": profile.get("registry_status", ""),
        }
    )

    if not no_data:
        if _safe_stage(row, "data", lambda: load_prices(cfg, canonical, update=True)) is None:
            return row

    st = _safe_stage(row, "data_status", lambda: data_status(cfg, canonical))
    if st is None:
        return row
    row.update(
        {
            "data_rows": st.get("rows", 0),
            "data_start": st.get("start") or "",
            "data_end": st.get("end") or "",
            "context_available": ",".join(st.get("context_tickers", []) or []),
            "context_requested": ",".join(st.get("requested_context_tickers", []) or []),
            "context_unavailable": ",".join(st.get("unavailable_context_tickers", []) or []),
            "linked_indices": ",".join((st.get("asset_profile", {}) or {}).get("linked_indices", []) or []),
        }
    )
    min_rows = int(cfg.get("data", {}).get("min_rows", 150))
    data_rows = int(st.get("rows", 0) or 0)
    if data_rows < min_rows:
        return _skip_insufficient_rows(row, stage="data_rows", rows=data_rows, min_rows=min_rows)

    try:
        fstat = fundamentals_data_status(cfg, canonical)
        row["fundamentals_status"] = fstat.get("status", "")
        row["fundamentals_source"] = fstat.get("source", "")
    except Exception as exc:
        row["fundamentals_status"] = "error"
        row["fundamentals_source"] = str(exc)[:80]
    try:
        sstat = sentiment_data_status(cfg, canonical)
        row["sentiment_status"] = sstat.get("status", "")
        row["sentiment_cache_rows"] = sstat.get("cache_rows", 0)
        row["sentiment_new_items"] = sstat.get("new_items", 0)
    except Exception as exc:
        row["sentiment_status"] = "error"
        row["sentiment_cache_rows"] = 0
        row["sentiment_new_items"] = 0
        row["sentiment_error"] = str(exc)[:80]

    built = _safe_stage(row, "dataset", lambda: build_current_dataset(cfg, canonical, update=False))
    if built is None:
        return row
    raw_X, all_y, meta = built
    if len(raw_X) < min_rows:
        return _skip_insufficient_rows(row, stage="dataset_rows", rows=len(raw_X), min_rows=min_rows)

    for t_col in ["target_return_d1", "target_return_d5", "target_return_d20"]:
        horizon = t_col.split("_")[-1]
        y_series = all_y[t_col].dropna()
        if len(y_series) < min_rows:
            return _skip_insufficient_rows(row, stage=f"target_rows_{horizon}", rows=len(y_series), min_rows=min_rows)

        h_meta = meta.copy()
        h_meta["horizon"] = horizon
        manifest = _safe_stage(
            row,
            f"train_{horizon}",
            lambda: train_models(
                cfg,
                canonical,
                raw_X,
                all_y[t_col],
                h_meta,
                autotune=bool(autotune),
                horizon=horizon,
                inner_threads=inner_threads,
            ),
        )
        if manifest is None:
            return row

        if horizon == "d1":
            metrics = manifest.get("metrics", {}) or {}
            ridge_metrics = metrics.get("ridge_arbiter", {}) or {}
            row.update(
                {
                    "run_id": manifest.get("run_id", ""),
                    "train_rows": manifest.get("train_rows", ""),
                    "test_rows": manifest.get("test_rows", ""),
                    "features": len(manifest.get("features", []) or []),
                    "top_features": _top_feature_line(manifest.get("top_features", []) or []),
                    "feature_family_profile": json.dumps(
                        manifest.get("feature_family_profile", {}) or {},
                        ensure_ascii=False,
                    ),
                    "mae_arbiter": ridge_metrics.get("mae_return", ""),
                    "engine_dispersion": manifest.get("engine_dispersion", 0.0),
                    "train_prediction_pct": _fmt_pct(manifest.get("latest_prediction_return", 0.0)),
                    "train_quality_pct": float(manifest.get("quality", manifest.get("confidence", 0.0)) or 0.0) * 100.0,
                }
            )

    signal = _safe_stage(row, "predict", lambda: make_signal(cfg, canonical, update=False))
    if signal is None:
        return row
    policy = signal.get("policy", {}) or {}
    horizons = signal.get("horizons", {}) or {}
    row.update(
        {
            "signal": policy.get("label", ""),
            "posture": policy.get("posture", ""),
            "prediction_pct": float(policy.get("score_pct", 0.0) or 0.0),
            "quality_pct": float(policy.get("quality_pct", policy.get("confidence_pct", 0.0)) or 0.0),
            "d5_ret": float(horizons.get("d5", {}).get("prediction_return", 0.0)) * 100.0,
            "d20_ret": float(horizons.get("d20", {}).get("prediction_return", 0.0)) * 100.0,
            "reasons": "; ".join(policy.get("reasons", []) or []),
        }
    )
    row["stage"] = "done"
    return row
