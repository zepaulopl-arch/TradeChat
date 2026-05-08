from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import load_prices
from .features import build_dataset
from .models import train_models
from .config import artifact_dir, models_dir
from .feature_audit import feature_family_profile, selected_feature_scores
from .presentation import banner, divider, render_table, screen_width
from .utils import normalize_ticker, read_json, safe_ticker, write_json


HORIZONS = ["d1", "d5", "d20"]
FAMILIES = ["technical", "context", "fundamentals", "sentiment"]
REMOVAL_PROFILES = ["full", "technical_only", "no_context", "no_fundamentals", "no_sentiment"]


def _latest_manifest_path(cfg: dict[str, Any], ticker: str, horizon: str) -> Any:
    return models_dir(cfg) / safe_ticker(ticker) / f"latest_train_{horizon}.json"


def _family_relevance_share(manifest: dict[str, Any]) -> dict[str, float]:
    features = [str(item) for item in manifest.get("features", []) or []]
    prep = manifest.get("preparation", {}) or {}
    scores = selected_feature_scores(prep, features)
    totals = {family: 0.0 for family in FAMILIES}
    for feature, score in scores.items():
        family = next((fam for fam, count in feature_family_profile([feature]).items() if count), "technical")
        totals[family] = totals.get(family, 0.0) + max(0.0, float(score))
    total = sum(totals.values())
    if total <= 0:
        return {family: 0.0 for family in FAMILIES}
    return {family: float(value / total * 100.0) for family, value in totals.items()}


def _decision_from_manifest(manifest: dict[str, Any]) -> str:
    metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
    mae = float(metrics.get("mae_return", 0.0) or 0.0)
    confidence = float(manifest.get("confidence", 0.0) or 0.0)
    selected = int(len(manifest.get("features", []) or []))
    if selected == 0:
        return "invalid"
    if mae <= 0.015 and confidence >= 0.45:
        return "keep"
    if mae <= 0.030:
        return "watch"
    return "review"


def collect_refine_summary(cfg: dict[str, Any], tickers: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for raw_ticker in tickers:
        ticker = normalize_ticker(raw_ticker)
        for horizon in HORIZONS:
            path = _latest_manifest_path(cfg, ticker, horizon)
            if not path.exists():
                missing.append({"ticker": ticker, "horizon": horizon, "path": str(path)})
                continue
            manifest = read_json(path)
            features = [str(item) for item in manifest.get("features", []) or []]
            family_counts = feature_family_profile(features)
            family_share = _family_relevance_share(manifest)
            metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "run_id": str(manifest.get("run_id", "")),
                    "mae_return": float(metrics.get("mae_return", 0.0) or 0.0),
                    "latest_prediction_return": float(manifest.get("latest_prediction_return", 0.0) or 0.0),
                    "confidence": float(manifest.get("confidence", 0.0) or 0.0),
                    "selected_feature_count": len(features),
                    "family_counts": family_counts,
                    "family_relevance_share_pct": family_share,
                    "decision": _decision_from_manifest(manifest),
                }
            )
    return {"rows": rows, "missing": missing}


def _parse_horizons(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return list(HORIZONS)
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in raw if str(item).strip()]
    selected = [item for item in items if item in HORIZONS]
    if not selected:
        raise ValueError(f"horizons must include at least one of: {', '.join(HORIZONS)}")
    return selected


def _parse_profiles(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return list(REMOVAL_PROFILES)
    if isinstance(raw, str):
        items = [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in raw if str(item).strip()]
    invalid = [item for item in items if item not in REMOVAL_PROFILES]
    if invalid:
        raise ValueError(f"unknown removal profile(s): {', '.join(invalid)}")
    return items or list(REMOVAL_PROFILES)


def _shadow_cfg(cfg: dict[str, Any], run_id: str, profile: str) -> dict[str, Any]:
    shadow = copy.deepcopy(cfg)
    app_cfg = dict(shadow.get("app", {}) or {})
    app_cfg["artifact_dir"] = f"artifacts/refine/{run_id}/{profile}"
    shadow["app"] = app_cfg
    return shadow


def _removal_cfg(cfg: dict[str, Any], profile: str, run_id: str) -> dict[str, Any]:
    shadow = _shadow_cfg(cfg, run_id, profile)
    features = shadow.setdefault("features", {})
    for family in FAMILIES:
        features.setdefault(family, {})

    if profile == "technical_only":
        features["context"]["enabled"] = False
        features["fundamentals"]["enabled"] = False
        features["sentiment"]["enabled"] = False
    elif profile == "no_context":
        features["context"]["enabled"] = False
    elif profile == "no_fundamentals":
        features["fundamentals"]["enabled"] = False
    elif profile == "no_sentiment":
        features["sentiment"]["enabled"] = False
    elif profile != "full":
        raise ValueError(f"unknown removal profile: {profile}")
    return shadow


def refine_dir(cfg: dict[str, Any], run_id: str) -> Path:
    path = artifact_dir(cfg) / "refine" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_removal_artifacts(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    out_dir = refine_dir(cfg, str(summary.get("run_id", "refine")))
    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"
    results_csv = out_dir / "removal_results.csv"

    write_json(summary_json, summary)
    rows = list(summary.get("rows", []) or [])
    fieldnames = [
        "ticker",
        "horizon",
        "profile",
        "mae_return",
        "latest_prediction_return",
        "confidence",
        "selected_feature_count",
        "technical_count",
        "context_count",
        "fundamentals_count",
        "sentiment_count",
        "artifact_dir",
        "run_id",
    ]
    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            counts = row.get("family_counts", {}) or {}
            writer.writerow(
                {
                    "ticker": row.get("ticker", ""),
                    "horizon": row.get("horizon", ""),
                    "profile": row.get("profile", ""),
                    "mae_return": row.get("mae_return", 0.0),
                    "latest_prediction_return": row.get("latest_prediction_return", 0.0),
                    "confidence": row.get("confidence", 0.0),
                    "selected_feature_count": row.get("selected_feature_count", 0),
                    "technical_count": counts.get("technical", 0),
                    "context_count": counts.get("context", 0),
                    "fundamentals_count": counts.get("fundamentals", 0),
                    "sentiment_count": counts.get("sentiment", 0),
                    "artifact_dir": row.get("artifact_dir", ""),
                    "run_id": row.get("run_id", ""),
                }
            )

    lines = render_removal_summary(summary)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dir": str(out_dir),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "results_csv": str(results_csv),
    }


def run_feature_removal(
    cfg: dict[str, Any],
    tickers: list[str],
    *,
    horizons: str | list[str] | None = None,
    profiles: str | list[str] | None = None,
    update: bool = False,
    autotune: bool = False,
    inner_threads: int | None = 1,
) -> dict[str, Any]:
    selected_horizons = _parse_horizons(horizons)
    selected_profiles = _parse_profiles(profiles)
    canonical = [normalize_ticker(ticker) for ticker in tickers]
    run_id = f"refine_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_ticker(canonical[0])}"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for ticker in canonical:
        prices = load_prices(cfg, ticker, update=update)
        for profile in selected_profiles:
            profile_cfg = _removal_cfg(cfg, profile, run_id)
            try:
                raw_X, all_y, meta = build_dataset(profile_cfg, prices, ticker)
            except Exception as exc:
                errors.append({"ticker": ticker, "profile": profile, "horizon": "*", "error": str(exc)})
                continue
            for horizon in selected_horizons:
                target_col = f"target_return_{horizon}"
                if target_col not in all_y:
                    errors.append({"ticker": ticker, "profile": profile, "horizon": horizon, "error": f"missing {target_col}"})
                    continue
                try:
                    manifest = train_models(
                        profile_cfg,
                        ticker,
                        raw_X,
                        all_y[target_col],
                        {**meta, "removal_profile": profile, "refine_run_id": run_id},
                        autotune=autotune,
                        horizon=horizon,
                        inner_threads=inner_threads,
                    )
                except Exception as exc:
                    errors.append({"ticker": ticker, "profile": profile, "horizon": horizon, "error": str(exc)})
                    continue
                metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
                features = [str(item) for item in manifest.get("features", []) or []]
                rows.append(
                    {
                        "ticker": ticker,
                        "profile": profile,
                        "horizon": horizon,
                        "run_id": str(manifest.get("run_id", "")),
                        "mae_return": float(metrics.get("mae_return", 0.0) or 0.0),
                        "latest_prediction_return": float(manifest.get("latest_prediction_return", 0.0) or 0.0),
                        "confidence": float(manifest.get("confidence", 0.0) or 0.0),
                        "selected_feature_count": len(features),
                        "family_counts": feature_family_profile(features),
                        "artifact_dir": str(models_dir(profile_cfg).parent),
                    }
                )
    summary = {
        "run_id": run_id,
        "horizons": selected_horizons,
        "profiles": selected_profiles,
        "rows": rows,
        "errors": errors,
    }
    summary["artifacts"] = _write_removal_artifacts(cfg, summary)
    return summary


def render_refine_summary(summary: dict[str, Any]) -> list[str]:
    width = screen_width()
    lines: list[str] = []
    rows = list(summary.get("rows", []) or [])
    missing = list(summary.get("missing", []) or [])
    lines.extend(banner("REFINE", "feature-family audit", width=width))
    if not rows:
        lines.append("No trained manifests found for the requested assets.")
        lines.extend(divider(width).splitlines())
        return lines

    table_rows = []
    for row in rows:
        counts = row.get("family_counts", {}) or {}
        shares = row.get("family_relevance_share_pct", {}) or {}
        table_rows.append(
            [
                str(row.get("ticker", "n/a")),
                str(row.get("horizon", "n/a")).upper(),
                f"{float(row.get('mae_return', 0.0) or 0.0) * 100:.2f}%",
                f"{float(row.get('latest_prediction_return', 0.0) or 0.0) * 100:+.2f}%",
                f"{float(row.get('confidence', 0.0) or 0.0) * 100:.0f}%",
                f"T{counts.get('technical', 0)}/C{counts.get('context', 0)}/F{counts.get('fundamentals', 0)}/S{counts.get('sentiment', 0)}",
                f"T{shares.get('technical', 0.0):.0f}/C{shares.get('context', 0.0):.0f}/F{shares.get('fundamentals', 0.0):.0f}/S{shares.get('sentiment', 0.0):.0f}",
                str(row.get("decision", "watch")),
            ]
        )
    lines.extend(
        render_table(
            ["Ticker", "Hz", "MAE", "Pred", "Qual", "Count", "Share", "Decision"],
            table_rows,
            width=width,
            aligns=["left", "left", "right", "right", "right", "left", "left", "left"],
            min_widths=[10, 3, 7, 7, 5, 13, 13, 8],
        )
    )
    if missing:
        lines.append("")
        lines.append(f"Missing manifests: {len(missing)}. Train the missing horizons before treating refine as complete.")
    lines.extend(divider(width).splitlines())
    return lines


def render_removal_summary(summary: dict[str, Any]) -> list[str]:
    width = screen_width()
    lines: list[str] = []
    rows = list(summary.get("rows", []) or [])
    errors = list(summary.get("errors", []) or [])
    lines.extend(banner("REFINE", "feature removal", str(summary.get("run_id", "")), width=width))
    if not rows:
        lines.append("No removal result was produced.")
        if errors:
            lines.append(f"Errors: {len(errors)}")
        lines.extend(divider(width).splitlines())
        return lines

    by_key = {(row["ticker"], row["horizon"], row["profile"]): row for row in rows}
    table_rows = []
    for row in rows:
        baseline = by_key.get((row["ticker"], row["horizon"], "full"))
        base_mae = float((baseline or row).get("mae_return", 0.0) or 0.0)
        mae = float(row.get("mae_return", 0.0) or 0.0)
        delta = mae - base_mae
        counts = row.get("family_counts", {}) or {}
        verdict = "base" if row.get("profile") == "full" else ("better" if delta < 0 else "worse" if delta > 0 else "tie")
        table_rows.append(
            [
                str(row.get("ticker", "n/a")),
                str(row.get("horizon", "n/a")).upper(),
                str(row.get("profile", "n/a")),
                f"{mae * 100:.2f}%",
                f"{delta * 100:+.2f}%",
                f"{float(row.get('confidence', 0.0) or 0.0) * 100:.0f}%",
                f"T{counts.get('technical', 0)}/C{counts.get('context', 0)}/F{counts.get('fundamentals', 0)}/S{counts.get('sentiment', 0)}",
                verdict,
            ]
        )
    lines.extend(
        render_table(
            ["Ticker", "Hz", "Profile", "MAE", "Delta", "Qual", "Count", "Verdict"],
            table_rows,
            width=width,
            aligns=["left", "left", "left", "right", "right", "right", "left", "left"],
            min_widths=[10, 3, 14, 7, 7, 5, 13, 7],
        )
    )
    if errors:
        lines.append("")
        lines.append(f"Removal errors: {len(errors)}. Use the artifacts and logs before drawing conclusions.")
    artifacts = summary.get("artifacts", {}) or {}
    if artifacts:
        lines.append("")
        lines.append(f"Artifacts: {artifacts.get('dir', 'n/a')}")
    lines.extend(divider(width).splitlines())
    return lines
