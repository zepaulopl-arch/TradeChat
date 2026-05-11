from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import artifact_dir, models_dir
from .data import load_prices
from .feature_audit import feature_family_profile, selected_feature_scores
from .features import build_dataset
from .models import train_models
from .presentation import banner, divider, render_table, screen_width
from .refine_decision import build_refine_decision_matrix
from .simulation.runner import run_pybroker_replay
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
        family = next(
            (fam for fam, count in feature_family_profile([feature]).items() if count), "technical"
        )
        totals[family] = totals.get(family, 0.0) + max(0.0, float(score))
    total = sum(totals.values())
    if total <= 0:
        return {family: 0.0 for family in FAMILIES}
    return {family: float(value / total * 100.0) for family, value in totals.items()}


def _decision_from_manifest(manifest: dict[str, Any]) -> str:
    metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
    mae = float(metrics.get("mae_return", 0.0) or 0.0)
    quality = float(manifest.get("quality", manifest.get("confidence", 0.0)) or 0.0)
    selected = int(len(manifest.get("features", []) or []))
    if selected == 0:
        return "invalid"
    if mae <= 0.015 and quality >= 0.45:
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
                    "latest_prediction_return": float(
                        manifest.get("latest_prediction_return", 0.0) or 0.0
                    ),
                    "quality": float(
                        manifest.get("quality", manifest.get("confidence", 0.0)) or 0.0
                    ),
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


def _baseline_manifest_summary(cfg: dict[str, Any], ticker: str, horizon: str) -> dict[str, Any]:
    path = _latest_manifest_path(cfg, ticker, horizon)
    if not path.exists():
        return {}
    manifest = read_json(path)
    metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
    features = [str(item) for item in manifest.get("features", []) or []]
    return {
        "baseline_mae_return": float(metrics.get("mae_return", 0.0) or 0.0),
        "baseline_quality": float(manifest.get("quality", manifest.get("confidence", 0.0)) or 0.0),
        "baseline_selected_feature_count": len(features),
        "baseline_family_counts": feature_family_profile(features),
    }


def refine_dir(cfg: dict[str, Any], run_id: str) -> Path:
    path = artifact_dir(cfg) / "refine" / str(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_decision_artifacts(
    summary: dict[str, Any], decision_csv: Path, decision_txt: Path
) -> None:
    decisions = list(summary.get("decisions", []) or [])
    fieldnames = [
        "ticker",
        "horizon",
        "profile",
        "removed_family",
        "return_delta_pct",
        "drawdown_delta_pct",
        "profit_factor_delta",
        "trade_count_delta",
        "exposure_delta_pct",
        "mae_delta",
        "quality_delta",
        "decision",
        "rationale",
    ]
    with decision_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            out = {name: row.get(name, "") for name in fieldnames}
            out["rationale"] = " | ".join(str(item) for item in row.get("rationale", []) or [])
            writer.writerow(out)

    lines = render_refine_decision_table(decisions)
    decision_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_removal_artifacts(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    out_dir = refine_dir(cfg, str(summary.get("run_id", "refine")))
    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"
    results_csv = out_dir / "removal_results.csv"
    decision_csv = out_dir / "decision_matrix.csv"
    decision_txt = out_dir / "decision_summary.txt"

    write_json(summary_json, summary)
    rows = list(summary.get("rows", []) or [])
    fieldnames = [
        "ticker",
        "horizon",
        "profile",
        "mae_return",
        "latest_prediction_return",
        "quality",
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
                    "quality": row.get("quality", 0.0),
                    "selected_feature_count": row.get("selected_feature_count", 0),
                    "technical_count": counts.get("technical", 0),
                    "context_count": counts.get("context", 0),
                    "fundamentals_count": counts.get("fundamentals", 0),
                    "sentiment_count": counts.get("sentiment", 0),
                    "artifact_dir": row.get("artifact_dir", ""),
                    "run_id": row.get("run_id", ""),
                }
            )

    _write_decision_artifacts(summary, decision_csv, decision_txt)

    lines = render_removal_summary(summary)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dir": str(out_dir),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "results_csv": str(results_csv),
        "decision_matrix_csv": str(decision_csv),
        "decision_summary_txt": str(decision_txt),
    }


def _write_walkforward_removal_artifacts(
    cfg: dict[str, Any], summary: dict[str, Any]
) -> dict[str, str]:
    out_dir = refine_dir(cfg, str(summary.get("run_id", "refine")))
    summary_json = out_dir / "walkforward_summary.json"
    summary_txt = out_dir / "walkforward_summary.txt"
    results_csv = out_dir / "walkforward_results.csv"
    decision_csv = out_dir / "decision_matrix.csv"
    decision_txt = out_dir / "decision_summary.txt"

    write_json(summary_json, summary)
    rows = list(summary.get("rows", []) or [])
    fieldnames = [
        "profile",
        "run_id",
        "mode",
        "total_return_pct",
        "return_delta_pct",
        "max_drawdown_pct",
        "trade_count",
        "hit_rate_pct",
        "profit_factor",
        "turnover_pct",
        "active_exposure_pct",
        "baseline_decision",
        "beat_rate_pct",
        "artifact_dir",
    ]
    with results_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    _write_decision_artifacts(summary, decision_csv, decision_txt)

    lines = render_removal_walkforward_summary(summary)
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "dir": str(out_dir),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "results_csv": str(results_csv),
        "decision_matrix_csv": str(decision_csv),
        "decision_summary_txt": str(decision_txt),
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
                errors.append(
                    {"ticker": ticker, "profile": profile, "horizon": "*", "error": str(exc)}
                )
                continue
            for horizon in selected_horizons:
                target_col = f"target_return_{horizon}"
                if target_col not in all_y:
                    errors.append(
                        {
                            "ticker": ticker,
                            "profile": profile,
                            "horizon": horizon,
                            "error": f"missing {target_col}",
                        }
                    )
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
                    errors.append(
                        {
                            "ticker": ticker,
                            "profile": profile,
                            "horizon": horizon,
                            "error": str(exc),
                        }
                    )
                    continue
                metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
                features = [str(item) for item in manifest.get("features", []) or []]
                baseline = _baseline_manifest_summary(cfg, ticker, horizon)
                rows.append(
                    {
                        **baseline,
                        "ticker": ticker,
                        "profile": profile,
                        "horizon": horizon,
                        "run_id": str(manifest.get("run_id", "")),
                        "mae_return": float(metrics.get("mae_return", 0.0) or 0.0),
                        "latest_prediction_return": float(
                            manifest.get("latest_prediction_return", 0.0) or 0.0
                        ),
                        "quality": float(
                            manifest.get("quality", manifest.get("confidence", 0.0)) or 0.0
                        ),
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
    summary["decisions"] = build_refine_decision_matrix(rows)
    summary["artifacts"] = _write_removal_artifacts(cfg, summary)
    return summary


def run_feature_removal_walkforward(
    cfg: dict[str, Any],
    tickers: list[str],
    *,
    profiles: str | list[str] | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    rebalance_days: int = 5,
    warmup_bars: int = 150,
    initial_cash: float | None = None,
    max_positions: int | None = None,
    allow_short: bool = False,
    autotune: bool = False,
    inner_threads: int | None = 1,
) -> dict[str, Any]:
    selected_profiles = _parse_profiles(profiles)
    canonical = [normalize_ticker(ticker) for ticker in tickers]
    run_id = f"refine_wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_ticker(canonical[0])}"
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for profile in selected_profiles:
        profile_cfg = _removal_cfg(cfg, profile, run_id)
        try:
            result = run_pybroker_replay(
                profile_cfg,
                canonical,
                mode="walkforward",
                start_date=start_date,
                end_date=end_date,
                rebalance_days=rebalance_days,
                warmup_bars=warmup_bars,
                initial_cash=initial_cash,
                max_positions=max_positions,
                allow_short=allow_short,
                walkforward_autotune=autotune,
                inner_threads=inner_threads,
            )
        except Exception as exc:
            errors.append({"profile": profile, "error": str(exc)})
            continue

        metrics = result.get("metrics", {}) or {}
        comparison = result.get("baseline_comparison", {}) or {}
        artifacts = result.get("artifacts", {}) or {}
        rows.append(
            {
                "profile": profile,
                "run_id": str(result.get("run_id", "")),
                "mode": str(result.get("mode", "")),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0) or 0.0),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "hit_rate_pct": float(
                    metrics.get("hit_rate_pct", metrics.get("win_rate", 0.0)) or 0.0
                ),
                "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
                "turnover_pct": float(metrics.get("turnover_pct", 0.0) or 0.0),
                "active_exposure_pct": float(metrics.get("active_exposure_pct", 0.0) or 0.0),
                "baseline_decision": str(comparison.get("decision", "n/a")),
                "beat_rate_pct": float(comparison.get("beat_rate_pct", 0.0) or 0.0),
                "artifact_dir": str(artifacts.get("dir", "")),
            }
        )

    base_return = next(
        (
            float(row.get("total_return_pct", 0.0) or 0.0)
            for row in rows
            if row.get("profile") == "full"
        ),
        None,
    )
    for row in rows:
        row["return_delta_pct"] = (
            0.0
            if base_return is None
            else float(row.get("total_return_pct", 0.0) or 0.0) - base_return
        )

    summary = {
        "run_id": run_id,
        "mode": "feature_removal_walkforward",
        "tickers": canonical,
        "profiles": selected_profiles,
        "start_date": str(start_date) if start_date is not None else None,
        "end_date": str(end_date) if end_date is not None else None,
        "rebalance_days": int(rebalance_days),
        "warmup_bars": int(warmup_bars),
        "allow_short": bool(allow_short),
        "autotune": bool(autotune),
        "rows": rows,
        "errors": errors,
    }
    summary["decisions"] = build_refine_decision_matrix(rows)
    summary["artifacts"] = _write_walkforward_removal_artifacts(cfg, summary)
    return summary


def _family_counts_text(counts: dict[str, Any]) -> str:
    return (
        f"T{counts.get('technical', 0)}"
        f"/C{counts.get('context', 0)}"
        f"/F{counts.get('fundamentals', 0)}"
        f"/S{counts.get('sentiment', 0)}"
    )


def _family_shares_text(shares: dict[str, Any]) -> str:
    return (
        f"T{shares.get('technical', 0.0):.0f}"
        f"/C{shares.get('context', 0.0):.0f}"
        f"/F{shares.get('fundamentals', 0.0):.0f}"
        f"/S{shares.get('sentiment', 0.0):.0f}"
    )


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
                f"{float(row.get('quality', row.get('confidence', 0.0)) or 0.0) * 100:.0f}%",
                _family_counts_text(counts),
                _family_shares_text(shares),
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
        lines.append(
            f"Missing manifests: {len(missing)}. Train missing horizons before refine is complete."
        )
    lines.extend(divider(width).splitlines())
    return lines


def render_removal_summary(summary: dict[str, Any]) -> list[str]:
    width = screen_width()
    lines: list[str] = []
    rows = list(summary.get("rows", []) or [])
    errors = list(summary.get("errors", []) or [])
    lines.extend(banner("REFINE", "controlled removal", width=width))
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
        if baseline is not None:
            base_mae = float(baseline.get("mae_return", 0.0) or 0.0)
            base_counts = baseline.get("family_counts", {}) or {}
        else:
            base_mae = float(row.get("baseline_mae_return", row.get("mae_return", 0.0)) or 0.0)
            base_counts = row.get("baseline_family_counts", row.get("family_counts", {})) or {}
        mae = float(row.get("mae_return", 0.0) or 0.0)
        delta = mae - base_mae
        counts = row.get("family_counts", {}) or {}
        verdict = (
            "base"
            if row.get("profile") == "full"
            else ("better" if delta < 0 else "worse" if delta > 0 else "tie")
        )
        table_rows.append(
            [
                str(row.get("ticker", "n/a")),
                str(row.get("horizon", "n/a")).upper(),
                str(row.get("profile", "n/a")),
                f"{base_mae * 100:.2f}%",
                f"{mae * 100:.2f}%",
                f"{delta * 100:+.2f}%",
                f"{float(row.get('quality', row.get('confidence', 0.0)) or 0.0) * 100:.0f}%",
                _family_counts_text(base_counts),
                _family_counts_text(counts),
                verdict,
            ]
        )
    lines.extend(
        render_table(
            [
                "Ticker",
                "Hz",
                "Profile",
                "Full MAE",
                "Profile MAE",
                "Delta",
                "Qual",
                "Full Count",
                "Profile Count",
                "Verdict",
            ],
            table_rows,
            width=width,
            aligns=[
                "left",
                "left",
                "left",
                "right",
                "right",
                "right",
                "right",
                "left",
                "left",
                "left",
            ],
            min_widths=[10, 3, 14, 9, 11, 8, 5, 13, 13, 7],
        )
    )
    if errors:
        lines.append("")
        lines.append(
            f"Removal errors: {len(errors)}. Use the artifacts and logs before drawing conclusions."
        )

    decisions = list(summary.get("decisions", []) or [])
    if decisions:
        lines.append("")
        lines.extend(render_refine_decision_table(decisions, width=width))
    lines.extend(divider(width).splitlines())
    return lines


def render_removal_walkforward_summary(summary: dict[str, Any]) -> list[str]:
    width = screen_width()
    lines: list[str] = []
    rows = list(summary.get("rows", []) or [])
    errors = list(summary.get("errors", []) or [])
    lines.extend(banner("REFINE", "controlled removal walk-forward", width=width))
    if not rows:
        lines.append("No walk-forward removal result was produced.")
        if errors:
            lines.append(f"Errors: {len(errors)}")
        lines.extend(divider(width).splitlines())
        return lines

    table_rows = []
    for row in rows:
        delta = float(row.get("return_delta_pct", 0.0) or 0.0)
        profile = str(row.get("profile", "n/a"))
        verdict = (
            "base"
            if profile == "full"
            else ("better" if delta > 0 else "worse" if delta < 0 else "tie")
        )
        table_rows.append(
            [
                profile,
                f"{float(row.get('total_return_pct', 0.0) or 0.0):+.2f}%",
                f"{delta:+.2f}%",
                f"{float(row.get('max_drawdown_pct', 0.0) or 0.0):+.2f}%",
                str(int(row.get("trade_count", 0) or 0)),
                f"{float(row.get('hit_rate_pct', 0.0) or 0.0):.1f}%",
                f"{float(row.get('profit_factor', 0.0) or 0.0):.2f}",
                f"{float(row.get('beat_rate_pct', 0.0) or 0.0):.0f}%",
                verdict,
            ]
        )
    lines.extend(
        render_table(
            ["Profile", "Return", "Delta", "DD", "Trades", "Hit", "PF", "Beat", "Verdict"],
            table_rows,
            width=width,
            aligns=["left", "right", "right", "right", "right", "right", "right", "right", "left"],
            min_widths=[14, 8, 8, 8, 6, 6, 5, 6, 7],
        )
    )
    if errors:
        lines.append("")
        lines.append(
            f"Walk-forward removal errors: {len(errors)}. Review them before drawing conclusions."
        )

    decisions = list(summary.get("decisions", []) or [])
    if decisions:
        lines.append("")
        lines.extend(render_refine_decision_table(decisions, width=width))
    lines.extend(divider(width).splitlines())
    return lines


def render_refine_decision_table(
    decisions: list[dict[str, Any]], *, width: int | None = None
) -> list[str]:
    width = width or screen_width()
    grouped = _consolidate_refine_decisions(decisions)
    table_rows = []
    notes: list[str] = []
    for row in grouped:
        table_rows.append(
            [
                str(row.get("profile", "n/a")),
                str(row.get("removed_family", "n/a")),
                _fmt_delta(row.get("return_delta_pct")),
                _fmt_delta(row.get("drawdown_delta_pct")),
                _fmt_delta(row.get("profit_factor_delta"), decimals=2, suffix=""),
                str(row.get("decision", "inconclusive")),
                str(row.get("scope", "tickers=0")),
            ]
        )
        for note in row.get("notes", []) or []:
            if note not in notes:
                notes.append(str(note))
    lines = ["REFINE DECISION"]
    if not grouped:
        lines.append("No decision matrix was produced.")
        return lines
    lines.extend(
        render_table(
            ["Profile", "Removed", "Return d", "DD d", "PF d", "Decision", "Scope"],
            table_rows,
            width=width,
            aligns=["left", "left", "right", "right", "right", "left", "left"],
            min_widths=[14, 14, 9, 8, 7, 16, 9],
        )
    )
    lines.extend(notes)
    return lines


def _consolidate_refine_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for row in decisions:
        profile = str(row.get("profile", "n/a"))
        removed = str(row.get("removed_family", "n/a"))
        key = (profile, removed)
        if key not in groups:
            groups[key] = {
                "profile": profile,
                "removed_family": removed,
                "return_delta_pct": row.get("return_delta_pct"),
                "drawdown_delta_pct": row.get("drawdown_delta_pct"),
                "profit_factor_delta": row.get("profit_factor_delta"),
                "decisions": [],
                "tickers": set(),
                "samples": 0,
                "notes": [],
            }
            order.append(key)
        group = groups[key]
        group["samples"] += 1
        ticker = str(row.get("ticker", "") or "")
        if ticker:
            group["tickers"].add(ticker)
        decision = str(row.get("decision", "inconclusive"))
        if decision not in group["decisions"]:
            group["decisions"].append(decision)
        for note in (row.get("no_op_notes", []) or []) + (row.get("rationale", []) or []):
            note = str(note)
            if "no-op for this run" in note and note not in group["notes"]:
                group["notes"].append(note)

    consolidated: list[dict[str, Any]] = []
    for key in order:
        group = groups[key]
        decisions = group["decisions"]
        tickers = group["tickers"]
        scope = f"tickers={len(tickers)}" if tickers else f"runs={group['samples']}"
        consolidated.append(
            {
                "profile": group["profile"],
                "removed_family": group["removed_family"],
                "return_delta_pct": group["return_delta_pct"],
                "drawdown_delta_pct": group["drawdown_delta_pct"],
                "profit_factor_delta": group["profit_factor_delta"],
                "decision": decisions[0] if len(decisions) == 1 else "mixed",
                "scope": scope,
                "notes": group["notes"],
            }
        )
    return consolidated


def _fmt_delta(value: Any, *, decimals: int = 2, suffix: str = "%") -> str:
    if value in {"", None}:
        return "n/a"
    try:
        return f"{float(value):+.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"
