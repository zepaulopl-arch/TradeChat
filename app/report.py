from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import artifact_dir
from .feature_audit import abbreviate_feature_name, feature_family, selected_feature_scores, top_selected_features
from .utils import run_id, safe_ticker, write_json


def _money(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def print_data_summary(status: dict[str, Any]) -> None:
    """Print data update/cache status using the same compact screen style as train/predict."""
    ticker = status.get("ticker", "")
    profile = status.get("asset_profile", {}) or {}
    context = status.get("context_tickers", []) or []
    fundamentals = status.get("fundamentals", {}) or {}
    sentiment = status.get("sentiment", {}) or {}
    period = status.get("period") or "n/a"
    min_rows = status.get("min_rows")
    cache_path = str(status.get("path", ""))
    cache_display = cache_path
    if "data_cache" in cache_display:
        cache_display = cache_display[cache_display.rfind("data_cache"):]

    print("\n" + "=" * 72)
    print(f"TRADEGEM DATA | {ticker}")
    print("=" * 72)
    print(f"status      : {status.get('status', 'updated')}")
    if status.get("ticker_changed"):
        requested = status.get("requested_ticker")
        alias = status.get("alias", {}) or {}
        reason = alias.get("reason", "ticker_alias")
        print(f"alias       : {requested} -> {ticker} | reason={reason}")
    rows = int(status.get("rows", 0) or 0)
    if min_rows is not None:
        validation = "ok" if rows >= int(min_rows) else "low_rows"
        print(f"rows        : {rows} | min={int(min_rows)} | validation={validation}")
    else:
        print(f"rows        : {rows}")
    print(f"range       : {status.get('start')}..{status.get('end')}")
    print(f"period      : {period}")
    print(f"cache       : {cache_display}")
    print("")
    print("context     : " + (", ".join([str(x) for x in context]) if context else "none"))
    skipped_context = status.get("unavailable_context_tickers", []) or []
    if skipped_context:
        print("ctx skipped : " + ", ".join([str(x) for x in skipped_context[:6]]))
    group = profile.get("group") or "n/a"
    subgroup = profile.get("subgroup") or "n/a"
    klass = profile.get("financial_class") or "n/a"
    print(f"registry    : group={group} | subgroup={subgroup} | class={klass}")
    print(f"cnpj        : {profile.get('cnpj') or 'not registered'}")
    linked = profile.get("linked_indices") or []
    if linked:
        print("indices     : " + ", ".join([str(x) for x in linked]))
    fund_status = fundamentals.get("status", "disabled")
    fund_source = fundamentals.get("source", "n/a")
    print(f"fundamentals: {fund_status} | source={fund_source}")
    sent_status = sentiment.get("status", "disabled")
    sent_cache = sentiment.get("cache", "n/a")
    sent_items = sentiment.get("new_items")
    if sent_items is None:
        print(f"sentiment   : {sent_status} | daily_cache={sent_cache}")
    else:
        print(f"sentiment   : {sent_status} | daily_cache={sent_cache} | entries={sent_items}")
    print("=" * 72)


def print_train_summary(manifest: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"TRADEGEM TRAIN | {manifest['ticker']}")
    print("=" * 72)
    print(f"run_id      : {manifest['run_id']}")
    print(f"rows        : {manifest['rows']} ({manifest['train_rows']} train / {manifest['test_rows']} test)")
    base_engines = manifest.get('base_engines', manifest.get('engines', []))
    print(f"base engines: {', '.join(base_engines)}")
    print(f"arbiter     : {manifest.get('arbiter', 'ridge')}")
    print(f"features    : {len(manifest['features'])}")
    top_feats = manifest.get("top_features") or []
    if top_feats:
        print("top feats   : " + ", ".join([str(item.get("short") or abbreviate_feature_name(item.get("name", ""))) for item in top_feats[:5]]))
    arbiter_metrics = manifest.get('metrics', {}).get('ridge_arbiter') or manifest.get('metrics', {}).get('ensemble', {})
    print(f"mae arbiter : {arbiter_metrics.get('mae_return', 0.0):.5f}")
    print(f"prediction  : {manifest['latest_prediction_return']*100:+.2f}% D+1")
    print(f"confidence  : {manifest['confidence']*100:.0f}%")
    print(f"artifact    : {manifest['model_path']}")
    print("=" * 72)


def print_signal(signal: dict[str, Any]) -> None:
    ticker = signal["ticker"]
    price = signal["latest_price"]
    target = signal["target_price"]
    policy = signal["policy"]
    fundamentals = signal.get("fundamentals", {})
    engines = signal.get("prediction", {}).get("by_engine", {})
    print("\n" + "=" * 72)
    print(f"TRADEGEM SIGNAL | {ticker}")
    print("=" * 72)
    print(f"date        : {signal.get('latest_date')}")
    print(f"last price  : {_money(price)}")
    print(f"target D+1  : {_money(target)} ({policy['score_pct']:+.2f}%)")
    print(f"signal      : {policy['label']} | posture: {policy['posture']}")
    print(f"confidence  : {policy['confidence_pct']:.0f}%")
    print(f"fundamentals: P/L {float(fundamentals.get('pl', 0) or 0):.2f} | DY {float(fundamentals.get('dy', 0) or 0)*100:.1f}% | P/VP {float(fundamentals.get('pvp', 0) or 0):.2f} | ROE {float(fundamentals.get('roe', 0) or 0)*100:.1f}%")
    if engines:
        line = " | ".join([f"{k} {v*100:+.2f}%" for k, v in engines.items()])
        arbiter = signal.get('prediction', {}).get('arbiter', 'ridge')
        print(f"base engines: {line}")
        raw_engines = signal.get("prediction", {}).get("raw_by_engine", {}) or {}
        discarded = signal.get("prediction", {}).get("discarded_engines", []) or []
        used = signal.get("prediction", {}).get("used_engines", []) or []
        if raw_engines and any(abs(float(raw_engines.get(k, 0))) > abs(float(v)) + 1e-12 for k, v in engines.items()):
            raw_line = " | ".join([f"{k} {v*100:+.2f}%" for k, v in raw_engines.items()])
            print(f"raw engines : {raw_line}")
        if discarded:
            print(f"used engines: {', '.join(used) if used else 'none'}")
            print(f"guard       : {', '.join(discarded)} neutralized before Ridge")
        print(f"arbiter     : {arbiter}")
    print("reasons     : " + "; ".join(policy.get("reasons", [])))
    print(f"train run   : {signal.get('train_run_id')}")
    print("=" * 72)


def _fmt_bool(value: Any) -> str:
    return "on" if bool(value) else "off"


def _feature_flags(cfg: dict[str, Any]) -> dict[str, Any]:
    features = cfg.get("features", {})
    return {
        "technical": features.get("technical", {}).get("enabled", True),
        "market_context": features.get("context", {}).get("enabled", False),
        "fundamentals": features.get("fundamentals", {}).get("enabled", False),
        "fundamental_regimes": features.get("fundamentals", {}).get("add_regime_features", False),
        "sentiment": features.get("sentiment", {}).get("enabled", False),
    }


def render_txt_report(cfg: dict[str, Any], signal: dict[str, Any]) -> str:
    """Build a detailed audit report. Unlike predict, this is meant for file output."""
    policy = signal.get("policy", {})
    prediction = signal.get("prediction", {})
    train_manifest = prediction.get("train_manifest", {}) or {}
    fundamentals = signal.get("fundamentals", {}) or {}
    engines = prediction.get("by_engine", {}) or {}
    flags = _feature_flags(cfg)
    features_used = signal.get("features_used", []) or []
    prep_meta = (train_manifest.get("dataset_meta", {}) or {}).get("preparation", {}) or {}
    tune_summary = train_manifest.get("tune_summary", {}) or {}
    metrics = train_manifest.get("metrics", {}) or {}

    lines: list[str] = []
    lines.append("TRADEGEM AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"ticker        : {signal.get('ticker')}")
    lines.append(f"latest_date   : {signal.get('latest_date')}")
    lines.append(f"train_run_id  : {signal.get('train_run_id')}")
    lines.append(f"architecture  : {train_manifest.get('architecture', 'XGB + RandomForest + MLP -> Ridge arbiter')}")
    lines.append(f"autotune      : {_fmt_bool(train_manifest.get('autotune', False))}")
    lines.append("")
    lines.append("SIGNAL")
    lines.append("-" * 80)
    lines.append(f"last_price    : {_money(float(signal.get('latest_price', 0) or 0))}")
    lines.append(f"target_d1     : {_money(float(signal.get('target_price', 0) or 0))}")
    lines.append(f"return_d1_pct : {float(policy.get('score_pct', 0) or 0):+.2f}%")
    lines.append(f"label         : {policy.get('label')}")
    lines.append(f"posture       : {policy.get('posture')}")
    lines.append(f"confidence    : {float(policy.get('confidence_pct', 0) or 0):.0f}%")
    lines.append("reasons       : " + "; ".join(policy.get("reasons", [])))
    lines.append("")
    lines.append("BASE ENGINES AND ARBITER")
    lines.append("-" * 80)
    if engines:
        for name, value in engines.items():
            lines.append(f"{name:<14}: {float(value)*100:+.4f}%")
    lines.append(f"arbiter       : {prediction.get('arbiter', 'ridge')}")
    lines.append(f"dispersion    : {float(prediction.get('dispersion', 0) or 0)*100:.4f}%")
    lines.append("")
    lines.append("MODEL METRICS")
    lines.append("-" * 80)
    if metrics:
        for name, item in metrics.items():
            if isinstance(item, dict):
                lines.append(f"{name:<14}: mae_return={float(item.get('mae_return', 0) or 0):.6f}")
    else:
        lines.append("not available in latest signal")
    lines.append("")
    lines.append("CONFIGURED FEATURE BLOCKS")
    lines.append("-" * 80)
    for name, value in flags.items():
        lines.append(f"{name:<22}: {_fmt_bool(value)}")
    lines.append("")
    lines.append("DATA PREPARATION")
    lines.append("-" * 80)
    if prep_meta:
        lines.append(f"generated     : {prep_meta.get('input_feature_count', 'n/a')} features")
        lines.append(f"selected      : {prep_meta.get('selected_feature_count', len(features_used))} features")
        lines.append(f"constant_drop : {len(prep_meta.get('dropped_constant_features', []) or [])}")
        sel = prep_meta.get('selection', {}) or {}
        lines.append(f"method        : {sel.get('method', 'n/a')}")
        lines.append(f"corr_limit    : {sel.get('correlation_threshold', 'n/a')}")
        fams = prep_meta.get('families', {}) or sel.get('families', {}) or {}
        if fams:
            lines.append("families      : " + ", ".join([f"{k}={v}" for k, v in fams.items()]))
    else:
        lines.append("not available")
    lines.append("")
    lines.append("FEATURES USED")
    lines.append("-" * 80)
    lines.append(f"count         : {len(features_used)}")
    top_feats = train_manifest.get("top_features") or top_selected_features(prep_meta, features_used, n=5)
    if top_feats:
        lines.append("top_5         : " + ", ".join([str(item.get("short") or abbreviate_feature_name(item.get("name", ""))) for item in top_feats[:5]]))
    scores = selected_feature_scores(prep_meta, features_used)
    for feature in features_used:
        family = feature_family(str(feature))
        score = float(scores.get(feature, 0.0) or 0.0)
        short = abbreviate_feature_name(str(feature), max_len=24)
        lines.append(f"- {short:<24} | family={family:<12} | relevance={score:.6f} | source={feature}")
    lines.append("")
    lines.append("FUNDAMENTALS")
    lines.append("-" * 80)
    if fundamentals:
        for key in sorted(fundamentals):
            lines.append(f"{key:<18}: {fundamentals[key]}")
    else:
        lines.append("not available")
    lines.append("")
    lines.append("AUTOTUNE SUMMARY")
    lines.append("-" * 80)
    if tune_summary:
        for name, item in tune_summary.items():
            lines.append(f"{name}:")
            if isinstance(item, dict):
                for key, value in item.items():
                    lines.append(f"  {key:<12}: {value}")
            else:
                lines.append(f"  {item}")
    else:
        lines.append("not used or not available")
    lines.append("")
    lines.append("RAW ARTIFACT POINTERS")
    lines.append("-" * 80)
    lines.append(f"latest_signal : artifacts/{signal.get('ticker', '').replace('.', '_')}/latest_signal.json")
    if train_manifest.get("model_path"):
        lines.append(f"model         : {train_manifest.get('model_path')}")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_txt_report(cfg: dict[str, Any], signal: dict[str, Any]) -> Path:
    ticker = signal["ticker"]
    reports_dir = artifact_dir(cfg) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = run_id("report", ticker)
    path = reports_dir / f"{report_id}.txt"
    path.write_text(render_txt_report(cfg, signal), encoding="utf-8")
    index_path = reports_dir / f"latest_{safe_ticker(ticker)}.json"
    write_json(index_path, {"ticker": ticker, "latest_report": str(path), "train_run_id": signal.get("train_run_id")})
    return path
