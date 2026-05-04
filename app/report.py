from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import artifact_dir
from .feature_audit import abbreviate_feature_name, feature_family, selected_feature_scores, top_selected_features
from .utils import run_id, safe_ticker, write_json


class C:
    """Discrete, opaque color palette for professional CLI."""
    HEADER = '\033[90m'  # Dark Gray
    BLUE = '\033[38;5;67m'  # Steel Blue (discrete)
    CYAN = '\033[38;5;109m' # Muted Cyan
    GREEN = '\033[38;5;108m' # Sage Green
    YELLOW = '\033[38;5;144m' # Sand/Beige
    RED = '\033[38;5;131m'   # Muted Red
    DIM = '\033[2m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def _money(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def print_data_summary(status: dict[str, Any]) -> None:
    """Print data update/cache status using the same compact screen style as train/predict."""
    ticker = status.get("ticker", "")
    profile = status.get("asset_profile", {}) or {}
    context = status.get("context_tickers", []) or []
    fundamentals = status.get("fundamentals", {}) or {}
    fund_status = fundamentals.get("status", "disabled")
    fund_source = fundamentals.get("source", "n/a")
    sentiment = status.get("sentiment", {}) or {}
    period = status.get("period") or "n/a"
    min_rows = status.get("min_rows")
    cache_path = str(status.get("path", ""))
    cache_display = cache_path
    if "data_cache" in cache_display:
        cache_display = cache_display[cache_display.rfind("data_cache"):]

    print("\n" + C.HEADER + "=" * 72 + C.RESET)
    print(f"{C.BOLD}TRADECHAT DATA{C.RESET} | {C.BLUE}{ticker}{C.RESET}")
    print(C.HEADER + "=" * 72 + C.RESET)
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
    print(f"fundamentals: {fund_status} | source={fund_source}")
    sent_status = sentiment.get("status", "disabled")
    sent_cache = sentiment.get("cache", "n/a")
    sent_items = sentiment.get("new_items")
    if sent_items is None:
        print(f"sentiment   : {sent_status} | daily_cache={sent_cache}")
    else:
        print(f"sentiment   : {sent_status} | daily_cache={sent_cache} | entries={sent_items}")
    print(f"models      : {status.get('models', 'n/a')}")
    print("=" * 72)


def print_train_summary(manifest: dict[str, Any]) -> None:
    horizon = manifest.get("horizon", "d1")
    ticker = manifest.get("ticker", "N/A")
    print(f"  {C.BLUE}{ticker}{C.RESET} | {C.CYAN}{horizon.upper()}{C.RESET} | mae: {C.YELLOW}{manifest.get('metrics', {}).get('ridge_arbiter', {}).get('mae_return', 0.0):.5f}{C.RESET} | conf: {C.GREEN}{manifest.get('confidence', 0.0)*100:.0f}%{C.RESET} | features: {len(manifest.get('features', []))}")


def print_signal(signal: dict[str, Any]) -> None:
    ticker = signal["ticker"]
    price = signal["latest_price"]
    policy = signal["policy"]
    fundamentals = signal.get("fundamentals", {})
    horizons = signal.get("horizons", {})
    
    print("\n" + C.HEADER + "=" * 72 + C.RESET)
    print(f"{C.BOLD}TRADECHAT SIGNAL{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {signal.get('latest_date')}")
    print(C.HEADER + "=" * 72 + C.RESET)
    print(f"last price  : {C.BOLD}{_money(price)}{C.RESET}")
    print(f"signal      : {C.BOLD}{policy['label']}{C.RESET} | posture: {C.CYAN}{policy['posture']}{C.RESET}")
    print(f"confidence  : {C.YELLOW}{policy['confidence_pct']:.0f}%{C.RESET} (D+1)")
    print("-" * 72)
    
    # Table of Horizons
    print(f"{'HORIZON':<12} {'EXP. RETURN':>14} {'TARGET PRICE':>14} {'CONFIDENCE':>12}")
    for h in ["d1", "d5", "d20"]:
        h_data = horizons.get(h, {})
        if "error" in h_data:
            print(f"{h:<12} {C.RED}{'n/a':>14}{C.RESET} {'-':>14} {'-':>12}")
            continue
            
        ret = float(h_data.get("prediction_return", 0.0))
        conf = float(h_data.get("confidence", 0.0)) * 100
        t_price = price * (1 + ret)
        
        color = C.GREEN if ret > 0 else C.RED if ret < 0 else C.RESET
        print(f"{h:<12} {color}{ret*100:>+13.2f}%{C.RESET} {_money(t_price):>14} {conf:>11.0f}%")
    
    print("-" * 72)
    print(f"fundamentals: P/L {C.DIM}{float(fundamentals.get('pl', 0) or 0):.2f}{C.RESET} | DY {C.DIM}{float(fundamentals.get('dy', 0) or 0)*100:.1f}%{C.RESET} | P/VP {C.DIM}{float(fundamentals.get('pvp', 0) or 0):.2f}{C.RESET} | ROE {C.DIM}{float(fundamentals.get('roe', 0) or 0)*100:.1f}%{C.RESET}")
    
    d1_engines = horizons.get("d1", {}).get("by_engine", {})
    if d1_engines:
        line = " | ".join([f"{k} {v*100:+.2f}%" for k, v in d1_engines.items()])
        print(f"base (d1)   : {C.DIM}{line}{C.RESET}")
    
    print(f"reasons     : " + "; ".join(policy.get("reasons", [])))
    print(f"train run   : {C.DIM}{signal.get('train_run_id')}{C.RESET}")
    print(C.HEADER + "=" * 72 + C.RESET)


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
    lines.append("TRADECHAT AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"ticker        : {signal.get('ticker')}")
    lines.append(f"latest_date   : {signal.get('latest_date')}")
    lines.append(f"train_run_id  : {signal.get('train_run_id')}")
    lines.append(f"architecture  : {train_manifest.get('architecture', 'XGB + CatBoost + ExtraTrees -> Ridge arbiter')}")
    lines.append(f"autotune      : {_fmt_bool(train_manifest.get('autotune', False))}")
    lines.append("")
    lines.append("SIGNAL")
    lines.append("-" * 80)
    lines.append(f"last_price    : {_money(float(signal.get('latest_price', 0) or 0))}")
    lines.append(f"label_d1      : {policy.get('label')}")
    lines.append(f"posture_d1    : {policy.get('posture')}")
    lines.append(f"confidence_d1 : {float(policy.get('confidence_pct', 0) or 0):.0f}%")
    lines.append("")
    lines.append("HORIZONS PREDICTION TABLE")
    lines.append("-" * 80)
    lines.append(f"{'HORIZON':<12} {'EXP. RETURN':>14} {'TARGET PRICE':>14} {'CONFIDENCE':>12}")
    horizons = signal.get("horizons", {})
    for h in ["d1", "d5", "d20"]:
        h_data = horizons.get(h, {})
        if "error" in h_data:
            lines.append(f"{h:<12} {'error':>14} {'-':>14} {'-':>12}")
            continue
        ret = float(h_data.get("prediction_return", 0.0))
        conf = float(h_data.get("confidence", 0.0)) * 100
        t_price = float(signal.get('latest_price', 0)) * (1 + ret)
        lines.append(f"{h:<12} {ret*100:>+13.2f}% {_money(t_price):>14} {conf:>11.0f}%")
    lines.append("-" * 80)
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
