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
    ticker = status.get("ticker", "N/A")
    profile = status.get("asset_profile", {}) or {}
    context = status.get("context_tickers", []) or []
    fundamentals = status.get("fundamentals", {}) or {}
    fund_source = "cvm" if "cvm" in str(fundamentals.get("source", "")) else "yfinance"
    sentiment = status.get("sentiment", {}) or {}
    cache_path = str(status.get("path", ""))
    cache_display = cache_path[cache_path.rfind("data_cache"):] if "data_cache" in cache_path else cache_path

    print("\n" + C.HEADER + "─" * 72 + C.RESET)
    print(f"{C.BOLD}DATA REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {status.get('end')}")
    print(C.HEADER + "─" * 72 + C.RESET)
    
    # Block 1: Status & Rows
    print(f"STATUS      : {status.get('status', 'updated').upper()}")
    print(f"ROWS        : {int(status.get('rows', 0))} (range: {status.get('start')}..{status.get('end')})")
    print(f"CACHE       : {cache_display}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 2: Registry
    group = profile.get("group") or "n/a"
    subgroup = profile.get("subgroup") or "n/a"
    print(f"REGISTRY    : {group} | {subgroup} | CNPJ: {profile.get('cnpj') or 'n/a'}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 3: Intelligence & Support
    print(f"CONTEXT     : " + (", ".join([str(x) for x in context]) if context else "none"))
    print(f"FUNDAMENTALS: {fundamentals.get('status', 'available')} ({fund_source})")
    sent_status = sentiment.get("status", "updated")
    if sentiment.get("is_fresh"):
        sent_info = f"cached ({sentiment.get('cache_rows', 0)} days)"
    else:
        sent_info = f"entries: {sentiment.get('new_items', 0)}"
    print(f"SENTIMENT   : {sent_status} | {sent_info}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 4: Models
    models_str = status.get('models', 'n/a')
    if "ok" in models_str:
        m1 = f"{C.CYAN}{C.BOLD}D1{C.RESET}: Ok"
        m5 = f"{C.CYAN}{C.BOLD}D5{C.RESET}: Ok"
        m20 = f"{C.CYAN}{C.BOLD}D20{C.RESET}: Ok"
        models_display = f"{m1} | {m5} | {m20}"
    else:
        models_display = models_str
    print(f"MODELS      : {models_display}")
    print(C.HEADER + "─" * 72 + C.RESET)


def print_train_summary(manifest: dict[str, Any]) -> None:
    horizon = manifest.get("horizon", "d1").upper()
    ticker = manifest.get("ticker", "N/A")
    metrics = manifest.get("metrics", {})
    families = manifest.get("feature_family_profile", {})
    top_feats = manifest.get("top_features", [])
    
    print("\n" + C.HEADER + "─" * 72 + C.RESET)
    print(f"{C.BOLD}TRAIN REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {C.CYAN}{C.BOLD}{horizon}{C.RESET}")
    print(C.HEADER + "─" * 72 + C.RESET)
    
    # Block 1: Engine Performance
    print(f"{'ENGINE':<15} {'MAE (TEST)':>12} {'LATEST RAW':>15}")
    for name in manifest.get("base_engines", []):
        m = metrics.get(name, {})
        mae = m.get("mae_return_raw", 0.0)
        latest = m.get("latest_raw_return", 0.0)
        print(f"{name:<15} {C.YELLOW}{mae:>12.5f}{C.RESET} {latest*100:>14.2f}%")
    
    arb_mae = metrics.get("ridge_arbiter", {}).get("mae_return", 0.0)
    print(f"{C.BOLD}{'RIDGE ARBITER':<15} {C.GREEN}{arb_mae:>12.5f}{C.RESET}{C.BOLD} {manifest.get('latest_prediction_return', 0.0)*100:>14.2f}%{C.RESET}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 2: Intelligence Mix
    line = " | ".join([f"{k}: {v}" for k, v in families.items() if v > 0])
    print(f"INTELLIGENCE: {C.DIM}{line}{C.RESET}")
    if top_feats:
        top_str = ", ".join([f"{f['short']} ({f['score']:.3f})" for f in top_feats[:3]])
        print(f"TOP FEATURES: {C.DIM}{top_str}{C.RESET}")
    print(C.HEADER + "─" * 72 + C.RESET)


def print_signal(signal: dict[str, Any]) -> None:
    ticker = signal["ticker"]
    price = signal["latest_price"]
    policy = signal["policy"]
    fundamentals = signal.get("fundamentals", {})
    horizons = signal.get("horizons", {})
    
    print("\n" + C.HEADER + "─" * 72 + C.RESET)
    print(f"{C.BOLD}SIGNAL REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {signal.get('latest_date')}")
    print(C.HEADER + "─" * 72 + C.RESET)
    
    # Block 1: Core Signal
    print(f"LAST PRICE  : {C.BOLD}{_money(price)}{C.RESET}")
    print(f"SIGNAL      : {C.BOLD}{policy['label']}{C.RESET} | POSTURE: {C.CYAN}{policy['posture']}{C.RESET}")
    print(f"CONFIDENCE  : {C.YELLOW}{policy['confidence_pct']:.0f}%{C.RESET} ({C.CYAN}{C.BOLD}D1{C.RESET})")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 2: Horizons Table
    print(f"{'HORIZON':<12} {'EXP. RETURN':>14} {'TARGET PRICE':>14} {'CONFIDENCE':>12}")
    for h in ["d1", "d5", "d20"]:
        h_data = horizons.get(h, {})
        h_label = h.upper()
        if "error" in h_data:
            print(f"{C.CYAN}{C.BOLD}{h_label:<12}{C.RESET} {C.RED}{'n/a':>14}{C.RESET} {'-':>14} {'-':>12}")
            continue
        ret = float(h_data.get("prediction_return", 0.0))
        conf = float(h_data.get("confidence", 0.0)) * 100
        t_price = price * (1 + ret)
        color = C.GREEN if ret > 0.001 else C.RED if ret < -0.001 else C.RESET
        print(f"{C.CYAN}{C.BOLD}{h_label:<12}{C.RESET} {color}{ret*100:>+13.2f}%{C.RESET} {_money(t_price):>14} {conf:>11.0f}%")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 3: Context & Fundamentals
    sent = float(signal.get("sentiment_value", 0.0))
    context_desc = "Aligned with IBOV" if "ctx_BVSP_corr_20" in signal.get("features_used", []) else "Neutral Context"
    sent_color = C.GREEN if sent > 0.1 else C.RED if sent < -0.1 else C.DIM
    print(f"SENTIMENT   : {sent_color}{sent:>+5.2f}{C.RESET} | CONTEXT: {C.DIM}{context_desc}{C.RESET}")
    print(f"FUNDAMENTALS: P/L {C.DIM}{float(fundamentals.get('pl', 0) or 0):.1f}{C.RESET} | ROE {C.DIM}{float(fundamentals.get('roe', 0) or 0)*100:.1f}%{C.RESET}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 4: Audit
    print(f"REASONS     : " + "; ".join(policy.get("reasons", [])))
    print(f"RUN ID      : {C.DIM}{signal.get('train_run_id')}{C.RESET}")
    print(C.HEADER + "─" * 72 + C.RESET)


def write_txt_report(cfg: dict[str, Any], signal: dict[str, Any]) -> Path:
    ticker = signal["ticker"]
    path = artifact_dir(cfg) / safe_ticker(ticker) / "latest_signal_audit.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TRADECHAT AUDIT | {ticker} | {signal.get('latest_date')}\n")
        f.write("-" * 72 + "\n")
        f.write(f"LAST PRICE : {signal['latest_price']}\n")
        f.write(f"SIGNAL     : {signal['policy']['label']} ({signal['policy']['posture']})\n")
        f.write(f"CONFIDENCE : {signal['policy']['confidence_pct']}%\n")
        f.write("-" * 72 + "\n")
        f.write(f"REASONS:\n")
        for r in signal["policy"].get("reasons", []):
            f.write(f" - {r}\n")
        f.write("-" * 72 + "\n")
        f.write(f"RUN ID: {signal.get('train_run_id')}\n")
    return path


def _fmt_bool(value: Any) -> str:
    return "ok" if bool(value) else "n/a"
