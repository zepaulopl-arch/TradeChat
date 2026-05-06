from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import reports_dir
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
    fund_source = fundamentals.get("source", "n/a")
    sentiment = status.get("sentiment", {}) or {}
    
    gray_line = C.DIM + "─" * 72 + C.RESET
    
    print("\n" + gray_line)
    print(f"{C.BOLD}DATA REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {status.get('end')}")
    print(gray_line)
    
    # Block 1: Inventory Status
    st_val = status.get('status', 'updated').upper()
    rows_val = str(status.get('rows', 0))
    period_val = status.get('period', 'max')
    print(f"STATUS: {C.GREEN}{st_val:<12}{C.RESET} | ROWS: {C.BOLD}{rows_val:<12}{C.RESET} | PERIOD: {C.DIM}{period_val}{C.RESET}")
    
    # Block 2: Registry & Profile
    group = (profile.get("group") or "n/a").upper()
    cnpj = profile.get("cnpj") or "n/a"
    source = status.get("source", "yfinance")
    print(f"GROUP : {C.BOLD}{group:<12}{C.RESET} | CNPJ: {C.DIM}{cnpj:<12}{C.RESET} | SOURCE: {C.DIM}{source}{C.RESET}")
    
    print(gray_line)
    # Block 3: Support Intelligence
    ctx_str = ", ".join([str(x) for x in context]) if context else "none"
    print(f"CONTEXT     : {C.DIM}{ctx_str}{C.RESET}")
    print(f"FUNDAMENTALS: {C.BOLD}{fundamentals.get('status', 'available').upper():<10}{C.RESET} | SOURCE: {C.DIM}{fund_source}{C.RESET}")
    sent_status = sentiment.get("status", "updated").upper()
    sent_info = "FRESH" if sentiment.get("is_fresh") else f"CACHE: {sentiment.get('cache_rows', 0)}d"
    print(f"SENTIMENT   : {C.BOLD}{sent_status:<10}{C.RESET} | {C.DIM}{sent_info}{C.RESET}")
    
    print(gray_line)


def print_multi_horizon_train_summary(manifests: list[dict[str, Any]]) -> None:
    if not manifests:
        return
    
    # Sort by horizon to ensure order D1, D5, D20
    order = {"d1": 0, "d5": 1, "d20": 2}
    sorted_m = sorted(manifests, key=lambda x: order.get(x.get("horizon", "").lower(), 99))
    
    m0 = sorted_m[0]
    ticker = m0.get("ticker", "N/A")
    base_engines = m0.get("base_engines", [])
    
    gray_line = C.DIM + "─" * 72 + C.RESET
    
    print("\n" + gray_line)
    print(f"{C.BOLD}TRAIN REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {C.CYAN}{C.BOLD}MULTI-HORIZON{C.RESET}")
    print(gray_line)
    
    # Header for Grid
    print(f"{'ENGINE':<15} {'MAE (D1)':>12} {'MAE (D5)':>12} {'MAE (D20)':>12}   {'STATUS'}")
    
    # Table Rows
    for name in base_engines:
        row = f"{name:<15}"
        for m in sorted_m:
            mae = m.get("metrics", {}).get(name, {}).get("mae_return_raw", 0.0)
            row += f" {C.YELLOW}{mae:>12.5f}{C.RESET}"
        row += f"   {C.GREEN}OK{C.RESET}"
        print(row)
    
    # Arbiter Row
    arb_row = f"{C.BOLD}{'ridge_arbiter':<15}{C.RESET}"
    for m in sorted_m:
        mae = m.get("metrics", {}).get("ridge_arbiter", {}).get("mae_return", 0.0)
        arb_row += f" {C.GREEN}{mae:>12.5f}{C.RESET}"
    arb_row += f"   {C.BOLD}FINAL{C.RESET}"
    print(arb_row)
    
    print(gray_line)
    # Block 3: Intelligence Mix (from latest/largest model)
    m_last = sorted_m[-1]
    families = m_last.get("feature_family_profile", {})
    top_feats = m_last.get("top_features", [])
    prep = m_last.get("preparation", {}) or {}
    samples = prep.get("output_rows", 0)
    features = prep.get("selected_feature_count", 0)
    
    print(f"SAMPLES     : {C.DIM}{samples:<10}{C.RESET} | FEATURES: {C.DIM}{features}{C.RESET}")
    line = " | ".join([f"{k}: {v}" for k, v in families.items() if v > 0])
    print(f"INTELLIGENCE: {C.DIM}{line}{C.RESET}")
    if top_feats:
        top_str = ", ".join([f"{f['short']}" for f in top_feats[:4]])
        print(f"TOP FEATURES: {C.DIM}{top_str}{C.RESET}")
    print(gray_line)


def print_signal(signal: dict[str, Any]) -> None:
    ticker = signal["ticker"]
    price = signal["latest_price"]
    policy = signal["policy"]
    fundamentals = signal.get("fundamentals", {})
    horizons = signal.get("horizons", {})
    is_neutral = policy['label'] in ["NEUTRAL", "LATERAL"]
    trigger_h = policy.get("horizon", "d1").upper()
    
    # Values for Tactical Block
    p_val = _money(price)
    s_val = f"{policy['label']} ({policy['posture']})"
    c_val = f"{policy['confidence_pct']:.0f}% | {trigger_h}"
    
    if is_neutral:
        t_val, st_val, rr_val = "---", "---", "---"
        sz_val, pr_val, be_val = "0", "---", "---"
    else:
        t_val = _money(policy.get('target_price', 0.0))
        st_val = _money(policy.get('stop_loss_price', 0.0))
        rr_val = f"{policy.get('risk_reward_ratio', 0.0):.2f}"
        sz_val = str(policy.get('position_size', 0))
        pr_val = _money(policy.get('target_partial', 0.0))
        be_val = _money(policy.get('breakeven_trigger', 0.0))

    print("\n" + C.HEADER + "─" * 72 + C.RESET)
    print(f"{C.BOLD}SIGNAL REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {signal.get('latest_date')}")
    print(C.HEADER + "─" * 72 + C.RESET)
    
    # Block 1: Cockpit Tático
    print(f"PRICE: {C.BOLD}{p_val:<12}{C.RESET} | SIGNAL: {C.BOLD}{s_val:<18}{C.RESET} | CONF: {C.YELLOW}{c_val}{C.RESET}")
    print(f"TARGET: {C.GREEN}{t_val:<11}{C.RESET} | STOP: {C.RED}{st_val:<20}{C.RESET} | R/R: {C.CYAN}{rr_val}{C.RESET}")
    print(f"SIZE: {C.BOLD}{sz_val:<13}{C.RESET} | PARTIAL: {C.CYAN}{pr_val:<17}{C.RESET} | BE: {C.DIM}{be_val}{C.RESET}")
    
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
    pl_val = f"{float(fundamentals.get('pl', 0) or 0):.1f}"
    roe_val = f"{float(fundamentals.get('roe', 0) or 0)*100:.1f}%"
    
    print(f"SENTIMENT: {sent_color}{sent:>+6.2f}{C.RESET}     | CONTEXT: {C.DIM}{context_desc}{C.RESET}")
    print(f"P/L      : {C.DIM}{pl_val:<10}{C.RESET} | ROE    : {C.DIM}{roe_val}{C.RESET}")
    
    print(C.HEADER + "─" * 72 + C.RESET)
    # Block 4: Audit
    # Filter out technical reasons containing '='
    cleaned_reasons = [r for r in policy.get("reasons", []) if "=" not in r]
    
    print(f"REASONS     : " + "; ".join(cleaned_reasons))
    print(f"RUN ID      : {C.DIM}{signal.get('train_run_id')}{C.RESET}")
    print(C.HEADER + "─" * 72 + C.RESET)


def print_signal_brief(signal: dict[str, Any]) -> None:
    ticker = signal["ticker"]
    price = signal["latest_price"]
    policy = signal["policy"]
    is_neutral = policy['label'] in ["NEUTRAL", "LATERAL"]
    trigger_h = policy.get("horizon", "d1").upper()
    
    # Values
    p_val = _money(price)
    s_val = f"{policy['label']} ({policy['posture']})"
    c_val = f"{policy['confidence_pct']:.0f}% | {trigger_h}"
    
    if is_neutral:
        t_val, st_val, rr_val = "---", "---", "---"
        sz_val, pr_val, be_val = "0", "---", "---"
    else:
        t_val = _money(policy.get('target_price', 0.0))
        st_val = _money(policy.get('stop_loss_price', 0.0))
        rr_val = f"{policy.get('risk_reward_ratio', 0.0):.2f}"
        sz_val = str(policy.get('position_size', 0))
        pr_val = _money(policy.get('target_partial', 0.0))
        be_val = _money(policy.get('breakeven_trigger', 0.0))

    print("\n" + C.HEADER + "─" * 72 + C.RESET)
    print(f"{C.BOLD}SIGNAL REPORT{C.RESET} | {C.BLUE}{ticker}{C.RESET} | {signal.get('latest_date')}")
    print(C.HEADER + "─" * 72 + C.RESET)
    
    # Line 1: Status
    print(f"PRICE: {C.BOLD}{p_val:<12}{C.RESET} | SIGNAL: {C.BOLD}{s_val:<18}{C.RESET} | CONF: {C.YELLOW}{c_val}{C.RESET}")
    
    # Line 2: Targets
    print(f"TARGET: {C.GREEN}{t_val:<11}{C.RESET} | STOP: {C.RED}{st_val:<20}{C.RESET} | R/R: {C.CYAN}{rr_val}{C.RESET}")
    
    # Line 3: Execution
    print(f"SIZE: {C.BOLD}{sz_val:<13}{C.RESET} | PARTIAL: {C.CYAN}{pr_val:<17}{C.RESET} | BE: {C.DIM}{be_val}{C.RESET}")
    
    print(C.HEADER + "─" * 72 + C.RESET)


def write_txt_report(cfg: dict[str, Any], signal: dict[str, Any]) -> Path:
    ticker = signal["ticker"]
    policy = signal["policy"]
    path = reports_dir(cfg) / f"{safe_ticker(ticker)}_audit.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TRADECHAT AUDIT | {ticker} | {signal.get('latest_date')}\n")
        f.write("-" * 72 + "\n")
        f.write(f"LAST PRICE : {signal['latest_price']}\n")
        f.write(f"SIGNAL     : {signal['policy']['label']} ({signal['policy']['posture']})\n")
        trigger_h = policy.get("horizon", "d1").upper()
        f.write(f"CONFIDENCE : {int(policy.get('confidence_pct', 0))}% | {trigger_h}\n")
        is_n = policy['label'] in ["NEUTRAL", "LATERAL"]
        f.write(f"SUGG. SIZE : {policy.get('position_size', 0) if not is_n else 0} units\n")
        f.write(f"PARTIAL T1 : R$ {policy.get('target_partial', 0.0) if not is_n else 0.0:.2f} (Realize 50%)\n")
        f.write(f"TARGET T2  : R$ {policy.get('target_price', 0.0) if not is_n else 0.0:.2f}\n")
        f.write(f"STOP-LOSS  : R$ {policy.get('stop_loss_price', 0.0) if not is_n else 0.0:.2f}\n")
        f.write(f"BE TRIGGER : R$ {policy.get('breakeven_trigger', 0.0) if not is_n else 0.0:.2f} (Move Stop to Entry)\n")
        f.write(f"R/R RATIO  : {policy.get('risk_reward_ratio', 0.0) if not is_n else 0.0:.2f}\n")
        f.write("-" * 72 + "\n")
        f.write(f"REASONS:\n")
        # Filter out technical reasons containing '='
        cleaned = [r for r in policy.get("reasons", []) if "=" not in r]
        for r in cleaned:
            f.write(f" - {r}\n")
        f.write("-" * 72 + "\n")
        f.write(f"RUN ID: {signal.get('train_run_id')}\n")
    return path


def _fmt_bool(value: Any) -> str:
    return "ok" if bool(value) else "n/a"
