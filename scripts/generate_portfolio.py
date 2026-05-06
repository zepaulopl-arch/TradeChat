import yfinance as yf
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Silence all warnings and yfinance logs
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Tactical Colors
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

def get_live_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    return None

def load_portfolio():
    path = Path("data/portfolio.json")
    if not path.exists():
        return {"positions": {}}
    with open(path, "r") as f:
        return json.load(f)

def get_latest_signal(ticker):
    safe_t = ticker.replace(".", "_").replace("=", "_")
    path = Path(f"artifacts/models/{safe_t}/latest_signal.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_portfolio(portfolio):
    with open("data/portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)

def main():
    portfolio = load_portfolio()
    positions = portfolio.get("positions", {})
    history = portfolio.get("history", [])
    account = portfolio.get("account", {"cash": 10000.0, "initial_capital": 10000.0})
    
    gray_line = C.DIM + "-" * 105 + C.RESET
    
    # --- PHASE 1: ACTIVE MANAGEMENT (AUTO-EXITS) ---
    closed_this_run = []
    active_tickers = list(positions.keys())
    
    for ticker in active_tickers:
        pos = positions[ticker]
        signal_data = get_latest_signal(ticker)
        if not signal_data: continue
        
        live_price = get_live_price(ticker)
        if live_price is None: continue
        
        policy = signal_data.get("policy", {})
        target = policy.get("target_price", 0.0)
        stop = policy.get("stop_loss_price", 0.0)
        m_shares = pos.get("shares", 0)
        is_long = m_shares > 0
        
        exit_reason = None
        if is_long:
            if live_price >= target and target > 0: exit_reason = "TARGET"
            elif live_price <= stop and stop > 0: exit_reason = "STOP"
        else: # Short
            if live_price <= target and target > 0: exit_reason = "TARGET"
            elif live_price >= stop and stop > 0: exit_reason = "STOP"
            
        if exit_reason:
            entry_price = pos.get("entry_price", 0.0)
            pl_cash = (live_price - entry_price) * m_shares
            account["cash"] += (entry_price * abs(m_shares)) + pl_cash
            
            trade_record = {
                "ticker": ticker,
                "action": exit_reason,
                "price": live_price,
                "shares": m_shares,
                "pl_cash": pl_cash,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            history.append(trade_record)
            closed_this_run.append(trade_record)
            del positions[ticker]

    if closed_this_run:
        save_portfolio(portfolio)

    # --- PHASE 2: TACTICAL REPORTING ---
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_activity = [h for h in history if h.get('date', '').startswith(today_str)]

    total_stock_value = 0
    for ticker, pos in positions.items():
        signal_data = get_latest_signal(ticker)
        live_price = get_live_price(ticker)
        curr_p = live_price if live_price is not None else (signal_data.get("latest_price", 0.0) if signal_data else 0.0)
        v_shares = pos.get("shares", 0)
        total_stock_value += abs(v_shares) * curr_p

    total_nav = account['cash'] + total_stock_value
    perf_pct = (total_nav / account['initial_capital'] - 1) * 100
    perf_color = C.GREEN if perf_pct >= 0 else C.RED

    print("\n" + gray_line)
    print(f"{C.BOLD}TRADECHAT TACTICAL PORTFOLIO{C.RESET} | {C.BLUE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(gray_line)
    print(f"CASH: R$ {account['cash']:,.2f} | STOCK: R$ {total_stock_value:,.2f} | {C.BOLD}TOTAL: {perf_color}R$ {total_nav:,.2f} ({perf_pct:+.2f}%){C.RESET}")
    print(gray_line)

    if today_activity:
        print(f"{C.BOLD}{C.CYAN}>>> TODAY'S ACTIVITY:{C.RESET}")
        print(f"   {'ACTION':<10} | {'TICKER':<12} | {'PRICE':>10} | {'SHARES':>8} | {'P/L CASH'}")
        print(f"   " + "-" * 70)
        for h in today_activity:
            h_action = h.get('action', h.get('exit_reason', 'N/A'))
            h_price = h.get('price', h.get('exit_price', 0.0))
            h_shares = h.get('shares', 'N/A')
            shares_str = f"{h_shares:>8}" if h_shares != 'N/A' else f"{'N/A':>8}"
            h_pl = h.get('pl_cash', 0)
            pl_c = C.GREEN if h_pl >= 0 else C.RED
            pl_info = f"{pl_c}R$ {h_pl:>9.2f}{C.RESET}" if 'pl_cash' in h else f"{C.DIM}N/A{C.RESET}"
            
            print(f"   {h_action:<10} | {C.BOLD}{h['ticker']:<12}{C.RESET} | {h_price:>10.2f} | {shares_str} | {pl_info}")
        print(gray_line)

    if not positions:
        print(f"{C.YELLOW}   No active positions in portfolio.{C.RESET}")
        print(gray_line)
        return

    print(f"{'TICKER':<12} | {'SHARES':>6} | {'ENTRY':>10} | {'CURRENT':>10} | {'P/L %':>9} | {'R/R':>5} | {'SIGNAL':<8} | {'TARGET':>10} | {'STOP':>10}")
    print(gray_line)
    
    for ticker, pos in positions.items():
        signal_data = get_latest_signal(ticker)
        live_price = get_live_price(ticker)
        current_price = live_price if live_price is not None else (signal_data.get("latest_price", 0.0) if signal_data else 0.0)
        
        entry_price = pos.get("entry_price", 0.0)
        p_shares = pos.get("shares", 0)
        pl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0
        if p_shares < 0:
            pl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0.0
        
        policy = signal_data.get("policy", {}) if signal_data else {}
        target = policy.get("target_price", 0.0)
        stop = policy.get("stop_loss_price", 0.0)
        current_signal = policy.get("label", "N/A")
        
        rr = policy.get("rr_ratio", 0.0)
        if rr <= 0 and abs(entry_price - stop) > 0.001:
            rr = abs(target - entry_price) / abs(entry_price - stop)
        
        pl_color = C.GREEN if pl_pct >= 0 else C.RED
        sig_color = C.GREEN if "BUY" in current_signal else (C.RED if "SELL" in current_signal else C.RESET)
        price_tag = "LIVE" if live_price is not None else "LAST"
        
        print(f"{C.BOLD}{ticker:<12}{C.RESET} | {p_shares:>6} | {entry_price:>10.2f} | {current_price:>10.2f} | {pl_color}{pl_pct:>8.2f}%{C.RESET} | {rr:>5.1f} | {sig_color}{current_signal:<8}{C.RESET} | {C.CYAN}{target:>10.2f}{C.RESET} | {C.YELLOW}{stop:>10.2f}{C.RESET}")

    print(gray_line)
    print(f"{C.DIM}Status: Monitoring active positions. Targets/Stops will auto-trigger on next consultation.{C.RESET}")
    print(gray_line)

if __name__ == "__main__":
    main()
