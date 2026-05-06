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
        shares = pos.get("shares", 0)
        is_long = shares > 0
        
        exit_reason = None
        if is_long:
            if live_price >= target and target > 0: exit_reason = "TARGET"
            elif live_price <= stop and stop > 0: exit_reason = "STOP"
        else: # Short
            if live_price <= target and target > 0: exit_reason = "TARGET"
            elif live_price >= stop and stop > 0: exit_reason = "STOP"
            
        if exit_reason:
            entry_price = pos.get("entry_price", 0.0)
            pl_cash = (live_price - entry_price) * shares
            account["cash"] += (entry_price * abs(shares)) + pl_cash
            
            trade_record = {
                "ticker": ticker,
                "exit_price": live_price,
                "exit_reason": exit_reason,
                "pl_cash": pl_cash,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            history.append(trade_record)
            closed_this_run.append(trade_record)
            del positions[ticker]

    if closed_this_run:
        save_portfolio(portfolio)

    # --- PHASE 2: TACTICAL REPORTING ---
    print("\n" + gray_line)
    print(f"{C.BOLD}TRADECHAT TACTICAL PORTFOLIO{C.RESET} | {C.BLUE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(gray_line)
    print(f"CASH: R$ {account['cash']:,.2f} | EQUITY: R$ {account['initial_capital']:,.2f}")
    print(gray_line)

    if closed_this_run:
        print(f"{C.BOLD}{C.YELLOW}>>> RECENT EXITS EXECUTED:{C.RESET}")
        for c in closed_this_run:
            color = C.GREEN if c['pl_cash'] >= 0 else C.RED
            print(f"   {c['ticker']:<12} | {c['exit_reason']:<8} | P/L: {color}R$ {c['pl_cash']:>8.2f}{C.RESET}")
        print(gray_line)

    if not positions:
        print(f"{C.YELLOW}   No active positions in portfolio.{C.RESET}")
        print(gray_line)
        return

    print(f"{'TICKER':<12} | {'ENTRY':>10} | {'CURRENT':>10} | {'P/L %':>9} | {'TARGET':>10} | {'STOP':>10} | {'SIGNAL'}")
    print(gray_line)
    
    for ticker, pos in positions.items():
        signal_data = get_latest_signal(ticker)
        live_price = get_live_price(ticker)
        current_price = live_price if live_price is not None else (signal_data.get("latest_price", 0.0) if signal_data else 0.0)
        
        entry_price = pos.get("entry_price", 0.0)
        pl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0
        # Correct P/L for Short positions
        if pos.get("shares", 0) < 0:
            pl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0.0
        
        policy = signal_data.get("policy", {}) if signal_data else {}
        target = policy.get("target_price", 0.0)
        stop = policy.get("stop_loss_price", 0.0)
        current_signal = policy.get("label", "N/A")
        
        pl_color = C.GREEN if pl_pct >= 0 else C.RED
        sig_color = C.RESET
        if "BUY" in current_signal: sig_color = C.GREEN
        elif "SELL" in current_signal: sig_color = C.RED
        
        price_tag = "LIVE" if live_price is not None else "LAST"
        print(f"{C.BOLD}{ticker:<12}{C.RESET} | {entry_price:>10.2f} | {current_price:>10.2f} | {pl_color}{pl_pct:>8.2f}%{C.RESET} | {C.CYAN}{target:>10.2f}{C.RESET} | {C.YELLOW}{stop:>10.2f}{C.RESET} | {sig_color}{current_signal:<10}{C.RESET} ({C.DIM}{price_tag}{C.RESET})")

    print(gray_line)
    print(f"{C.DIM}Status: Monitoring active positions. Targets/Stops will auto-trigger on next consultation.{C.RESET}")
    print(gray_line)

if __name__ == "__main__":
    main()
