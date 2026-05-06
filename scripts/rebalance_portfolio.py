import json
import os
import math
from datetime import datetime
from pathlib import Path

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

def load_config():
    with open("config/config.yaml", "r") as f:
        import yaml
        return yaml.safe_load(f)

def get_all_signals():
    signals = []
    models_dir = Path("artifacts/models")
    if not models_dir.exists():
        return []
    
    for ticker_dir in models_dir.iterdir():
        if ticker_dir.is_dir():
            sig_path = ticker_dir / "latest_signal.json"
            if sig_path.exists():
                with open(sig_path, "r") as f:
                    signals.append(json.load(f))
    return signals

def load_portfolio():
    path = Path("data/portfolio.json")
    if not path.exists():
        return {"positions": {}}
    with open(path, "r") as f:
        return json.load(f)

def main():
    total_capital_target = 10000.0
    portfolio = load_portfolio()
    current_positions = portfolio.get("positions", {})
    account = portfolio.get("account", {"cash": 10000.0, "initial_capital": 10000.0})
    
    signals = get_all_signals()
    
    # 1. ANALYZE CURRENT POSITIONS VS NEW SIGNALS
    gray_line = C.DIM + "-" * 105 + C.RESET
    print("\n" + gray_line)
    print(f"{C.BOLD}TRADECHAT TACTICAL REBALANCE{C.RESET} | {C.BLUE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(gray_line)
    
    # Identify active signals
    active_signals_map = {s["ticker"]: s for s in signals if any(l in s.get("policy", {}).get("label", "") for l in ["BUY", "SELL"])}
    
    # Close positions that are now NEUTRAL or OPPOSITE
    closed_count = 0
    for ticker, pos in list(current_positions.items()):
        sig = active_signals_map.get(ticker)
        shares = pos.get("shares", 0)
        is_long = shares > 0
        
        should_close = False
        reason = ""
        
        if not sig:
            should_close = True
            reason = "NEUTRAL / NO SIGNAL"
        else:
            new_label = sig.get("policy", {}).get("label", "")
            if is_long and "SELL" in new_label:
                should_close = True
                reason = "SIGNAL FLIPPED TO SELL"
            elif not is_long and "BUY" in new_label:
                should_close = True
                reason = "SIGNAL FLIPPED TO BUY"
        
        if should_close:
            exit_price = sig.get("latest_price") if sig else pos.get("entry_price")
            pl_cash = (exit_price - pos.get("entry_price", 0.0)) * shares
            account["cash"] += (pos.get("entry_price", 0.0) * abs(shares)) + pl_cash
            print(f"   {C.RED}CLOSE{C.RESET}: {C.BOLD}{ticker:<12}{C.RESET} | Reason: {C.DIM}{reason}{C.RESET}")
            closed_count += 1
            del current_positions[ticker]

    if closed_count > 0: print(gray_line)

    # 2. CALCULATE TOTAL EQUITY FOR RE-ALLOCATION
    kept_value = 0
    for ticker, pos in current_positions.items():
        sig = active_signals_map.get(ticker)
        price = sig.get("latest_price") if sig else pos.get("entry_price")
        kept_value += abs(pos.get("shares", 0)) * price
    
    total_equity = account["cash"] + kept_value
    total_to_allocate = max(total_equity, total_capital_target)
    
    # 3. RE-ALLOCATE BASED ON SCORE
    total_score = 0
    for ticker, s in active_signals_map.items():
        policy = s.get("policy", {})
        horizons = s.get("horizons", {})
        trigger_h = policy.get("horizon", "d1")
        trigger_pred = horizons.get(trigger_h, horizons.get("d1", {}))
        h_days_map = {"d1": 1, "d5": 5, "d20": 20}
        h_days = h_days_map.get(trigger_h.lower(), 1)
        conf_pct = float(policy.get("confidence_pct", 0.0))
        ret = abs(float(trigger_pred.get("prediction_return", 0.0)) * 100)
        score = conf_pct * (ret / math.sqrt(h_days))
        s["_temp_score"] = score
        total_score += score

    print(f"{'STATUS':<10} | {'TICKER':<12} | {'SIDE':<8} | {'WEIGHT':>8} | {'SHARES':>10} | {'PRICE':>10}")
    print(gray_line)

    final_positions = {}
    for ticker, s in active_signals_map.items():
        label = s.get("policy", {}).get("label", "")
        is_short = "SELL" in label
        weight = s["_temp_score"] / total_score
        amount_to_invest = total_to_allocate * weight
        price = s["latest_price"]
        shares = int(amount_to_invest / price)
        
        if shares > 0:
            display_shares = -shares if is_short else shares
            final_positions[ticker] = {
                "shares": display_shares, "entry_price": price, "date": s["latest_date"], "side": "SHORT" if is_short else "LONG"
            }
            
            # Action Label Logic
            old_shares = current_positions.get(ticker, {}).get("shares", 0)
            if old_shares == 0:
                status, stat_color = "ENTER", C.GREEN
            elif (old_shares > 0 and display_shares < 0) or (old_shares < 0 and display_shares > 0):
                status, stat_color = "REVERSE", C.YELLOW + C.BOLD
            elif old_shares != display_shares:
                status, stat_color = "ADJUST", C.CYAN
            else:
                status, stat_color = "KEEP", C.DIM
                
            side_str = f"{C.RED}SHORT{C.RESET}" if is_short else f"{C.GREEN}LONG {C.RESET}"
            print(f"{stat_color}{status:<10}{C.RESET} | {C.BOLD}{ticker:<12}{C.RESET} | {side_str} | {weight*100:>7.1f}% | {display_shares:>10} | {price:>10.2f}")

    # Update portfolio.json
    portfolio["positions"] = final_positions
    portfolio["account"]["cash"] = total_to_allocate - sum([abs(p['shares']) * p['entry_price'] for p in final_positions.values()])
    
    with open("data/portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)
    
    print(gray_line)
    print(f"{C.BOLD}Rebalance Complete.{C.RESET} Total Equity: {C.GREEN}R$ {total_to_allocate:,.2f}{C.RESET} | Active: {len(final_positions)}")
    print(gray_line)

if __name__ == "__main__":
    main()
