import json
import os
import math
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

def main():
    total_capital = 10000.0
    signals = get_all_signals()
    
    # Filter BUY and SELL signals
    active_signals = [s for s in signals if any(label in s.get("policy", {}).get("label", "") for label in ["BUY", "SELL"])]
    
    if not active_signals:
        print("No BUY/SELL signals found to allocate capital.")
        return

    # Calculate Total Score for weighting
    total_score = 0
    for s in active_signals:
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

    new_positions = {}
    print(f"\n--- REBALANCING PORTFOLIO (Capital: R$ {total_capital:,.2f}) ---")
    
    for s in active_signals:
        ticker = s["ticker"]
        label = s.get("policy", {}).get("label", "")
        is_short = "SELL" in label
        
        weight = s["_temp_score"] / total_score
        amount_to_invest = total_capital * weight
        price = s["latest_price"]
        shares = int(amount_to_invest / price)
        
        if shares > 0:
            display_shares = -shares if is_short else shares
            new_positions[ticker] = {
                "shares": display_shares,
                "entry_price": price,
                "date": s["latest_date"],
                "allocated_capital": round(amount_to_invest, 2),
                "score_weight": round(weight * 100, 2),
                "side": "SHORT" if is_short else "LONG"
            }
            side_str = f"{C.RED}SHORT{C.RESET}" if is_short else f"{C.GREEN}LONG {C.RESET}"
            print(f"ALLOCATED: {ticker:<12} | {side_str} | Weight: {weight*100:>5.1f}% | Amount: R$ {amount_to_invest:>8.2f} | Shares: {display_shares}")

    # Update portfolio.json
    portfolio = {
        "account": {
            "initial_capital": total_capital,
            "cash": 0.0, # Full allocation
            "currency": "BRL"
        },
        "positions": new_positions,
        "history": [] # Reset history for fresh start
    }
    
    with open("data/portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)
    
    print("\nPortfolio updated successfully. Check 'run_portfolio.bat' for live monitoring.")

if __name__ == "__main__":
    main()
