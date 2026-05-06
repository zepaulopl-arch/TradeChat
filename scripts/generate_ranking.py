import json
from pathlib import Path
import pandas as pd
from datetime import datetime

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

def generate_ranking(artifact_dir: Path):
    signals = []
    # Search recursively in all artifact folders
    for signal_file in artifact_dir.rglob("latest_signal.json"):
        try:
            with open(signal_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            horizons = data.get("horizons", {})
            policy = data.get("policy", {"label": "NEUTRAL", "horizon": "d1"})
            
            # Extract basic returns
            d1_ret = float(horizons.get("d1", {}).get("prediction_return", 0.0)) * 100
            d5_ret = float(horizons.get("d5", {}).get("prediction_return", 0.0)) * 100
            d20_ret = float(horizons.get("d20", {}).get("prediction_return", 0.0)) * 100
            
            # Triggered horizon and its metrics
            trigger_h = policy.get("horizon", "d1")
            trigger_pred = horizons.get(trigger_h, horizons.get("d1", {}))
            
            # Horizon mapping for normalization
            h_days_map = {"d1": 1, "d5": 5, "d20": 20}
            h_days = h_days_map.get(trigger_h.lower(), 1)
            
            conf_pct = float(policy.get("confidence_pct", 0.0))
            triggered_ret = float(trigger_pred.get("prediction_return", 0.0)) * 100
            
            # Intelligent Score: Intensity normalized by time (Annualized logic)
            # Score = Confidence * (|Return| / sqrt(Days))
            import math
            score = conf_pct * (abs(triggered_ret) / math.sqrt(h_days))
            
            priority_map = {
                "STRONG BUY": 100,
                "BUY": 80,
                "STRONG SELL": 70,
                "SELL": 60,
                "NEUTRAL": 10
            }
            priority = priority_map.get(policy.get("label", "NEUTRAL"), 0)
            
            signals.append({
                "ticker": data.get("ticker", "N/A"),
                "signal": policy.get("label", "NEUTRAL"),
                "horizon": trigger_h.upper(),
                "d1_ret": d1_ret,
                "d5_ret": d5_ret,
                "d20_ret": d20_ret,
                "confidence_pct": conf_pct,
                "score": score,
                "priority": priority,
                "rr": float(policy.get("risk_reward_ratio", 0.0)),
                "date": data.get("latest_date", "N/A")
            })
        except Exception:
            continue
            
    if not signals:
        print("No signals found. Run diagnostics or daily first.")
        return

    df = pd.DataFrame(signals)
    df = df.sort_values(by=["priority", "score"], ascending=[False, False])
    
    gray_line = C.DIM + "-" * 115 + C.RESET
    
    print("\n" + gray_line)
    print(f"{C.BOLD}TRADECHAT MULTI-HORIZON RANKING{C.RESET} | {C.BLUE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(gray_line)
    print(f"{'TICKER':<12} | {'SIGNAL':<14} | {'H':<3} | {'D1 %':>9} | {'D5 %':>9} | {'D20 %':>9} | {'CONF':>5} | {'R/R':>5} | {'SCORE':>8}")
    print(gray_line)
    
    for _, row in df.head(40).iterrows():
        color = C.RESET
        if "BUY" in row["signal"]: color = C.GREEN
        elif "SELL" in row["signal"]: color = C.RED
        
        sig_display = f"{color}{row['signal']:<14}{C.RESET}"
        
        print(f"{C.BOLD}{row['ticker']:<12}{C.RESET} | {sig_display} | {C.CYAN}{row['horizon']:<3}{C.RESET} | {row['d1_ret']:>+8.2f}% | {row['d5_ret']:>+8.2f}% | {row['d20_ret']:>+8.2f}% | {C.YELLOW}{row['confidence_pct']:>4.0f}%{C.RESET} | {row['rr']:>5.1f} | {C.BOLD}{row['score']:>8.1f}{C.RESET}")
    
    print(gray_line)
    print(f"{C.DIM}Priority: BUY Signals | Score: Time-Weighted Signal Intensity | R/R Target: 1.5+{C.RESET}")
    print(gray_line)

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    artifact_path = root / "artifacts"
    generate_ranking(artifact_path)
