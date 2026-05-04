import json
from pathlib import Path
import pandas as pd
from datetime import datetime

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
            
            conf_pct = float(policy.get("confidence_pct", 0.0))
            # Use return of the horizon that triggered the signal for the score
            triggered_ret = float(trigger_pred.get("prediction_return", 0.0)) * 100
            
            # Opportunity Score based on the triggered conviction
            score = (conf_pct / 100.0) * abs(triggered_ret) * 100
            
            # Signal priority for sorting: BUYs at top, SELLs at bottom (or also at top?)
            # Let's put STRONG signals first, then BUY/SELL, then NEUTRAL
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
                "date": data.get("latest_date", "N/A")
            })
        except Exception:
            continue
            
    if not signals:
        print("No signals found. Run diagnostics or daily first.")
        return

    df = pd.DataFrame(signals)
    # Sort by priority desc, then score desc
    df = df.sort_values(by=["priority", "score"], ascending=[False, False])
    
    print("\n" + "="*115)
    print(f" TRADECHAT MULTI-HORIZON RANKING | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*115)
    print(f"{'TICKER':<12} {'SIGNAL':<14} {'H':<4} {'D1 %':>9} {'D5 %':>9} {'D20 %':>9} {'CONF %':>8} {'OPP SCORE':>12}")
    print("-" * 115)
    
    for _, row in df.head(30).iterrows():
        color = ""
        # Green for BUY, Red for SELL
        if "BUY" in row["signal"]: color = "\033[92m"
        elif "SELL" in row["signal"]: color = "\033[91m"
        reset = "\033[0m"
        
        sig_display = f"{color}{row['signal']:<14}{reset}"
        
        print(f"{row['ticker']:<12} {sig_display} {row['horizon']:<4} {row['d1_ret']:>+8.2f}% {row['d5_ret']:>+8.2f}% {row['d20_ret']:>+8.2f}% {row['confidence_pct']:>7.0f}% {row['score']:>12.2f}")
    
    print("-" * 115)
    print("Priority: BUY signals first | Score = Trigger_Confidence * |Trigger_Return|")
    print("="*115)

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    artifact_path = root / "artifacts"
    generate_ranking(artifact_path)
