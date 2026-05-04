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
                
            ticker = data["ticker"]
            pred = data["prediction"]
            policy = data["policy"]
            
            # Core metrics
            ret = float(pred["prediction_return"]) * 100
            conf = float(pred["confidence"])
            disp = float(pred.get("dispersion", 0)) * 100
            
            # Opportunity Score: Confidence * Magnitude of move
            # We want high confidence on significant moves.
            score = conf * abs(ret)
            
            signals.append({
                "ticker": ticker,
                "signal": policy["label"],
                "return_pct": ret,
                "confidence_pct": conf * 100, # Multiply by 100 for display
                "dispersion_pct": disp,
                "score": score * 100, # Score scale adjustment
                "target": data["target_price"],
                "last_price": data["latest_price"],
                "date": data["latest_date"]
            })
        except Exception:
            continue
            
    if not signals:
        print("No signals found. Run diagnostics or daily first.")
        return

    df = pd.DataFrame(signals)
    
    # Sort by Opportunity Score
    df = df.sort_values(by="score", ascending=False)
    
    print("\n" + "="*90)
    print(f" TRADECHAT OPPORTUNITY RANKING | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90)
    print(f"{'TICKER':<12} {'SIGNAL':<12} {'RETURN %':>10} {'CONF %':>8} {'DISP %':>8} {'OPP SCORE':>12}")
    print("-" * 90)
    
    for _, row in df.head(20).iterrows():
        color = ""
        if "BUY" in row["signal"]: color = "\033[92m" # Green
        elif "SELL" in row["signal"]: color = "\033[91m" # Red
        reset = "\033[0m"
        
        print(f"{row['ticker']:<12} {color}{row['signal']:<12}{reset} {row['return_pct']:>+9.2f}% {row['confidence_pct']:>7.0f}% {row['dispersion_pct']:>7.2f}% {row['score']:>12.2f}")
    
    print("-" * 90)
    print("Opp Score = Confidence * |Expected Return|")
    print("="*90)

if __name__ == "__main__":
    # Assuming standard project structure
    root = Path(__file__).parent.parent
    artifact_path = root / "artifacts"
    generate_ranking(artifact_path)
