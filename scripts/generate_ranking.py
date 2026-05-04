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
                
            # Extract all horizons
            horizons = data.get("horizons", {})
            d1_ret = float(horizons.get("d1", {}).get("prediction_return", 0.0)) * 100
            d5_ret = float(horizons.get("d5", {}).get("prediction_return", 0.0)) * 100
            d20_ret = float(horizons.get("d20", {}).get("prediction_return", 0.0)) * 100
            
            # Confidence from d1 (primary)
            conf = float(horizons.get("d1", {}).get("confidence", 0.0))
            disp = float(horizons.get("d1", {}).get("dispersion", 0.0)) * 100
            
            # Opportunity Score based on d1 conviction
            score = conf * abs(d1_ret) * 100
            
            ticker = data.get("ticker", "N/A")
            policy = data.get("policy", {"label": "N/A"})
            
            signals.append({
                "ticker": ticker,
                "signal": policy.get("label", "N/A"),
                "d1_ret": d1_ret,
                "d5_ret": d5_ret,
                "d20_ret": d20_ret,
                "confidence_pct": conf * 100,
                "dispersion_pct": disp,
                "score": score,
                "date": data["latest_date"]
            })
        except Exception:
            continue
            
    if not signals:
        print("No signals found. Run diagnostics or daily first.")
        return

    df = pd.DataFrame(signals)
    df = df.sort_values(by="score", ascending=False)
    
    print("\n" + "="*105)
    print(f" TRADECHAT MULTI-HORIZON RANKING | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*105)
    print(f"{'TICKER':<12} {'SIGNAL':<12} {'D1 %':>9} {'D5 %':>9} {'D20 %':>9} {'CONF %':>8} {'OPP SCORE':>12}")
    print("-" * 105)
    
    for _, row in df.head(25).iterrows():
        color = ""
        if "BUY" in row["signal"]: color = "\033[92m"
        elif "SELL" in row["signal"]: color = "\033[91m"
        reset = "\033[0m"
        
        print(f"{row['ticker']:<12} {color}{row['signal']:<12}{reset} {row['d1_ret']:>+8.2f}% {row['d5_ret']:>+8.2f}% {row['d20_ret']:>+8.2f}% {row['confidence_pct']:>7.0f}% {row['score']:>12.2f}")
    
    print("-" * 105)
    print("Score = D1_Confidence * |D1_Return|")
    print("="*105)

if __name__ == "__main__":
    # Assuming standard project structure
    root = Path(__file__).parent.parent
    artifact_path = root / "artifacts"
    generate_ranking(artifact_path)
