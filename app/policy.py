from __future__ import annotations

from typing import Any


def _confidence_floor_pct(pcfg: dict[str, Any]) -> float:
    raw = float(pcfg.get("min_confidence_pct", 0.55))
    return raw * 100 if raw <= 1.0 else raw


def classify_signal(cfg: dict[str, Any], results: dict[str, Any], dataset_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Classify signal based on multi-horizon predictions.
    It checks D1, D5, and D20, picking the most significant valid signal.
    """
    pcfg = cfg.get("policy", {})
    floor_pct = _confidence_floor_pct(pcfg)
    
    # Thresholds for different horizons (scaled conservatively)
    # Base is the D1 threshold from config
    buy_base = float(pcfg.get("buy_return_pct", 0.45))
    strong_buy_base = float(pcfg.get("strong_buy_return_pct", 1.5))
    sell_base = float(pcfg.get("sell_return_pct", -0.45))
    strong_sell_base = float(pcfg.get("strong_sell_return_pct", -1.5))

    # Horizon configuration: (name, threshold_multiplier, posture_suffix)
    horizons_config = [
        ("d1", 1.0, "day"),
        ("d5", 2.2, "swing"),
        ("d20", 4.5, "position")
    ]

    best_signal = {
        "label": "NEUTRAL",
        "posture": "wait",
        "score_pct": 0.0,
        "confidence_pct": 0.0,
        "horizon": "d1",
        "reasons": []
    }

    # Evaluate each horizon
    candidates = []
    for h_name, multiplier, posture_type in horizons_config:
        pred = results.get(h_name, {})
        if "error" in pred or not pred:
            continue
            
        ret_pct = float(pred.get("prediction_return", 0.0) * 100)
        conf_pct = float(pred.get("confidence", 0.0) * 100)
        
        # Reasons for this specific horizon
        h_reasons = [f"{h_name}_ret={ret_pct:+.2f}%", f"{h_name}_conf={conf_pct:.0f}%"]
        
        # Check confidence floor
        if conf_pct < floor_pct:
            continue

        h_label = "NEUTRAL"
        h_posture = "wait"
        
        # Scaled thresholds
        h_buy = buy_base * multiplier
        h_strong_buy = strong_buy_base * (multiplier * 0.8) # Strong signals need slightly less scaling to be catchable
        h_sell = sell_base * multiplier
        h_strong_sell = strong_sell_base * (multiplier * 0.8)

        if ret_pct >= h_strong_buy:
            h_label, h_posture = "STRONG BUY", f"buy_{posture_type}_aggressive"
        elif ret_pct >= h_buy:
            h_label, h_posture = "BUY", f"buy_{posture_type}_selective"
        elif ret_pct <= h_strong_sell:
            h_label, h_posture = "STRONG SELL", f"sell_{posture_type}_aggressive"
        elif ret_pct <= h_sell:
            h_label, h_posture = "SELL", f"sell_{posture_type}_selective"

        if h_label != "NEUTRAL":
            candidates.append({
                "label": h_label,
                "posture": h_posture,
                "score_pct": ret_pct,
                "confidence_pct": conf_pct,
                "horizon": h_name,
                "reasons": h_reasons
            })

    # Pick the "best" candidate (Strongest label first, then highest confidence)
    # Strength priority: STRONG BUY/SELL > BUY/SELL
    priority = {"STRONG BUY": 4, "STRONG SELL": 4, "BUY": 3, "SELL": 3, "NEUTRAL": 1}
    
    if candidates:
        # Sort by priority desc, then confidence desc
        candidates.sort(key=lambda x: (priority.get(x["label"], 0), x["confidence_pct"]), reverse=True)
        best_signal = candidates[0]
    else:
        # If no horizon triggered, use D1 as the neutral reference
        d1 = results.get("d1", {})
        ret_d1 = float(d1.get("prediction_return", 0.0) * 100)
        conf_d1 = float(d1.get("confidence", 0.0) * 100)
        best_signal["score_pct"] = ret_d1
        best_signal["confidence_pct"] = conf_d1
        best_signal["reasons"] = [f"D1={ret_d1:+.2f}%", f"conf={conf_d1:.0f}%", f"all horizons below threshold or confidence floor ({floor_pct:.0f}%)"]

    # Final Filters (Fundamentals/Sentiment) only for BUY signals
    if "BUY" in best_signal["label"]:
        fundamentals = dataset_meta.get("fundamentals", {}) or {}
        sent = float(dataset_meta.get("sentiment_value", 0.0) or 0.0)
        risk_pct = float(dataset_meta.get("latest_risk_pct", 0.0) or 0.0)
        pl = float(fundamentals.get("pl", 0.0) or 0.0)
        roe = float(fundamentals.get("roe", 0.0) or 0.0)
        
        filters = []
        if best_signal["label"] == "STRONG BUY":
            if sent < float(pcfg.get("min_sentiment_for_strong_buy", -0.1)):
                filters.append(f"sentiment {sent:+.2f} below strong-buy limit")
            
            if bool(pcfg.get("use_fundamentals_as_filter", True)):
                if pl > 0 and pl > float(pcfg.get("max_pl_for_strong_buy", 18.0)):
                    filters.append(f"P/L {pl:.2f} above strong-buy limit")
                if roe and roe < float(pcfg.get("min_roe_for_strong_buy", 0.08)):
                    filters.append(f"ROE {roe*100:.1f}% below strong-buy limit")
        
        if risk_pct and risk_pct > float(pcfg.get("max_risk_pct_for_buy", 4.0)):
            filters.append(f"risk {risk_pct:.2f}% above buy limit")
            
        if filters:
            # Downgrade or Neutralize
            if best_signal["label"] == "STRONG BUY":
                best_signal["label"] = "BUY"
                best_signal["posture"] = best_signal["posture"].replace("aggressive", "selective")
            else:
                # If it was already just BUY and failed risk, we could neutralize, but let's keep it selective BUY for now
                # unless risk is really high.
                pass
            best_signal["reasons"].extend(filters)

    return best_signal
