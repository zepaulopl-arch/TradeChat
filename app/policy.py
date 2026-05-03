from __future__ import annotations

from typing import Any


def _confidence_floor_pct(pcfg: dict[str, Any]) -> float:
    raw = float(pcfg.get("min_confidence_pct", 0.55))
    # Accept both 0.55 and 55.0 in YAML; normalize to percent.
    return raw * 100 if raw <= 1.0 else raw


def classify_signal(cfg: dict[str, Any], prediction: dict[str, Any], dataset_meta: dict[str, Any]) -> dict[str, Any]:
    pcfg = cfg.get("policy", {})
    ret_pct = float(prediction["prediction_return"] * 100)
    confidence_pct = float(prediction.get("confidence", 0.0) * 100)
    fundamentals = dataset_meta.get("fundamentals", {}) or {}
    sent = 0.0
    try:
        # sentiment is stored as feature value only when current dataset uses it; meta tells source, not value.
        sent = float(dataset_meta.get("sentiment_value", 0.0))
    except Exception:
        sent = 0.0

    risk_pct = float(dataset_meta.get("latest_risk_pct", 0.0) or 0.0)
    pl = float(fundamentals.get("pl", 0.0) or 0.0)
    roe = float(fundamentals.get("roe", 0.0) or 0.0)

    reasons = [f"expected_return={ret_pct:+.2f}%", f"confidence={confidence_pct:.0f}%"]
    floor_pct = _confidence_floor_pct(pcfg)
    if confidence_pct < floor_pct:
        return {"label": "NEUTRAL", "posture": "wait", "score_pct": ret_pct, "confidence_pct": confidence_pct, "reasons": reasons + [f"confidence below floor ({floor_pct:.0f}%)"]}

    label = "NEUTRAL"
    posture = "wait"
    if ret_pct >= float(pcfg.get("strong_buy_return_pct", 1.5)):
        label, posture = "STRONG BUY", "buy_aggressive"
    elif ret_pct >= float(pcfg.get("buy_return_pct", 0.45)):
        label, posture = "BUY", "buy_selective"
    elif ret_pct <= float(pcfg.get("strong_sell_return_pct", -1.5)):
        label, posture = "STRONG SELL", "sell_aggressive"
    elif ret_pct <= float(pcfg.get("sell_return_pct", -0.45)):
        label, posture = "SELL", "sell_selective"

    if label == "STRONG BUY" and sent < float(pcfg.get("min_sentiment_for_strong_buy", -999.0)):
        label, posture = "BUY", "buy_selective"
        reasons.append(f"sentiment {sent:+.2f} below strong-buy limit")

    if label == "STRONG SELL" and sent > float(pcfg.get("max_sentiment_for_strong_sell", 999.0)):
        label, posture = "SELL", "sell_selective"
        reasons.append(f"sentiment {sent:+.2f} above strong-sell limit")

    if label == "STRONG BUY" and bool(pcfg.get("use_fundamentals_as_filter", True)):
        filters = []
        if pl > 0 and pl > float(pcfg.get("max_pl_for_strong_buy", 18.0)):
            filters.append(f"P/L {pl:.2f} above strong-buy limit")
        if roe and roe < float(pcfg.get("min_roe_for_strong_buy", 0.08)):
            filters.append(f"ROE {roe*100:.1f}% below strong-buy limit")
        if risk_pct and risk_pct > float(pcfg.get("max_risk_pct_for_buy", 4.0)):
            filters.append(f"risk {risk_pct:.2f}% above buy limit")
        if filters:
            label, posture = "BUY", "buy_selective"
            reasons.extend(filters)

    return {"label": label, "posture": posture, "score_pct": ret_pct, "confidence_pct": confidence_pct, "reasons": reasons}
