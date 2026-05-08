from __future__ import annotations

from typing import Any


def _confidence_floor_pct(pcfg: dict[str, Any]) -> float:
    raw = float(pcfg.get("min_confidence_pct", 0.55))
    return raw * 100 if raw <= 1.0 else raw


def _as_pct(value: Any, default: float) -> float:
    raw = float(default if value is None else value)
    return raw * 100 if raw <= 1.0 else raw


def _ridge_mae_pct(prediction: dict[str, Any]) -> float:
    manifest = prediction.get("train_manifest", {}) or {}
    metrics = manifest.get("metrics", {}) or {}
    ridge = metrics.get("ridge_arbiter", {}) or {}
    return abs(float(ridge.get("mae_return", 0.0) or 0.0)) * 100.0


def classify_signal(
    cfg: dict[str, Any], results: dict[str, Any], dataset_meta: dict[str, Any]
) -> dict[str, Any]:
    """Classify the tactical signal using D1/D5/D20 predictions."""
    pcfg = cfg.get("policy", {}) or {}
    floor_pct = _confidence_floor_pct(pcfg)
    high_conf_pct = _as_pct(pcfg.get("high_confidence_pct"), 0.70)
    mae_threshold_multiplier = float(pcfg.get("mae_threshold_multiplier", 0.25))

    buy_base = float(pcfg.get("buy_return_pct", 0.45))
    strong_buy_base = float(pcfg.get("strong_buy_return_pct", 1.5))
    sell_base = float(pcfg.get("sell_return_pct", -0.45))
    strong_sell_base = float(pcfg.get("strong_sell_return_pct", -1.5))

    horizons_config = [
        ("d1", 1.0, "day"),
        ("d5", 2.2, "swing"),
        ("d20", 4.5, "position"),
    ]

    best_signal = {
        "label": "NEUTRAL",
        "posture": "wait",
        "score_pct": 0.0,
        "confidence_pct": 0.0,
        "quality_pct": 0.0,
        "horizon": "d1",
        "reasons": [],
    }

    candidates = []
    skipped_by_confidence = []
    for h_name, multiplier, posture_type in horizons_config:
        pred = results.get(h_name, {}) or {}
        if "error" in pred or not pred:
            continue

        ret_pct = float(pred.get("prediction_return", 0.0) * 100)
        conf_pct = float(pred.get("confidence", 0.0) * 100)
        h_reasons = [f"{h_name}_ret={ret_pct:+.2f}%", f"{h_name}_quality={conf_pct:.0f}%"]

        if conf_pct < floor_pct:
            skipped_by_confidence.append(
                f"{h_name} quality below floor ({conf_pct:.0f}% < {floor_pct:.0f}%)"
            )
            continue

        h_label = "NEUTRAL"
        h_posture = "wait"
        h_buy = buy_base * multiplier
        h_strong_buy = strong_buy_base * (multiplier * 0.8)
        h_sell = sell_base * multiplier
        h_strong_sell = strong_sell_base * (multiplier * 0.8)
        mae_floor = _ridge_mae_pct(pred) * mae_threshold_multiplier
        if mae_floor > 0:
            h_buy = max(h_buy, mae_floor)
            h_sell = min(h_sell, -mae_floor)
            h_reasons.append(f"edge_floor={mae_floor:.2f}%")

        if ret_pct >= h_strong_buy:
            h_label, h_posture = "STRONG BUY", f"buy_{posture_type}_aggressive"
        elif ret_pct >= h_buy:
            h_label, h_posture = "BUY", f"buy_{posture_type}_selective"
        elif ret_pct <= h_strong_sell:
            h_label, h_posture = "STRONG SELL", f"sell_{posture_type}_aggressive"
        elif ret_pct <= h_sell:
            h_label, h_posture = "SELL", f"sell_{posture_type}_selective"

        if h_label == "STRONG BUY" and conf_pct < high_conf_pct:
            h_label, h_posture = "BUY", f"buy_{posture_type}_selective"
            h_reasons.append(
                f"strong signal downgraded: quality {conf_pct:.0f}% < {high_conf_pct:.0f}%"
            )
        elif h_label == "STRONG SELL" and conf_pct < high_conf_pct:
            h_label, h_posture = "SELL", f"sell_{posture_type}_selective"
            h_reasons.append(
                f"strong signal downgraded: quality {conf_pct:.0f}% < {high_conf_pct:.0f}%"
            )

        if h_label != "NEUTRAL":
            candidates.append(
                {
                    "label": h_label,
                    "posture": h_posture,
                    "score_pct": ret_pct,
                    "confidence_pct": conf_pct,
                    "quality_pct": conf_pct,
                    "horizon": h_name,
                    "reasons": h_reasons,
                }
            )

    priority = {"STRONG BUY": 4, "STRONG SELL": 4, "BUY": 3, "SELL": 3, "NEUTRAL": 1}
    if candidates:
        candidates.sort(
            key=lambda x: (priority.get(x["label"], 0), x["confidence_pct"]), reverse=True
        )
        best_signal = candidates[0]
    else:
        d1 = results.get("d1", {}) or {}
        ret_d1 = float(d1.get("prediction_return", 0.0) * 100)
        conf_d1 = float(d1.get("confidence", 0.0) * 100)
        best_signal["score_pct"] = ret_d1
        best_signal["confidence_pct"] = conf_d1
        best_signal["quality_pct"] = conf_d1
        best_signal["reasons"] = [
            f"D1={ret_d1:+.2f}%",
            f"quality={conf_d1:.0f}%",
            f"all horizons below threshold or quality floor ({floor_pct:.0f}%)",
        ]
        best_signal["reasons"].extend(skipped_by_confidence[:3])

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
    elif best_signal["label"] == "STRONG SELL":
        max_sent = float(pcfg.get("max_sentiment_for_strong_sell", -0.1))
        if sent > max_sent:
            filters.append(f"sentiment {sent:+.2f} above strong-sell limit")

    if (
        "BUY" in best_signal["label"]
        and risk_pct
        and risk_pct > float(pcfg.get("max_risk_pct_for_buy", 4.0))
    ):
        filters.append(f"risk {risk_pct:.2f}% above buy limit")

    if filters:
        if best_signal["label"] == "STRONG BUY":
            best_signal["label"] = "BUY"
            best_signal["posture"] = best_signal["posture"].replace("aggressive", "selective")
        elif best_signal["label"] == "STRONG SELL":
            best_signal["label"] = "SELL"
            best_signal["posture"] = best_signal["posture"].replace("aggressive", "selective")
        best_signal["reasons"].extend(filters)

    latest_price = float(dataset_meta.get("latest_price", 0.0))
    risk_pct = float(dataset_meta.get("latest_risk_pct", 2.0))
    rm_cfg = pcfg.get("risk_management", {}) or {}
    agg_mult = float(rm_cfg.get("aggressive_multiplier", 1.2))
    sel_mult = float(rm_cfg.get("selective_multiplier", 1.8))

    risk_multiplier = agg_mult if "aggressive" in best_signal["posture"] else sel_mult
    stop_dist_pct = (risk_pct * risk_multiplier) / 100.0
    score_pct = best_signal["score_pct"] / 100.0

    if "BUY" in best_signal["label"]:
        best_signal["target_price"] = latest_price * (1 + score_pct)
        best_signal["stop_loss_price"] = latest_price * (1 - stop_dist_pct)
    elif "SELL" in best_signal["label"]:
        best_signal["target_price"] = latest_price * (1 + score_pct)
        best_signal["stop_loss_price"] = latest_price * (1 + stop_dist_pct)
    else:
        best_signal["target_price"] = latest_price
        best_signal["stop_loss_price"] = latest_price * (1 - stop_dist_pct)

    tcfg = cfg.get("trading", {}) or {}
    capital = float(tcfg.get("capital", 10000.0))
    max_loss = capital * (float(tcfg.get("risk_per_trade_pct", 1.0)) / 100.0)
    risk_per_share = abs(latest_price - best_signal["stop_loss_price"])
    if risk_per_share > 0 and best_signal["label"] != "NEUTRAL":
        best_signal["position_size"] = int(max_loss / risk_per_share)
    else:
        best_signal["position_size"] = 0

    move_pct = score_pct * 0.5
    if "BUY" in best_signal["label"]:
        best_signal["target_partial"] = latest_price * (1 + move_pct)
        best_signal["breakeven_trigger"] = best_signal["target_partial"]
    elif "SELL" in best_signal["label"]:
        best_signal["target_partial"] = latest_price * (1 + move_pct)
        best_signal["breakeven_trigger"] = best_signal["target_partial"]
    else:
        best_signal["target_partial"] = latest_price
        best_signal["breakeven_trigger"] = latest_price

    reward_amt = abs(best_signal["target_price"] - latest_price)
    rr = reward_amt / risk_per_share if risk_per_share > 0 else 0.0
    min_rr = float(rm_cfg.get("min_rr_threshold", pcfg.get("min_rr_threshold", 0.0)) or 0.0)
    if best_signal["label"] != "NEUTRAL" and min_rr > 0 and rr < min_rr:
        blocked = best_signal["label"]
        best_signal["label"] = "NEUTRAL"
        best_signal["posture"] = "wait_rr_filter"
        best_signal["blocked_signal"] = blocked
        best_signal["position_size"] = 0
        best_signal["target_price"] = latest_price
        best_signal["stop_loss_price"] = latest_price * (1 - stop_dist_pct)
        best_signal["target_partial"] = latest_price
        best_signal["breakeven_trigger"] = latest_price
        best_signal["reasons"].append(f"{blocked} blocked: R/R {rr:.2f} < {min_rr:.2f}")
        rr = 0.0

    allow_short = bool(
        pcfg.get(
            "allow_short",
            cfg.get("trading", {}).get(
                "allow_short", cfg.get("simulation", {}).get("allow_short", False)
            ),
        )
    )
    actionable = best_signal["label"] != "NEUTRAL"
    if "SELL" in best_signal["label"] and not allow_short:
        actionable = False
        best_signal["position_size"] = 0
        best_signal["reasons"].append("short disabled: SELL is exit/avoid only")
    best_signal["actionable"] = bool(actionable)
    best_signal["risk_reward_ratio"] = rr
    return best_signal
