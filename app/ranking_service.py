from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .policy import active_policy_profile, signal_policy_summary
from .portfolio_service import iter_latest_signals
from .presentation import (
    C,
    banner,
    divider,
    paint,
    render_facts,
    render_table,
    screen_width,
    tone_signal,
)
from .scoring import signal_priority, signal_score, trigger_horizon, trigger_result
from .utils import normalize_ticker


def collect_ranked_signals(
    cfg: dict[str, Any],
    *,
    limit: int = 40,
    tickers: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = {normalize_ticker(ticker) for ticker in (tickers or [])}
    signals: list[dict[str, Any]] = []
    for data in iter_latest_signals(cfg):
        ticker = normalize_ticker(str(data.get("ticker", "")))
        if allowed and ticker not in allowed:
            continue
        try:
            horizons = data.get("horizons", {}) or {}
            policy = data.get("policy", {"label": "NEUTRAL", "horizon": "d1"}) or {}
            trigger_pred = trigger_result(data)
            quality_pct = float(policy.get("quality_pct", policy.get("confidence_pct", 0.0)) or 0.0)
            triggered_ret = float(trigger_pred.get("prediction_return", 0.0) or 0.0) * 100.0
            signals.append(
                {
                    "ticker": ticker or data.get("ticker", "N/A"),
                    "signal": str(policy.get("label", "NEUTRAL")),
                    "horizon": trigger_horizon(data).upper(),
                    "trigger_ret": triggered_ret,
                    "d1_ret": float(horizons.get("d1", {}).get("prediction_return", 0.0) or 0.0)
                    * 100.0,
                    "d5_ret": float(horizons.get("d5", {}).get("prediction_return", 0.0) or 0.0)
                    * 100.0,
                    "d20_ret": float(horizons.get("d20", {}).get("prediction_return", 0.0) or 0.0)
                    * 100.0,
                    "quality_pct": quality_pct,
                    "score": signal_score(data),
                    "priority": signal_priority(data),
                    "rr": float(policy.get("risk_reward_ratio", 0.0) or 0.0),
                    "blocker": signal_policy_summary(cfg, data),
                }
            )
        except Exception:
            continue

    signals.sort(key=lambda row: (row["priority"], row["score"]), reverse=True)
    return signals[:limit] if limit > 0 else signals


def render_ranking(
    cfg: dict[str, Any],
    *,
    limit: int = 40,
    tickers: list[str] | tuple[str, ...] | set[str] | None = None,
    diagnostic: bool = False,
) -> list[str]:
    rows_data = collect_ranked_signals(cfg, limit=limit, tickers=tickers)
    if not rows_data:
        if tickers:
            return ["No signals found for requested tickers. Run signal generate first."]
        return ["No signals found. Run signal generate first."]

    df = pd.DataFrame(rows_data)
    width = screen_width()
    compact = width < 104 and not diagnostic
    buy_count = int(df["signal"].astype(str).str.contains("BUY").sum())
    sell_count = int(df["signal"].astype(str).str.contains("SELL").sum())
    lines: list[str] = []

    lines.append("")
    lines.extend(
        banner(
            "TRADECHAT RANKING",
            paint(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), C.BLUE),
            width=width,
        )
    )
    lines.extend(
        render_facts(
            [
                ("Signals", len(df)),
                ("Shown", len(rows_data)),
                ("Scope", "explicit" if tickers else "all latest"),
                ("Policy", active_policy_profile(cfg)),
                ("Buy / Sell", f"{buy_count} / {sell_count}"),
                ("Sort", "priority -> score"),
            ],
            width=width,
            max_columns=2,
        )
    )

    if compact:
        headers = ["TICKER", "SIGNAL", "H", "RETURN", "QUAL", "SCORE"]
        rows = [
            [
                paint(row["ticker"], C.BOLD),
                paint(row["signal"], tone_signal(row["signal"])),
                paint(row["horizon"], C.CYAN),
                f"{row['trigger_ret']:+.2f}%",
                f"{row['quality_pct']:.0f}%",
                f"{row['score']:.1f}",
            ]
            for row in rows_data
        ]
        aligns = ["left", "left", "left", "right", "right", "right"]
        min_widths = [8, 8, 3, 8, 6, 7]
    elif diagnostic:
        headers = ["TICKER", "SIGNAL", "H", "D1 %", "D5 %", "D20 %", "QUAL", "BLOCKER"]
        rows = [
            [
                paint(row["ticker"], C.BOLD),
                paint(row["signal"], tone_signal(row["signal"])),
                paint(row["horizon"], C.CYAN),
                f"{row['d1_ret']:+.2f}%",
                f"{row['d5_ret']:+.2f}%",
                f"{row['d20_ret']:+.2f}%",
                f"{row['quality_pct']:.0f}%",
                str(row.get("blocker", "n/a")),
            ]
            for row in rows_data
        ]
        aligns = ["left", "left", "left", "right", "right", "right", "right", "left"]
        min_widths = [8, 8, 3, 7, 7, 8, 6, 18]
    else:
        headers = ["TICKER", "SIGNAL", "H", "D1 %", "D5 %", "D20 %", "QUAL", "R/R", "SCORE"]
        rows = [
            [
                paint(row["ticker"], C.BOLD),
                paint(row["signal"], tone_signal(row["signal"])),
                paint(row["horizon"], C.CYAN),
                f"{row['d1_ret']:+.2f}%",
                f"{row['d5_ret']:+.2f}%",
                f"{row['d20_ret']:+.2f}%",
                f"{row['quality_pct']:.0f}%",
                f"{row['rr']:.1f}",
                f"{row['score']:.1f}",
            ]
            for row in rows_data
        ]
        aligns = ["left", "left", "left", "right", "right", "right", "right", "right", "right"]
        min_widths = [8, 8, 3, 7, 7, 8, 6, 5, 7]

    lines.extend(render_table(headers, rows, width=width, aligns=aligns, min_widths=min_widths))
    lines.append(f"{paint('Score', C.DIM)} = signal quality x |trigger return| / sqrt(days)")
    if diagnostic:
        lines.append("Diagnostic = main policy blocker or selected actionable signal.")
    lines.append(divider(width))
    return lines
