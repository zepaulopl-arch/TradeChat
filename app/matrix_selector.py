# app/matrix_selector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MatrixSelectorConfig:
    min_profit_factor: float = 1.15
    min_trades: int = 20
    min_win_rate: float = 0.48
    max_drawdown: float = 0.12


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_valid(
    row: dict[str, Any],
    cfg: MatrixSelectorConfig,
) -> bool:
    profit_factor = _safe_float(row.get("profit_factor"))
    win_rate = _safe_float(row.get("win_rate"))
    max_drawdown = _safe_float(row.get("max_drawdown"))
    trade_count = _safe_int(row.get("trade_count"))

    if profit_factor < cfg.min_profit_factor:
        return False

    if win_rate < cfg.min_win_rate:
        return False

    if max_drawdown > cfg.max_drawdown:
        return False

    if trade_count < cfg.min_trades:
        return False

    return True


def _compute_score(row: dict[str, Any]) -> float:
    profit_factor = _safe_float(row.get("profit_factor"))
    win_rate = _safe_float(row.get("win_rate"))
    avg_return = _safe_float(row.get("avg_return"))
    max_drawdown = _safe_float(row.get("max_drawdown"))
    trade_count = _safe_int(row.get("trade_count"))

    trade_score = min(trade_count / 100.0, 1.0)

    score = (
        (profit_factor * 0.40)
        + (win_rate * 0.25)
        + (avg_return * 0.20)
        - (max_drawdown * 0.10)
        + (trade_score * 0.05)
    )

    return round(score, 6)


def select_policy(
    rows: list[dict[str, Any]],
    cfg: MatrixSelectorConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or MatrixSelectorConfig()

    valid_rows: list[dict[str, Any]] = []

    for row in rows:
        if not _is_valid(row, cfg):
            continue

        row = dict(row)
        row["_score"] = _compute_score(row)

        valid_rows.append(row)

    if not valid_rows:
        return {
            "applicable": False,
            "reason": "Nenhuma policy válida.",
        }

    best = sorted(
        valid_rows,
        key=lambda x: x["_score"],
        reverse=True,
    )[0]

    return {
        "applicable": True,
        "policy": best.get("policy"),
        "horizon": best.get("horizon"),
        "score": best["_score"],
        "evidence": {
            "profit_factor": best.get("profit_factor"),
            "win_rate": best.get("win_rate"),
            "max_drawdown": best.get("max_drawdown"),
            "trade_count": best.get("trade_count"),
            "avg_return": best.get("avg_return"),
        },
    }
