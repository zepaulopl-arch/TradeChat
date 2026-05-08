from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionCostConfig:
    fee_mode: str | None = None
    fee_amount: float = 0.0
    slippage_pct: float = 0.0


@dataclass(frozen=True)
class ValidationRunConfig:
    tickers: tuple[str, ...]
    mode: str = "replay"
    start_date: str | None = None
    end_date: str | None = None
    rebalance_days: int = 5
    warmup_bars: int = 150
    initial_cash: float | None = None
    max_positions: int | None = None
    allow_short: bool = False


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_indices: tuple[int, ...] = field(default_factory=tuple)
    test_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    orders: Any = None
    trades: Any = None
    stops: Any = None
