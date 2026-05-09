from __future__ import annotations

import os
from typing import Any

from .models import train_models
from .pipeline_service import build_current_dataset, canonical_ticker


def safe_worker_count(total_items: int, requested: int | None = None, default: int = 1) -> int:
    total = max(1, int(total_items or 1))
    desired = default if requested is None or int(requested) <= 0 else int(requested)
    cpu_total = os.cpu_count() or 1
    cpu_cap = max(1, cpu_total - 1) if cpu_total > 2 else 1
    return max(1, min(total, desired, cpu_cap))


def train_one_asset(
    cfg: dict[str, Any],
    ticker: str,
    *,
    update: bool = False,
    autotune: bool = False,
    inner_threads: int | None = None,
) -> dict[str, Any]:
    ticker = canonical_ticker(cfg, ticker)
    targets = ["target_return_d1", "target_return_d5", "target_return_d20"]
    raw_X, all_y, meta = build_current_dataset(cfg, ticker, update=update)
    manifests: list[dict[str, Any]] = []

    for t_col in targets:
        horizon = t_col.split("_")[-1]
        y_series = all_y[t_col]
        h_meta = meta.copy()
        h_meta["horizon"] = horizon
        manifests.append(
            train_models(
                cfg,
                ticker,
                raw_X,
                y_series,
                h_meta,
                autotune=bool(autotune),
                horizon=horizon,
                inner_threads=inner_threads,
            )
        )

    return {
        "ticker": ticker,
        "ok": True,
        "rows": len(raw_X),
        "autotune": bool(autotune),
        "update": bool(update),
        "manifests": manifests,
    }
