from __future__ import annotations

from typing import Any

import pandas as pd


def metrics_frame_to_dict(metrics_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if metrics_df is None or metrics_df.empty:
        return out
    for row in metrics_df.to_dict(orient="records"):
        out[str(row.get("name"))] = row.get("value")
    return out
