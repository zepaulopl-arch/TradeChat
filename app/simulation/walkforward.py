from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .types import WalkForwardWindow


def build_walkforward_windows(
    dates: Iterable[object],
    *,
    min_train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    embargo_bars: int = 0,
) -> list[WalkForwardWindow]:
    """Create strictly temporal walk-forward windows.

    The train window always ends before the test window starts. If embargo_bars
    is positive, that many bars are skipped between train_end and test_start.
    """
    index = pd.Index(pd.to_datetime(list(dates))).sort_values().drop_duplicates()
    if len(index) == 0:
        return []
    min_train = max(1, int(min_train_bars))
    test_size = max(1, int(test_bars))
    step = max(1, int(step_bars or test_size))
    embargo = max(0, int(embargo_bars))
    windows: list[WalkForwardWindow] = []

    test_start_idx = min_train + embargo
    while test_start_idx < len(index):
        train_end_idx = test_start_idx - embargo - 1
        if train_end_idx < 0:
            break
        test_end_idx = min(len(index) - 1, test_start_idx + test_size - 1)
        train_indices = tuple(range(0, train_end_idx + 1))
        test_indices = tuple(range(test_start_idx, test_end_idx + 1))
        if train_indices and test_indices:
            windows.append(
                WalkForwardWindow(
                    train_start=index[train_indices[0]],
                    train_end=index[train_indices[-1]],
                    test_start=index[test_indices[0]],
                    test_end=index[test_indices[-1]],
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )
        test_start_idx += step
    return windows
