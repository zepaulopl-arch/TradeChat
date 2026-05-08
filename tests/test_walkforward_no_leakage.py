import pandas as pd

from app.simulation.walkforward import build_walkforward_windows


def test_walkforward_windows_are_temporal():
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    windows = build_walkforward_windows(
        dates,
        min_train_bars=20,
        test_bars=5,
        step_bars=5,
        embargo_bars=2,
    )

    assert windows
    for window in windows:
        assert window.train_end < window.test_start
        assert max(window.train_indices) + 2 < min(window.test_indices)
        assert all(dates[idx] <= window.train_end for idx in window.train_indices)
        assert all(dates[idx] >= window.test_start for idx in window.test_indices)
