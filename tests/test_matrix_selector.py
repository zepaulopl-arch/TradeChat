# tests/test_matrix_selector.py

from app.matrix_selector import (
    MatrixSelectorConfig,
    select_policy,
)


def test_selects_best_policy():
    rows = [
        {
            "policy": "policy_a",
            "horizon": "d5",
            "profit_factor": 1.20,
            "win_rate": 0.51,
            "max_drawdown": 0.08,
            "trade_count": 30,
            "avg_return": 0.010,
        },
        {
            "policy": "policy_b",
            "horizon": "d5",
            "profit_factor": 1.45,
            "win_rate": 0.56,
            "max_drawdown": 0.07,
            "trade_count": 40,
            "avg_return": 0.018,
        },
    ]

    result = select_policy(rows)

    assert result["applicable"] is True
    assert result["policy"] == "policy_b"


def test_ignores_invalid_policy():
    rows = [
        {
            "policy": "bad_policy",
            "horizon": "d5",
            "profit_factor": 0.90,
            "win_rate": 0.42,
            "max_drawdown": 0.25,
            "trade_count": 8,
            "avg_return": -0.010,
        },
        {
            "policy": "good_policy",
            "horizon": "d5",
            "profit_factor": 1.30,
            "win_rate": 0.55,
            "max_drawdown": 0.06,
            "trade_count": 35,
            "avg_return": 0.014,
        },
    ]

    result = select_policy(rows)

    assert result["applicable"] is True
    assert result["policy"] == "good_policy"


def test_returns_not_applicable_when_all_invalid():
    rows = [
        {
            "policy": "bad_policy",
            "horizon": "d5",
            "profit_factor": 0.80,
            "win_rate": 0.30,
            "max_drawdown": 0.30,
            "trade_count": 5,
            "avg_return": -0.020,
        }
    ]

    result = select_policy(rows)

    assert result["applicable"] is False
    assert "reason" in result


def test_custom_thresholds():
    rows = [
        {
            "policy": "policy_a",
            "horizon": "d5",
            "profit_factor": 1.18,
            "win_rate": 0.50,
            "max_drawdown": 0.08,
            "trade_count": 25,
            "avg_return": 0.010,
        }
    ]

    cfg = MatrixSelectorConfig(
        min_profit_factor=1.10,
        min_trades=20,
        min_win_rate=0.49,
        max_drawdown=0.10,
    )

    result = select_policy(rows, cfg)

    assert result["applicable"] is True
    assert result["policy"] == "policy_a"
