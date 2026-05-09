from app.evaluation_decision import make_validation_decision


def _baseline_comparison(beaten: int = 4, total: int = 5) -> dict:
    names = [
        "zero_return_no_trade",
        "buy_and_hold_equal_weight",
        "mean_return_long_flat",
        "last_return_long_flat",
        "random_long_flat",
    ]
    return {
        "rows": [
            {"baseline": name, "beat_return": idx < beaten}
            for idx, name in enumerate(names[:total])
        ],
        "beat_rate_pct": beaten / total * 100.0,
        "beat_count": beaten,
        "baseline_count": total,
    }


def test_validation_decision_promote():
    decision = make_validation_decision(
        {
            "total_return_pct": 8.0,
            "max_drawdown_pct": -6.0,
            "profit_factor": 1.4,
            "trade_count": 12,
            "active_exposure_pct": 45.0,
            "hit_rate_pct": 55.0,
            "avg_return_pct": 0.3,
        },
        _baseline_comparison(beaten=4),
    )
    assert decision["final_decision"] == "promote"


def test_validation_decision_does_not_promote_low_trade_count():
    decision = make_validation_decision(
        {
            "total_return_pct": 8.0,
            "max_drawdown_pct": -6.0,
            "profit_factor": 1.4,
            "trade_count": 2,
            "active_exposure_pct": 45.0,
        },
        _baseline_comparison(beaten=4),
    )
    assert decision["final_decision"] in {"observe", "inconclusive"}


def test_validation_decision_rejects_clear_economic_failure():
    decision = make_validation_decision(
        {
            "total_return_pct": -2.0,
            "max_drawdown_pct": -10.0,
            "profit_factor": 0.7,
            "trade_count": 10,
            "active_exposure_pct": 40.0,
        },
        _baseline_comparison(beaten=1),
    )
    assert decision["final_decision"] == "reject"


def test_validation_decision_inconclusive_when_metrics_are_missing():
    decision = make_validation_decision({}, {"rows": [], "beat_rate_pct": 0.0})
    assert decision["final_decision"] == "inconclusive"
