from app.evaluation_decision import make_validation_decision


def _comparison(beaten=4, total=5):
    names = [
        "zero_return_no_trade",
        "buy_and_hold_equal_weight",
        "mean_return_long_flat",
        "last_return_long_flat",
        "random_long_flat",
    ]
    rows = []
    for idx, name in enumerate(names[:total]):
        rows.append({"baseline": name, "beat_return": idx < beaten})
    return {
        "rows": rows,
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
        _comparison(beaten=4, total=5),
    )
    assert decision["final_decision"] == "promote"


def test_validation_decision_observe_when_low_trade_count():
    decision = make_validation_decision(
        {
            "total_return_pct": 8.0,
            "max_drawdown_pct": -6.0,
            "profit_factor": 1.4,
            "trade_count": 2,
            "active_exposure_pct": 45.0,
        },
        _comparison(beaten=4, total=5),
    )
    assert decision["final_decision"] in {"observe", "inconclusive"}
    assert decision["final_decision"] != "promote"


def test_validation_decision_reject_when_fails_baselines():
    decision = make_validation_decision(
        {
            "total_return_pct": -2.0,
            "max_drawdown_pct": -10.0,
            "profit_factor": 0.7,
            "trade_count": 10,
            "active_exposure_pct": 40.0,
        },
        _comparison(beaten=1, total=5),
    )
    assert decision["final_decision"] == "reject"


def test_validation_decision_inconclusive_when_missing_metrics():
    decision = make_validation_decision({}, {"rows": [], "beat_rate_pct": 0.0})
    assert decision["final_decision"] == "inconclusive"
