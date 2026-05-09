from app import report


def _sample_signal():
    return {
        "ticker": "PETR4.SA",
        "latest_date": "2026-05-09",
        "latest_price": 45.67,
        "train_run_id": "train_hidden_id",
        "horizons": {
            "d1": {"prediction_return": 0.001, "target_price": 45.71, "confidence": 0.34},
        },
        "policy": {
            "label": "NEUTRAL",
            "horizon": "d1",
            "quality_pct": 34.0,
            "reasons": [],
        },
        "features_used": [],
        "sentiment": {"score": 0.0},
        "fundamentals": {},
    }


def test_print_signal_hides_run_id_by_default(capsys):
    report.print_signal(_sample_signal())

    assert "Run Id" not in capsys.readouterr().out


def test_print_signal_shows_run_id_when_verbose(capsys):
    report.print_signal(_sample_signal(), verbose=True)

    assert "Run Id" in capsys.readouterr().out
