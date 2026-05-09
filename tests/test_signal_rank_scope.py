from app import ranking_service


def test_collect_ranked_signals_respects_explicit_ticker_scope(monkeypatch):
    fake_signals = [
        {"ticker": "PETR4.SA", "policy": {"label": "NEUTRAL", "horizon": "d1"}, "horizons": {}},
        {"ticker": "VALE3.SA", "policy": {"label": "BUY", "horizon": "d5"}, "horizons": {}},
        {"ticker": "PRIO3.SA", "policy": {"label": "BUY", "horizon": "d20"}, "horizons": {}},
    ]
    monkeypatch.setattr(ranking_service, "iter_latest_signals", lambda cfg: fake_signals)
    monkeypatch.setattr(ranking_service, "signal_score", lambda data: 1.0)
    monkeypatch.setattr(ranking_service, "signal_priority", lambda data: 1)
    monkeypatch.setattr(ranking_service, "trigger_horizon", lambda data: data["policy"]["horizon"])
    monkeypatch.setattr(ranking_service, "trigger_result", lambda data: {"prediction_return": 0.0})

    rows = ranking_service.collect_ranked_signals({}, tickers=["PETR4.SA", "VALE3.SA"], limit=40)

    assert {row["ticker"] for row in rows} == {"PETR4.SA", "VALE3.SA"}
    assert "PRIO3.SA" not in {row["ticker"] for row in rows}


def test_render_ranking_empty_scoped_message(monkeypatch):
    monkeypatch.setattr(ranking_service, "iter_latest_signals", lambda cfg: [])

    assert ranking_service.render_ranking({}, tickers=["PETR4.SA"]) == [
        "No signals found for requested tickers. Run signal generate first."
    ]
