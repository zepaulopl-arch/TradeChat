from app.cli import build_parser
from app import ranking_service, report
from app.policy import signal_policy_diagnostic, signal_policy_summary


def _cfg():
    return {
        "policy": {
            "buy_return_pct": 0.20,
            "sell_return_pct": -0.20,
            "min_confidence_pct": 0.45,
            "mae_threshold_multiplier": 0.25,
            "risk_management": {"min_rr_threshold": 1.5},
        }
    }


def _neutral_signal():
    return {
        "ticker": "PETR4.SA",
        "latest_date": "2026-05-09",
        "latest_price": 45.67,
        "train_run_id": "hidden",
        "horizons": {
            "d1": {"prediction_return": 0.0009, "confidence": 0.34},
            "d5": {"prediction_return": 0.0015, "confidence": 0.31},
            "d20": {"prediction_return": 0.0069, "confidence": 0.37},
        },
        "policy": {
            "label": "NEUTRAL",
            "posture": "wait",
            "horizon": "d1",
            "quality_pct": 34.0,
            "confidence_pct": 34.0,
            "reasons": ["all horizons below threshold or quality floor (45%)"],
        },
        "features_used": [],
        "sentiment_value": 0.0,
        "fundamentals": {},
    }


def test_cli_accepts_signal_diagnostic_flags():
    parser = build_parser()

    assert parser.parse_args(["signal", "generate", "PETR4.SA", "--diagnostic"]).diagnostic
    assert parser.parse_args(["signal", "rank", "PETR4.SA", "--diagnostic"]).diagnostic


def test_policy_diagnostic_explains_neutral_blockers():
    diag = signal_policy_diagnostic(_cfg(), _neutral_signal())

    assert diag["final_label"] == "NEUTRAL"
    assert diag["main_blocker"] == "quality floor"
    assert {row["horizon"] for row in diag["rows"]} == {"d1", "d5", "d20"}
    assert all(not row["quality_ok"] for row in diag["rows"])
    assert "quality" in diag["rows"][0]["blocker"]


def test_print_signal_diagnostic_renders_policy_table(capsys):
    report.print_signal(_neutral_signal(), cfg=_cfg(), diagnostic=True)
    out = capsys.readouterr().out

    assert "POLICY DIAGNOSTIC" in out
    assert "Main blocker" in out
    assert "BUY >=" in out
    assert "quality" in out
    assert "Run Id" not in out


def test_rank_diagnostic_renders_blocker_summary(monkeypatch):
    fake_signals = [_neutral_signal()]
    monkeypatch.setattr(ranking_service, "iter_latest_signals", lambda cfg: fake_signals)
    monkeypatch.setattr(ranking_service, "screen_width", lambda: 120)

    lines = ranking_service.render_ranking(_cfg(), tickers=["PETR4.SA"], diagnostic=True)
    text = "\n".join(lines)

    assert "BLOCKER" in text
    assert "quality floor" in text


def test_signal_policy_summary_reports_actionable_label():
    sig = _neutral_signal()
    sig["policy"] = {"label": "BUY", "horizon": "d5", "posture": "buy_swing_selective"}

    assert signal_policy_summary(_cfg(), sig) == "BUY via D5"
