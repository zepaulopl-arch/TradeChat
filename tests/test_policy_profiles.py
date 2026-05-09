from app.cli import build_parser
from app.policy import apply_policy_profile, classify_signal, signal_policy_diagnostic


def _cfg():
    return {
        "policy": {
            "buy_return_pct": 0.20,
            "sell_return_pct": -0.20,
            "min_confidence_pct": 0.45,
            "mae_threshold_multiplier": 0.25,
            "risk_management": {"min_rr_threshold": 1.5},
            "profiles": {
                "relaxed": {
                    "buy_return_pct": 0.08,
                    "sell_return_pct": -0.08,
                    "min_confidence_pct": 0.35,
                    "mae_threshold_multiplier": 0.08,
                    "risk_management": {"min_rr_threshold": 0.0},
                }
            },
        },
        "trading": {"capital": 10000, "risk_per_trade_pct": 1.0},
    }


def _results():
    return {
        "d1": {"prediction_return": 0.0009, "confidence": 0.34},
        "d5": {"prediction_return": 0.0015, "confidence": 0.31},
        "d20": {"prediction_return": 0.0069, "confidence": 0.37},
    }


def _meta():
    return {
        "latest_price": 45.67,
        "latest_risk_pct": 2.0,
        "fundamentals": {},
        "sentiment_value": 0.0,
    }


def test_cli_accepts_policy_profile_flags():
    parser = build_parser()

    assert (
        parser.parse_args(
            ["signal", "generate", "PETR4.SA", "--policy-profile", "relaxed"]
        ).policy_profile
        == "relaxed"
    )
    assert (
        parser.parse_args(
            ["signal", "rank", "PETR4.SA", "--policy-profile", "balanced"]
        ).policy_profile
        == "balanced"
    )


def test_policy_profile_relaxes_thresholds_without_mutating_base_config():
    cfg = _cfg()
    relaxed = apply_policy_profile(cfg, "relaxed")

    assert cfg["policy"]["min_confidence_pct"] == 0.45
    assert relaxed["policy"]["_active_profile"] == "relaxed"
    assert relaxed["policy"]["min_confidence_pct"] == 0.35
    assert relaxed["policy"]["buy_return_pct"] == 0.08
    assert relaxed["policy"]["risk_management"]["min_rr_threshold"] == 0.0


def test_relaxed_profile_can_create_actionable_signal_when_strict_blocks():
    strict = classify_signal(_cfg(), _results(), _meta())
    relaxed = classify_signal(apply_policy_profile(_cfg(), "relaxed"), _results(), _meta())

    assert strict["label"] == "NEUTRAL"
    assert relaxed["label"] == "BUY"
    assert relaxed["horizon"] == "d20"
    assert relaxed["profile"] == "relaxed"


def test_diagnostic_reports_active_policy_profile():
    signal = {
        "ticker": "PETR4.SA",
        "horizons": _results(),
        "policy": classify_signal(apply_policy_profile(_cfg(), "relaxed"), _results(), _meta()),
    }
    diag = signal_policy_diagnostic(apply_policy_profile(_cfg(), "relaxed"), signal)

    assert diag["profile"] == "relaxed"
    assert diag["final_label"] == "BUY"
