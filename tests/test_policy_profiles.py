from app.cli import build_parser
from app.policy import (
    available_policy_profiles,
    apply_policy_profile,
    classify_signal,
    signal_policy_diagnostic,
)


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
    assert (
        parser.parse_args(
            ["signal", "rank", "PETR4.SA", "--policy-profile", "active"]
        ).policy_profile
        == "active"
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


def _active_results():
    return {
        "d1": {"prediction_return": 0.0004, "confidence": 0.37},
        "d5": {"prediction_return": 0.0011, "confidence": 0.36},
        "d20": {"prediction_return": 0.0106, "confidence": 0.71},
    }


def test_active_profile_exists_between_balanced_and_relaxed():
    cfg = _cfg()
    assert "active" in available_policy_profiles(cfg)

    balanced = apply_policy_profile(cfg, "balanced")
    active = apply_policy_profile(cfg, "active")
    relaxed = apply_policy_profile(cfg, "relaxed")

    assert (
        balanced["policy"]["risk_management"]["min_rr_threshold"]
        > active["policy"]["risk_management"]["min_rr_threshold"]
    )
    assert (
        active["policy"]["risk_management"]["min_rr_threshold"]
        > relaxed["policy"]["risk_management"]["min_rr_threshold"]
    )
    assert balanced["policy"]["min_confidence_pct"] > active["policy"]["min_confidence_pct"]
    assert active["policy"]["min_confidence_pct"] > relaxed["policy"]["min_confidence_pct"]


def test_active_profile_can_pass_when_balanced_rr_blocks():
    cfg = _cfg()
    meta = {
        "latest_price": 100.0,
        "latest_risk_pct": 2.0,
        "fundamentals": {},
        "sentiment_value": 0.0,
    }

    balanced = classify_signal(apply_policy_profile(cfg, "balanced"), _active_results(), meta)
    active = classify_signal(apply_policy_profile(cfg, "active"), _active_results(), meta)

    assert balanced["label"] == "NEUTRAL"
    assert "R/R" in " ".join(balanced.get("reasons", []))
    assert active["label"] == "BUY"
    assert active["horizon"] == "d20"
    assert active["profile"] == "active"


def test_asset_specific_active_preferred_horizon_wins_same_signal_strength():
    cfg = apply_policy_profile(_cfg(), "active")
    cfg["policy"]["buy_return_pct"] = 0.05
    cfg["policy"]["sell_return_pct"] = -0.05
    cfg["policy"]["min_confidence_pct"] = 0.30
    cfg["policy"]["preferred_horizon"] = "d5"
    cfg["policy"]["risk_management"]["min_rr_threshold"] = 0.0

    results = {
        "d1": {"prediction_return": 0.0002, "confidence": 0.50},
        "d5": {"prediction_return": 0.0015, "confidence": 0.55},
        "d20": {"prediction_return": 0.0060, "confidence": 0.90},
    }

    signal = classify_signal(cfg, results, _meta())

    assert signal["label"] == "BUY"
    assert signal["horizon"] == "d5"


def test_asset_specific_active_risk_budget_and_position_cap_are_applied():
    cfg = apply_policy_profile(_cfg(), "active")
    cfg["policy"]["buy_return_pct"] = 0.05
    cfg["policy"]["sell_return_pct"] = -0.05
    cfg["policy"]["min_confidence_pct"] = 0.30
    cfg["policy"]["risk_management"]["min_rr_threshold"] = 0.0
    cfg["policy"]["risk_management"]["risk_per_trade_pct"] = 0.25
    cfg["policy"]["risk_management"]["max_position_pct"] = 5.0

    results = {
        "d1": {"prediction_return": 0.0010, "confidence": 0.80},
    }
    meta = {
        "latest_price": 100.0,
        "latest_risk_pct": 2.0,
        "fundamentals": {},
        "sentiment_value": 0.0,
    }

    signal = classify_signal(cfg, results, meta)

    assert signal["label"] == "BUY"
    assert signal["position_size"] == 5
