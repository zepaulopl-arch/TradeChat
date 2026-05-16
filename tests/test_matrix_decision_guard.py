from app.runtime_policy import (
    apply_matrix_decision_guard,
)


def test_matrix_decision_guard_blocks_buy_when_observe():
    signal = {
        "ticker": "PETR4.SA",
        "policy": {
            "label": "BUY",
            "posture": "buy_day_selective",
            "actionable": True,
            "position_size": 62,
            "risk_reward_ratio": 0.30,
            "reasons": [],
        },
    }

    evidence = {
        "decision": "OBSERVE",
    }

    guard_cfg = {
        "enabled": True,
        "decisions": {
            "OBSERVE": {
                "max_signal": "NEUTRAL",
                "posture": "wait_matrix_observe",
                "reason": "signal blocked: Matrix decision is OBSERVE",
            }
        },
    }

    result = apply_matrix_decision_guard(
        signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    assert result["policy"]["label"] == "NEUTRAL"
    assert result["policy"]["posture"] == "wait_matrix_observe"
    assert result["policy"]["blocked_signal"] == "BUY"
    assert result["policy"]["actionable"] is False
    assert result["policy"]["position_size"] == 0
    assert result["matrix_decision_guard"]["blocked"] is True
    assert result["matrix_decision_guard"]["matrix_decision"] == "OBSERVE"
    assert result["matrix_decision_guard"]["blocked_signal"] == "BUY"


def test_matrix_decision_guard_marks_blocked_when_signal_already_neutral_but_has_blocked_signal():
    signal = {
        "ticker": "PETR4.SA",
        "policy": {
            "label": "NEUTRAL",
            "posture": "wait_rr_filter",
            "blocked_signal": "BUY",
            "actionable": False,
            "position_size": 0,
            "risk_reward_ratio": 0.0,
            "reasons": [
                "BUY blocked: R/R 0.03 < 0.20",
            ],
        },
    }

    evidence = {
        "decision": "OBSERVE",
    }

    guard_cfg = {
        "enabled": True,
        "decisions": {
            "OBSERVE": {
                "max_signal": "NEUTRAL",
                "posture": "wait_matrix_observe",
                "reason": "signal blocked: Matrix decision is OBSERVE",
            }
        },
    }

    result = apply_matrix_decision_guard(
        signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    assert result["policy"]["label"] == "NEUTRAL"
    assert result["policy"]["posture"] == "wait_rr_filter"
    assert result["policy"]["blocked_signal"] == "BUY"
    assert result["policy"]["actionable"] is False
    assert result["matrix_decision_guard"]["blocked"] is True
    assert result["matrix_decision_guard"]["current_label"] == "NEUTRAL"
    assert result["matrix_decision_guard"]["previous_blocked_signal"] == "BUY"
    assert result["matrix_decision_guard"]["blocked_signal"] == "BUY"
    assert "BUY blocked: R/R 0.03 < 0.20" in result["policy"]["reasons"]
    assert "signal blocked: Matrix decision is OBSERVE" in result["policy"]["reasons"]


def test_matrix_decision_guard_allows_approve():
    signal = {
        "ticker": "PETR4.SA",
        "policy": {
            "label": "BUY",
            "posture": "buy_day_selective",
            "actionable": True,
            "position_size": 62,
            "risk_reward_ratio": 0.30,
            "reasons": [],
        },
    }

    evidence = {
        "decision": "APPROVE",
    }

    guard_cfg = {
        "enabled": True,
        "decisions": {
            "APPROVE": {
                "max_signal": "ACTIONABLE",
            }
        },
    }

    result = apply_matrix_decision_guard(
        signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    assert result["policy"]["label"] == "BUY"
    assert result["policy"]["actionable"] is True
    assert result["matrix_decision_guard"]["blocked"] is False
    assert result["matrix_decision_guard"]["matrix_decision"] == "APPROVE"


def test_matrix_decision_guard_does_not_mutate_original_signal():
    signal = {
        "ticker": "PETR4.SA",
        "policy": {
            "label": "BUY",
            "posture": "buy_day_selective",
            "actionable": True,
            "position_size": 62,
            "risk_reward_ratio": 0.30,
            "reasons": [],
        },
    }

    evidence = {
        "decision": "REJECT",
    }

    guard_cfg = {
        "enabled": True,
        "decisions": {
            "REJECT": {
                "max_signal": "NEUTRAL",
            }
        },
    }

    result = apply_matrix_decision_guard(
        signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    assert result["policy"]["label"] == "NEUTRAL"
    assert signal["policy"]["label"] == "BUY"
    assert signal["policy"]["position_size"] == 62


def test_matrix_decision_guard_disabled_does_nothing():
    signal = {
        "ticker": "PETR4.SA",
        "policy": {
            "label": "BUY",
            "posture": "buy_day_selective",
            "actionable": True,
            "position_size": 62,
            "risk_reward_ratio": 0.30,
            "reasons": [],
        },
    }

    evidence = {
        "decision": "OBSERVE",
    }

    guard_cfg = {
        "enabled": False,
    }

    result = apply_matrix_decision_guard(
        signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    assert result["policy"]["label"] == "BUY"
    assert "matrix_decision_guard" not in result
