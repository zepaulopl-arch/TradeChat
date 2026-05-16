import argparse

from app.commands import signal_command


def test_smart_rank_generates_compact_table(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "PETR4.SA",
            "VALE3.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        if ticker == "PETR4.SA":
            return {
                "profile": "relaxed",
                "source": "policy_matrix",
                "overrides": {},
                "evidence": {
                    "ticker": ticker,
                    "decision": "OBSERVE",
                    "profit_factor": 1.16,
                    "trades": 18,
                    "score": 83.3,
                },
                "selection": {},
            }

        return {
            "profile": "active",
            "source": "policy_matrix",
            "overrides": {},
            "evidence": {
                "ticker": ticker,
                "decision": "APPROVE",
                "profit_factor": 1.80,
                "trades": 31,
                "score": 92.0,
            },
            "selection": {},
        }

    def fake_apply_policy_profile(
        cfg,
        profile,
    ):
        new_cfg = dict(cfg)

        new_cfg["policy"] = {
            "_active_profile": profile,
        }

        return new_cfg

    def fake_runtime_policy_overrides_for_profile(
        profile,
    ):
        return {}

    def fake_runtime_decision_guard_config():
        return {
            "enabled": True,
            "decisions": {
                "OBSERVE": {
                    "max_signal": "NEUTRAL",
                    "reason": "signal blocked: Matrix decision is OBSERVE",
                },
                "APPROVE": {
                    "max_signal": "ACTIONABLE",
                },
            },
        }

    def fake_make_signal(
        cfg,
        ticker,
        update=False,
    ):
        if ticker == "PETR4.SA":
            return {
                "ticker": ticker,
                "policy": {
                    "label": "BUY",
                    "posture": "buy_day_selective",
                    "risk_reward_ratio": 0.30,
                    "position_size": 10,
                    "actionable": True,
                    "reasons": [],
                },
            }

        return {
            "ticker": ticker,
            "policy": {
                "label": "BUY",
                "posture": "buy_swing_selective",
                "risk_reward_ratio": 0.75,
                "position_size": 20,
                "actionable": True,
                "reasons": [],
            },
        }

    def fake_latest_signal_path(
        cfg,
        ticker,
    ):
        return (
            tmp_path
            / ticker.replace(
                ".",
                "_",
            )
            / "latest_signal.json"
        )

    monkeypatch.setattr(
        signal_command,
        "resolve_cli_tickers",
        fake_resolve_cli_tickers,
    )

    monkeypatch.setattr(
        signal_command,
        "resolve_policy_selection",
        fake_resolve_policy_selection,
    )

    monkeypatch.setattr(
        signal_command,
        "apply_policy_profile",
        fake_apply_policy_profile,
    )

    monkeypatch.setattr(
        signal_command,
        "runtime_policy_overrides_for_profile",
        fake_runtime_policy_overrides_for_profile,
    )

    monkeypatch.setattr(
        signal_command,
        "runtime_decision_guard_config",
        fake_runtime_decision_guard_config,
    )

    monkeypatch.setattr(
        signal_command,
        "make_signal",
        fake_make_signal,
    )

    monkeypatch.setattr(
        signal_command,
        "latest_signal_path",
        fake_latest_signal_path,
    )

    args = argparse.Namespace(
        tickers=[],
        asset_list="validacao",
        policy_profile=None,
        update=False,
        diagnostic=False,
        smart=True,
        rank_limit=40,
    )

    signal_command._smart_rank(
        {},
        args,
    )

    out = capsys.readouterr().out

    assert "SMART RANK" in out
    assert "PETR4.SA" in out
    assert "VALE3.SA" in out
    assert "OBSERVE" in out
    assert "APPROVE" in out
    assert "BLOCK" in out
    assert "OK" in out
    assert "signal blocked: Matrix decision is OBSERVE" in out


def test_rank_dispatches_to_smart_rank_when_flag_is_set(
    monkeypatch,
):
    calls = {}

    def fake_smart_rank(
        cfg,
        args,
    ):
        calls["called"] = True

    monkeypatch.setattr(
        signal_command,
        "_smart_rank",
        fake_smart_rank,
    )

    args = argparse.Namespace(
        smart=True,
    )

    signal_command._rank(
        {},
        args,
    )

    assert calls["called"] is True
