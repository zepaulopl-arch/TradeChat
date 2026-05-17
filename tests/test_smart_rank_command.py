import argparse

from app.commands import signal_command
from app.smart_rank_service import smart_rank_preflight_warnings, smart_rank_summary
from app.smart_report import print_smart_rank


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


def test_smart_rank_limit_limits_processing_and_display(
    monkeypatch,
    capsys,
):
    calls = {
        "resolved": [],
        "built": [],
    }

    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "PETR4.SA",
            "VALE3.SA",
            "ITUB4.SA",
            "BBDC4.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        calls["resolved"].append(ticker)

        return {
            "profile": "relaxed",
            "source": "policy_matrix",
            "promoted": True,
            "promotion_status": "promoted",
            "rejection_reasons": [],
            "evidence": {
                "ticker": ticker,
                "decision": "APPROVE",
                "profit_factor": 1.20,
                "trades": 20,
                "score": 90.0,
            },
        }

    def fake_build_smart_signal(
        cfg,
        args,
        ticker,
    ):
        calls["built"].append(ticker)

        return (
            {
                "ticker": ticker,
                "policy": {
                    "label": "NEUTRAL",
                    "risk_reward_ratio": 0.0,
                },
                "smart_signal": {
                    "profile": "relaxed",
                    "promotion_status": "promoted",
                    "rejection_reasons": [],
                    "evidence": {
                        "ticker": ticker,
                        "decision": "APPROVE",
                        "profit_factor": 1.20,
                        "trades": 20,
                        "score": 90.0,
                    },
                    "matrix_decision_guard": {},
                },
            },
            {},
            None,
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
        "_build_smart_signal",
        fake_build_smart_signal,
    )

    args = argparse.Namespace(
        tickers=[],
        asset_list="ibov",
        policy_profile=None,
        update=False,
        diagnostic=False,
        smart=True,
        rank_limit=2,
    )

    signal_command._smart_rank(
        {},
        args,
    )

    out = capsys.readouterr().out

    assert calls["resolved"] == [
        "PETR4.SA",
        "VALE3.SA",
    ]
    assert calls["built"] == [
        "PETR4.SA",
        "VALE3.SA",
    ]
    assert "Rows: 2 of 2" in out
    assert "ITUB4.SA" not in out
    assert "BBDC4.SA" not in out


def test_smart_rank_rejected_does_not_build_signal(
    monkeypatch,
    capsys,
):
    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "ALOS3.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        return {
            "profile": "active",
            "source": "policy_matrix",
            "promoted": False,
            "promotion_status": "rejected_by_constraints",
            "rejection_reasons": [
                "trades 0 < 15",
            ],
            "evidence": {
                "ticker": ticker,
                "decision": "REJECT",
                "profit_factor": 0.0,
                "trades": 0,
                "score": 16.7,
            },
        }

    def fake_build_smart_signal(
        cfg,
        args,
        ticker,
    ):
        raise AssertionError("rejected asset must not build a smart signal")

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
        "_build_smart_signal",
        fake_build_smart_signal,
    )

    args = argparse.Namespace(
        tickers=[],
        asset_list="ibov",
        policy_profile=None,
        update=False,
        diagnostic=False,
        smart=True,
        rank_limit=20,
    )

    signal_command._smart_rank(
        {},
        args,
    )

    out = capsys.readouterr().out

    assert "ALOS3.SA" in out
    assert "REJECTED" in out
    assert "BLOCK" in out
    assert "trades 0 < 15" in out
    assert "NO_MATRIX" not in out


def test_smart_rank_no_matrix_only_for_absent_runtime_assets(
    monkeypatch,
    capsys,
):
    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "ALOS3.SA",
            "MISSING.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        if ticker == "ALOS3.SA":
            return {
                "profile": "active",
                "source": "policy_matrix",
                "promoted": False,
                "promotion_status": "rejected_by_constraints",
                "rejection_reasons": [
                    "trades 0 < 15",
                ],
                "evidence": {
                    "ticker": ticker,
                    "decision": "REJECT",
                    "profit_factor": 0.0,
                    "trades": 0,
                    "score": 16.7,
                },
            }

        return {
            "profile": "balanced",
            "source": "fallback",
            "promoted": False,
            "promotion_status": "fallback",
            "rejection_reasons": [],
            "evidence": {},
        }

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

    args = argparse.Namespace(
        tickers=[],
        asset_list="ibov",
        policy_profile=None,
        update=False,
        diagnostic=False,
        smart=True,
        rank_limit=20,
    )

    signal_command._smart_rank(
        {},
        args,
    )

    out = capsys.readouterr().out

    assert "ALOS3.SA    REJECTED" in out
    assert "MISSING.SA  NO_MATRIX" in out
    assert "SKIP" in out


def test_smart_rank_table_body_strips_ansi_sequences(
    capsys,
):
    print_smart_rank(
        [
            {
                "ticker": "\x1b[31mPETR4.SA\x1b[0m",
                "signal": "\x1b[32mBUY\x1b[0m",
                "profile": "relaxed",
                "matrix": "APPROVE",
                "pf": "inf",
                "trades": 18,
                "rr": 0.3,
                "guard": "OK",
                "blocker": "\x1b[33mnone\x1b[0m",
            }
        ],
        limit=20,
    )

    out = capsys.readouterr().out

    assert "\x1b[" not in out
    assert "PETR4.SA" in out
    assert "BUY" in out
    assert "Rows: 1 of 1 | OK=1 | BLOCK=0 | REJECTED=0 | SKIP=0 | ERROR=0" in out


def test_smart_rank_prints_fingerprint_warnings_and_summary(
    capsys,
):
    rows = [
        {
            "ticker": "PETR4.SA",
            "signal": "BUY",
            "profile": "active",
            "matrix": "APPROVE",
            "pf": 2.10,
            "trades": 34,
            "rr": 1.45,
            "guard": "OK",
            "blocker": "none",
        },
        {
            "ticker": "ALOS3.SA",
            "signal": "REJECTED",
            "profile": "active",
            "matrix": "REJECT",
            "pf": 0.0,
            "trades": 0,
            "rr": 0.0,
            "guard": "BLOCK",
            "blocker": "trades 0 < 15",
        },
    ]

    print_smart_rank(
        rows,
        limit=20,
        fingerprint="Policy: config/runtime_policy.yaml | Universe: ibov | Assets: 2/79",
        warnings=[
            "runtime_policy has 0 rejected assets. Check promote_policy output.",
        ],
        summary=smart_rank_summary(rows),
    )

    out = capsys.readouterr().out

    assert "Policy: config/runtime_policy.yaml | Universe: ibov | Assets: 2/79" in out
    assert "WARNING: runtime_policy has 0 rejected assets" in out
    assert "Top actionable: PETR4.SA" in out
    assert "Main blocker: trades < minimum" in out


def test_smart_rank_preflight_warns_about_degenerate_runtime():
    warnings = smart_rank_preflight_warnings(
        runtime_data={
            "assets_total": 1,
            "assets_promoted": 1,
            "assets_rejected": 0,
            "assets": {
                "PETR4.SA": {
                    "promoted": True,
                }
            },
        },
        tickers=[
            "PETR4.SA",
        ],
    )

    assert "runtime_policy has 0 rejected assets. Check promote_policy output." in warnings
