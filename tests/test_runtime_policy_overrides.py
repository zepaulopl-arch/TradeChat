import argparse
import json

from app import runtime_policy
from app.commands import signal_command


def test_apply_runtime_policy_overrides_deep_merges_policy():
    cfg = {
        "policy": {
            "buy_return_pct": 0.12,
            "min_confidence_pct": 0.38,
            "risk_management": {
                "min_rr_threshold": 0.30,
                "aggressive_multiplier": 1.2,
            },
        }
    }

    overrides = {
        "buy_return_pct": 0.06,
        "risk_management": {
            "min_rr_threshold": 0.0,
        },
    }

    result = runtime_policy.apply_runtime_policy_overrides(
        cfg,
        overrides,
    )

    assert result["policy"]["buy_return_pct"] == 0.06
    assert result["policy"]["min_confidence_pct"] == 0.38
    assert result["policy"]["risk_management"]["min_rr_threshold"] == 0.0
    assert result["policy"]["risk_management"]["aggressive_multiplier"] == 1.2

    assert cfg["policy"]["buy_return_pct"] == 0.12
    assert cfg["policy"]["risk_management"]["min_rr_threshold"] == 0.30


def test_resolve_policy_selection_returns_overrides(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "PETR4.SA": {
                        "profile": "active",
                        "policy_type": "asset_specific_active",
                        "source": "policy_matrix",
                        "overrides": {
                            "buy_return_pct": 0.06,
                            "min_confidence_pct": 0.32,
                        },
                        "evidence": {
                            "profit_factor": 1.16,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_policy,
        "POLICY_PATH",
        policy_path,
    )

    selection = runtime_policy.resolve_policy_selection(
        "PETR4.SA",
    )

    assert selection["profile"] == "active"
    assert selection["policy_type"] == "asset_specific_active"
    assert selection["source"] == "policy_matrix"
    assert selection["overrides"]["buy_return_pct"] == 0.06
    assert selection["overrides"]["min_confidence_pct"] == 0.32
    assert selection["evidence"]["profit_factor"] == 1.16


def test_merge_runtime_overrides_prefers_asset_specific_runtime():
    stored = {
        "buy_return_pct": 0.06,
        "min_confidence_pct": 0.32,
        "risk_management": {
            "min_rr_threshold": 0.0,
            "aggressive_multiplier": 1.2,
        },
    }

    live = {
        "risk_management": {
            "min_rr_threshold": 0.20,
        },
    }

    result = runtime_policy.merge_runtime_overrides(
        stored,
        live,
    )

    assert result["buy_return_pct"] == 0.06
    assert result["min_confidence_pct"] == 0.32
    assert result["risk_management"]["min_rr_threshold"] == 0.0
    assert result["risk_management"]["aggressive_multiplier"] == 1.2


def test_runtime_policy_overrides_for_profile_reads_live_yaml(
    monkeypatch,
):
    def fake_load_runtime_policy_config():
        return {
            "promotion": {
                "runtime_overrides": {
                    "enabled": True,
                    "profiles": {
                        "active": {
                            "buy_return_pct": 0.06,
                            "risk_management": {
                                "min_rr_threshold": 0.20,
                            },
                        }
                    },
                }
            }
        }

    monkeypatch.setattr(
        runtime_policy,
        "load_runtime_policy_config",
        fake_load_runtime_policy_config,
    )

    result = runtime_policy.runtime_policy_overrides_for_profile(
        "active",
    )

    assert result["buy_return_pct"] == 0.06
    assert result["risk_management"]["min_rr_threshold"] == 0.20


def test_smart_signal_prefers_asset_specific_runtime_over_live_yaml(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fake_resolve_cli_tickers(
        cfg,
        args,
        required,
    ):
        return [
            "PETR4.SA",
        ]

    def fake_resolve_policy_selection(
        ticker,
        fallback=None,
    ):
        return {
            "profile": "active",
            "policy_type": "asset_specific_active",
            "source": "policy_matrix",
            "overrides": {
                "buy_return_pct": 0.06,
                "min_confidence_pct": 0.32,
                "risk_management": {
                    "min_rr_threshold": 0.0,
                },
            },
            "evidence": {
                "profit_factor": 1.16,
                "trades": 18,
            },
            "selection": {
                "metric": "profit_factor",
            },
        }

    def fake_runtime_policy_overrides_for_profile(
        profile,
    ):
        calls["live_profile"] = profile

        return {
            "risk_management": {
                "min_rr_threshold": 0.20,
            }
        }

    def fake_apply_policy_profile(
        cfg,
        profile,
    ):
        new_cfg = dict(cfg)

        new_cfg["policy"] = {
            "buy_return_pct": 0.08,
            "min_confidence_pct": 0.35,
            "_active_profile": profile,
            "risk_management": {
                "min_rr_threshold": 0.15,
                "aggressive_multiplier": 1.2,
            },
        }

        return new_cfg

    def fake_make_signal(
        cfg,
        ticker,
        update=False,
    ):
        calls["cfg"] = cfg
        calls["ticker"] = ticker
        calls["update"] = update

        return {
            "ticker": ticker,
            "label": "NEUTRAL",
            "policy": {
                "profile": cfg["policy"]["_active_profile"],
                "buy_return_pct": cfg["policy"]["buy_return_pct"],
                "min_confidence_pct": cfg["policy"]["min_confidence_pct"],
                "min_rr_threshold": cfg["policy"]["risk_management"]["min_rr_threshold"],
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

    def fake_print_signal(
        signal,
        verbose=False,
        diagnostic=False,
        cfg=None,
    ):
        calls["printed"] = True
        calls["printed_signal"] = signal

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
        "runtime_policy_overrides_for_profile",
        fake_runtime_policy_overrides_for_profile,
    )

    monkeypatch.setattr(
        signal_command,
        "apply_policy_profile",
        fake_apply_policy_profile,
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

    monkeypatch.setattr(
        signal_command,
        "print_signal",
        fake_print_signal,
    )

    args = argparse.Namespace(
        tickers=[
            "PETR4.SA",
        ],
        asset_list=None,
        policy_profile=None,
        update=True,
        verbose=True,
        diagnostic=True,
    )

    signal_command._smart(
        {},
        args,
    )

    assert calls["live_profile"] == "active"
    assert calls["cfg"]["policy"]["buy_return_pct"] == 0.06
    assert calls["cfg"]["policy"]["min_confidence_pct"] == 0.32
    assert calls["cfg"]["policy"]["risk_management"]["min_rr_threshold"] == 0.0
    assert calls["cfg"]["policy"]["risk_management"]["aggressive_multiplier"] == 1.2
    assert calls["ticker"] == "PETR4.SA"
    assert calls["update"] is True
    assert calls["printed"] is True

    smart_path = tmp_path / "PETR4_SA" / "latest_smart_signal.json"

    assert smart_path.exists()

    written = smart_path.read_text(
        encoding="utf-8",
    )

    assert '"stored_overrides"' in written
    assert '"live_overrides"' in written
    assert '"min_rr_threshold": 0.0' in written
