from app.cli import build_parser
from app.commands import validate_command
from app.policy import active_policy_profile
from app.validation_view import render_validation_summary


def test_validate_cli_accepts_policy_profile():
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate",
            "PETR4.SA",
            "--mode",
            "replay",
            "--policy-profile",
            "balanced",
        ]
    )

    assert args.policy_profile == "balanced"


def test_validate_command_applies_policy_profile(monkeypatch):
    captured = {}

    def fake_load_config(path=None):
        return {
            "policy": {
                "buy_return_pct": 0.45,
                "sell_return_pct": -0.45,
                "min_confidence_pct": 0.45,
            },
            "simulation": {"rebalance_days": 5, "warmup_bars": 150},
            "data": {},
        }

    def fake_resolve_cli_tickers(cfg, args):
        captured["profile_in_resolve"] = active_policy_profile(cfg)
        return ["PETR4.SA"]

    def fake_run_pybroker_replay(cfg, tickers, **kwargs):
        captured["profile_in_runner"] = active_policy_profile(cfg)
        captured["kwargs"] = kwargs
        return {
            "mode": "pybroker_artifact_replay",
            "tickers": tickers,
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
            "policy_profile": active_policy_profile(cfg),
            "metrics": {},
            "baselines": {},
            "baseline_comparison": {},
            "validation_decision": {},
        }

    monkeypatch.setattr(validate_command, "load_config", fake_load_config)
    monkeypatch.setattr(validate_command, "resolve_cli_tickers", fake_resolve_cli_tickers)
    monkeypatch.setattr(validate_command, "run_pybroker_replay", fake_run_pybroker_replay)
    monkeypatch.setattr(validate_command, "render_validation_summary", lambda *a, **k: [])

    parser = build_parser()
    args = parser.parse_args(
        [
            "validate",
            "PETR4.SA",
            "--mode",
            "replay",
            "--policy-profile",
            "relaxed",
        ]
    )
    validate_command.run(args)

    assert captured["profile_in_resolve"] == "relaxed"
    assert captured["profile_in_runner"] == "relaxed"


def test_validation_summary_renders_policy_profile():
    lines = render_validation_summary(
        {
            "mode": "pybroker_artifact_replay",
            "tickers": ["PETR4.SA"],
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
            "policy_profile": "balanced",
            "metrics": {},
            "baselines": {},
            "baseline_comparison": {},
            "validation_decision": {},
        },
        mode="replay",
    )

    assert "Policy" in "\n".join(lines)
    assert "balanced" in "\n".join(lines)
