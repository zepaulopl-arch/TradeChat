import json
from pathlib import Path

from app import runtime_policy


def test_resolve_policy_selection_supports_legacy_string(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps({"assets": {"PETR4.SA": "balanced"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_policy,
        "POLICY_PATH",
        policy_path,
    )

    selection = runtime_policy.resolve_policy_selection("PETR4.SA")

    assert selection["profile"] == "balanced"
    assert selection["source"] == "runtime_policy_legacy"
    assert selection["evidence"] == {}


def test_resolve_policy_selection_supports_rich_object(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "PETR4.SA": {
                        "profile": "aggressive",
                        "source": "policy_matrix",
                        "evidence": {
                            "profit_factor": 1.42,
                            "trades": 34,
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

    selection = runtime_policy.resolve_policy_selection("PETR4.SA")

    assert selection["profile"] == "aggressive"
    assert selection["source"] == "policy_matrix"
    assert selection["evidence"]["profit_factor"] == 1.42
    assert selection["evidence"]["trades"] == 34


def test_resolve_policy_profile_remains_backward_compatible(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "PETR4.SA": {
                        "profile": "conservative",
                        "source": "policy_matrix",
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

    profile = runtime_policy.resolve_policy_profile("PETR4.SA")

    assert profile == "conservative"


def test_resolve_policy_selection_uses_fallback_when_missing(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps({"assets": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_policy,
        "POLICY_PATH",
        policy_path,
    )

    selection = runtime_policy.resolve_policy_selection(
        "VALE3.SA",
        fallback="balanced",
    )

    assert selection["profile"] == "balanced"
    assert selection["source"] == "fallback"
