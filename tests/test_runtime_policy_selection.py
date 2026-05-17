import json
from pathlib import Path

from app import runtime_policy


def test_resolve_policy_selection_rejects_legacy_string_as_fallback(
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

    assert selection["profile"] == "active"
    assert selection["source"] == "fallback"
    assert selection["promoted"] is False
    assert selection["evidence"] == {}


def test_resolve_policy_selection_consolidates_rich_object_to_active(
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
                        "policy_type": "asset_specific_active",
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

    assert selection["profile"] == "active"
    assert selection["policy_type"] == "asset_specific_active"
    assert selection["source"] == "policy_matrix"
    assert selection["evidence"]["profit_factor"] == 1.42
    assert selection["evidence"]["trades"] == 34


def test_resolve_policy_selection_preserves_matrix_rejection_metadata(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "ALOS3.SA": {
                        "profile": "active",
                        "policy_type": "asset_specific_active",
                        "source": "policy_matrix",
                        "promoted": False,
                        "promotion_status": "rejected_by_constraints",
                        "rejection_reasons": [
                            "trades 0 < 15",
                        ],
                        "overrides": {
                            "buy_return_pct": 0.08,
                        },
                        "evidence": {
                            "decision": "REJECT",
                            "trades": 0,
                        },
                        "selection": {
                            "metric": "profit_factor",
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

    selection = runtime_policy.resolve_policy_selection("ALOS3.SA")

    assert selection["profile"] == "active"
    assert selection["source"] == "policy_matrix"
    assert selection["promoted"] is False
    assert selection["promotion_status"] == "rejected_by_constraints"
    assert selection["rejection_reasons"] == [
        "trades 0 < 15",
    ]
    assert selection["overrides"]["buy_return_pct"] == 0.08
    assert selection["evidence"]["decision"] == "REJECT"
    assert selection["selection"]["metric"] == "profit_factor"


def test_resolve_policy_selection_preserves_operational_decision_fields(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "EMBJ3.SA": {
                        "profile": "active",
                        "policy_type": "asset_specific_active",
                        "source": "data_eligibility",
                        "evaluated": True,
                        "ineligible_data": True,
                        "promoted": False,
                        "actionable_candidate": False,
                        "promotion_status": "ineligible_data",
                        "rejection_reasons": [
                            "insufficient history",
                        ],
                        "blocker": "insufficient history",
                        "overrides": {},
                        "evidence": {
                            "decision": "INELIGIBLE_DATA",
                            "data_rows": 8,
                        },
                        "selection": {
                            "metric": "profit_factor",
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

    selection = runtime_policy.resolve_policy_selection("EMBJ3.SA")

    assert selection["source"] == "data_eligibility"
    assert selection["evaluated"] is True
    assert selection["ineligible_data"] is True
    assert selection["promoted"] is False
    assert selection["actionable_candidate"] is False
    assert selection["promotion_status"] == "ineligible_data"
    assert selection["rejection_reasons"] == [
        "insufficient history",
    ]
    assert selection["blocker"] == "insufficient history"


def test_resolve_policy_selection_infers_rejected_status_when_promoted_missing(
    tmp_path,
    monkeypatch,
):
    policy_path = tmp_path / "runtime_policy.json"

    policy_path.write_text(
        json.dumps(
            {
                "assets": {
                    "ALOS3.SA": {
                        "profile": "active",
                        "policy_type": "asset_specific_active",
                        "source": "policy_matrix",
                        "promotion_status": "rejected_by_constraints",
                        "rejection_reasons": "trades 0 < 15",
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

    selection = runtime_policy.resolve_policy_selection("ALOS3.SA")

    assert selection["promoted"] is False
    assert selection["rejection_reasons"] == [
        "trades 0 < 15",
    ]


def test_resolve_policy_profile_returns_operational_active(
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
                        "policy_type": "asset_specific_active",
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

    assert profile == "active"


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

    assert selection["profile"] == "active"
    assert selection["source"] == "fallback"
