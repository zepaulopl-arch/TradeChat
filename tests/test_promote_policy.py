import csv
import importlib
import json

from app.commands.promote_policy import (
    promote_policy,
)


def test_promote_policy_import():

    assert callable(
        promote_policy
    )


def test_promote_policy_writes_promoted_and_rejected_assets(
    tmp_path,
    monkeypatch,
    capsys,
):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()

    rows = [
        {
            "phase": "04_validate_per_asset",
            "ticker": "PETR4.SA",
            "scope": "PETR4.SA",
            "profile": "active",
            "policy": "active",
            "decision": "APPROVE",
            "score": "92.0",
            "return_pct": "2.5",
            "trades": "18",
            "drawdown_pct": "-1.2",
            "hit_pct": "70.0",
            "avg_trade_pct": "0.4",
            "profit_factor": "inf",
            "profit_factor_display": "inf",
            "exposure_pct": "20.0",
            "threshold": "0.11",
            "horizon": "d5",
            "matrix_rr": "1.80",
            "log_path": "logs/policy_matrix/x/PETR4.SA.log",
        },
        {
            "phase": "04_validate_per_asset",
            "ticker": "ALOS3.SA",
            "scope": "ALOS3.SA",
            "profile": "active",
            "policy": "active",
            "decision": "REJECT",
            "score": "16.7",
            "return_pct": "0.0",
            "trades": "0",
            "drawdown_pct": "0.0",
            "hit_pct": "0.0",
            "avg_trade_pct": "0.0",
            "profit_factor": "0.00",
            "profit_factor_display": "0.00",
            "exposure_pct": "0.0",
            "threshold": "",
            "horizon": "",
            "matrix_rr": "",
            "log_path": "logs/policy_matrix/x/ALOS3.SA.log",
        },
    ]

    with (matrix_dir / "validation_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    def fake_load_runtime_policy_config():
        return {
            "promotion": {
                "mode": "by-asset",
                "metric": "profit_factor",
                "constraints": {
                    "max_drawdown_pct": 15.0,
                    "min_trades": 15,
                    "min_sharpe": 0.80,
                    "max_exposure_pct": 80.0,
                },
                "tie_break": [
                    "hit_pct",
                ],
                "runtime_overrides": {
                    "enabled": True,
                    "profiles": {
                        "active": {
                            "buy_return_pct": 0.06,
                        }
                    },
                },
            }
        }

    promote_module = importlib.import_module("app.commands.promote_policy")

    monkeypatch.setattr(
        promote_module,
        "load_runtime_policy_config",
        fake_load_runtime_policy_config,
    )
    monkeypatch.chdir(tmp_path)

    promote_policy(str(matrix_dir))

    out = capsys.readouterr().out

    assert "Assets total: 2" in out
    assert "Assets promoted: 1" in out
    assert "Assets rejected: 1" in out

    runtime = json.loads(
        (tmp_path / "runtime" / "runtime_policy.json").read_text(
            encoding="utf-8",
        )
    )

    assert runtime["assets_total"] == 2
    assert runtime["assets_promoted"] == 1
    assert runtime["assets_rejected"] == 1

    petr = runtime["assets"]["PETR4.SA"]
    assert petr["source"] == "policy_matrix"
    assert petr["profile"] == "active"
    assert petr["policy_type"] == "asset_specific_active"
    assert petr["evaluated"] is True
    assert petr["ineligible_data"] is False
    assert petr["promoted"] is True
    assert petr["actionable_candidate"] is True
    assert petr["promotion_status"] == "promoted"
    assert petr["rejection_reasons"] == []
    assert petr["overrides"]["buy_return_pct"] == 0.11
    assert petr["overrides"]["sell_return_pct"] == -0.11
    assert petr["overrides"]["min_confidence_pct"] == 0.32
    assert petr["overrides"]["preferred_horizon"] == "d5"
    assert petr["overrides"]["risk_management"]["min_rr_threshold"] == 0.6
    assert petr["overrides"]["risk_management"]["risk_per_trade_pct"] == 0.872
    assert petr["overrides"]["risk_management"]["max_position_pct"] == 34.4
    assert petr["overrides"]["validation_constraints"]["min_trades"] == 15
    assert petr["overrides"]["validation_constraints"]["observed_trades"] == 18
    assert petr["overrides"]["validation_constraints"]["observed_drawdown_pct"] == -1.2
    assert petr["overrides"]["validation_constraints"]["observed_exposure_pct"] == 20.0
    assert petr["overrides"]["validation_constraints"]["observed_profit_factor"] == "inf"
    assert petr["overrides"]["asset_specific_active"]["policy_type"] == "asset_specific_active"
    assert petr["overrides"]["asset_specific_active"]["matrix_decision"] == "APPROVE"
    assert petr["overrides"]["asset_specific_active"]["matrix_profile"] == "active"
    assert petr["evidence"]["decision"] == "APPROVE"
    assert petr["evidence"]["profit_factor"] == "inf"
    assert petr["evidence"]["profit_factor_display"] == "inf"
    assert petr["evidence"]["matrix_rr"] == 1.8
    assert petr["selection"]["sort_columns"] == [
        "profit_factor",
        "hit_pct",
    ]
    assert petr["selection"]["matrix_profile"] == "active"
    assert "risk_management.min_rr_threshold" in petr["selection"]["asset_specific_parameters"]

    alos = runtime["assets"]["ALOS3.SA"]
    assert alos["source"] == "policy_matrix"
    assert alos["profile"] == "active"
    assert alos["policy_type"] == "asset_specific_active"
    assert alos["evaluated"] is True
    assert alos["ineligible_data"] is False
    assert alos["promoted"] is False
    assert alos["actionable_candidate"] is False
    assert alos["promotion_status"] == "rejected_by_constraints"
    assert alos["rejection_reasons"] == [
        "trades 0 < 15",
    ]
    assert alos["overrides"] == {}
    assert alos["evidence"]["decision"] == "REJECT"


def test_promote_policy_marks_ineligible_data_assets(
    tmp_path,
    monkeypatch,
    capsys,
):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()

    rows = [
        {
            "phase": "04_validate_per_asset",
            "ticker": "EMBJ3.SA",
            "scope": "EMBJ3.SA",
            "profile": "active",
            "policy": "active",
            "decision": "INELIGIBLE_DATA",
            "score": "",
            "return_pct": "",
            "trades": "",
            "drawdown_pct": "",
            "hit_pct": "",
            "avg_trade_pct": "",
            "profit_factor": "",
            "profit_factor_display": "",
            "exposure_pct": "",
            "eligibility_status": "ineligible_data",
            "ineligible_reason": "insufficient history: rows 8 < 60",
            "data_rows": "8",
            "log_path": "logs/policy_matrix/x/EMBJ3.SA.log",
        }
    ]

    with (matrix_dir / "validation_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    def fake_load_runtime_policy_config():
        return {
            "promotion": {
                "mode": "by-asset",
                "metric": "profit_factor",
                "constraints": {
                    "max_drawdown_pct": 15.0,
                    "min_trades": 15,
                    "min_sharpe": 0.80,
                    "max_exposure_pct": 80.0,
                },
                "tie_break": [],
                "runtime_decision_guard": {
                    "decisions": {
                        "APPROVE": {
                            "max_signal": "ACTIONABLE",
                        }
                    }
                },
            }
        }

    promote_module = importlib.import_module("app.commands.promote_policy")

    monkeypatch.setattr(
        promote_module,
        "load_runtime_policy_config",
        fake_load_runtime_policy_config,
    )
    monkeypatch.chdir(tmp_path)

    promote_policy(str(matrix_dir))

    out = capsys.readouterr().out

    assert "Assets total: 1" in out
    assert "Assets ineligible: 1" in out

    runtime = json.loads(
        (tmp_path / "runtime" / "runtime_policy.json").read_text(
            encoding="utf-8",
        )
    )
    embj = runtime["assets"]["EMBJ3.SA"]

    assert embj["source"] == "data_eligibility"
    assert embj["profile"] == "active"
    assert embj["policy_type"] == "asset_specific_active"
    assert embj["evaluated"] is True
    assert embj["ineligible_data"] is True
    assert embj["promoted"] is False
    assert embj["actionable_candidate"] is False
    assert embj["promotion_status"] == "ineligible_data"
    assert embj["rejection_reasons"] == [
        "insufficient history: rows 8 < 60",
    ]
    assert embj["evidence"]["data_rows"] == 8
