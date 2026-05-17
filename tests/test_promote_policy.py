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
            "profile": "relaxed",
            "policy": "relaxed",
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
                        "relaxed": {
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
    assert petr["promoted"] is True
    assert petr["promotion_status"] == "promoted"
    assert petr["rejection_reasons"] == []
    assert petr["overrides"]["buy_return_pct"] == 0.06
    assert petr["evidence"]["decision"] == "APPROVE"
    assert petr["evidence"]["profit_factor"] == "inf"
    assert petr["evidence"]["profit_factor_display"] == "inf"
    assert petr["selection"]["sort_columns"] == [
        "profit_factor",
        "hit_pct",
    ]

    alos = runtime["assets"]["ALOS3.SA"]
    assert alos["source"] == "policy_matrix"
    assert alos["promoted"] is False
    assert alos["promotion_status"] == "rejected_by_constraints"
    assert alos["rejection_reasons"] == [
        "trades 0 < 15",
    ]
    assert alos["overrides"] == {}
    assert alos["evidence"]["decision"] == "REJECT"
