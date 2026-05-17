import argparse
import json
from pathlib import Path

from scripts.check_runtime_policy import check_runtime_policy
from scripts.run_operational_matrix import OPERATIONAL_PROFILE, build_operational_plan
from scripts.validate_smart_rank_output import validate_smart_rank_output


def _write_runtime(path: Path, assets: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "assets_total": len(assets),
                "assets_promoted": sum(1 for item in assets.values() if item["promoted"]),
                "assets_rejected": sum(1 for item in assets.values() if not item["promoted"]),
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )


def _asset(
    *,
    promoted: bool,
    status: str,
    reasons: list[str],
) -> dict:
    return {
        "profile": "active",
        "policy_type": "asset_specific_active",
        "source": "policy_matrix",
        "evaluated": True,
        "ineligible_data": False,
        "promoted": promoted,
        "actionable_candidate": promoted,
        "promotion_status": status,
        "rejection_reasons": reasons,
        "blocker": reasons[0] if reasons else None,
        "overrides": {},
        "evidence": {
            "decision": "APPROVE" if promoted else "REJECT",
            "profit_factor": 2.0 if promoted else 0.0,
            "trades": 20 if promoted else 0,
        },
        "selection": {},
    }


def test_check_runtime_policy_accepts_promoted_and_rejected_assets(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    config_path = tmp_path / "runtime_policy.yaml"
    config_path.write_text("promotion: {}\n", encoding="utf-8")
    _write_runtime(
        runtime_path,
        {
            "PETR4.SA": _asset(
                promoted=True,
                status="promoted",
                reasons=[],
            ),
            "ALOS3.SA": _asset(
                promoted=False,
                status="rejected_by_constraints",
                reasons=["trades 0 < 15"],
            ),
        },
    )

    errors, warnings, summary = check_runtime_policy(runtime_path, config_path)

    assert errors == []
    assert warnings == []
    assert summary["assets_total"] == 2
    assert summary["assets_promoted"] == 1
    assert summary["assets_rejected"] == 1


def test_check_runtime_policy_rejects_runtime_without_rejected_assets(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    config_path = tmp_path / "runtime_policy.yaml"
    config_path.write_text("promotion: {}\n", encoding="utf-8")
    _write_runtime(
        runtime_path,
        {
            "PETR4.SA": _asset(
                promoted=True,
                status="promoted",
                reasons=[],
            )
        },
    )

    errors, _warnings, _summary = check_runtime_policy(runtime_path, config_path)

    assert "runtime_policy has 0 rejected assets. Check promote_policy output." in errors


def test_validate_smart_rank_output_accepts_clean_policy_first_rank(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    _write_runtime(
        runtime_path,
        {
            "PETR4.SA": _asset(
                promoted=True,
                status="promoted",
                reasons=[],
            ),
            "ALOS3.SA": _asset(
                promoted=False,
                status="rejected_by_constraints",
                reasons=["trades 0 < 15"],
            ),
        },
    )

    text = """
SMART RANK
------------------------------------------------------------------------------------------------
#   TICKER      ACTION      SIGNAL     MATRIX          PF   TRD     RR GUARD  REASON
------------------------------------------------------------------------------------------------
  1 PETR4.SA    ACTIONABLE  BUY        APPROVE       2.10    34   1.45 OK      none
  2 ALOS3.SA    REJECTED    REJECTED   REJECT        0.00     0    n/a BLOCK   trades 0 < 15
------------------------------------------------------------------------------------------------
Processed: 2
Displayed: 2 of 2
Rows: 2 of 2
ACTIONABLE=1 | WATCH=0 | BLOCKED=0 | REJECTED=1 | INELIGIBLE=0 | ERROR=0
Top actionable: PETR4.SA
Main blocker: trades < minimum
"""

    errors, warnings, result = validate_smart_rank_output(
        text,
        expected_rows=2,
        runtime_path=runtime_path,
    )

    assert errors == []
    assert warnings == []
    assert result["summary"]["total"] == 2


def test_validate_smart_rank_output_rejects_actionable_without_matrix(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    _write_runtime(
        runtime_path,
        {
            "PETR4.SA": _asset(
                promoted=True,
                status="promoted",
                reasons=[],
            )
        },
    )

    text = """
SMART RANK
------------------------------------------------------------------------------------------------
#   TICKER      ACTION      SIGNAL     MATRIX          PF   TRD     RR GUARD  REASON
------------------------------------------------------------------------------------------------
  1 PETR4.SA    ACTIONABLE  BUY        n/a            n/a   n/a   0.00 OK     none
------------------------------------------------------------------------------------------------
Processed: 1
Displayed: 1 of 1
Rows: 1 of 1
ACTIONABLE=1 | WATCH=0 | BLOCKED=0 | REJECTED=0 | INELIGIBLE=0 | ERROR=0
"""

    errors, _warnings, _result = validate_smart_rank_output(
        text,
        expected_rows=1,
        runtime_path=runtime_path,
    )

    assert "PETR4.SA: actionable signal without Matrix decision found." in errors


def test_validate_smart_rank_output_parses_multi_word_signal(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    _write_runtime(
        runtime_path,
        {
            "PETR4.SA": _asset(
                promoted=True,
                status="promoted",
                reasons=[],
            )
        },
    )

    text = """
SMART RANK
------------------------------------------------------------------------------------------------
#   TICKER      ACTION      SIGNAL     MATRIX          PF   TRD     RR GUARD  REASON
------------------------------------------------------------------------------------------------
  1 PETR4.SA    ACTIONABLE  STRONG BUY APPROVE       2.10    34   1.45 OK      none
------------------------------------------------------------------------------------------------
Processed: 1
Displayed: 1 of 1
Rows: 1 of 1
ACTIONABLE=1 | WATCH=0 | BLOCKED=0 | REJECTED=0 | INELIGIBLE=0 | ERROR=0
"""

    errors, _warnings, result = validate_smart_rank_output(
        text,
        expected_rows=1,
        runtime_path=runtime_path,
    )

    assert errors == []
    assert result["rows"][0]["signal"] == "STRONG BUY"


def test_operational_matrix_plan_runs_active_only_without_powershell(tmp_path):
    args = argparse.Namespace(
        universe="ibov",
        rank_limit=20,
        jobs=4,
        allow_untrained=False,
        skip_tests=True,
        python="python",
    )
    matrix_dir = tmp_path / "logs" / "policy_matrix" / "run"

    steps = build_operational_plan(
        args,
        out_dir=tmp_path / "artifacts",
        matrix_dir=matrix_dir,
    )
    commands = [" ".join(step.command) for step in steps]

    assert OPERATIONAL_PROFILE == "active"
    assert commands[0].startswith("python trade.py validate matrix --universe ibov --jobs 4 --policy-profile active")
    assert "--policy-profile active" in commands[0]
    assert "--skip-pytest" in commands[0]
    assert "python trade.py validate report --latest" in commands
    assert "python trade.py signal rank --list ibov --smart --rank-limit 20" in commands
    assert any("scripts/validate_smart_rank_output.py" in command for command in commands)
    assert "powershell" not in " ".join(commands).lower()
    assert any(matrix_dir.as_posix() in command for command in commands if "promote_policy" in command)
    assert any("smart_rank.txt" in command for command in commands)
