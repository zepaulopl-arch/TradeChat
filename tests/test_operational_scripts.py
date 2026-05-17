import argparse
import json
from pathlib import Path

from scripts.check_runtime_policy import check_runtime_policy
from scripts.run_daily_smart_rank import LATEST_MATRIX_PLACEHOLDER, build_daily_plan
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
        "source": "policy_matrix",
        "promoted": promoted,
        "promotion_status": status,
        "rejection_reasons": reasons,
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
TICKER      SIGNAL     PROFILE   MATRIX          PF   TRD    RR GUARD  BLOCKER
------------------------------------------------------------------------------------------------
PETR4.SA    BUY        active    APPROVE       2.10    34  1.45 OK      none
ALOS3.SA    REJECTED   active    REJECT        0.00     0  0.00 BLOCK   trades 0 < 15
------------------------------------------------------------------------------------------------
Rows: 2 of 2 | OK=1 | BLOCK=1 | REJECTED=1 | SKIP=0 | ERROR=0
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


def test_validate_smart_rank_output_rejects_false_no_matrix(tmp_path):
    runtime_path = tmp_path / "runtime_policy.json"
    _write_runtime(
        runtime_path,
        {
            "ALOS3.SA": _asset(
                promoted=False,
                status="rejected_by_constraints",
                reasons=["trades 0 < 15"],
            )
        },
    )

    text = """
SMART RANK
------------------------------------------------------------------------------------------------
TICKER      SIGNAL     PROFILE   MATRIX          PF   TRD    RR GUARD  BLOCKER
------------------------------------------------------------------------------------------------
ALOS3.SA    NO_MATRIX  n/a       n/a            n/a   n/a  0.00 SKIP   not found in runtime
------------------------------------------------------------------------------------------------
Rows: 1 of 1 | OK=0 | BLOCK=0 | REJECTED=0 | SKIP=1 | ERROR=0
"""

    errors, _warnings, _result = validate_smart_rank_output(
        text,
        expected_rows=1,
        runtime_path=runtime_path,
    )

    assert "ALOS3.SA: NO_MATRIX row exists but ticker is present in runtime." in errors


def test_validate_smart_rank_output_rejects_balanced_na_actionable_signal(tmp_path):
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
TICKER      SIGNAL     PROFILE   MATRIX          PF   TRD    RR GUARD  BLOCKER
------------------------------------------------------------------------------------------------
PETR4.SA    BUY        balanced  n/a            n/a   n/a  0.00 OK     none
------------------------------------------------------------------------------------------------
Rows: 1 of 1 | OK=1 | BLOCK=0 | REJECTED=0 | SKIP=0 | ERROR=0
"""

    errors, _warnings, _result = validate_smart_rank_output(
        text,
        expected_rows=1,
        runtime_path=runtime_path,
    )

    assert "PETR4.SA: actionable balanced n/a signal found in smart rank." in errors


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
TICKER      SIGNAL     PROFILE   MATRIX          PF   TRD    RR GUARD  BLOCKER
------------------------------------------------------------------------------------------------
PETR4.SA    STRONG BUY active    APPROVE       2.10    34  1.45 OK      none
------------------------------------------------------------------------------------------------
Rows: 1 of 1 | OK=1 | BLOCK=0 | REJECTED=0 | SKIP=0 | ERROR=0
"""

    errors, _warnings, result = validate_smart_rank_output(
        text,
        expected_rows=1,
        runtime_path=runtime_path,
    )

    assert errors == []
    assert result["rows"][0]["signal"] == "STRONG BUY"


def test_daily_plan_is_light_by_default_and_validates_rank_output(tmp_path):
    args = argparse.Namespace(
        universe="ibov",
        rank_limit=20,
        jobs=1,
        skip_data=False,
        with_matrix=False,
        skip_promote=False,
        matrix_dir=None,
        python="python",
    )

    steps = build_daily_plan(args, out_dir=tmp_path)
    commands = [" ".join(step.command) for step in steps]

    assert commands[0] == "python trade.py data load --list ibov"
    assert "python trade.py signal rank --list ibov --smart --rank-limit 20" in commands
    assert any("scripts/validate_smart_rank_output.py" in command for command in commands)
    assert not any("validate matrix" in command for command in commands)


def test_daily_plan_adds_matrix_and_promote_only_when_requested(tmp_path):
    args = argparse.Namespace(
        universe="ibov",
        rank_limit=20,
        jobs=1,
        skip_data=True,
        with_matrix=True,
        skip_promote=False,
        matrix_dir=None,
        python="python",
    )

    steps = build_daily_plan(args, out_dir=tmp_path)
    commands = [" ".join(step.command) for step in steps]

    assert commands[0] == "python trade.py validate matrix --universe ibov --jobs 1"
    assert any("python trade.py validate report --latest" == command for command in commands)
    assert any(LATEST_MATRIX_PLACEHOLDER in command for command in commands)
