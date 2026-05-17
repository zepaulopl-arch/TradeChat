from pathlib import Path


def test_policy_matrix_analyzer_exists_and_writes_outputs():
    path = Path("tools/analyze_policy_matrix.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "asset_eligibility_suggested.yaml" in text
    assert "policy_matrix_profile_summary.csv" in text
    assert "policy_matrix_eligibility_suggested.csv" in text
    assert "validation_summary.csv" in text
    assert "rebuild_validation_summary" in text


def test_policy_matrix_analyzer_does_not_import_app():
    text = Path("tools/analyze_policy_matrix.py").read_text(encoding="utf-8")
    assert "from app" not in text
    assert "import app" not in text


def test_policy_matrix_analyzer_rebuilds_summary_from_logs(tmp_path):
    from tools.analyze_policy_matrix import main

    run_dir = tmp_path / "matrix"
    log_dir = run_dir / "04_validate_per_asset" / "relaxed"
    log_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"ticker_count": 1, "profiles": ["relaxed"], "tickers": ["PETR4.SA"]}',
        encoding="utf-8",
    )
    (log_dir / "0001_PETR4.SA.log").write_text(
        """
VALIDATE - PYBROKER - REPLAY
RESUMO
Amostra              : 2025-01-01 ate 2026-01-01 | 1 ativos
Policy               : relaxed
VALIDATION DECISION
Decision   : [OBSERVE]
Score      : 83.3
RESULTADO
Configuracao         Retorno  Trades   Drawdown  Decisao
----------------------------------------------------------------------------------------------------
replay operacional    +1.70%      12     -0.87%  [MANTER EM OBSERVACAO]
ECONOMIA
    Hit  Avg Trade  Profit F   Turnover   Exposure      Cost
----------------------------------------------------------------------------------------------------
  83.3%     +0.51%      2.39     722.4%       8.2%    +21.65
P&L AUDIT
  Gross +    Gross -        Net  Before Cost  After Cost
----------------------------------------------------------------------------------------------------
  +291.55    -121.86    +169.69       +1.91%      +1.70%
MODELO VS BASELINES
Beat rate  : 20.0%
""",
        encoding="utf-8",
    )

    rc = main([str(run_dir)])
    assert rc == 0
    assert (run_dir / "validation_summary.csv").exists()
    report = (run_dir / "analysis" / "policy_matrix_analysis_report.md").read_text(encoding="utf-8")
    assert "PETR4.SA" in report
    assert "Main candidates" in report
    assert "+1.70%" in report


def test_policy_matrix_analyzer_marks_insufficient_history_as_ineligible(tmp_path):
    from tools.analyze_policy_matrix import rebuild_validation_summary

    run_dir = tmp_path / "matrix"
    log_dir = run_dir / "04_validate_per_asset" / "active"
    log_dir.mkdir(parents=True)
    (log_dir / "0001_EMBJ3.SA.log").write_text(
        """
VALIDATE - PYBROKER - REPLAY
Policy               : active
ERROR: insufficient prepared train rows: 8 < 60
""",
        encoding="utf-8",
    )

    summary = rebuild_validation_summary(
        run_dir,
        force=True,
    )
    text = summary.read_text(encoding="utf-8")

    assert "EMBJ3.SA" in text
    assert "INELIGIBLE_DATA" in text
    assert "ineligible_data" in text
    assert "insufficient history: rows 8 < 60" in text
