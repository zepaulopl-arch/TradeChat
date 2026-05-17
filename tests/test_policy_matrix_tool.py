from pathlib import Path


def test_policy_matrix_tool_exists_and_documents_outputs():
    path = Path("tools/run_policy_matrix.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "status.csv" in text
    assert "validation_summary.csv" in text
    assert "rebuild_validation_summary" in text
    assert "--asset-list" in text
    assert "--resume" in text
    assert "--jobs" in text


def test_policy_matrix_tool_uses_existing_tradechat_cli():
    text = Path("tools/run_policy_matrix.py").read_text(encoding="utf-8")
    assert "trade.py" in text
    assert "validate" in text
    assert "signal" in text
    assert "rank" in text
    assert "data" in text
    assert "audit" in text


def test_policy_matrix_tool_has_parallel_execution_guardrails():
    text = Path("tools/run_policy_matrix.py").read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" in text
    assert "STATUS_LOCK" in text
    assert "jobs" in text
    assert "pytest" in text
