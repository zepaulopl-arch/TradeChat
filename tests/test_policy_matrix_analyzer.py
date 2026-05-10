from pathlib import Path


def test_policy_matrix_analyzer_exists_and_writes_outputs():
    path = Path("tools/analyze_policy_matrix.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "asset_eligibility_suggested.yaml" in text
    assert "policy_matrix_profile_summary.csv" in text
    assert "policy_matrix_eligibility_suggested.csv" in text
    assert "validation_summary.csv" in text


def test_policy_matrix_analyzer_does_not_import_app():
    text = Path("tools/analyze_policy_matrix.py").read_text(encoding="utf-8")
    assert "from app" not in text
    assert "import app" not in text
