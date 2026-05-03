from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_script_exists_and_keeps_cli_separate():
    script = ROOT / "scripts" / "diagnose_assets.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "data -> train -> predict" in text
    assert "def main" in text
    assert "train_models" in text
    assert "_make_signal" in text
    assert "assets_diagnostic.csv" in text


def test_diagnostic_runner_exists():
    runner = ROOT / "run_diagnostics.bat"
    assert runner.exists()
    assert "scripts\\diagnose_assets.py" in runner.read_text(encoding="utf-8")


def test_diagnostic_docs_describe_outputs():
    doc = ROOT / "docs" / "ASSET_DIAGNOSTIC.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "artifacts/diagnostics" in text
    assert "raw/guarded engine outputs" in text
    assert "feature selection" in text
