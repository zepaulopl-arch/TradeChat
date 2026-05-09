from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_shortcut_bat_remains():
    root_bats = sorted(path.name for path in ROOT.glob("*.bat"))
    assert root_bats == ["run.bat"]
    assert not (ROOT / "scripts" / "diagnose_assets.py").exists()
    assert not (ROOT / "scripts" / "analyze_assets.py").exists()


def test_no_legacy_cli_facade_or_simulator_facade_remain():
    assert not (ROOT / "app" / "cli_handlers.py").exists()
    assert not (ROOT / "app" / "simulator_service.py").exists()


def test_docs_do_not_recommend_legacy_surface_or_bat_scripts():
    for rel_path in ["README.md", "OPERATIONAL_MANUAL.md"]:
        text = (ROOT / rel_path).read_text(encoding="utf-8").lower()
        assert ".bat" not in text
        assert "trade.py predict" not in text
        assert "predict --rank" not in text
        assert "deprecated alias" not in text


def test_app_code_does_not_import_legacy_cli_facade():
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "cli_handlers" not in text
