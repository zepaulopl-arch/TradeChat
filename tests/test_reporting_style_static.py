from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_presentation_module_provides_shared_layout_helpers():
    text = (ROOT / "app" / "presentation.py").read_text(encoding="utf-8")
    for token in [
        "def screen_width",
        "def banner",
        "def render_facts",
        "def render_table",
        "class C:",
    ]:
        assert token in text


def test_operational_reports_use_shared_presentation_layer():
    for rel_path in [
        "scripts/diagnose_assets.py",
        "app/report.py",
        "app/ranking_service.py",
        "app/rebalance_service.py",
        "app/portfolio_monitor_service.py",
    ]:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "from app.presentation" in text or "from .presentation" in text
        assert "screen_width" in text
        assert "render_table" in text
