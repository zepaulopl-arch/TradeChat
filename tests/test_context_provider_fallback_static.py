from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_config_period_defaults_to_max():
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    assert "period: max" in text


def test_data_download_has_context_period_fallback():
    text = (ROOT / "app" / "data.py").read_text(encoding="utf-8")
    assert "def _period_fallbacks" in text
    assert 'attempts.extend(["10y", "5y", "2y", "1y"])' in text
    assert "is_context=idx > 0" in text


def test_data_screen_shows_skipped_context():
    text = (ROOT / "app" / "report.py").read_text(encoding="utf-8")
    assert "ctx skipped" in text
    assert "unavailable_context_tickers" in text


def test_data_yaml_has_group_and_subgroup_context_candidates():
    text = (ROOT / "config" / "data.yaml").read_text(encoding="utf-8")
    assert "group_defaults:" in text
    assert "subgroup_defaults:" in text
    assert "IDIV:" in text
    assert "pending_provider_validation" in text
