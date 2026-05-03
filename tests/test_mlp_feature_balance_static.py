from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_features_yaml_has_family_limits():
    text = (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")
    assert "family_limits:" in text
    assert "context:" in text
    assert "max: 5" in text
    assert "target_relevance_low_correlation_greedy_family_balanced" in text


def test_mlp_has_own_scaler_and_subset():
    text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "def _engine_feature_columns" in text
    assert "def _fit_engine_input" in text
    assert "def _latest_engine_guard" in text
    assert "feature_family" in text


def test_signal_reports_used_and_discarded_engines():
    text = (ROOT / "app" / "report.py").read_text(encoding="utf-8")
    assert "used engines:" in text
    assert "neutralized before Ridge" in text
