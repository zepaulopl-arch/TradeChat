from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_preparation_module_exists_and_is_used():
    assert (ROOT / "app" / "preparation.py").exists()
    features_py = (ROOT / "app" / "features.py").read_text(encoding="utf-8")
    assert "prepare_training_matrix" in features_py
    assert "generated_feature_count" in features_py
    assert "selected_features" in features_py


def test_features_yaml_has_preparation_contract():
    text = (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")
    assert "preparation:" in text
    assert "max_features: 20" in text
    assert "target_relevance_low_correlation_greedy" in text
    assert "normalization:" in text


def test_models_use_configurable_scaler():
    text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "def _make_scaler" in text
    assert "RobustScaler" in text
    assert "normalization" in text
