from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_preparation_module_exists_and_is_used():
    assert (ROOT / "app" / "preparation.py").exists()
    models_py = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    features_py = (ROOT / "app" / "features.py").read_text(encoding="utf-8")
    assert "prepare_training_matrix" in models_py
    assert "generated_feature_count" in features_py
    assert "selected_features" in features_py


def test_features_yaml_has_preparation_contract():
    text = (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")
    assert "preparation:" in text
    assert "max_features: 24" in text
    assert "stationarity:" in text
    assert "drop_raw_price: true" in text
    assert "target_relevance_low_correlation_greedy" in text
    assert "normalization:" in text
    assert "scaler: robust" in text


def test_models_use_configurable_scaler():
    text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "def _make_scaler" in text
    assert "RobustScaler" in text
    assert "normalization" in text
