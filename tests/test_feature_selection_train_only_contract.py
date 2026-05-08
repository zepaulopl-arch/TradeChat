from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_selection_train_only_contract():
    models_text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "prepare_training_matrix(X_train_raw" in models_text
    assert "X_test_raw" in models_text
    assert "embargo_bars" in models_text
