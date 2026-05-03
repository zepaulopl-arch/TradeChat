from pathlib import Path


def test_autotune_restores_original_working_pattern():
    source = Path("app/models.py").read_text(encoding="utf-8")
    assert "from skopt import BayesSearchCV" in source
    assert "MLPRegressor" in source
    assert "model__hidden_layer_sizes" not in source.split("def _tune_base_engines", 1)[1].split("def _make_arbiter", 1)[0]
    assert "hidden_layer_sizes" in source
    assert "StandardScaler()" in source


def test_autotune_keeps_ridge_out_of_base_search():
    source = Path("app/models.py").read_text(encoding="utf-8")
    tune_block = source.split("def _tune_base_engines", 1)[1].split("def _make_arbiter", 1)[0]
    assert "elif name == \"mlp\"" in tune_block
    assert "elif name == \"ridge\"" not in tune_block
    assert "tuned_original_pattern" in tune_block
