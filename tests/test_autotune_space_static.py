from pathlib import Path


def test_autotune_uses_explicit_skopt_dimensions_for_tabular_engines():
    source = Path("app/models.py").read_text(encoding="utf-8")
    tune_block = source.split("def _tune_base_engines", 1)[1].split("def _make_arbiter", 1)[0]
    assert "from skopt import BayesSearchCV" in tune_block
    assert "_space_from_pair" in source
    assert "for name, engine in base_engines.items()" in tune_block
    assert 'raw_space = spaces.get(name, {})' in tune_block
    assert 'elif name == "mlp"' not in tune_block
    assert 'elif name == "ridge"' not in tune_block
    assert "tuned_tabular" in tune_block
    assert "cv_splitter = _time_series_cv" in tune_block
    assert "cv=cv_splitter" in tune_block


def test_mlp_removed_from_operational_models():
    source = Path("app/models.py").read_text(encoding="utf-8")
    assert "MLPRegressor" not in source
    assert "CatBoostRegressor" in source
    assert "ExtraTreesRegressor" in source
