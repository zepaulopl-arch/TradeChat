from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mlp_fixed_config_preserves_two_layer_architecture():
    source = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    block = source.split('if engines_cfg.get("mlp"', 1)[1].split('return engines', 1)[0]
    assert "tuple(int(v) for v in hidden)" in block
    assert "hidden_layer_sizes=hidden_layer_sizes" in block
    assert "hidden = int(hidden[-1])" not in block


def test_engine_consensus_guard_exists_for_stacking_inputs():
    source = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "def _apply_consensus_guard" in source
    assert "meta_train, train_consensus_meta = _apply_consensus_guard" in source
    assert "meta_test, test_consensus_meta = _apply_consensus_guard" in source
    assert "meta_latest, latest_consensus_meta = _apply_consensus_guard" in source


def test_engine_safety_is_configurable():
    text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    assert "engine_safety:" in text
    assert "consensus_guard_enabled: true" in text
    assert "max_deviation_from_median:" in text


def test_yaml_files_live_under_config_directory():
    assert (ROOT / "config" / "config.yaml").exists()
    assert (ROOT / "config" / "data.yaml").exists()
    assert (ROOT / "config" / "features.yaml").exists()
    assert not (ROOT / "config.yaml").exists()
    assert not (ROOT / "data.yaml").exists()
    assert not (ROOT / "features.yaml").exists()
