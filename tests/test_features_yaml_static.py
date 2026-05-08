from pathlib import Path

import yaml

from app.config import load_config, load_features_config

ROOT = Path(__file__).resolve().parents[1]


def test_features_yaml_exists_and_config_points_to_it():
    cfg_raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_raw["features_file"] == "features.yaml"
    assert "features" not in cfg_raw
    features = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    for section in ["selection", "generation", "families"]:
        assert section in features


def test_each_feature_family_owns_its_windows_and_presets():
    features = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    families = features["families"]
    for family in ["technical", "context", "fundamentals", "sentiment"]:
        assert family in families
        assert "presets" in families[family]
        for preset in families[family]["presets"].values():
            assert "windows" in preset
            assert "features" in preset
            assert isinstance(preset["windows"], dict)
            assert isinstance(preset["features"], dict)


def test_selection_profile_chooses_family_presets():
    features = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    default = features["selection"]["profiles"][features["selection"]["active_profile"]]
    for family in ["technical", "context", "fundamentals", "sentiment"]:
        assert default["families"][family]["enabled"] is True
        assert default["families"][family]["preset"] in features["families"][family]["presets"]


def test_load_config_merges_features_yaml_for_runtime_code():
    cfg = load_config(None)
    assert "features" in cfg
    assert cfg["features"]["context"]["enabled"] is True
    assert cfg["features"]["fundamentals"]["enabled"] is True
    assert cfg["features"]["sentiment"]["enabled"] is True
    assert cfg["features"]["generation"]["multicollinearity_threshold"] == 0.92
    assert cfg["features"]["multicollinearity_threshold"] == 0.92


def test_feature_profiles_control_family_enabled_flags():
    features = load_features_config(ROOT / "config" / "features.yaml")
    assert features["selection"]["active_profile"] == "default"
    assert features["technical"]["enabled"] is True
    assert features["context"]["enabled"] is True
    assert features["fundamentals"]["enabled"] is True
    assert features["sentiment"]["enabled"] is True


def test_runtime_context_and_sentiment_keep_family_specific_windows():
    cfg = load_features_config(ROOT / "config" / "features.yaml")
    assert cfg["context"]["return_windows"] == [5, 20, 60]
    assert cfg["context"]["beta_windows"] == [20, 60]
    assert cfg["context"]["feature_windows"]["alignment"] == {"short": 5, "long": 20}
    assert cfg["sentiment"]["window_groups"]["mean"] == [1, 3, 7]
    assert cfg["sentiment"]["window_groups"]["delta"] == [3]
