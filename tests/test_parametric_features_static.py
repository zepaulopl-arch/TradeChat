from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_no_new_cli_surface_for_context_fundamentals_sentiment():
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert 'sub.add_parser("context"' not in cli
    assert 'sub.add_parser("fundamentals"' not in cli
    assert 'sub.add_parser("sentiment"' not in cli
    assert "context:" in (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")


def test_context_fundamentals_sentiment_are_yaml_controlled():
    features = yaml.safe_load((ROOT / "config" / "features.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    default = features["selection"]["profiles"][features["selection"]["active_profile"]]["families"]
    assert default["context"]["enabled"] is True
    assert default["fundamentals"]["enabled"] is True
    assert default["sentiment"]["enabled"] is True
    assert isinstance(features["families"]["context"]["presets"]["asset_linked"]["windows"], dict)
    assert (
        features["families"]["fundamentals"]["presets"]["temporal_cvm_safe"]["features"][
            "regime_score"
        ]
        is True
    )
    assert (
        "enabled_by_default" not in features["families"]["sentiment"]["presets"]["temporal_cache"]
    )
    assert "as_feature" not in features["families"]["sentiment"]["presets"]["temporal_cache"]
    assert "features" not in cfg
    assert "enabled_by_default" in cfg["model"]["autotune"]


def test_tabular_model_contract_preserved():
    models = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "XGB + CatBoost + ExtraTrees -> Ridge arbiter" in models
    assert "CatBoostRegressor" in models
    assert "ExtraTreesRegressor" in models
    assert "MLPRegressor" not in models
    assert "Ridge" in models
    assert "BayesSearchCV" in models
