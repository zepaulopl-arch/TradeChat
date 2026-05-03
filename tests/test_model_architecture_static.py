from app.config import load_config
from app.models import _make_base_engines


def test_base_engines_are_tabular_and_exclude_ridge_and_mlp():
    cfg = load_config(None)
    cfg["model"]["engines"]["xgb"]["enabled"] = False
    cfg["model"]["engines"]["catboost"]["enabled"] = False
    engines = _make_base_engines(cfg)
    assert "extratrees" in engines
    assert "mlp" not in engines
    assert "rf" not in engines
    assert "ridge" not in engines


def test_ridge_is_declared_as_arbiter_in_config():
    cfg = load_config(None)
    assert cfg["model"]["arbiter"]["ridge"]["enabled"] is True
    assert "xgb" in cfg["model"]["engines"]
    assert "catboost" in cfg["model"]["engines"]
    assert "extratrees" in cfg["model"]["engines"]
    assert "mlp" not in cfg["model"]["engines"]


def test_autotune_is_available_but_not_default_daily():
    from app.cli import build_parser
    cfg = load_config(None)
    assert cfg["model"]["autotune"]["enabled_by_default"] is False
    parser = build_parser()
    train = parser.parse_args(["train", "PETR4", "--autotune"])
    assert train.autotune is True
    daily = parser.parse_args(["daily", "PETR4"])
    assert not hasattr(daily, "autotune")
