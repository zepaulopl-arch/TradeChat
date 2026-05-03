from app.config import load_config, load_data_registry
from app.data import resolve_asset
from app.models import _confidence_from, _make_base_engines


def test_known_b3_ticker_migrations_resolve_to_current_codes():
    cfg = load_config()
    expected = {
        "ARZZ3": "AZZA3.SA",
        "SOMA3": "AZZA3.SA",
        "CCRO3": "MOTV3.SA",
        "RRRP3": "BRAV3.SA",
        "BRFS3": "MBRF3.SA",
        "MRFG3": "MBRF3.SA",
        "AZUL4": "AZUL54.SA",
        "GOLL4": "GOLL54.SA",
        "EMBR3": "EMBJ3.SA",
    }
    for old, new in expected.items():
        resolved = resolve_asset(cfg, old)
        assert resolved["canonical"] == new
        assert resolved["changed"] is True


def test_legacy_ticker_aliases_are_excluded_from_reference_sample():
    registry = load_data_registry(load_config())
    for old in ["ARZZ3.SA", "SOMA3.SA", "CCRO3.SA", "RRRP3.SA", "BRFS3.SA", "MRFG3.SA", "AZUL4.SA", "GOLL4.SA", "EMBR3.SA"]:
        meta = registry["assets"][old]
        assert meta["registry_status"] == "inactive_alias"
        assert meta["use_in_reference_sample"] is False


def test_operational_engines_are_three_tabular_specialists():
    engines = _make_base_engines(load_config())
    assert set(engines) == {"xgb", "catboost", "extratrees"}


def test_confidence_penalizes_tiny_prediction_and_discarded_engine():
    cfg = load_config()
    dispersion, conf = _confidence_from(
        [0.0013, 0.0014, 0.00135],
        0.0013,
        cfg,
        mae=0.012,
        guard_meta={"used": ["xgb", "catboost"], "discarded": ["extratrees"]},
        train_rows=1000,
    )
    assert dispersion >= 0
    assert conf < 0.70
