from app.config import load_config
from app.data import resolve_asset
from app.models import _confidence_from, _make_base_engines


def test_old_b3_tickers_are_not_resolved_as_aliases():
    cfg = load_config()
    for ticker in ["ARZZ3", "SOMA3", "CCRO3", "RRRP3", "BRFS3", "MRFG3", "AZUL4", "GOLL4", "EMBR3"]:
        resolved = resolve_asset(cfg, ticker)
        assert resolved["canonical"] == f"{ticker}.SA"
        assert resolved["changed"] is False
        assert resolved["profile"] == {}


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
