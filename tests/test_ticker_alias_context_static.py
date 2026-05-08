from app.config import load_config, load_data_registry
from app.data import resolve_asset, resolve_context_tickers


def test_elet3_is_not_resolved_as_alias():
    cfg = load_config()
    resolved = resolve_asset(cfg, "ELET3")
    assert resolved["canonical"] == "ELET3.SA"
    assert resolved["changed"] is False
    assert resolved["profile"] == {}


def test_axia3_has_electric_sector_context():
    cfg = load_config()
    ctx = resolve_context_tickers(cfg, "AXIA3")
    assert "^BVSP" in ctx
    assert "^IEE" in ctx
    assert "^IBX50" in ctx


def test_disabled_context_indices_do_not_leak_as_provider_tickers():
    cfg = load_config()
    ctx = resolve_context_tickers(cfg, "PETR4.SA")
    assert "^BVSP" in ctx
    assert "USDBRL=X" in ctx
    assert "BZ=F" in ctx
    assert "IDIV" not in ctx


def test_group_and_subgroup_context_catalog_has_fetchable_indices():
    registry = load_data_registry(load_config())
    catalog = registry["indices"]["catalog"]
    assert catalog["IFNC"]["yahoo_ticker"] == "IFNC.SA"
    assert catalog["IEEX"]["yahoo_ticker"] == "^IEE"
    assert catalog["IBRX50"]["yahoo_ticker"] == "^IBX50"
