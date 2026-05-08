from pathlib import Path

import yaml

from app.config import load_config
from app.pipeline_service import resolve_tickers

ROOT = Path(__file__).resolve().parents[1]


def test_data_yaml_exists_and_has_reference_sample():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    assert "assets" in data
    assert len(data["assets"]) >= 50


def test_config_uses_data_registry_file_instead_of_context_registry():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["registry_file"] == "data.yaml"
    assert "context_registry" not in cfg["data"]


def test_reference_assets_have_cadastral_fields():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    required = {
        "name",
        "group",
        "subgroup",
        "financial_class",
        "cnpj",
        "context_tickers",
        "linked_indices",
    }
    for ticker, profile in data["assets"].items():
        assert required.issubset(profile), ticker
        assert isinstance(profile["context_tickers"], list), ticker
        assert isinstance(profile["linked_indices"], list), ticker


def test_data_registry_has_current_assets_only():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    assert "aliases" not in data
    for ticker, profile in data["assets"].items():
        assert profile.get("registry_status") in {"active", "inactive"}, ticker
        if profile.get("registry_status") == "inactive":
            assert profile.get("use_in_reference_sample") is False, ticker
            assert profile.get("inactive_reason"), ticker
        assert "canonical_ticker" not in profile
        assert "predecessor_ticker" not in profile


def test_goll54_is_inactive_and_excluded_from_reference_sample():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    goll = data["assets"]["GOLL54.SA"]
    assert goll["registry_status"] == "inactive"
    assert goll["use_in_reference_sample"] is False
    assert goll["provider_status"] == "delisted_b3"
    assert "GOLL54.SA" not in resolve_tickers(load_config(), ["ALL"])
    assert resolve_tickers(load_config(), ["GOLL54.SA"]) == []


def test_key_assets_keep_specific_context():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    assert "BZ=F" in data["assets"]["PETR4.SA"]["context_tickers"]
    assert "BZ=F" not in data["assets"]["ITUB4.SA"]["context_tickers"]
    assert data["assets"]["ITUB4.SA"]["financial_class"] == "financial"
