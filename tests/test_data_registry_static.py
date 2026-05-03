from pathlib import Path
import yaml

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
    required = {"name", "group", "subgroup", "financial_class", "cnpj", "context_tickers", "linked_indices"}
    for ticker, profile in data["assets"].items():
        assert required.issubset(profile), ticker
        assert isinstance(profile["context_tickers"], list), ticker
        assert isinstance(profile["linked_indices"], list), ticker


def test_key_assets_keep_specific_context():
    data = yaml.safe_load((ROOT / "config" / "data.yaml").read_text(encoding="utf-8"))
    assert "BZ=F" in data["assets"]["PETR4.SA"]["context_tickers"]
    assert "BZ=F" not in data["assets"]["ITUB4.SA"]["context_tickers"]
    assert data["assets"]["ITUB4.SA"]["financial_class"] == "financial"
