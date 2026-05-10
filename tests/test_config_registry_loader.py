from pathlib import Path

import yaml

from app.config import load_config, load_data_registry
from app.config_registry import load_config_registry, split_registry_available


def test_split_config_registry_assembles_current_ibov_files():
    cfg = load_config()
    registry = load_data_registry(cfg)

    assert registry["_registry_source"] == "split"
    assert "assets" in registry
    assert "indices" in registry
    assert "context" in registry
    assert "sources" in registry
    assert "defaults" in registry
    assert "universes" in registry

    assert "ibov" in registry["universes"]
    assert len(registry["universes"]["ibov"]["tickers"]) == 79
    assert len(registry["ibov_universe"]) == 79
    assert "ibov" in registry["lists"]
    assert len(registry["lists"]["ibov"]) == 79

    alos = registry["assets"]["ALOS3.SA"]
    assert alos["cnpj"] == "05.878.397/0001-32"
    assert alos["context_tickers"] == ["^BVSP", "USDBRL=X"]
    assert alos["ibov_component"] is True
    assert alos["ibov"]["weight_pct"] == 0.567


def test_explicit_registry_path_preserves_legacy_file_loading(tmp_path):
    legacy = tmp_path / "data.yaml"
    legacy.write_text(
        yaml.dump(
            {
                "version": 1,
                "assets": {
                    "TEST3.SA": {
                        "name": "TEST",
                        "registry_status": "active",
                        "context_tickers": ["^BVSP"],
                    }
                },
                "indices": {"default_context": ["^BVSP"]},
                "context": {"global": ["^BVSP"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = load_config_registry(tmp_path, registry_path=legacy, prefer_split=True)

    assert split_registry_available(tmp_path) is False
    assert list(registry["assets"]) == ["TEST3.SA"]
    assert "universes" not in registry


def test_load_data_registry_path_argument_uses_exact_file(tmp_path):
    data = tmp_path / "custom.yaml"
    data.write_text(
        yaml.dump(
            {
                "version": 1,
                "assets": {"ONLY3.SA": {"name": "ONLY", "registry_status": "active"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = load_data_registry(path=data)

    assert list(registry["assets"]) == ["ONLY3.SA"]


def test_registry_list_tickers_resolves_split_ibov_universe():
    from app.config import load_config
    from app.commands._shared import registry_list_tickers

    cfg = load_config()
    tickers = list(registry_list_tickers(cfg, "ibov"))

    assert tickers
    assert any(t.endswith(".SA") for t in tickers)
