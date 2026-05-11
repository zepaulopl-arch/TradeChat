from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_ibov_universe_assets_exist_in_asset_registry():
    registry = load_yaml("config/assets/ibov_assets.yaml").get("assets", {})
    universe = load_yaml("config/universes/ibov.yaml")

    components = universe.get("components") or universe.get("assets") or {}
    tickers = components.keys() if isinstance(components, dict) else components

    missing = [ticker for ticker in tickers if ticker not in registry]

    assert not missing, f"Universe assets missing from registry: {missing}"


def test_ibov_universe_does_not_disagree_with_registry_core_fields():
    registry = load_yaml("config/assets/ibov_assets.yaml").get("assets", {})
    universe = load_yaml("config/universes/ibov.yaml")

    components = universe.get("components") or universe.get("assets") or {}

    if not isinstance(components, dict):
        return

    checked_fields = ("ticker", "b3_code", "yahoo_ticker", "cnpj")

    disagreements = []

    for ticker, universe_meta in components.items():
        registry_meta = registry.get(ticker, {})
        if not isinstance(universe_meta, dict):
            continue

        for field in checked_fields:
            universe_value = universe_meta.get(field)
            registry_value = registry_meta.get(field)

            if universe_value is not None and registry_value is not None:
                if str(universe_value) != str(registry_value):
                    disagreements.append(
                        {
                            "ticker": ticker,
                            "field": field,
                            "universe": universe_value,
                            "registry": registry_value,
                        }
                    )

    assert not disagreements, f"Universe/registry disagreements: {disagreements}"
