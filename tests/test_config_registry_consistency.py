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


def test_ibov_universe_does_not_duplicate_registry_metadata():
    universe = load_yaml("config/universes/ibov.yaml")
    components = universe.get("components") or universe.get("assets") or {}

    if not isinstance(components, dict):
        return

    forbidden_fields = {
        "name",
        "asset_type",
        "country",
        "currency",
        "group",
        "subgroup",
        "financial_class",
        "cnpj",
        "cnpj_status",
        "registry_status",
        "b3_name",
        "share_class",
        "b3_type_raw",
        "b3_listing_segment",
    }

    duplicated = []

    for ticker, meta in components.items():
        if not isinstance(meta, dict):
            continue

        repeated = sorted(forbidden_fields.intersection(meta.keys()))
        if repeated:
            duplicated.append({"ticker": ticker, "fields": repeated})

    assert not duplicated, f"Universe duplicates registry metadata: {duplicated}"


def test_context_asset_keys_exist_in_asset_registry():
    registry = load_yaml("config/assets/ibov_assets.yaml").get("assets", {})
    layers = load_yaml("config/context/layers.yaml")

    asset_contexts = layers.get("context", {}).get("assets", {}) or {}

    missing = [ticker for ticker in asset_contexts.keys() if ticker not in registry]

    assert not missing, f"Context asset keys missing from registry: {missing}"


def test_context_candidate_tickers_are_declared_or_market_symbols():
    indices = load_yaml("config/indices/catalog.yaml").get("indices", {})
    catalog = indices.get("catalog", {}) or indices.get("available_now", {}) or {}
    layers = load_yaml("config/context/layers.yaml")

    allowed_codes = set(catalog.keys())
    asset_contexts = layers.get("context", {}).get("assets", {}) or {}

    unresolved = []

    for ticker, meta in asset_contexts.items():
        for candidate in meta.get("candidate_context_tickers", []) or []:
            value = str(candidate)

            is_market_symbol = value.startswith("^") or value.endswith(".SA") or "=" in value

            if value not in allowed_codes and not is_market_symbol:
                unresolved.append({"asset": ticker, "candidate": value})

    assert not unresolved, f"Unresolved context candidates: {unresolved}"
