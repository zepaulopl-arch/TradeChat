from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SPLIT_DIRECTORIES = ("assets", "universes", "indices", "context", "sources", "defaults")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"yaml file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.yml") if path.is_file()) + sorted(
        path for path in directory.glob("*.yaml") if path.is_file()
    )


def split_registry_available(config_dir: str | Path) -> bool:
    """Return True when at least one split registry YAML exists.

    The split layout is optional during the transition. The legacy data.yaml file
    remains valid and is used as fallback when no split registry files exist.
    """
    root = Path(config_dir)
    return any(_yaml_files(root / name) for name in _SPLIT_DIRECTORIES)


def _load_split_assets(config_dir: Path) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "assets"):
        doc = _load_yaml(path)
        assets = _merge_dicts(assets, doc.get("assets", {}) or {})
    return assets


def _load_split_indices(config_dir: Path) -> dict[str, Any]:
    indices: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "indices"):
        doc = _load_yaml(path)
        indices = _merge_dicts(indices, doc.get("indices", {}) or {})
    return indices


def _load_split_context(config_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    context: dict[str, Any] = {}
    asset_overrides: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "context"):
        doc = _load_yaml(path)
        context = _merge_dicts(context, doc.get("context", {}) or {})
        asset_overrides = _merge_dicts(
            asset_overrides,
            doc.get("asset_context_overrides", {}) or {},
        )
    return context, asset_overrides


def _load_split_sources(config_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources: dict[str, Any] = {}
    enrichment_metadata: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "sources"):
        doc = _load_yaml(path)
        sources = _merge_dicts(sources, doc.get("sources", {}) or {})
        enrichment_metadata = _merge_dicts(
            enrichment_metadata,
            doc.get("enrichment_metadata", {}) or {},
        )
    return sources, enrichment_metadata


def _load_split_defaults(config_dir: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "defaults"):
        doc = _load_yaml(path)
        defaults = _merge_dicts(defaults, doc.get("defaults", {}) or {})
    return defaults


def _load_split_universes(config_dir: Path, assets: dict[str, Any]) -> dict[str, Any]:
    universes: dict[str, Any] = {}
    for path in _yaml_files(config_dir / "universes"):
        doc = _load_yaml(path)
        name = str(doc.get("universe") or path.stem).lower()
        components = doc.get("components", {}) or {}
        tickers = list(components.keys())
        universes[name] = {
            "name": name,
            "metadata": doc.get("metadata", {}) or {},
            "asset_count": int(doc.get("asset_count") or len(tickers)),
            "tickers": tickers,
            "components": components,
            "source_file": str(path.relative_to(config_dir)),
        }

        for ticker, component in components.items():
            profile = dict(assets.get(ticker, {}) or {})
            profile.setdefault("asset_type", "stock")
            profile.setdefault("country", "BR")
            profile.setdefault("currency", "BRL")
            profile["registry_status"] = profile.get("registry_status", "active")
            profile[f"{name}_component"] = True
            if name.upper() not in [
                str(item).upper() for item in profile.get("linked_indices", []) or []
            ]:
                profile["linked_indices"] = list(profile.get("linked_indices", []) or []) + [
                    name.upper()
                ]
            if component.get("use_in_reference_sample") is not None:
                profile["use_in_reference_sample"] = bool(component.get("use_in_reference_sample"))
            for key in (
                "composition_date",
                "source_file",
                "theoretical_quantity",
                "weight_pct",
            ):
                if key in component and key not in profile:
                    profile[key] = component[key]
            if "composition_date" in component or "weight_pct" in component:
                profile[name] = {
                    "composition_date": component.get("composition_date"),
                    "source_file": component.get("source_file"),
                    "theoretical_quantity": component.get("theoretical_quantity"),
                    "weight_pct": component.get("weight_pct"),
                }
            assets[ticker] = profile
    return universes


def _apply_asset_context_overrides(
    assets: dict[str, Any],
    asset_context_overrides: dict[str, Any],
) -> None:
    for ticker, override in (asset_context_overrides or {}).items():
        if not isinstance(override, dict):
            continue
        candidates = (
            override.get("candidate_context_tickers")
            or override.get("context_tickers")
            or override.get("tickers")
        )
        if not candidates:
            continue
        profile = dict(assets.get(ticker, {}) or {})
        # Split context files store candidate contexts. Runtime keeps the legacy
        # key context_tickers; context_policy still decides whether each series
        # passes coverage before becoming a feature.
        profile["context_tickers"] = list(candidates)
        assets[ticker] = profile


def _build_lists(registry: dict[str, Any]) -> dict[str, list[str]]:
    assets = registry.get("assets", {}) or {}
    active = [
        ticker
        for ticker, meta in assets.items()
        if isinstance(meta, dict) and str(meta.get("registry_status", "active")).lower() == "active"
    ]
    reference = [
        ticker
        for ticker, meta in assets.items()
        if isinstance(meta, dict)
        and str(meta.get("registry_status", "active")).lower() == "active"
        and bool(meta.get("use_in_reference_sample", False))
    ]

    lists = dict(registry.get("lists", {}) or {})
    lists.setdefault("all", active)
    lists.setdefault("validacao", reference or active)

    for name, universe in (registry.get("universes", {}) or {}).items():
        tickers = [
            ticker
            for ticker in universe.get("tickers", []) or []
            if ticker in assets
            and str((assets.get(ticker, {}) or {}).get("registry_status", "active")).lower()
            == "active"
        ]
        lists.setdefault(str(name).lower(), tickers)
        lists.setdefault(str(name).upper(), tickers)
    return lists


def load_config_registry(
    config_dir: str | Path,
    *,
    registry_path: str | Path | None = None,
    prefer_split: bool = True,
) -> dict[str, Any]:
    """Load the asset/data registry with split-config support.

    When split files exist, they are assembled into the same runtime shape used
    by the legacy data.yaml registry. If split files do not exist, the legacy
    file is loaded directly. When both exist, legacy data is used as a base and
    split files override/add the more granular registry content.

    This keeps the transition safe: existing code can continue reading
    registry["assets"], registry["indices"], registry["context"],
    registry["sources"] and registry["defaults"].
    """
    root = Path(config_dir)
    legacy_path = Path(registry_path) if registry_path else root / "data.yaml"
    if not legacy_path.is_absolute():
        legacy_path = root / legacy_path

    has_split = split_registry_available(root)
    if not prefer_split or not has_split:
        return _load_yaml(legacy_path)

    legacy = _load_yaml(legacy_path) if legacy_path.exists() else {}
    registry: dict[str, Any] = dict(legacy or {})
    registry["version"] = registry.get("version", 1)
    registry["description"] = registry.get(
        "description",
        "TradeChat split asset registry assembled at runtime.",
    )

    assets = _merge_dicts(registry.get("assets", {}) or {}, _load_split_assets(root))
    context, asset_context_overrides = _load_split_context(root)
    universes = _load_split_universes(root, assets)
    _apply_asset_context_overrides(assets, asset_context_overrides)

    sources, enrichment_metadata = _load_split_sources(root)

    registry["assets"] = assets
    registry["indices"] = _merge_dicts(registry.get("indices", {}) or {}, _load_split_indices(root))
    registry["context"] = _merge_dicts(registry.get("context", {}) or {}, context)
    registry["sources"] = _merge_dicts(registry.get("sources", {}) or {}, sources)
    registry["defaults"] = _merge_dicts(
        registry.get("defaults", {}) or {}, _load_split_defaults(root)
    )
    registry["universes"] = _merge_dicts(registry.get("universes", {}) or {}, universes)
    if enrichment_metadata:
        registry["enrichment_metadata"] = _merge_dicts(
            registry.get("enrichment_metadata", {}) or {},
            enrichment_metadata,
        )

    # Compatibility keys used by earlier IBOV data files.
    if "ibov" in registry["universes"]:
        registry["ibov_metadata"] = registry["universes"]["ibov"].get("metadata", {})
        registry["ibov_universe"] = registry["universes"]["ibov"].get("tickers", [])

    registry["lists"] = _build_lists(registry)
    registry["_registry_source"] = "split"
    registry["_registry_config_dir"] = str(root)
    return registry
