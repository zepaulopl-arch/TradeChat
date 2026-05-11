from __future__ import annotations

from typing import Any


def asset_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("assets_registry") or {}).get("assets") or {}


def get_asset_meta(cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    registry = asset_registry(cfg)
    return registry.get(str(ticker), {}) or {}


def canonical_asset_ticker(cfg: dict[str, Any], ticker: str) -> str:
    raw = str(ticker).strip()
    raw_upper = raw.upper()

    registry = asset_registry(cfg)

    if raw in registry:
        return raw

    for canonical, meta in registry.items():
        canonical_upper = str(canonical).upper()

        if raw_upper == canonical_upper:
            return canonical

        aliases = meta.get("aliases") or []

        alias_set = {str(alias).upper() for alias in aliases}

        if raw_upper in alias_set:
            return canonical

        lineage = meta.get("lineage") or {}

        predecessors = lineage.get("predecessors") or []

        for predecessor in predecessors:
            predecessor_ticker = predecessor.get("ticker")

            if predecessor_ticker and (raw_upper == str(predecessor_ticker).upper()):
                return canonical

    return raw
