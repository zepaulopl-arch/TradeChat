from __future__ import annotations
from pathlib import Path

import argparse
from typing import Any

from ..config import load_data_registry
from ..pipeline_service import resolve_tickers
from ..utils import parse_tickers


def registry_list_tickers(cfg: dict[str, Any], list_name: str) -> list[str]:
    registry = load_data_registry(cfg)
    assets = registry.get("assets", {}) or {}
    name = str(list_name or "").strip().lower()
    if name in {"all", "todos"}:
        return [
            ticker
            for ticker, meta in assets.items()
            if isinstance(meta, dict) and meta.get("registry_status", "active") == "active"
        ]
    if name in {"validacao", "validation", "reference", "referencia"}:
        return [
            ticker
            for ticker, meta in assets.items()
            if isinstance(meta, dict)
            and meta.get("registry_status", "active") == "active"
            and bool(meta.get("use_in_reference_sample", False))
        ]
    raise SystemExit(f"unknown asset list: {list_name}")


def resolve_cli_tickers(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    attr: str = "tickers",
    required: bool = True,
) -> list[str]:
    list_name = getattr(args, "asset_list", None)
    if list_name:
        return registry_list_tickers(cfg, str(list_name))
    raw = list(getattr(args, attr, []) or [])
    if not raw:
        if required:
            raise SystemExit("at least one ticker or --list is required")
        return []
    return resolve_tickers(cfg, parse_tickers(raw))


# --- config registry universe support ---
def _component_to_ticker(key, value):
    """Return a ticker from a universe component entry."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for field in ("ticker", "yahoo_ticker", "asset", "symbol"):
            candidate = value.get(field)
            if candidate:
                return str(candidate)
    if key:
        return str(key)
    return None


def _tickers_from_universe(universe):
    """Resolve tickers from a split-config universe object."""
    if not universe:
        return []

    if isinstance(universe, list):
        resolved = []
        for item in universe:
            ticker = _component_to_ticker(None, item)
            if ticker:
                resolved.append(ticker)
        return resolved

    if isinstance(universe, dict):
        components = universe.get("components", universe)
        if isinstance(components, dict):
            resolved = []
            for key, value in components.items():
                ticker = _component_to_ticker(key, value)
                if ticker:
                    resolved.append(ticker)
            return resolved
        if isinstance(components, list):
            resolved = []
            for item in components:
                ticker = _component_to_ticker(None, item)
                if ticker:
                    resolved.append(ticker)
            return resolved

    return []


def registry_list_tickers(registry, list_name):
    """Resolve public asset lists/universes from the merged config registry.

    Supports legacy lists plus split-config universes such as `ibov`.
    """
    name = str(list_name or "all").strip()
    normalized = name.lower()

    assets = registry.get("assets") or {}

    if normalized in {"all", "todos", "*"}:
        return list(assets.keys())

    lists = registry.get("lists") or {}
    for key, value in lists.items():
        if str(key).lower() == normalized:
            if isinstance(value, dict):
                if "tickers" in value:
                    return list(value.get("tickers") or [])
                if "assets" in value:
                    return list(value.get("assets") or [])
                return _tickers_from_universe(value)
            return list(value or [])

    universes = registry.get("universes") or {}
    for key, value in universes.items():
        if str(key).lower() == normalized:
            return _tickers_from_universe(value)

    if normalized == "ibov":
        if registry.get("ibov_universe"):
            return list(registry.get("ibov_universe") or [])
        ibov = universes.get("ibov") or universes.get("IBOV")
        if ibov:
            return _tickers_from_universe(ibov)

    # Fallback for split config files while load_config migration is transitional.
    # This keeps validate matrix --universe ibov working even if the merged runtime
    # config has not exposed cfg["universes"] yet.
    config_dir = Path(str(registry.get("_config_dir") or "config"))
    universe_file = config_dir / "universes" / f"{normalized}.yaml"
    if universe_file.exists():
        import yaml

        payload = yaml.safe_load(universe_file.read_text(encoding="utf-8")) or {}
        tickers = _tickers_from_universe(payload)
        if tickers:
            return tickers

    raise SystemExit(f"unknown asset list: {list_name}")
