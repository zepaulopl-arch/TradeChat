from __future__ import annotations

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
