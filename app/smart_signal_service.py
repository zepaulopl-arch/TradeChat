from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .pipeline_service import latest_signal_path, make_signal
from .policy import apply_policy_profile
from .runtime_policy import (
    apply_matrix_decision_guard,
    apply_runtime_policy_overrides,
    merge_runtime_overrides,
    resolve_policy_selection,
    runtime_decision_guard_config,
    runtime_policy_overrides_for_profile,
)
from .utils import read_json


@dataclass(frozen=True)
class SmartSignalDependencies:
    resolve_policy_selection: Callable[..., dict[str, Any]]
    apply_policy_profile: Callable[..., dict[str, Any]]
    runtime_policy_overrides_for_profile: Callable[[str | None], dict[str, Any]]
    merge_runtime_overrides: Callable[..., dict[str, Any]]
    apply_runtime_policy_overrides: Callable[..., dict[str, Any]]
    make_signal: Callable[..., dict[str, Any]]
    runtime_decision_guard_config: Callable[[], dict[str, Any]]
    apply_matrix_decision_guard: Callable[..., dict[str, Any]]
    latest_signal_path: Callable[..., Path]


def default_smart_signal_dependencies() -> SmartSignalDependencies:
    return SmartSignalDependencies(
        resolve_policy_selection=resolve_policy_selection,
        apply_policy_profile=apply_policy_profile,
        runtime_policy_overrides_for_profile=runtime_policy_overrides_for_profile,
        merge_runtime_overrides=merge_runtime_overrides,
        apply_runtime_policy_overrides=apply_runtime_policy_overrides,
        make_signal=make_signal,
        runtime_decision_guard_config=runtime_decision_guard_config,
        apply_matrix_decision_guard=apply_matrix_decision_guard,
        latest_signal_path=latest_signal_path,
    )


def smart_signal_path(
    cfg: dict,
    ticker: str,
    *,
    deps: SmartSignalDependencies | None = None,
) -> Path:
    deps = deps or default_smart_signal_dependencies()
    base_path = deps.latest_signal_path(cfg, ticker)
    return base_path.with_name("latest_smart_signal.json")


def write_smart_signal(
    cfg: dict,
    ticker: str,
    signal: dict[str, Any],
    *,
    deps: SmartSignalDependencies | None = None,
) -> Path:
    path = smart_signal_path(
        cfg,
        ticker,
        deps=deps,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(signal, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


def build_smart_signal(
    cfg: dict,
    args: argparse.Namespace,
    ticker: str,
    *,
    deps: SmartSignalDependencies | None = None,
    prefer_latest: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    deps = deps or default_smart_signal_dependencies()
    if prefer_latest is None:
        prefer_latest = bool(getattr(args, "_prefer_latest_signal", False))

    selection = deps.resolve_policy_selection(
        str(ticker),
        fallback=getattr(args, "policy_profile", None),
    )

    policy_type = str(selection.get("policy_type", "asset_specific_active"))
    profile = "active" if policy_type == "asset_specific_active" else str(selection.get("profile", ""))

    if not profile:
        raise SystemExit(f"no runtime policy profile available for {ticker}")

    smart_cfg = deps.apply_policy_profile(cfg, profile)

    stored_overrides = selection.get("overrides", {}) or {}
    live_overrides = deps.runtime_policy_overrides_for_profile(profile)

    overrides = deps.merge_runtime_overrides(
        stored_overrides,
        live_overrides,
    )

    smart_cfg = deps.apply_runtime_policy_overrides(
        smart_cfg,
        overrides,
    )

    evidence = selection.get("evidence", {}) or {}

    latest_path = deps.latest_signal_path(
        smart_cfg,
        ticker,
    )

    if prefer_latest and latest_path.exists():
        raw_signal = read_json(latest_path)
    else:
        raw_signal = deps.make_signal(
            smart_cfg,
            ticker,
            update=bool(getattr(args, "update", False)),
        )

    signal = deps.apply_matrix_decision_guard(
        raw_signal,
        evidence=evidence,
        guard_cfg=deps.runtime_decision_guard_config(),
    )

    signal["smart_signal"] = {
        "enabled": True,
        "source": selection.get("source", "runtime_policy"),
        "profile": profile,
        "policy_type": policy_type,
        "evaluated": selection.get("evaluated", True),
        "ineligible_data": selection.get("ineligible_data", False),
        "promoted": selection.get("promoted", True),
        "actionable_candidate": selection.get("actionable_candidate", True),
        "promotion_status": selection.get("promotion_status", "promoted"),
        "rejection_reasons": selection.get("rejection_reasons", []) or [],
        "blocker": selection.get("blocker"),
        "overrides": overrides,
        "stored_overrides": stored_overrides,
        "live_overrides": live_overrides,
        "evidence": evidence,
        "selection": selection.get("selection", {}),
        "matrix_decision_guard": signal.get("matrix_decision_guard", {}),
    }

    out_path = write_smart_signal(
        smart_cfg,
        ticker,
        signal,
        deps=deps,
    )

    return signal, smart_cfg, out_path
