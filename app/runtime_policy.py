from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .runtime_policy_config import (
    load_runtime_policy_config,
)

POLICY_PATH = Path("runtime/runtime_policy.json")


def load_runtime_policy() -> dict[str, Any]:
    if not POLICY_PATH.exists():
        return {}

    try:
        return json.loads(
            POLICY_PATH.read_text(
                encoding="utf-8",
            )
        )

    except Exception:
        return {}


def default_policy_profile() -> str:
    cfg = load_runtime_policy_config()

    promotion = cfg.get("promotion", {}) or {}

    return str(
        promotion.get(
            "fallback_profile",
            "active",
        )
    )


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def runtime_policy_overrides_for_profile(
    profile: str | None,
) -> dict[str, Any]:
    if not profile:
        return {}

    cfg = load_runtime_policy_config()

    promotion = cfg.get("promotion", {}) or {}

    runtime_overrides = promotion.get("runtime_overrides", {}) or {}

    if not bool(
        runtime_overrides.get(
            "enabled",
            False,
        )
    ):
        return {}

    profiles = runtime_overrides.get("profiles", {}) or {}

    overrides = profiles.get(str(profile), {}) or {}

    return deepcopy(overrides)


def runtime_decision_guard_config() -> dict[str, Any]:
    cfg = load_runtime_policy_config()

    promotion = cfg.get("promotion", {}) or {}

    guard = promotion.get("runtime_decision_guard", {}) or {}

    return deepcopy(guard)


def merge_runtime_overrides(
    stored_overrides: dict[str, Any] | None,
    live_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge live active defaults with stored asset-specific calibration.

    The YAML profile is the operational base. The Matrix/autotune runtime entry
    is ticker-specific and therefore has final precedence.
    """
    return _deep_merge(
        live_overrides or {},
        stored_overrides or {},
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return deepcopy(value)

    if isinstance(value, (tuple, set)):
        return list(value)

    return [value]


def _promoted_from_status(value: dict[str, Any], promotion_status: str) -> bool:
    if "promoted" in value:
        return bool(value.get("promoted"))

    return promotion_status.lower() not in {
        "rejected_by_constraints",
        "ineligible_data",
        "skip_data",
        "rejected",
        "no_matrix",
        "fallback",
    }


def _normalize_asset_selection(
    value: Any,
    fallback: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        profile = "active"

        promotion_status = str(
            value.get(
                "promotion_status",
                "promoted",
            )
        )

        return {
            "profile": profile,
            "policy_type": str(
                value.get(
                    "policy_type",
                    "asset_specific_active",
                )
            ),
            "source": str(
                value.get(
                    "source",
                    "runtime_policy",
                )
            ),
            "evaluated": bool(value.get("evaluated", True)),
            "ineligible_data": bool(value.get("ineligible_data", False)),
            "promoted": _promoted_from_status(
                value,
                promotion_status,
            ),
            "actionable_candidate": bool(
                value.get(
                    "actionable_candidate",
                    bool(value.get("promoted", False)),
                )
            ),
            "promotion_status": promotion_status,
            "rejection_reasons": _as_list(
                value.get(
                    "rejection_reasons",
                    [],
                )
            ),
            "blocker": deepcopy(value.get("blocker")),
            "overrides": deepcopy(value.get("overrides", {}) or {}),
            "evidence": deepcopy(value.get("evidence", {}) or {}),
            "selection": deepcopy(value.get("selection", {}) or {}),
        }

    return {
        "profile": "active",
        "policy_type": "asset_specific_active",
        "source": "fallback",
        "evaluated": False,
        "ineligible_data": False,
        "promoted": False,
        "actionable_candidate": False,
        "promotion_status": "fallback",
        "rejection_reasons": [],
        "blocker": None,
        "overrides": {},
        "evidence": {},
        "selection": {},
    }


def resolve_policy_selection(
    ticker: str,
    fallback: str | None = None,
) -> dict[str, Any]:
    data = load_runtime_policy()

    return resolve_policy_selection_from_data(
        data,
        ticker,
        fallback=fallback,
    )


def resolve_policy_selection_from_data(
    data: dict[str, Any],
    ticker: str,
    fallback: str | None = None,
) -> dict[str, Any]:

    assets = data.get("assets", {}) or {}

    if fallback is None:
        fallback = default_policy_profile()

    raw_value = assets.get(str(ticker))

    return _normalize_asset_selection(
        raw_value,
        fallback,
    )


def resolve_policy_profile(
    ticker: str,
    fallback: str | None = None,
) -> str:
    selection = resolve_policy_selection(
        ticker=ticker,
        fallback=fallback,
    )

    return str(
        selection.get(
            "profile",
            fallback or default_policy_profile(),
        )
    )


def apply_runtime_policy_overrides(
    cfg: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    overrides = overrides or {}

    if not overrides:
        return deepcopy(cfg)

    new_cfg = deepcopy(cfg)

    current_policy = new_cfg.get("policy", {}) or {}

    new_cfg["policy"] = _deep_merge(
        current_policy,
        overrides,
    )

    return new_cfg


def _current_signal_label(
    signal: dict[str, Any],
) -> str:
    policy = signal.get("policy", {}) or {}

    return str(
        policy.get(
            "label",
            signal.get("label", "NEUTRAL"),
        )
    ).upper()


def _previous_blocked_signal(
    signal: dict[str, Any],
) -> str:
    policy = signal.get("policy", {}) or {}

    return str(
        policy.get(
            "blocked_signal",
            "",
        )
        or ""
    ).upper()


def _append_policy_reason(
    policy: dict[str, Any],
    reason: str,
) -> None:
    reasons = list(policy.get("reasons", []) or [])

    if reason not in reasons:
        reasons.append(reason)

    policy["reasons"] = reasons


def _is_actionable_label(
    label: str,
) -> bool:
    label = str(label).upper()

    return label not in {
        "",
        "NEUTRAL",
        "WAIT",
        "WATCH",
        "AVOID",
    }


def apply_matrix_decision_guard(
    signal: dict[str, Any],
    evidence: dict[str, Any] | None,
    guard_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    result = deepcopy(signal)

    guard_cfg = guard_cfg or {}

    if not bool(
        guard_cfg.get(
            "enabled",
            False,
        )
    ):
        return result

    evidence = evidence or {}

    matrix_decision = str(
        evidence.get(
            "decision",
            "",
        )
        or ""
    ).upper()

    if not matrix_decision:
        return result

    decisions_cfg = (
        guard_cfg.get(
            "decisions",
            {},
        )
        or {}
    )

    decision_rule = (
        decisions_cfg.get(
            matrix_decision,
            {},
        )
        or {}
    )

    max_signal = str(
        decision_rule.get(
            "max_signal",
            "ACTIONABLE",
        )
        or "ACTIONABLE"
    ).upper()

    current_label = _current_signal_label(result)

    previous_blocked_signal = _previous_blocked_signal(result)

    candidate_label = (
        current_label if _is_actionable_label(current_label) else previous_blocked_signal
    )

    should_block = max_signal in {
        "NEUTRAL",
        "WAIT",
        "WATCH",
        "AVOID",
    } and _is_actionable_label(candidate_label)

    if not should_block:
        result["matrix_decision_guard"] = {
            "enabled": True,
            "blocked": False,
            "matrix_decision": matrix_decision,
            "max_signal": max_signal,
            "current_label": current_label,
            "previous_blocked_signal": previous_blocked_signal,
        }

        return result

    policy = result.get("policy", {}) or {}

    blocked_label = candidate_label

    reason = str(
        decision_rule.get(
            "reason",
            f"{blocked_label} blocked by Matrix decision guard: {matrix_decision}",
        )
    )

    policy["blocked_signal"] = blocked_label

    if _is_actionable_label(current_label):
        policy["label"] = "NEUTRAL"
        policy["posture"] = str(
            decision_rule.get(
                "posture",
                "wait_matrix_guard",
            )
        )

        if "label" in result:
            result["label"] = "NEUTRAL"

    policy["actionable"] = False
    policy["position_size"] = 0
    policy["risk_reward_ratio"] = 0.0

    _append_policy_reason(
        policy,
        reason,
    )

    result["policy"] = policy

    if "actionable" in result:
        result["actionable"] = False

    result["matrix_decision_guard"] = {
        "enabled": True,
        "blocked": True,
        "matrix_decision": matrix_decision,
        "max_signal": max_signal,
        "current_label": current_label,
        "previous_blocked_signal": previous_blocked_signal,
        "blocked_signal": blocked_label,
        "reason": reason,
    }

    return result
