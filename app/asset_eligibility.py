from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import normalize_ticker

DEFAULT_AGGRESSIVE_PROFILES = ("relaxed",)
BLOCKING_STATUSES = {"blocked", "blocked_aggressive", "blocked_all"}
OBSERVE_STATUSES = {"observe", "eligible_candidate", "eligible"}
UNTESTED_STATUSES = {"untested", "untested_no_trades", "observe_insufficient"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def asset_eligibility_path(cfg: dict[str, Any]) -> Path:
    configured = (cfg.get("asset_eligibility") or {}).get("file")
    config_dir = Path(str(cfg.get("_config_dir") or "config"))
    path = Path(str(configured or "asset_eligibility.yaml"))
    if not path.is_absolute():
        path = config_dir / path
    return path


def load_asset_eligibility(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load operational asset/profile eligibility produced from matrix analysis.

    The file is optional. Missing config means permissive mode: signals are not
    blocked by eligibility. This keeps the transition safe.
    """
    path = asset_eligibility_path(cfg)
    payload = _load_yaml(path)
    if "asset_eligibility" in payload:
        return payload.get("asset_eligibility") or {}
    if "asset_eligibility_suggested" in payload:
        suggested = payload.get("asset_eligibility_suggested") or {}
        return {
            "default_status": suggested.get("default_status", "untested"),
            "generated_from": suggested.get("generated_from"),
            "note": suggested.get("note"),
            "assets": suggested.get("assets", {}) or {},
        }
    return payload or {}


def _asset_entry(config: dict[str, Any], ticker: str) -> dict[str, Any]:
    assets = config.get("assets", {}) or {}
    normalized = normalize_ticker(ticker)
    for key, value in assets.items():
        if normalize_ticker(str(key)) == normalized and isinstance(value, dict):
            return dict(value)
    return {}


def _profile_entry(entry: dict[str, Any], profile: str) -> dict[str, Any]:
    profiles = entry.get("profiles", {}) or {}
    for key, value in profiles.items():
        if str(key).lower() == str(profile).lower() and isinstance(value, dict):
            return dict(value)
    return {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def resolve_asset_eligibility(
    cfg: dict[str, Any],
    ticker: str,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Return the operational eligibility decision for ticker/profile.

    Status meanings are intentionally conservative:
    - observe / eligible: allowed, but displayed.
    - blocked_aggressive: blocks aggressive profiles such as relaxed.
    - blocked: blocks the profile.
    - untested/no-trades: displayed but not blocked by default.
    """
    config = load_asset_eligibility(cfg)
    active_profile = str(profile or "default").lower()
    aggressive_profiles = {
        str(item).lower()
        for item in _as_list(config.get("aggressive_profiles") or DEFAULT_AGGRESSIVE_PROFILES)
    }
    default_status = str(config.get("default_status", "untested")).lower()
    entry = _asset_entry(config, ticker)
    profile_entry = _profile_entry(entry, active_profile)

    status = str(
        profile_entry.get("status") or entry.get("status") or default_status or "untested"
    ).lower()
    best_profile = str(entry.get("best_profile") or profile_entry.get("best_profile") or "none")
    reason = str(
        profile_entry.get("reason")
        or entry.get("reason")
        or config.get("default_reason")
        or "no matrix eligibility evidence"
    )

    blocked_profiles = {str(item).lower() for item in _as_list(entry.get("blocked_profiles"))}
    allowed_profiles = {str(item).lower() for item in _as_list(entry.get("allowed_profiles"))}

    blocked = False
    blocked_reason = ""

    if status in BLOCKING_STATUSES:
        if status == "blocked_aggressive":
            blocked = active_profile in aggressive_profiles or active_profile in blocked_profiles
            blocked_reason = reason if blocked else ""
        else:
            blocked = True
            blocked_reason = reason

    if active_profile in blocked_profiles:
        blocked = True
        blocked_reason = reason

    if allowed_profiles and active_profile not in allowed_profiles:
        blocked = True
        blocked_reason = f"profile {active_profile} not allowed for asset"

    label = status
    if best_profile and best_profile != "none":
        label = f"{status}/{best_profile}"

    return {
        "ticker": normalize_ticker(ticker),
        "profile": active_profile,
        "status": status,
        "label": label,
        "best_profile": best_profile,
        "reason": reason,
        "blocked": bool(blocked),
        "blocker": blocked_reason or reason,
        "source": str(
            config.get("generated_from") or config.get("source") or "config/asset_eligibility.yaml"
        ),
    }


def apply_eligibility_to_signal(
    cfg: dict[str, Any],
    signal: dict[str, Any],
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    """Return a copy of a signal with eligibility metadata applied.

    Blocking is operational: the original predictions remain in horizons, but the
    public label is neutralized so rank/portfolio do not treat it as actionable.
    """
    out = dict(signal or {})
    ticker = str(out.get("ticker") or "")
    decision = resolve_asset_eligibility(cfg, ticker, profile=profile)
    out["eligibility"] = decision
    if decision.get("blocked"):
        policy = dict(out.get("policy", {}) or {})
        policy["original_label"] = policy.get("label", "NEUTRAL")
        policy["label"] = "NEUTRAL"
        policy["posture"] = "blocked_by_eligibility"
        policy["eligibility_blocked"] = True
        reasons = list(policy.get("reasons", []) or [])
        reasons.append(str(decision.get("blocker") or "eligibility blocked"))
        policy["reasons"] = reasons
        out["policy"] = policy
    return out
