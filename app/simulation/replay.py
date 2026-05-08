from __future__ import annotations


def normalize_validation_mode(mode: str | None) -> str:
    normalized = str(mode or "replay").strip().lower()
    if normalized not in {"replay", "walkforward"}:
        raise ValueError("simulation mode must be 'replay' or 'walkforward'")
    return normalized
