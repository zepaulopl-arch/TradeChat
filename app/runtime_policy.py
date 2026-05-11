from __future__ import annotations

import json
from pathlib import Path

from .runtime_policy_config import (
    load_runtime_policy_config,
)


POLICY_PATH = Path(
    "runtime/runtime_policy.json"
)


def load_runtime_policy() -> dict:

    if not POLICY_PATH.exists():
        return {}

    try:

        return json.loads(
            POLICY_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return {}


def default_policy_profile() -> str:

    cfg = load_runtime_policy_config()

    promotion = (
        cfg.get("promotion", {})
        or {}
    )

    return str(
        promotion.get(
            "fallback_profile",
            "balanced",
        )
    )


def resolve_policy_profile(
    ticker: str,
    fallback: str | None = None,
) -> str:

    data = load_runtime_policy()

    assets = (
        data.get("assets", {})
        or {}
    )

    if fallback is None:
        fallback = default_policy_profile()

    return str(
        assets.get(
            str(ticker),
            fallback,
        )
    )
