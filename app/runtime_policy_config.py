from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(
    "config/runtime_policy.yaml"
)


def load_runtime_policy_config() -> dict:

    if not CONFIG_PATH.exists():
        return {}

    try:

        return yaml.safe_load(
            CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        ) or {}

    except Exception:

        return {}
