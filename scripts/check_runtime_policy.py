from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_PATH = PROJECT_ROOT / "runtime" / "runtime_policy.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "runtime_policy.yaml"

REQUIRED_ASSET_FIELDS = (
    "profile",
    "source",
    "promoted",
    "promotion_status",
    "rejection_reasons",
    "overrides",
    "evidence",
    "selection",
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_count(data: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(data.get(key, fallback) or 0)
    except Exception:
        return fallback


def _asset_counts(assets: dict[str, Any]) -> tuple[int, int, int]:
    total = len(assets)
    promoted = sum(1 for item in assets.values() if bool((item or {}).get("promoted")))
    rejected = sum(1 for item in assets.values() if not bool((item or {}).get("promoted")))
    return total, promoted, rejected


def check_runtime_policy(
    runtime_path: Path | str = DEFAULT_RUNTIME_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[list[str], list[str], dict[str, Any]]:
    runtime_path = Path(runtime_path)
    config_path = Path(config_path)

    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "runtime_path": _display_path(runtime_path),
        "config_path": _display_path(config_path),
        "assets_total": 0,
        "assets_promoted": 0,
        "assets_rejected": 0,
    }

    if not config_path.exists():
        errors.append(f"{_display_path(config_path)} not found.")

    if not runtime_path.exists():
        errors.append(
            f"{_display_path(runtime_path)} not found. Run promote_policy before smart rank."
        )
        return errors, warnings, summary

    try:
        data = _load_json(runtime_path)
    except Exception as exc:
        errors.append(f"{_display_path(runtime_path)} is not valid JSON: {exc}")
        return errors, warnings, summary

    assets = data.get("assets")

    if not isinstance(assets, dict) or not assets:
        errors.append("runtime_policy has 0 assets. Check promote_policy output.")
        return errors, warnings, summary

    actual_total, actual_promoted, actual_rejected = _asset_counts(assets)
    declared_total = _declared_count(data, "assets_total", actual_total)
    declared_promoted = _declared_count(data, "assets_promoted", actual_promoted)
    declared_rejected = _declared_count(data, "assets_rejected", actual_rejected)

    summary.update(
        {
            "assets_total": actual_total,
            "assets_promoted": actual_promoted,
            "assets_rejected": actual_rejected,
            "declared_assets_total": declared_total,
            "declared_assets_promoted": declared_promoted,
            "declared_assets_rejected": declared_rejected,
        }
    )

    if declared_total != actual_total:
        errors.append(
            f"assets_total={declared_total} but assets map has {actual_total} entries."
        )

    if declared_promoted != actual_promoted:
        errors.append(
            f"assets_promoted={declared_promoted} but runtime has "
            f"{actual_promoted} promoted assets."
        )

    if declared_rejected != actual_rejected:
        errors.append(
            f"assets_rejected={declared_rejected} but runtime has "
            f"{actual_rejected} rejected assets."
        )

    if actual_promoted <= 0:
        errors.append("runtime_policy has 0 promoted assets. Check promotion constraints.")

    if actual_rejected <= 0:
        errors.append("runtime_policy has 0 rejected assets. Check promote_policy output.")

    for ticker, asset in sorted(assets.items()):
        if not isinstance(asset, dict):
            errors.append(f"{ticker}: asset entry must be an object.")
            continue

        for field in REQUIRED_ASSET_FIELDS:
            if field not in asset:
                errors.append(f"{ticker}: missing field {field}.")

        if asset.get("source") != "policy_matrix":
            errors.append(f"{ticker}: source must be policy_matrix.")

        promoted = asset.get("promoted")

        if not isinstance(promoted, bool):
            errors.append(f"{ticker}: promoted must be true or false.")
            continue

        promotion_status = str(asset.get("promotion_status", ""))

        if promoted and promotion_status != "promoted":
            errors.append(f"{ticker}: promoted asset must use promotion_status=promoted.")

        if not promoted and promotion_status != "rejected_by_constraints":
            errors.append(
                f"{ticker}: rejected asset must use promotion_status=rejected_by_constraints."
            )

        rejection_reasons = asset.get("rejection_reasons")

        if not isinstance(rejection_reasons, list):
            errors.append(f"{ticker}: rejection_reasons must be a list.")
        elif not promoted and not rejection_reasons:
            warnings.append(f"{ticker}: rejected asset has no rejection_reasons.")

        if not isinstance(asset.get("evidence"), dict):
            errors.append(f"{ticker}: evidence must be an object.")

        if not isinstance(asset.get("selection"), dict):
            errors.append(f"{ticker}: selection must be an object.")

        if not isinstance(asset.get("overrides"), dict):
            errors.append(f"{ticker}: overrides must be an object.")

    return errors, warnings, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TradeChat runtime/runtime_policy.json health."
    )
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors, warnings, summary = check_runtime_policy(args.runtime, args.config)

    print("RUNTIME POLICY CHECK")
    print("-" * 80)
    print(f"Runtime: {summary['runtime_path']}")
    print(f"Config: {summary['config_path']}")
    print(
        "Assets: "
        f"{summary.get('assets_total', 0)} | "
        f"Promoted: {summary.get('assets_promoted', 0)} | "
        f"Rejected: {summary.get('assets_rejected', 0)}"
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
