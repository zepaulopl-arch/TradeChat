from __future__ import annotations

from typing import Any


TARGET_PREFIXES = ("target_",)
OPERATIONAL_MODEL_MARKER = "artifacts/models"
REFINE_SHADOW_MARKER = "artifacts/refine"


def check_no_target_features(features: list[str]) -> dict[str, Any]:
    leaked = [str(feature) for feature in features if str(feature).startswith(TARGET_PREFIXES)]
    return {
        "check": "no_target_features",
        "passed": not leaked,
        "details": {"leaked_features": leaked},
    }


def check_temporal_split(split_meta: dict[str, Any], *, horizon: str) -> dict[str, Any]:
    expected_gap = _horizon_bars(horizon)
    gap = int(split_meta.get("embargo_bars", 0) or 0)
    train_end = int(split_meta.get("train_end_index", 0) or 0)
    split_index = int(split_meta.get("split_index", 0) or 0)
    dropped = int(split_meta.get("dropped_embargo_rows", 0) or 0)
    passed = gap >= expected_gap and split_index >= train_end and dropped >= expected_gap
    return {
        "check": "temporal_split_embargo",
        "passed": bool(passed),
        "details": {
            "horizon": horizon,
            "expected_min_gap": expected_gap,
            "embargo_bars": gap,
            "train_end_index": train_end,
            "split_index": split_index,
            "dropped_embargo_rows": dropped,
        },
    }


def check_train_only_feature_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    prep = manifest.get("preparation", {}) or {}
    split = manifest.get("validation_split", {}) or {}
    selection = prep.get("selection", {}) or {}
    selected_count = int(prep.get("selected_feature_count", len(manifest.get("features", []) or [])) or 0)
    train_rows = int(manifest.get("train_rows", 0) or 0)
    test_rows = int(manifest.get("test_rows", 0) or 0)
    passed = bool(selection) and selected_count > 0 and train_rows > 0 and test_rows > 0 and split.get("test_target") == "raw_unclipped"
    return {
        "check": "train_only_feature_selection",
        "passed": bool(passed),
        "details": {
            "selected_feature_count": selected_count,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "test_target": split.get("test_target"),
            "selection_method": selection.get("method"),
        },
    }


def check_validation_has_baselines(summary: dict[str, Any]) -> dict[str, Any]:
    baselines = summary.get("baselines", {}) or {}
    comparison = summary.get("baseline_comparison", {}) or {}
    required = {"zero_return_no_trade", "buy_and_hold_equal_weight", "last_return_long_flat"}
    missing = sorted(required.difference(set(baselines)))
    passed = not missing and bool(comparison.get("rows"))
    return {
        "check": "validation_baselines_present",
        "passed": bool(passed),
        "details": {
            "missing": missing,
            "baseline_count": len(baselines),
            "comparison_rows": len(comparison.get("rows", []) or []),
        },
    }


def check_refine_removal_uses_shadow_artifacts(summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(summary.get("rows", []) or [])
    bad_paths = []
    for row in rows:
        path = str(row.get("artifact_dir", "")).replace("\\", "/")
        if REFINE_SHADOW_MARKER not in path or OPERATIONAL_MODEL_MARKER in path:
            bad_paths.append(path)
    artifacts = summary.get("artifacts", {}) or {}
    result_file = str(artifacts.get("results_csv", ""))
    passed = bool(rows) and not bad_paths and result_file.endswith("removal_results.csv")
    return {
        "check": "refine_removal_shadow_artifacts",
        "passed": bool(passed),
        "details": {
            "row_count": len(rows),
            "bad_paths": bad_paths,
            "results_csv": result_file,
        },
    }


def methodology_report(
    *,
    manifest: dict[str, Any] | None = None,
    validation_summary: dict[str, Any] | None = None,
    removal_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if manifest is not None:
        checks.append(check_no_target_features([str(item) for item in manifest.get("features", []) or []]))
        checks.append(check_temporal_split(manifest.get("validation_split", {}) or {}, horizon=str(manifest.get("horizon", "d1"))))
        checks.append(check_train_only_feature_selection(manifest))
    if validation_summary is not None:
        checks.append(check_validation_has_baselines(validation_summary))
    if removal_summary is not None:
        checks.append(check_refine_removal_uses_shadow_artifacts(removal_summary))
    failed = [check for check in checks if not bool(check.get("passed", False))]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def _horizon_bars(horizon: str) -> int:
    value = str(horizon or "d1").strip().lower()
    if value.startswith("d"):
        value = value[1:]
    try:
        return max(1, int(value))
    except Exception:
        return 1
