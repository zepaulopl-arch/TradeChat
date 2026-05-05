from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
DATA_REGISTRY_PATH = CONFIG_DIR / "data.yaml"
FEATURES_CONFIG_PATH = CONFIG_DIR / "features.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"yaml file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _legacy_family_block(family: str, preset_cfg: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Adapt the organized features.yaml schema to the runtime keys used by code.

    The public schema is family-first: each family owns windows, attributes and
    defaults. This adapter preserves the existing pipeline, CLI and model code.
    """
    preset_cfg = dict(preset_cfg or {})
    block: dict[str, Any] = {"enabled": bool(enabled)}
    windows = preset_cfg.get("windows", {}) or {}
    features = preset_cfg.get("features", {}) or {}

    if family == "technical":
        block.update({
            "legacy_macro_features": bool(features.get("legacy_macro_features", False)),
            "rsi_window": int(windows.get("rsi", 14)),
            "sma_short": int((windows.get("sma", {}) or {}).get("short", 10)),
            "sma_long": int((windows.get("sma", {}) or {}).get("long", 50)),
            "ema_short": int((windows.get("ema", {}) or {}).get("short", 20)),
            "ema_long": int((windows.get("ema", {}) or {}).get("long", 100)),
            "roc_window": int(windows.get("roc", 10)),
            "vol_window": int((windows.get("volatility", [20]) or [20])[0]),
            "frac_diff_d": float((windows.get("fractional_memory", {}) or {}).get("d", 0.5)),
            "windows": windows,
            "features": features,
        })
    elif family == "context":
        beta_windows = windows.get("beta", []) or []
        all_windows: list[int] = []
        for key in ("returns", "volatility", "correlation"):
            for w in windows.get(key, []) or []:
                if int(w) not in all_windows:
                    all_windows.append(int(w))
        for w in beta_windows:
            if int(w) not in all_windows:
                all_windows.append(int(w))
        rel = windows.get("relative_strength", {}) or {}
        align = windows.get("alignment", {}) or {}
        block.update({
            "benchmark": preset_cfg.get("benchmark", "^BVSP"),
            "source": preset_cfg.get("source", "data_yaml_asset_context"),
            "use_asset_registry": bool(preset_cfg.get("use_asset_registry", True)),
            "windows": sorted(all_windows) or [5, 20, 60],
            "return_windows": [int(w) for w in windows.get("returns", []) or []],
            "volatility_windows": [int(w) for w in windows.get("volatility", []) or []],
            "correlation_windows": [int(w) for w in windows.get("correlation", []) or []],
            "beta_windows": [int(w) for w in beta_windows],
            "alignment_short_window": int(align.get("short", 5)),
            "alignment_long_window": int(align.get("long", 20)),
            "relative_strength_short_window": int(rel.get("short", 5)),
            "relative_strength_long_window": int(rel.get("long", 20)),
            "add_alignment_features": bool(features.get("alignment", True)),
            "use_returns": bool(features.get("returns", True)),
            "use_volatility": bool(features.get("volatility", True)),
            "use_correlation": bool(features.get("correlation", True)),
            "use_beta": bool(features.get("beta", True)),
            "use_relative_strength": bool(features.get("relative_strength", True)),
            "use_alignment": bool(features.get("alignment", True)),
            "feature_windows": windows,
            "features": features,
        })
    elif family == "fundamentals":
        thresholds = preset_cfg.get("thresholds", {}) or {}
        weights = preset_cfg.get("weights", {}) or {}
        safety = preset_cfg.get("safety", {}) or {}
        block.update({
            "source_priority": preset_cfg.get("source_priority", []),
            "windows": windows,
            "features": features,
            "add_regime_features": bool(features.get("regime_score", True)),
            "cheap_pl": float(thresholds.get("cheap_pl", 10.0)),
            "expensive_pl": float(thresholds.get("expensive_pl", 22.0)),
            "weak_roe": float(thresholds.get("weak_roe", 0.04)),
            "good_roe": float(thresholds.get("good_roe", 0.12)),
            "good_dy": float(thresholds.get("good_dy", 0.06)),
            "value_weight": float(weights.get("value", 0.35)),
            "quality_weight": float(weights.get("quality", 0.45)),
            "yield_weight": float(weights.get("yield", 0.20)),
            "require_historical": bool(safety.get("require_historical", True)),
            "use_snapshot_as_features": bool(safety.get("use_snapshot_as_features", False)),
            "snapshot_only_in_report": bool(safety.get("snapshot_only_in_report", True)),
        })
    elif family == "sentiment":
        collection = preset_cfg.get("collection", {}) or {}
        flat_windows: list[int] = []
        for key in ("mean", "count", "delta", "std"):
            for w in windows.get(key, []) or []:
                if int(w) not in flat_windows:
                    flat_windows.append(int(w))
        block.update({
            "mode": preset_cfg.get("mode", "temporal_feature"),
            "provider": preset_cfg.get("provider", "rss_vader"),
            "windows": sorted(flat_windows) or [1, 3, 7],
            "window_groups": windows,
            "features": features,
            "max_news_entries": int(collection.get("max_news_entries", 20)),
            "cache_days": int(collection.get("cache_days", 365)),
            "min_items_for_feature": int(collection.get("min_items_for_feature", 1)),
            "fallback_to_zero": bool(collection.get("fallback_to_zero", True)),
            "vader_weight": float(collection.get("vader_weight", 1.0)),
        })
    else:
        block.update(preset_cfg)
    return block


def _normalize_features_config(features: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime features block expected by the pipeline.

    features.yaml is organized around feature families and presets. Runtime code still
    consumes cfg["features"]["technical"], cfg["features"]["context"],
    cfg["features"]["fundamentals"] and cfg["features"]["sentiment"].
    """
    features = dict(features or {})
    generation = features.get("generation", {}) or {}
    if "multicollinearity_threshold" not in features and "multicollinearity_threshold" in generation:
        features["multicollinearity_threshold"] = generation["multicollinearity_threshold"]

    families_cfg = features.get("families", {}) or {}
    selection = features.get("selection", {}) or {}
    active_profile = selection.get("active_profile", "default")
    profiles = selection.get("profiles", {}) or {}
    profile = profiles.get(active_profile, {}) or {}
    profile_families = profile.get("families", {}) or {}

    for family in ("technical", "context", "fundamentals", "sentiment"):
        family_choice = profile_families.get(family, {}) or {}
        family_def = families_cfg.get(family, {}) or {}
        preset_name = family_choice.get("preset") or family_def.get("default_preset") or next(iter((family_def.get("presets", {}) or {"default": {}}).keys()))
        preset_cfg = (family_def.get("presets", {}) or {}).get(preset_name, {}) or {}
        preset_cfg = _merge_dicts(preset_cfg, family_choice.get("overrides", {}) or {})
        enabled = bool(family_choice.get("enabled", preset_cfg.get("enabled", True)))
        block = _legacy_family_block(family, preset_cfg, enabled)
        block["preset"] = preset_name
        features[family] = block
    return features


def load_features_config(path: str | Path | None = None) -> dict[str, Any]:
    features_path = Path(path) if path else FEATURES_CONFIG_PATH
    if not features_path.exists():
        raise FileNotFoundError(f"features.yaml not found: {features_path}")
    return _normalize_features_config(_load_yaml(features_path))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)
    cfg["_config_dir"] = str(cfg_path.parent)
    features_file = cfg.get("features_file", "features.yaml")
    features_path = Path(features_file)
    if not features_path.is_absolute():
        features_path = cfg_path.parent / features_path
    cfg["features"] = load_features_config(features_path)
    return cfg


def load_data_registry(cfg: dict[str, Any] | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Load cadastral asset metadata from data.yaml.

    config.yaml remains operational. data.yaml stores asset metadata: groups,
    subgroups, CNPJ hints, linked indices and context baskets.
    """
    if path:
        registry_path = Path(path)
    elif cfg:
        registry_file = Path(cfg.get("data", {}).get("registry_file", "data.yaml"))
        base = Path(cfg.get("_config_dir", str(CONFIG_DIR)))
        registry_path = registry_file if registry_file.is_absolute() else base / registry_file
    else:
        registry_path = DATA_REGISTRY_PATH
    return _load_yaml(registry_path)


def artifact_dir(cfg: dict[str, Any]) -> Path:
    path = ROOT / cfg.get("app", {}).get("artifact_dir", "artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(cfg: dict[str, Any]) -> Path:
    path = ROOT / cfg.get("app", {}).get("data_cache_dir", "data/cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def historical_dir(cfg: dict[str, Any]) -> Path:
    path = ROOT / "data" / "historical"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir(cfg: dict[str, Any]) -> Path:
    path = artifact_dir(cfg) / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path
