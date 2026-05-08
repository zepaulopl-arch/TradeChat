from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


from .feature_audit import feature_family as _family_of


_DEFAULT_LEVEL_FEATURES = {
    "frac_mem",
    "sma_10",
    "sma_50",
    "ema_20",
    "ema_100",
    "bb_mid",
    "bb_std",
}


def _looks_like_raw_provider_column(name: str) -> bool:
    value = str(name)
    return value.startswith("^") or value.endswith(".SA") or "=" in value


def _drop_stationarity_blocked_features(X: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    prep = cfg.get("features", {}).get("preparation", {}) or {}
    stationarity = prep.get("stationarity", {}) or {}
    blocked_names: set[str] = set()
    if bool(stationarity.get("drop_level_features", True)):
        configured = stationarity.get("level_feature_names")
        blocked_names.update(str(item) for item in (configured or sorted(_DEFAULT_LEVEL_FEATURES)))

    dropped: list[str] = []
    keep: list[str] = []
    drop_raw_price = bool(stationarity.get("drop_raw_price", True))
    for col in X.columns:
        name = str(col)
        if name in blocked_names or (drop_raw_price and _looks_like_raw_provider_column(name)):
            dropped.append(name)
        else:
            keep.append(name)
    return X[keep], dropped


def _safe_abs_target_corr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    vals: dict[str, float] = {}
    yy = pd.Series(y, index=X.index).astype(float)
    for col in X.columns:
        xx = pd.Series(X[col], index=X.index).astype(float)
        if xx.nunique(dropna=True) <= 1:
            vals[col] = 0.0
            continue
        corr = xx.corr(yy)
        vals[col] = 0.0 if corr is None or not np.isfinite(corr) else abs(float(corr))
    return pd.Series(vals).sort_values(ascending=False)


def _drop_constant_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    keep: list[str] = []
    for col in X.columns:
        s = X[col]
        if s.nunique(dropna=True) <= 1:
            dropped.append(col)
        else:
            keep.append(col)
    return X[keep], dropped


def _greedy_low_correlation_selection(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int,
    threshold: float,
    family_minimums: dict[str, int] | None = None,
    family_limits: dict[str, dict[str, int]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Select a compact, useful and family-balanced feature set.

    Ranking uses absolute correlation with the target as a transparent relevance
    score. Candidates that are too correlated with already selected features are
    rejected. Family limits prevent the context block from occupying the entire
    feature set and keep the asset's own technical behaviour visible.
    """
    if X.empty:
        return [], {"reason": "empty_X"}

    max_features = max(1, int(max_features))
    threshold = float(threshold)
    family_minimums = {str(k): int(v) for k, v in (family_minimums or {}).items() if int(v) > 0}
    family_limits = family_limits or {}
    family_maximums: dict[str, int] = {}
    family_minimums_from_limits: dict[str, int] = {}
    for family, raw in family_limits.items():
        if not isinstance(raw, dict):
            continue
        fam = str(family)
        if "min" in raw and int(raw.get("min") or 0) > 0:
            family_minimums_from_limits[fam] = int(raw.get("min") or 0)
        if "max" in raw and int(raw.get("max") or 0) > 0:
            family_maximums[fam] = int(raw.get("max") or 0)
    family_minimums = {**family_minimums_from_limits, **family_minimums}

    relevance = _safe_abs_target_corr(X, y)
    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    selected: list[str] = []
    rejected: dict[str, str] = {}

    def family_count(family: str) -> int:
        return len([c for c in selected if _family_of(c) == family])

    def can_add(col: str, *, enforce_family_max: bool = True) -> tuple[bool, str]:
        if col in selected:
            return False, "already_selected"
        fam = _family_of(col)
        if enforce_family_max and fam in family_maximums and family_count(fam) >= family_maximums[fam]:
            return False, f"family_limit:{fam}:{family_maximums[fam]}"
        for chosen in selected:
            value = float(corr.loc[col, chosen]) if col in corr.index and chosen in corr.columns else 0.0
            if value > threshold:
                return False, f"correlated_with:{chosen}:{value:.4f}"
        return True, "ok"

    # First satisfy family minimums, still respecting the correlation rule.
    for family, minimum in family_minimums.items():
        fam_candidates = [c for c in relevance.index if _family_of(c) == family]
        for col in fam_candidates:
            if family_count(family) >= minimum:
                break
            if len(selected) >= max_features:
                break
            ok, reason = can_add(col, enforce_family_max=False)
            if ok:
                selected.append(col)
            else:
                rejected[col] = reason

    # Then fill by relevance, respecting both correlation and family maximums.
    for col in relevance.index:
        if len(selected) >= max_features:
            break
        ok, reason = can_add(col, enforce_family_max=True)
        if ok:
            selected.append(col)
        else:
            rejected.setdefault(col, reason)

    # If correlation is too strict, fill remaining slots without violating family maxima.
    forced: list[str] = []
    if len(selected) < min(max_features, len(relevance)):
        for col in relevance.index:
            if len(selected) >= min(max_features, len(relevance)):
                break
            if col in selected:
                continue
            fam = _family_of(col)
            if fam in family_maximums and family_count(fam) >= family_maximums[fam]:
                rejected.setdefault(col, f"family_limit:{fam}:{family_maximums[fam]}")
                continue
            selected.append(col)
            forced.append(col)

    meta = {
        "method": "target_relevance_low_correlation_greedy_family_balanced",
        "max_features": max_features,
        "correlation_threshold": threshold,
        "family_minimums": family_minimums,
        "family_maximums": family_maximums,
        "selected_count": len(selected),
        "relevance": {k: float(v) for k, v in relevance.loc[selected].items()},
        "families": {fam: len([c for c in selected if _family_of(c) == fam]) for fam in ("technical", "context", "fundamentals", "sentiment")},
        "forced_fill": forced,
        "rejected_count": len(rejected),
        "rejected_sample": dict(list(rejected.items())[:20]),
    }
    return selected, meta

def prepare_training_matrix(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Pre-model data preparation: clean, reduce collinearity, select features.

    This is intentionally internal. The CLI stays unchanged; features.yaml controls
    whether the step is active and how aggressive it is.
    """
    fcfg = cfg.get("features", {}) or {}
    prep = fcfg.get("preparation", {}) or {}
    enabled = bool(prep.get("enabled", True))

    X0 = X.copy().replace([np.inf, -np.inf], np.nan)
    y0 = pd.Series(y, index=X0.index).replace([np.inf, -np.inf], np.nan)
    X0 = X0.dropna(axis=0, how="any")
    y0 = y0.loc[X0.index].dropna()
    X0 = X0.loc[y0.index]

    meta: dict[str, Any] = {
        "enabled": enabled,
        "input_features": list(X.columns),
        "input_feature_count": len(X.columns),
        "input_rows": int(len(X)),
        "output_rows": int(len(X0)),
    }

    X0, stationarity_drops = _drop_stationarity_blocked_features(X0, cfg)
    meta["dropped_stationarity_features"] = stationarity_drops

    X1, constant_drops = _drop_constant_features(X0)
    meta["dropped_constant_features"] = constant_drops

    if not enabled:
        meta["selected_features"] = list(X1.columns)
        meta["selected_feature_count"] = len(X1.columns)
        return X1, y0, meta

    # Target clipping protects engines, especially divergent base engines, from very old/extreme daily
    # returns dominating a long history. It is configurable and audited.
    target_cfg = prep.get("target", {}) or {}
    y1 = y0.astype(float)
    if bool(target_cfg.get("clip", True)):
        limit_pct = float(target_cfg.get("max_abs_return_pct", 8.0))
        limit = max(0.0, limit_pct / 100.0)
        if limit > 0:
            y1 = y1.clip(lower=-limit, upper=limit)
        meta["target_clip"] = {"enabled": True, "max_abs_return_pct": limit_pct}
    else:
        meta["target_clip"] = {"enabled": False}

    select_cfg = prep.get("selection", {}) or {}
    if bool(select_cfg.get("enabled", True)):
        max_features = int(select_cfg.get("max_features", 20))
        threshold = float(select_cfg.get("correlation_threshold", fcfg.get("multicollinearity_threshold", 0.88)))
        family_minimums = select_cfg.get("family_minimums", {}) or {}
        family_limits = select_cfg.get("family_limits", {}) or {}
        selected, selection_meta = _greedy_low_correlation_selection(X1, y1, max_features, threshold, family_minimums, family_limits)
        X2 = X1[selected]
        meta["selection"] = selection_meta
    else:
        X2 = X1
        meta["selection"] = {"enabled": False, "selected_count": len(X2.columns)}

    meta["selected_features"] = list(X2.columns)
    meta["selected_feature_count"] = len(X2.columns)
    meta["families"] = {fam: len([c for c in X2.columns if _family_of(c) == fam]) for fam in ("technical", "context", "fundamentals", "sentiment")}
    return X2, y1.loc[X2.index], meta
