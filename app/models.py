from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import models_dir
from .feature_audit import feature_family_profile, top_selected_features
from .preparation import prepare_training_matrix
from .utils import normalize_ticker, read_json, safe_ticker, write_json


def _make_scaler(cfg: dict[str, Any]):
    prep = cfg.get("features", {}).get("preparation", {}) or {}
    norm = prep.get("normalization", {}) or {}
    if not bool(norm.get("enabled", True)):
        return None
    scaler_name = str(norm.get("scaler", "robust")).lower()
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _make_named_scaler(name: str):
    scaler_name = str(name or "robust").lower()
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _engine_feature_columns(
    cfg: dict[str, Any], engine_name: str, all_features: list[str]
) -> list[str]:
    """Return the prepared feature subset used by each tabular engine.

    Operational engines are tree/boosting tabular models. They receive the same
    prepared feature matrix so the Ridge arbiter compares specialists on a
    consistent input contract.
    """
    return list(all_features)


def _make_engine_scaler(cfg: dict[str, Any], engine_name: str):
    return _make_scaler(cfg)


def _oof_valid_mask(oof_predictions: dict[str, np.ndarray]) -> np.ndarray:
    if not oof_predictions:
        return np.array([], dtype=bool)
    stacked = np.column_stack(
        [np.asarray(values, dtype=float) for values in oof_predictions.values()]
    )
    return np.all(np.isfinite(stacked), axis=1)


def _fit_engine_input(
    cfg: dict[str, Any],
    engine_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_latest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, list[str]]:
    cols = _engine_feature_columns(cfg, engine_name, list(X_train.columns))
    scaler = _make_engine_scaler(cfg, engine_name)
    train_df = X_train[cols]
    test_df = X_test[cols]
    latest_df = X_latest[cols]
    if scaler is not None:
        train_arr = scaler.fit_transform(train_df)
        test_arr = scaler.transform(test_df)
        latest_arr = scaler.transform(latest_df)
    else:
        train_arr = train_df.to_numpy(dtype=float)
        test_arr = test_df.to_numpy(dtype=float)
        latest_arr = latest_df.to_numpy(dtype=float)
    return train_arr, test_arr, latest_arr, scaler, cols


def _transform_engine_input(
    engine_meta: dict[str, Any], engine_name: str, X: pd.DataFrame
) -> np.ndarray:
    meta = engine_meta.get(engine_name, {}) or {}
    cols = meta.get("features") or list(X.columns)
    scaler = meta.get("scaler")
    X_engine = X[cols]
    if scaler is not None:
        return scaler.transform(X_engine)
    return X_engine.to_numpy(dtype=float)


def _latest_engine_guard(
    raw_by_engine: dict[str, float], cfg: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Neutralize divergent latest engine predictions before Ridge.

    A divergent tabular engine should not keep pushing the arbiter. If an engine is outside the
    absolute return guard or too far from the median, it is replaced by the median
    of valid engines for this one signal and reported as discarded/neutralized.
    """
    if not raw_by_engine:
        return {}, {"discarded": [], "used": []}
    guards = _prediction_guard_cfg(cfg)
    safety = _engine_safety_cfg(cfg)
    engine_clip = float(guards.get("max_engine_return_abs", 0.12))
    band = float(safety.get("max_deviation_from_median", 0.025))
    values = np.array(list(raw_by_engine.values()), dtype=float)
    median_all = float(np.nanmedian(values)) if values.size else 0.0
    discarded: list[str] = []
    for name, value in raw_by_engine.items():
        extreme = engine_clip > 0 and abs(float(value)) > engine_clip
        divergent = (
            bool(safety.get("consensus_guard_enabled", True))
            and band > 0
            and abs(float(value) - median_all) > band
        )
        if extreme or divergent:
            discarded.append(name)
    valid_values = [
        float(v) for k, v in raw_by_engine.items() if k not in discarded and np.isfinite(float(v))
    ]
    replacement = float(np.nanmedian(valid_values)) if valid_values else median_all
    guarded: dict[str, float] = {}
    for name, value in raw_by_engine.items():
        guarded[name] = (
            replacement if name in discarded else _clip_return_float(float(value), engine_clip)
        )
    return guarded, {
        "discarded": discarded,
        "used": [k for k in raw_by_engine if k not in discarded],
        "replacement": replacement,
        "reason": "extreme_or_divergent_base_prediction" if discarded else "ok",
    }


def _prediction_guard_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    guards = cfg.get("model", {}).get("prediction_guards", {})
    return {
        "max_engine_return_abs": float(guards.get("max_engine_return_abs", 0.12)),
        "max_final_return_abs": float(guards.get("max_final_return_abs", 0.08)),
    }


def _clip_return_array(values: np.ndarray, limit: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if limit <= 0:
        return arr
    return np.clip(arr, -limit, limit)


def _clip_return_float(value: float, limit: float) -> float:
    if limit <= 0:
        return float(value)
    return float(np.clip(float(value), -limit, limit))


def _engine_safety_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    safety = cfg.get("model", {}).get("engine_safety", {}) or {}
    return {
        "consensus_guard_enabled": bool(safety.get("consensus_guard_enabled", True)),
        "max_deviation_from_median": float(safety.get("max_deviation_from_median", 0.025)),
    }


def _resolve_inner_threads(inner_threads: int | None) -> int | None:
    if inner_threads is None:
        return None
    return max(1, int(inner_threads))


def _apply_consensus_guard(
    matrix: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Limit one divergent engine without changing the architecture.

    The Ridge arbiter should judge specialists, not be dominated by a single
    divergent engine excursion. This guard is row-wise: each engine prediction is clipped to a
    configurable band around the row median of all base engines. Raw predictions
    are still audited separately.
    """
    arr = np.asarray(matrix, dtype=float)
    safety = _engine_safety_cfg(cfg)
    if arr.size == 0 or not safety["consensus_guard_enabled"]:
        return arr, {"enabled": False, "changed_values": 0}
    band = float(safety["max_deviation_from_median"])
    if band <= 0:
        return arr, {"enabled": False, "changed_values": 0}
    med = np.nanmedian(arr, axis=1, keepdims=True)
    guarded = np.clip(arr, med - band, med + band)
    changed = int(np.sum(np.abs(guarded - arr) > 1e-12))
    return guarded, {
        "enabled": True,
        "max_deviation_from_median": band,
        "changed_values": changed,
        "changed_ratio": float(changed / arr.size) if arr.size else 0.0,
    }


def _make_base_engines(cfg: dict[str, Any], inner_threads: int | None = None) -> dict[str, Any]:
    """Create the specialist/base tabular engines.

    Operational architecture:
      XGB + CatBoost + ExtraTrees -> Ridge arbiter.

    Neural experiments belong in a separate benchmark until they outperform this
    tabular stack with stable walk-forward evidence.
    """
    model_cfg = cfg.get("model", {})
    random_state = int(model_cfg.get("random_state", 42))
    engines_cfg = model_cfg.get("engines", {})
    threads = _resolve_inner_threads(inner_threads)
    engines: dict[str, Any] = {}

    if engines_cfg.get("xgb", {}).get("enabled", True):
        try:
            from xgboost import XGBRegressor

            ecfg = engines_cfg.get("xgb", {})
            engines["xgb"] = XGBRegressor(
                n_estimators=int(ecfg.get("n_estimators", 120)),
                max_depth=int(ecfg.get("max_depth", 3)),
                learning_rate=float(ecfg.get("learning_rate", 0.05)),
                subsample=float(ecfg.get("subsample", 0.9)),
                objective="reg:squarederror",
                random_state=random_state,
                verbosity=0,
                n_jobs=threads,
            )
        except Exception:
            # Keep the CLI usable when xgboost is not installed.
            pass

    if engines_cfg.get("catboost", {}).get("enabled", True):
        try:
            from catboost import CatBoostRegressor

            ecfg = engines_cfg.get("catboost", {})
            engines["catboost"] = CatBoostRegressor(
                iterations=int(ecfg.get("iterations", 220)),
                depth=int(ecfg.get("depth", 4)),
                learning_rate=float(ecfg.get("learning_rate", 0.04)),
                loss_function=str(ecfg.get("loss_function", "MAE")),
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=threads,
            )
        except Exception:
            # CatBoost is optional at import time, but listed in requirements.
            pass

    if engines_cfg.get("extratrees", {}).get("enabled", True):
        ecfg = engines_cfg.get("extratrees", {})
        engines["extratrees"] = ExtraTreesRegressor(
            n_estimators=int(ecfg.get("n_estimators", 260)),
            max_depth=ecfg.get("max_depth", 10),
            min_samples_leaf=int(ecfg.get("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=-1 if threads is None else threads,
        )
    return engines


def _space_from_pair(values: Any, kind: str):
    """Build skopt dimensions explicitly from YAML ranges."""
    from skopt.space import Categorical, Integer, Real

    if (
        isinstance(values, list)
        and len(values) == 2
        and all(isinstance(v, (int, float)) for v in values)
    ):
        lo, hi = values
        if kind == "int":
            return Integer(int(lo), int(hi))
        return Real(float(lo), float(hi))
    if isinstance(values, list):
        return Categorical(values)
    return Categorical([values])


def _horizon_bars(horizon: str) -> int:
    value = str(horizon or "d1").lower().strip()
    if value.startswith("d"):
        value = value[1:]
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _validation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("model", {}).get("validation", {}) or {}


def _embargo_gap(cfg: dict[str, Any], horizon: str) -> int:
    vcfg = _validation_cfg(cfg)
    if not bool(vcfg.get("embargo_by_horizon", True)):
        return 0
    raw_gap = vcfg.get("embargo_bars", "auto")
    if str(raw_gap).lower() == "auto":
        return _horizon_bars(horizon)
    return max(0, int(raw_gap or 0))


def _safe_cv_gap(train_rows: int, n_splits: int, requested_gap: int) -> int:
    if requested_gap <= 0:
        return 0
    rows_per_fold = max(1, int(train_rows) // max(2, int(n_splits) + 1))
    return max(0, min(int(requested_gap), rows_per_fold - 1))


def _time_series_cv(n_splits: int, train_rows: int, gap: int) -> TimeSeriesSplit:
    safe_gap = _safe_cv_gap(train_rows, n_splits, gap)
    return TimeSeriesSplit(n_splits=int(n_splits), gap=safe_gap)


def _tune_base_engines(
    cfg: dict[str, Any],
    base_engines: dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    inner_threads: int | None = None,
    horizon: str = "d1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune tabular specialists before Ridge arbitration."""
    try:
        from skopt import BayesSearchCV
    except Exception as exc:
        raise RuntimeError(
            "--autotune requires scikit-optimize. Install package: scikit-optimize"
        ) from exc

    model_cfg = cfg.get("model", {})
    tune_cfg = model_cfg.get("autotune", {})
    n_iter = int(tune_cfg.get("n_iter", 5))
    cv = int(tune_cfg.get("cv", 2))
    scoring = tune_cfg.get("scoring", "neg_mean_absolute_error")
    random_state = int(model_cfg.get("random_state", 42))
    spaces = tune_cfg.get("spaces", {})
    threads = _resolve_inner_threads(inner_threads)
    cv_splitter = _time_series_cv(cv, len(X_train), _embargo_gap(cfg, horizon))

    tuned: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for name, engine in base_engines.items():
        raw_space = spaces.get(name, {})
        if not raw_space:
            tuned[name] = engine
            summary[name] = {"status": "skipped", "reason": "no search space in config"}
            continue

        search_space = {}
        for param, values in raw_space.items():
            kind = "real"
            if param in [
                "n_estimators",
                "max_depth",
                "iterations",
                "depth",
                "l2_leaf_reg",
                "min_samples_leaf",
            ]:
                kind = "int"
            search_space[param] = _space_from_pair(values, kind)

        opt = BayesSearchCV(
            estimator=engine,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1 if threads is None else threads,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="skopt.space.space")
            warnings.filterwarnings("ignore", category=FutureWarning)
            opt.fit(X_train, y_train)
        tuned[name] = opt.best_estimator_
        summary[name] = {
            "status": "tuned_tabular",
            "best_score": float(opt.best_score_),
            "best_params": opt.best_params_,
        }

    return tuned, summary


def _make_arbiter(cfg: dict[str, Any]) -> Ridge:
    """Create the final stacking arbiter from config."""
    model_cfg = cfg.get("model", {}) or {}
    arbiter_cfg = model_cfg.get("arbiter", {}) or {}
    ridge_cfg = arbiter_cfg.get("ridge", {}) or {}
    return Ridge(
        alpha=float(ridge_cfg.get("alpha", 1.0)),
        fit_intercept=bool(ridge_cfg.get("fit_intercept", True)),
    )


def _stacking_cv_splits(cfg: dict[str, Any], train_rows: int) -> int:
    stack_cfg = cfg.get("model", {}).get("stacking", {}) or {}
    configured = int(stack_cfg.get("cv", cfg.get("model", {}).get("autotune", {}).get("cv", 3)))
    return max(2, min(configured, int(train_rows) - 1))


def _stacking_cv(cfg: dict[str, Any], train_rows: int, horizon: str) -> TimeSeriesSplit:
    n_splits = _stacking_cv_splits(cfg, train_rows)
    return _time_series_cv(n_splits, train_rows, _embargo_gap(cfg, horizon))


def _minimum_train_rows(cfg: dict[str, Any]) -> int:
    min_rows = int(cfg.get("data", {}).get("min_rows", 220))
    vcfg = _validation_cfg(cfg)
    return max(40, int(vcfg.get("min_train_rows", max(60, int(min_rows * 0.60)))))


def _align_training_input(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X0 = X.copy().replace([np.inf, -np.inf], np.nan)
    y0 = pd.Series(y, index=X0.index).replace([np.inf, -np.inf], np.nan)
    valid_idx = y0.dropna().index
    X0 = X0.loc[valid_idx]
    y0 = y0.loc[valid_idx].astype(float)
    return X0, y0


def _split_train_test_raw(
    cfg: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    horizon: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any]]:
    X0, y0 = _align_training_input(X, y)
    test_size = float(cfg.get("model", {}).get("test_size", 0.20))
    split = max(1, int(len(X0) * (1 - test_size)))
    gap = _embargo_gap(cfg, horizon)
    train_end = max(1, split - gap)
    min_train_rows = _minimum_train_rows(cfg)
    if train_end < min_train_rows:
        raise RuntimeError(f"insufficient train rows after embargo: {train_end} < {min_train_rows}")
    if split >= len(X0):
        raise RuntimeError("insufficient test rows for holdout evaluation")

    X_train_raw = X0.iloc[:train_end]
    y_train_raw = y0.iloc[:train_end]
    X_test_raw = X0.iloc[split:]
    y_test_raw = y0.iloc[split:]
    split_meta = {
        "test_size": test_size,
        "target_rows": int(len(X0)),
        "embargo_bars": int(gap),
        "split_index": int(split),
        "train_end_index": int(train_end),
        "dropped_embargo_rows": int(max(0, split - train_end)),
        "test_target": "raw_unclipped",
    }
    return X_train_raw, y_train_raw, X_test_raw, y_test_raw, split_meta


def _apply_selected_features(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X1 = X[list(features)].replace([np.inf, -np.inf], np.nan)
    y1 = pd.Series(y, index=X1.index).replace([np.inf, -np.inf], np.nan).astype(float)
    valid = X1.notna().all(axis=1) & y1.notna()
    return X1.loc[valid], y1.loc[valid]


def _confidence_from(
    values: list[float],
    prediction: float,
    cfg: dict[str, Any] | None = None,
    *,
    mae: float | None = None,
    guard_meta: dict[str, Any] | None = None,
    train_rows: int | None = None,
) -> tuple[float, float]:
    """Calibrate operational confidence for D+1 equity forecasts.

    This version is adjusted for OOF (Out-of-Fold) stacking, which is more honest
    but naturally has higher errors than biased in-sample training.
    """
    clean = [float(v) for v in values if np.isfinite(float(v))]
    dispersion = float(np.std(clean)) if clean else 0.0
    if not clean:
        return dispersion, 0.0

    ccfg = (cfg or {}).get("model", {}).get("confidence", {})

    # Agreement: how much base engines agree on the direction/magnitude
    agreement_scale = float(ccfg.get("agreement_scale_return", 0.025))  # Increased from 0.015
    scale = max(agreement_scale, float(np.mean(np.abs(clean))) if clean else 0.0)
    agreement = float(scale / (scale + dispersion)) if scale > 0 else 0.0

    # MAE Component: more forgiving sigmoid
    mae_scale = float(ccfg.get("mae_reference_return", 0.020))  # Increased from 0.015
    if mae is None or not np.isfinite(float(mae)) or mae_scale <= 0:
        mae_component = 0.80
    else:
        mae_ratio = max(0.0, float(mae)) / mae_scale
        mae_component = 1.0 / (1.0 + 0.4 * (mae_ratio**1.2))

    # Magnitude: predictions near zero are less 'confident'
    action_scale = float(ccfg.get("action_scale_return", 0.0050))
    magnitude_ratio = min(1.0, abs(float(prediction)) / action_scale) if action_scale > 0 else 1.0
    neutral_floor = float(ccfg.get("neutral_prediction_component_floor", 0.60))  # Higher floor 0.60
    magnitude_component = neutral_floor + (1.0 - neutral_floor) * magnitude_ratio

    # Engine component
    guard_meta = guard_meta or {}
    used = len(guard_meta.get("used", []) or clean)
    discarded = len(guard_meta.get("discarded", []) or [])
    total = max(used + discarded, len(clean), 1)
    engine_component = max(0.50, used / total)  # Higher floor 0.50
    discarded_engine_penalty = float(ccfg.get("discarded_engine_penalty", 0.75))
    if discarded > 0:
        engine_component *= max(0.20, discarded_engine_penalty**discarded)

    # Sample Size
    if train_rows is None or int(train_rows or 0) <= 0:
        sample_component = 1.0
    else:
        reference_rows = max(1, int(ccfg.get("sample_reference_rows", 600)))  # Lowered from 800
        sample_component = min(
            1.0, max(0.65, float(train_rows) / reference_rows)
        )  # Higher floor 0.65

    # New Weighted Combination: Less aggressive multiplication
    # We mix Agreement and MAE as the 'Core Quality'
    core_quality = (0.60 * agreement) + (0.40 * mae_component)

    # Then we multiply by situational factors
    confidence = core_quality * magnitude_component * engine_component * sample_component

    # Final scaling to mapping it to a more useful range
    # Boost by a factor to utilize the 0-100 range better
    confidence = confidence * 1.15

    floor = float(ccfg.get("minimum_when_engines_exist", 0.10))
    cap = float(ccfg.get("maximum_confidence", 0.95))
    confidence = min(cap, max(floor, confidence))
    return dispersion, confidence


def train_models(
    cfg: dict[str, Any],
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_meta: dict[str, Any],
    autotune: bool = False,
    horizon: str = "d1",
    inner_threads: int | None = None,
) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    min_rows = int(cfg.get("data", {}).get("min_rows", 220))
    target_rows = int(pd.Series(y, index=X.index).dropna().shape[0])
    if target_rows < min_rows:
        raise RuntimeError(f"insufficient rows for training: {target_rows} < {min_rows}")

    X_train_raw, y_train_raw, X_test_raw, y_test_raw, split_meta = _split_train_test_raw(
        cfg,
        X,
        y,
        horizon=horizon,
    )
    X_train, y_train, prep_meta = prepare_training_matrix(X_train_raw, y_train_raw, cfg)
    if len(X_train) < _minimum_train_rows(cfg):
        raise RuntimeError(
            f"insufficient prepared train rows: {len(X_train)} < {_minimum_train_rows(cfg)}"
        )
    if not list(X_train.columns):
        raise RuntimeError("feature preparation selected no usable columns")

    X_test, y_test = _apply_selected_features(X_test_raw, y_test_raw, list(X_train.columns))
    if len(X_test) == 0:
        raise RuntimeError("holdout has no rows after applying train-selected features")

    latest_raw = X[list(X_train.columns)].replace([np.inf, -np.inf], np.nan).dropna().tail(1)
    if latest_raw.empty:
        raise RuntimeError("latest row has no valid train-selected features")

    # Model-preparation contract: feature selection is fitted on train only; the
    # selected columns are then applied to holdout/latest without re-ranking.
    X_latest = latest_raw
    dataset_meta = dict(dataset_meta or {})
    dataset_meta["preparation"] = prep_meta
    dataset_meta["validation_split"] = split_meta
    base_engines = _make_base_engines(cfg, inner_threads=inner_threads)
    if len(base_engines) < 2:
        raise RuntimeError("at least two base engines are required for Ridge stacking")

    engine_inputs: dict[str, dict[str, Any]] = {}
    for name in base_engines:
        train_arr, test_arr, latest_arr, input_scaler, input_features = _fit_engine_input(
            cfg, name, X_train, X_test, X_latest
        )
        engine_inputs[name] = {
            "train": train_arr,
            "test": test_arr,
            "latest": latest_arr,
            "scaler": input_scaler,
            "features": input_features,
        }

    tune_summary: dict[str, Any] = {}
    if autotune:
        tuned: dict[str, Any] = {}
        for name, engine in base_engines.items():
            tuned_one, summary_one = _tune_base_engines(
                cfg,
                {name: engine},
                engine_inputs[name]["train"],
                y_train,
                inner_threads=inner_threads,
                horizon=horizon,
            )
            tuned.update(tuned_one)
            tune_summary.update(summary_one)
        base_engines = tuned

    base_order = list(base_engines.keys())
    guards = _prediction_guard_cfg(cfg)
    engine_clip = guards["max_engine_return_abs"]
    final_clip = guards["max_final_return_abs"]

    # --- Stacking with TimeSeriesSplit to avoid leakage ---
    tscv = _stacking_cv(cfg, len(X_train), horizon)
    oof_predictions = {name: np.full(len(X_train), np.nan, dtype=float) for name in base_order}

    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        for name in base_order:
            # Fit on fold
            from sklearn.base import clone

            fold_engine = clone(base_engines[name])

            # Prepare fold input (subset of training)
            cols = engine_inputs[name]["features"]
            scaler = _make_engine_scaler(cfg, name)
            if scaler is None:
                fold_train_arr = X_fold_train[cols].to_numpy(dtype=float)
                fold_val_arr = X_fold_val[cols].to_numpy(dtype=float)
            else:
                fold_train_arr = scaler.fit_transform(X_fold_train[cols])
                fold_val_arr = scaler.transform(X_fold_val[cols])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                fold_engine.fit(fold_train_arr, y_fold_train)

            # Predict out-of-fold
            fold_pred = fold_engine.predict(fold_val_arr)
            oof_predictions[name][val_idx] = fold_pred

    # Only use indices where we have OOF predictions (first fold is skipped by TSS)
    valid_idx = np.where(_oof_valid_mask(oof_predictions))[0]
    if len(valid_idx) == 0:
        raise RuntimeError(
            "time-series stacking produced no valid out-of-fold rows for ridge training"
        )
    meta_train_oof = np.column_stack([oof_predictions[name][valid_idx] for name in base_order])
    y_train_oof = y_train.iloc[valid_idx]

    # --- Final fit of base engines on full X_train ---
    fitted: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    raw_test_cols: list[np.ndarray] = []
    raw_latest_values: list[float] = []

    for name, engine in base_engines.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            engine.fit(engine_inputs[name]["train"], y_train)
        fitted[name] = engine

        raw_test_pred = np.asarray(engine.predict(engine_inputs[name]["test"]), dtype=float)
        raw_latest_pred = float(engine.predict(engine_inputs[name]["latest"])[0])

        raw_test_cols.append(raw_test_pred)
        raw_latest_values.append(raw_latest_pred)
        metrics[name] = {
            "mae_return_raw": float(mean_absolute_error(y_test, raw_test_pred)),
        }

    raw_meta_test = np.column_stack(raw_test_cols)
    raw_meta_latest = np.array([raw_latest_values], dtype=float)

    # Apply guards
    abs_meta_train = _clip_return_array(meta_train_oof, engine_clip)
    abs_meta_test = _clip_return_array(raw_meta_test, engine_clip)
    abs_meta_latest = _clip_return_array(raw_meta_latest, engine_clip)

    meta_train, train_consensus_meta = _apply_consensus_guard(abs_meta_train, cfg)
    meta_test, test_consensus_meta = _apply_consensus_guard(abs_meta_test, cfg)
    meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_meta_latest, cfg)

    raw_pred_latest = {name: float(raw_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    neutralized_pred_latest, latest_discard_meta = _latest_engine_guard(raw_pred_latest, cfg)
    if latest_discard_meta.get("discarded"):
        meta_latest = np.array(
            [[neutralized_pred_latest[name] for name in base_order]], dtype=float
        )

    pred_latest = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    abs_pred_latest = {name: float(abs_meta_latest[0, idx]) for idx, name in enumerate(base_order)}

    for idx, name in enumerate(base_order):
        metrics[name]["mae_return_abs_guarded"] = float(
            mean_absolute_error(y_test, abs_meta_test[:, idx])
        )
        metrics[name]["mae_return_consensus_guarded"] = float(
            mean_absolute_error(y_test, meta_test[:, idx])
        )
        metrics[name]["latest_raw_return"] = raw_pred_latest[name]
        metrics[name]["latest_guarded_return"] = pred_latest[name]

    # --- Fit Arbiter on OOF meta-features ---
    arbiter = _make_arbiter(cfg)
    arbiter.fit(meta_train, y_train_oof)

    arbiter_test_pred = np.asarray(arbiter.predict(meta_test), dtype=float)
    arbiter_latest_raw = float(arbiter.predict(meta_latest)[0])
    arbiter_latest = _clip_return_float(arbiter_latest_raw, final_clip)
    arbiter_mae = float(mean_absolute_error(y_test, arbiter_test_pred))

    dispersion, confidence = _confidence_from(
        list(pred_latest.values()),
        arbiter_latest,
        cfg,
        mae=arbiter_mae,
        guard_meta=latest_discard_meta,
        train_rows=len(X_train),
    )

    rid = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{horizon}_{safe_ticker(ticker)}"
    out_dir = models_dir(cfg) / safe_ticker(ticker) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{horizon}.pkl"
    selected_features = list(X_train.columns)
    payload = {
        "ticker": ticker,
        "features": selected_features,
        "engine_inputs": {
            name: {
                "features": engine_inputs[name]["features"],
                "scaler": engine_inputs[name]["scaler"],
            }
            for name in base_order
        },
        "normalization": (cfg.get("features", {}).get("preparation", {}) or {}).get(
            "normalization", {}
        ),
        "base_models": fitted,
        "base_order": base_order,
        "arbiter": arbiter,
        "architecture": "xgb_catboost_extratrees__ridge_arbiter",
        "autotune": bool(autotune),
        "tune_summary": tune_summary,
        "pred_latest_by_engine": pred_latest,
        "raw_pred_latest_by_engine": raw_pred_latest,
        "prediction_guards": guards,
        "engine_safety": _engine_safety_cfg(cfg),
        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
        "latest_engine_guard": latest_discard_meta,
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)

    prep_meta = (dataset_meta or {}).get("preparation", {}) or {}

    # populate feature relevance from trained ExtraTrees

    try:
        et_model = fitted.get("extratrees")

        if et_model is not None and hasattr(et_model, "feature_importances_"):

            relevance = {
                str(feature): float(score)
                for feature, score in zip(
                    selected_features,
                    et_model.feature_importances_,
                )
            }

            prep_meta.setdefault("selection", {})
            prep_meta.setdefault("selection", {})
        prep_meta["selection"]["model_relevance"] = relevance

    except Exception:
        pass

    top_features = top_selected_features(prep_meta, selected_features, n=5)
    family_profile = feature_family_profile(selected_features)

    manifest = {
        "run_id": rid,
        "ticker": ticker,
        "horizon": horizon,
        "model_path": str(model_path),
        "created_at": pd.Timestamp.now().isoformat(),
        "architecture": "XGB + CatBoost + ExtraTrees -> Ridge arbiter",
        "autotune": bool(autotune),
        "tune_summary": tune_summary,
        "rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": selected_features,
        "top_features": top_features,
        "feature_family_profile": family_profile,
        "preparation": prep_meta,
        "validation_split": split_meta,
        "autotune_performed": bool(autotune),
        "latest_price": dataset_meta.get("latest_price"),
        "latest_date": dataset_meta.get("latest_date"),
        "base_engines": base_order,
        "arbiter": "ridge",
        "metrics": {**metrics, "ridge_arbiter": {"mae_return": arbiter_mae}},
        "latest_prediction_return": arbiter_latest,
        "latest_prediction_by_engine": pred_latest,
        "latest_prediction_by_engine_raw": raw_pred_latest,
        "latest_prediction_return_raw": arbiter_latest_raw,
        "prediction_guards": guards,
        "engine_safety": _engine_safety_cfg(cfg),
        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
        "latest_engine_guard": latest_discard_meta,
        "engine_input_features": {name: engine_inputs[name]["features"] for name in base_order},
        "consensus_guard": {
            "train": train_consensus_meta,
            "test": test_consensus_meta,
            "latest": latest_consensus_meta,
        },
        "engine_dispersion": dispersion,
        "confidence": confidence,
        "quality": confidence,
        "dataset_meta": dataset_meta,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(models_dir(cfg) / safe_ticker(ticker) / f"latest_train_{horizon}.json", manifest)
    return manifest


def load_latest_model(
    cfg: dict[str, Any], ticker: str, horizon: str = "d1"
) -> tuple[dict[str, Any], dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    latest = models_dir(cfg) / safe_ticker(ticker) / f"latest_train_{horizon}.json"
    if not latest.exists():
        raise FileNotFoundError(
            f"no trained model found for {ticker} horizon {horizon}; run train first"
        )
    manifest = read_json(latest)
    with Path(manifest["model_path"]).open("rb") as fh:
        model = pickle.load(fh)
    return model, manifest


def predict_with_model(
    cfg: dict[str, Any], ticker: str, X: pd.DataFrame, horizon: str = "d1"
) -> dict[str, Any]:
    model, manifest = load_latest_model(cfg, ticker, horizon=horizon)
    features = model["features"]
    X = X.copy()
    missing = [c for c in features if c not in X.columns]
    recoverable = [c for c in missing if str(c).startswith("sent_")]
    for col in recoverable:
        X[col] = 0.0
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise RuntimeError(f"current dataset is missing trained features: {missing[:8]}")

    latest_X = X[features].tail(1)

    # Current architecture: base specialists -> Ridge arbiter.
    if "base_models" in model and "arbiter" in model:
        base_order = model.get("base_order") or list(model["base_models"].keys())
        guards = model.get("prediction_guards") or _prediction_guard_cfg(cfg)
        engine_clip = float(guards.get("max_engine_return_abs", 0.12))
        final_clip = float(guards.get("max_final_return_abs", 0.08))
        engine_meta = model.get("engine_inputs") or {}
        raw_by_engine = {}
        for name in base_order:
            engine_input = (
                _transform_engine_input(engine_meta, name, latest_X)
                if engine_meta
                else latest_X.to_numpy(dtype=float)
            )
            raw_by_engine[name] = float(model["base_models"][name].predict(engine_input)[0])
        abs_values = np.array(
            [[_clip_return_float(raw_by_engine[name], engine_clip) for name in base_order]],
            dtype=float,
        )
        neutralized_by_engine, latest_engine_guard = _latest_engine_guard(raw_by_engine, cfg)
        if latest_engine_guard.get("discarded"):
            meta_latest = np.array(
                [[neutralized_by_engine[name] for name in base_order]], dtype=float
            )
            latest_consensus_meta = {
                "enabled": True,
                "changed_values": len(latest_engine_guard.get("discarded", [])),
                "neutralized": True,
            }
        else:
            meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_values, cfg)
        by_engine = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
        abs_by_engine = {name: float(abs_values[0, idx]) for idx, name in enumerate(base_order)}
        raw_prediction = float(model["arbiter"].predict(meta_latest)[0])
        prediction = _clip_return_float(raw_prediction, final_clip)
        ridge_metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
        mae = ridge_metrics.get("mae_return")
        train_rows = manifest.get("train_rows")
        dispersion, confidence = _confidence_from(
            list(by_engine.values()),
            prediction,
            cfg,
            mae=mae,
            guard_meta=latest_engine_guard,
            train_rows=train_rows,
        )
        return {
            "ticker": normalize_ticker(ticker),
            "architecture": "XGB + CatBoost + ExtraTrees -> Ridge arbiter",
            "prediction_return": prediction,
            "by_engine": by_engine,
            "raw_by_engine": raw_by_engine,
            "abs_guarded_by_engine": abs_by_engine,
            "raw_prediction_return": raw_prediction,
            "consensus_guard": latest_consensus_meta,
            "latest_engine_guard": latest_engine_guard,
            "used_engines": latest_engine_guard.get("used", base_order),
            "discarded_engines": latest_engine_guard.get("discarded", []),
            "prediction_guards": guards,
            "arbiter": "ridge",
            "dispersion": dispersion,
            "confidence": confidence,
            "quality": confidence,
            "train_manifest": manifest,
        }

    raise RuntimeError(
        "trained artifact does not match the current Ridge-arbiter architecture; run train again"
    )


def predict_multi_horizon(
    cfg: dict[str, Any], ticker: str, X: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Generate predictions for all available horizons."""
    horizons = ["d1", "d5", "d20"]
    results = {}
    for h in horizons:
        try:
            results[h] = predict_with_model(cfg, ticker, X, horizon=h)
        except Exception as exc:
            results[h] = {"error": str(exc)}
    return results

