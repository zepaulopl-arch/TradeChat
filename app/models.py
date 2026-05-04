from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from .feature_audit import feature_family

from .config import artifact_dir
from .feature_audit import top_selected_features, feature_family_profile
from .utils import latest_file, normalize_ticker, run_id, safe_ticker, write_json, read_json


def _make_scaler(cfg: dict[str, Any]):
    prep = cfg.get("features", {}).get("preparation", {}) or {}
    norm = prep.get("normalization", {}) or {}
    if not bool(norm.get("enabled", True)):
        return None
    scaler_name = str(norm.get("scaler", "standard")).lower()
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()



def _make_named_scaler(name: str):
    scaler_name = str(name or "standard").lower()
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _engine_feature_columns(cfg: dict[str, Any], engine_name: str, all_features: list[str]) -> list[str]:
    """Return the prepared feature subset used by each tabular engine.

    Operational engines are tree/boosting tabular models. They receive the same
    prepared feature matrix so the Ridge arbiter compares specialists on a
    consistent input contract.
    """
    return list(all_features)


def _make_engine_scaler(cfg: dict[str, Any], engine_name: str):
    return _make_scaler(cfg)


def _fit_engine_input(cfg: dict[str, Any], engine_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, X_latest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, list[str]]:
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


def _transform_engine_input(engine_meta: dict[str, Any], engine_name: str, X: pd.DataFrame) -> np.ndarray:
    meta = engine_meta.get(engine_name, {}) or {}
    cols = meta.get("features") or list(X.columns)
    scaler = meta.get("scaler")
    X_engine = X[cols]
    if scaler is not None:
        return scaler.transform(X_engine)
    return X_engine.to_numpy(dtype=float)


def _latest_engine_guard(raw_by_engine: dict[str, float], cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
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
        divergent = bool(safety.get("consensus_guard_enabled", True)) and band > 0 and abs(float(value) - median_all) > band
        if extreme or divergent:
            discarded.append(name)
    valid_values = [float(v) for k, v in raw_by_engine.items() if k not in discarded and np.isfinite(float(v))]
    replacement = float(np.nanmedian(valid_values)) if valid_values else median_all
    guarded: dict[str, float] = {}
    for name, value in raw_by_engine.items():
        guarded[name] = replacement if name in discarded else _clip_return_float(float(value), engine_clip)
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


def _apply_consensus_guard(matrix: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
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


def _make_base_engines(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create the specialist/base tabular engines.

    Operational architecture:
      XGB + CatBoost + ExtraTrees -> Ridge arbiter.

    The old MLP path was intentionally removed from the production stack because
    diagnostics showed frequent divergent raw predictions. Neural experiments
    should live in a separate benchmark, not in the daily arbiter.
    """
    model_cfg = cfg.get("model", {})
    random_state = int(model_cfg.get("random_state", 42))
    engines_cfg = model_cfg.get("engines", {})
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
            n_jobs=-1,
        )
    return engines

def _make_arbiter(cfg: dict[str, Any]) -> Ridge:
    arbiter_cfg = cfg.get("model", {}).get("arbiter", {}).get("ridge", {})
    return Ridge(alpha=float(arbiter_cfg.get("alpha", 1.0)))

def _space_from_pair(values: Any, kind: str):
    """Build skopt dimensions explicitly from YAML ranges."""
    from skopt.space import Integer, Real, Categorical

    if isinstance(values, list) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
        lo, hi = values
        if kind == "int":
            return Integer(int(lo), int(hi))
        return Real(float(lo), float(hi))
    if isinstance(values, list):
        return Categorical(values)
    return Categorical([values])


def _tune_base_engines(cfg: dict[str, Any], base_engines: dict[str, Any], X_train: np.ndarray, y_train: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune tabular specialists before Ridge arbitration."""
    try:
        from skopt import BayesSearchCV
    except Exception as exc:
        raise RuntimeError("--autotune requires scikit-optimize. Install package: scikit-optimize") from exc

    model_cfg = cfg.get("model", {})
    tune_cfg = model_cfg.get("autotune", {})
    n_iter = int(tune_cfg.get("n_iter", 5))
    cv = int(tune_cfg.get("cv", 2))
    scoring = tune_cfg.get("scoring", "neg_mean_absolute_error")
    random_state = int(model_cfg.get("random_state", 42))
    spaces = tune_cfg.get("spaces", {})

    tuned: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for name, engine in base_engines.items():
        if name == "xgb":
            raw = spaces.get("xgb", {})
            search_space = {
                "n_estimators": _space_from_pair(raw.get("n_estimators", [60, 180]), "int"),
                "max_depth": _space_from_pair(raw.get("max_depth", [2, 7]), "int"),
                "learning_rate": _space_from_pair(raw.get("learning_rate", [0.01, 0.12]), "real"),
            }
        elif name == "catboost":
            raw = spaces.get("catboost", {})
            search_space = {
                "iterations": _space_from_pair(raw.get("iterations", [80, 260]), "int"),
                "depth": _space_from_pair(raw.get("depth", [2, 7]), "int"),
                "learning_rate": _space_from_pair(raw.get("learning_rate", [0.01, 0.12]), "real"),
            }
        elif name == "extratrees":
            raw = spaces.get("extratrees", {})
            search_space = {
                "n_estimators": _space_from_pair(raw.get("n_estimators", [120, 360]), "int"),
                "max_depth": _space_from_pair(raw.get("max_depth", [3, 14]), "int"),
                "min_samples_leaf": _space_from_pair(raw.get("min_samples_leaf", [1, 8]), "int"),
            }
        else:
            tuned[name] = engine
            summary[name] = {"status": "skipped", "reason": "no search space"}
            continue

        opt = BayesSearchCV(
            estimator=engine,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
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
    arbiter_cfg = cfg.get("model", {}).get("arbiter", {}).get("ridge", {})
    return Ridge(alpha=float(arbiter_cfg.get("alpha", 1.0)))


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
    agreement_scale = float(ccfg.get("agreement_scale_return", 0.025)) # Increased from 0.015
    scale = max(agreement_scale, float(np.mean(np.abs(clean))) if clean else 0.0)
    agreement = float(scale / (scale + dispersion)) if scale > 0 else 0.0

    # MAE Component: more forgiving sigmoid
    mae_scale = float(ccfg.get("mae_reference_return", 0.020)) # Increased from 0.015
    if mae is None or not np.isfinite(float(mae)) or mae_scale <= 0:
        mae_component = 0.80
    else:
        mae_ratio = max(0.0, float(mae)) / mae_scale
        mae_component = 1.0 / (1.0 + 0.4 * (mae_ratio ** 1.2))

    # Magnitude: predictions near zero are less 'confident'
    action_scale = float(ccfg.get("action_scale_return", 0.0050))
    magnitude_ratio = min(1.0, abs(float(prediction)) / action_scale) if action_scale > 0 else 1.0
    neutral_floor = float(ccfg.get("neutral_prediction_component_floor", 0.60)) # Higher floor 0.60
    magnitude_component = neutral_floor + (1.0 - neutral_floor) * magnitude_ratio

    # Engine component
    guard_meta = guard_meta or {}
    used = len(guard_meta.get("used", []) or clean)
    discarded = len(guard_meta.get("discarded", []) or [])
    total = max(used + discarded, len(clean), 1)
    engine_component = max(0.50, used / total) # Higher floor 0.50

    # Sample Size
    if train_rows is None or int(train_rows or 0) <= 0:
        sample_component = 1.0
    else:
        reference_rows = max(1, int(ccfg.get("sample_reference_rows", 600))) # Lowered from 800
        sample_component = min(1.0, max(0.65, float(train_rows) / reference_rows)) # Higher floor 0.65

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


def train_models(cfg: dict[str, Any], ticker: str, X: pd.DataFrame, y: pd.Series, dataset_meta: dict[str, Any], autotune: bool = False, horizon: str = "d1") -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    min_rows = int(cfg.get("data", {}).get("min_rows", 220))
    if len(X) < min_rows:
        raise RuntimeError(f"insufficient rows for training: {len(X)} < {min_rows}")

    test_size = float(cfg.get("model", {}).get("test_size", 0.20))
    split = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Model-preparation contract: features are selected globally, but each engine
    # may have its own input scaler/subset. This keeps the tabular specialists on a consistent prepared data contract.
    X_latest = X.tail(1)
    base_engines = _make_base_engines(cfg)
    if len(base_engines) < 2:
        raise RuntimeError("at least two base engines are required for Ridge stacking")

    engine_inputs: dict[str, dict[str, Any]] = {}
    for name in base_engines:
        train_arr, test_arr, latest_arr, input_scaler, input_features = _fit_engine_input(cfg, name, X_train, X_test, X_latest)
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
            tuned_one, summary_one = _tune_base_engines(cfg, {name: engine}, engine_inputs[name]["train"], y_train)
            tuned.update(tuned_one)
            tune_summary.update(summary_one)
        base_engines = tuned

    base_order = list(base_engines.keys())
    guards = _prediction_guard_cfg(cfg)
    engine_clip = guards["max_engine_return_abs"]
    final_clip = guards["max_final_return_abs"]

    # --- Stacking with TimeSeriesSplit to avoid leakage ---
    tscv = TimeSeriesSplit(n_splits=int(cfg.get("model", {}).get("autotune", {}).get("cv", 3)))
    oof_predictions = {name: np.zeros(len(X_train)) for name in base_order}

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
            fold_train_arr = scaler.fit_transform(X_fold_train[cols])
            fold_val_arr = scaler.transform(X_fold_val[cols])
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                fold_engine.fit(fold_train_arr, y_fold_train)
            
            # Predict out-of-fold
            fold_pred = fold_engine.predict(fold_val_arr)
            oof_predictions[name][val_idx] = fold_pred

    # Only use indices where we have OOF predictions (first fold is skipped by TSS)
    valid_idx = np.where(np.any([oof_predictions[name] != 0 for name in base_order], axis=0))[0]
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

    meta_train_guarded, train_consensus_meta = _apply_consensus_guard(abs_meta_train, cfg)
    meta_test, test_consensus_meta = _apply_consensus_guard(abs_meta_test, cfg)
    meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_meta_latest, cfg)

    raw_pred_latest = {name: float(raw_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    neutralized_pred_latest, latest_discard_meta = _latest_engine_guard(raw_pred_latest, cfg)
    if latest_discard_meta.get("discarded"):
        meta_latest = np.array([[neutralized_pred_latest[name] for name in base_order]], dtype=float)
    
    pred_latest = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    abs_pred_latest = {name: float(abs_meta_latest[0, idx]) for idx, name in enumerate(base_order)}

    for idx, name in enumerate(base_order):
        metrics[name]["mae_return_abs_guarded"] = float(mean_absolute_error(y_test, abs_meta_test[:, idx]))
        metrics[name]["mae_return_consensus_guarded"] = float(mean_absolute_error(y_test, meta_test[:, idx]))
        metrics[name]["latest_raw_return"] = raw_pred_latest[name]
        metrics[name]["latest_guarded_return"] = pred_latest[name]

    # --- Fit Arbiter on OOF meta-features ---
    arbiter = _make_arbiter(cfg)
    arbiter.fit(meta_train_guarded, y_train_oof)
    
    arbiter_test_pred = np.asarray(arbiter.predict(meta_test), dtype=float)
    arbiter_latest_raw = float(arbiter.predict(meta_latest)[0])
    arbiter_latest = _clip_return_float(arbiter_latest_raw, final_clip)
    arbiter_mae = float(mean_absolute_error(y_test, arbiter_test_pred))

    dispersion, confidence = _confidence_from(list(pred_latest.values()), arbiter_latest, cfg, mae=arbiter_mae, guard_meta=latest_discard_meta, train_rows=len(X_train))

    rid = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{horizon}_{safe_ticker(ticker)}"
    out_dir = artifact_dir(cfg) / safe_ticker(ticker) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{horizon}.pkl"
    payload = {
        "ticker": ticker,
        "features": list(X.columns),
        "engine_inputs": {name: {"features": engine_inputs[name]["features"], "scaler": engine_inputs[name]["scaler"]} for name in base_order},
        "normalization": (cfg.get("features", {}).get("preparation", {}) or {}).get("normalization", {}),
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
    top_features = top_selected_features(prep_meta, list(X.columns), n=5)
    family_profile = feature_family_profile(list(X.columns))

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
        "features": list(X.columns),
        "top_features": top_features,
        "feature_family_profile": family_profile,
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
        "dataset_meta": dataset_meta,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(artifact_dir(cfg) / safe_ticker(ticker) / f"latest_train_{horizon}.json", manifest)
    return manifest


def load_latest_model(cfg: dict[str, Any], ticker: str, horizon: str = "d1") -> tuple[dict[str, Any], dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    latest = artifact_dir(cfg) / safe_ticker(ticker) / f"latest_train_{horizon}.json"
    if not latest.exists():
        alt = latest_file(artifact_dir(cfg) / safe_ticker(ticker), f"train_*_{horizon}_*/manifest.json")
        if not alt:
            # Fallback to generic search if suffix not found (for legacy models)
            alt = latest_file(artifact_dir(cfg) / safe_ticker(ticker), "train_*/manifest.json")
            if not alt:
                raise FileNotFoundError(f"no trained model found for {ticker} horizon {horizon}; run train first")
        latest = alt
    manifest = read_json(latest)
    with Path(manifest["model_path"]).open("rb") as fh:
        model = pickle.load(fh)
    return model, manifest


def predict_with_model(cfg: dict[str, Any], ticker: str, X: pd.DataFrame, horizon: str = "d1") -> dict[str, Any]:
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
            engine_input = _transform_engine_input(engine_meta, name, latest_X) if engine_meta else latest_X.to_numpy(dtype=float)
            raw_by_engine[name] = float(model["base_models"][name].predict(engine_input)[0])
        abs_values = np.array([[_clip_return_float(raw_by_engine[name], engine_clip) for name in base_order]], dtype=float)
        neutralized_by_engine, latest_engine_guard = _latest_engine_guard(raw_by_engine, cfg)
        if latest_engine_guard.get("discarded"):
            meta_latest = np.array([[neutralized_by_engine[name] for name in base_order]], dtype=float)
            latest_consensus_meta = {"enabled": True, "changed_values": len(latest_engine_guard.get("discarded", [])), "neutralized": True}
        else:
            meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_values, cfg)
        by_engine = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
        abs_by_engine = {name: float(abs_values[0, idx]) for idx, name in enumerate(base_order)}
        raw_prediction = float(model["arbiter"].predict(meta_latest)[0])
        prediction = _clip_return_float(raw_prediction, final_clip)
        ridge_metrics = (manifest.get("metrics", {}) or {}).get("ridge_arbiter", {}) or {}
        mae = ridge_metrics.get("mae_return")
        train_rows = manifest.get("train_rows")
        dispersion, confidence = _confidence_from(list(by_engine.values()), prediction, cfg, mae=mae, guard_meta=latest_engine_guard, train_rows=train_rows)
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
            "train_manifest": manifest,
        }

    # Compatibility path for older artifacts produced before this correction.
    by_engine = {name: float(est.predict(latest_X)[0]) for name, est in model["models"].items()}
    prediction = float(np.mean(list(by_engine.values())))
    dispersion, confidence = _confidence_from(list(by_engine.values()), prediction, cfg)
    return {
        "ticker": normalize_ticker(ticker),
        "architecture": "legacy_mean_ensemble",
        "prediction_return": prediction,
        "by_engine": by_engine,
        "dispersion": dispersion,
        "confidence": confidence,
        "train_manifest": manifest,
    }

def predict_multi_horizon(cfg: dict[str, Any], ticker: str, X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Generate predictions for all available horizons."""
    horizons = ["d1", "d5", "d20"]
    results = {}
    for h in horizons:
        try:
            results[h] = predict_with_model(cfg, ticker, X, horizon=h)
        except Exception as exc:
            results[h] = {"error": str(exc)}
    return results
