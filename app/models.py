from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning

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

    The Ridge arbiter should judge specialists, not be dominated by a single MLP
    excursion. This guard is row-wise: each engine prediction is clipped to a
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
    """Create the specialist/base engines.

    Original TradeGem architecture is preserved here:
      XGB + RandomForest + MLP -> Ridge arbiter.

    Ridge is deliberately not a base engine. It is the stacking judge trained on the
    predictions produced by the three specialists.
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
            # Keep the CLI usable when xgboost is not installed. RF + MLP can still train.
            pass

    if engines_cfg.get("rf", {}).get("enabled", True):
        ecfg = engines_cfg.get("rf", {})
        engines["rf"] = RandomForestRegressor(
            n_estimators=int(ecfg.get("n_estimators", 220)),
            max_depth=int(ecfg.get("max_depth", 8)),
            random_state=random_state,
            n_jobs=-1,
        )

    if engines_cfg.get("mlp", {}).get("enabled", True):
        ecfg = engines_cfg.get("mlp", {})
        hidden = ecfg.get("hidden_layer_sizes", [64, 32])
        # Fixed training config uses the sklearn meaning: [64, 32] = two hidden layers.
        # Autotune search config may still use scalar alternatives such as [32, 64].
        if isinstance(hidden, list):
            hidden_layer_sizes = tuple(int(v) for v in hidden) if hidden else (64,)
        elif isinstance(hidden, tuple):
            hidden_layer_sizes = tuple(int(v) for v in hidden) if hidden else (64,)
        else:
            hidden_layer_sizes = (int(hidden),)
        engines["mlp"] = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=str(ecfg.get("activation", "relu")),
            solver=str(ecfg.get("solver", "adam")),
            alpha=float(ecfg.get("alpha", 0.0005)),
            learning_rate=str(ecfg.get("learning_rate", "adaptive")),
            learning_rate_init=float(ecfg.get("learning_rate_init", 0.001)),
            max_iter=int(ecfg.get("max_iter", 800)),
            random_state=random_state,
            early_stopping=bool(ecfg.get("early_stopping", True)),
            validation_fraction=float(ecfg.get("validation_fraction", 0.12)),
            n_iter_no_change=int(ecfg.get("n_iter_no_change", 30)),
        )
    return engines


def _tune_base_engines(cfg: dict[str, Any], base_engines: dict[str, Any], X_train: np.ndarray, y_train: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune specialists using the original working TradeGem pattern.

    Contract restored from the original project:
      - global StandardScaler is applied before this function;
      - MLPRegressor is tuned directly, not through a Pipeline;
      - MLP hidden_layer_sizes uses scalar choices like [32, 64];
      - specialists remain XGB + RandomForest + MLP;
      - Ridge remains only the stacking arbiter.
    """
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
                "n_estimators": raw.get("n_estimators", [50, 150]),
                "max_depth": raw.get("max_depth", [3, 7]),
            }
        elif name == "rf":
            raw = spaces.get("rf", {})
            search_space = {
                "n_estimators": raw.get("n_estimators", [50, 150]),
                "max_depth": raw.get("max_depth", [5, 10]),
            }
        elif name == "mlp":
            raw = spaces.get("mlp", {})
            # Original working code used scalar hidden sizes directly: [32, 64].
            hidden = raw.get("hidden_layer_sizes", [32, 64])
            alpha = raw.get("alpha", [0.0001, 0.01])
            search_space = {
                "hidden_layer_sizes": hidden,
                "alpha": alpha,
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
            "status": "tuned_original_pattern",
            "best_score": float(opt.best_score_),
            "best_params": opt.best_params_,
        }

    return tuned, summary

def _make_arbiter(cfg: dict[str, Any]) -> Ridge:
    arbiter_cfg = cfg.get("model", {}).get("arbiter", {})
    return Ridge(alpha=float(arbiter_cfg.get("alpha", 1.0)))


def _confidence_from(values: list[float], prediction: float, cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """Estimate operational confidence from agreement among base engines.

    The previous formula divided dispersion by the absolute prediction. When the
    Ridge arbiter produced a small return, confidence collapsed to 0% even when
    the engines were merely disagreeing around a neutral forecast. This version
    uses a configurable return scale and never reports a hard zero when base
    engines exist. It is still an agreement score, not statistical certainty.
    """
    clean = [float(v) for v in values if np.isfinite(float(v))]
    dispersion = float(np.std(clean)) if clean else 0.0
    if not clean:
        return dispersion, 0.0

    ccfg = (cfg or {}).get("model", {}).get("confidence", {})
    scale = max(
        abs(float(prediction)),
        float(ccfg.get("agreement_scale_return", 0.010)),
        float(np.mean(np.abs(clean))) if clean else 0.0,
    )
    confidence = float(scale / (scale + dispersion)) if scale > 0 else 0.0
    floor = float(ccfg.get("minimum_when_engines_exist", 0.05))
    confidence = min(1.0, max(floor, confidence))
    return dispersion, confidence


def train_models(cfg: dict[str, Any], ticker: str, X: pd.DataFrame, y: pd.Series, dataset_meta: dict[str, Any], autotune: bool = False) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    min_rows = int(cfg.get("data", {}).get("min_rows", 220))
    if len(X) < min_rows:
        raise RuntimeError(f"insufficient rows for training: {len(X)} < {min_rows}")

    test_size = float(cfg.get("model", {}).get("test_size", 0.20))
    split = max(1, int(len(X) * (1 - test_size)))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Model-preparation contract: all engines receive the same normalized matrix.
    scaler = _make_scaler(cfg)
    if scaler is not None:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_latest_s = scaler.transform(X.tail(1))
    else:
        X_train_s = X_train.to_numpy(dtype=float)
        X_test_s = X_test.to_numpy(dtype=float)
        X_latest_s = X.tail(1).to_numpy(dtype=float)

    base_engines = _make_base_engines(cfg)
    if len(base_engines) < 2:
        raise RuntimeError("at least two base engines are required for Ridge stacking")

    tune_summary: dict[str, Any] = {}
    if autotune:
        base_engines, tune_summary = _tune_base_engines(cfg, base_engines, X_train_s, y_train)

    fitted: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    raw_train_cols: list[np.ndarray] = []
    raw_test_cols: list[np.ndarray] = []
    raw_latest_values: list[float] = []

    guards = _prediction_guard_cfg(cfg)
    engine_clip = guards["max_engine_return_abs"]
    final_clip = guards["max_final_return_abs"]

    for name, engine in base_engines.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            engine.fit(X_train_s, y_train)
        fitted[name] = engine
        raw_train_pred = np.asarray(engine.predict(X_train_s), dtype=float)
        raw_test_pred = np.asarray(engine.predict(X_test_s), dtype=float)
        raw_latest_pred = float(engine.predict(X_latest_s)[0])
        raw_train_cols.append(raw_train_pred)
        raw_test_cols.append(raw_test_pred)
        raw_latest_values.append(raw_latest_pred)
        metrics[name] = {
            "mae_return_raw": float(mean_absolute_error(y_test, raw_test_pred)),
        }

    base_order = list(fitted.keys())
    raw_meta_train = np.column_stack(raw_train_cols)
    raw_meta_test = np.column_stack(raw_test_cols)
    raw_meta_latest = np.array([raw_latest_values], dtype=float)

    abs_meta_train = _clip_return_array(raw_meta_train, engine_clip)
    abs_meta_test = _clip_return_array(raw_meta_test, engine_clip)
    abs_meta_latest = _clip_return_array(raw_meta_latest, engine_clip)

    meta_train, train_consensus_meta = _apply_consensus_guard(abs_meta_train, cfg)
    meta_test, test_consensus_meta = _apply_consensus_guard(abs_meta_test, cfg)
    meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_meta_latest, cfg)

    raw_pred_latest = {name: float(raw_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    pred_latest = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    abs_pred_latest = {name: float(abs_meta_latest[0, idx]) for idx, name in enumerate(base_order)}

    for idx, name in enumerate(base_order):
        metrics[name]["mae_return_abs_guarded"] = float(mean_absolute_error(y_test, abs_meta_test[:, idx]))
        metrics[name]["mae_return_consensus_guarded"] = float(mean_absolute_error(y_test, meta_test[:, idx]))
        metrics[name]["latest_raw_return"] = raw_pred_latest[name]
        metrics[name]["latest_guarded_return"] = pred_latest[name]

    arbiter = _make_arbiter(cfg)
    arbiter.fit(meta_train, y_train)
    arbiter_test_pred = np.asarray(arbiter.predict(meta_test), dtype=float)
    arbiter_latest_raw = float(arbiter.predict(meta_latest)[0])
    arbiter_latest = _clip_return_float(arbiter_latest_raw, final_clip)
    arbiter_mae = float(mean_absolute_error(y_test, arbiter_test_pred))

    dispersion, confidence = _confidence_from(list(pred_latest.values()), arbiter_latest, cfg)

    rid = run_id("train", ticker)
    out_dir = artifact_dir(cfg) / safe_ticker(ticker) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    payload = {
        "ticker": ticker,
        "features": list(X.columns),
        "scaler": scaler,
        "normalization": (cfg.get("features", {}).get("preparation", {}) or {}).get("normalization", {}),
        "base_models": fitted,
        "base_order": base_order,
        "arbiter": arbiter,
        "architecture": "xgb_rf_mlp__ridge_arbiter_original_autotune",
        "autotune": bool(autotune),
        "tune_summary": tune_summary,
        "pred_latest_by_engine": pred_latest,
        "raw_pred_latest_by_engine": raw_pred_latest,
        "prediction_guards": guards,
        "engine_safety": _engine_safety_cfg(cfg),
        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)

    prep_meta = (dataset_meta or {}).get("preparation", {}) or {}
    top_features = top_selected_features(prep_meta, list(X.columns), n=5)
    family_profile = feature_family_profile(list(X.columns))

    manifest = {
        "run_id": rid,
        "ticker": ticker,
        "model_path": str(model_path),
        "created_at": pd.Timestamp.now().isoformat(),
        "architecture": "XGB + RandomForest + MLP -> Ridge arbiter",
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
    write_json(artifact_dir(cfg) / safe_ticker(ticker) / "latest_train.json", manifest)
    return manifest


def load_latest_model(cfg: dict[str, Any], ticker: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ticker = normalize_ticker(ticker)
    latest = artifact_dir(cfg) / safe_ticker(ticker) / "latest_train.json"
    if not latest.exists():
        alt = latest_file(artifact_dir(cfg) / safe_ticker(ticker), "train_*/manifest.json")
        if not alt:
            raise FileNotFoundError(f"no trained model found for {ticker}; run train first")
        latest = alt
    manifest = read_json(latest)
    with Path(manifest["model_path"]).open("rb") as fh:
        model = pickle.load(fh)
    return model, manifest


def predict_with_model(cfg: dict[str, Any], ticker: str, X: pd.DataFrame) -> dict[str, Any]:
    model, manifest = load_latest_model(cfg, ticker)
    features = model["features"]
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise RuntimeError(f"current dataset is missing trained features: {missing[:8]}")

    latest_X = X[features].tail(1)

    # Current architecture: base specialists -> Ridge arbiter.
    if "base_models" in model and "arbiter" in model:
        base_order = model.get("base_order") or list(model["base_models"].keys())
        latest_input = model["scaler"].transform(latest_X) if model.get("scaler") is not None else latest_X.to_numpy(dtype=float)
        guards = model.get("prediction_guards") or _prediction_guard_cfg(cfg)
        engine_clip = float(guards.get("max_engine_return_abs", 0.12))
        final_clip = float(guards.get("max_final_return_abs", 0.08))
        raw_by_engine = {name: float(model["base_models"][name].predict(latest_input)[0]) for name in base_order}
        abs_values = np.array([[_clip_return_float(raw_by_engine[name], engine_clip) for name in base_order]], dtype=float)
        meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_values, cfg)
        by_engine = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
        abs_by_engine = {name: float(abs_values[0, idx]) for idx, name in enumerate(base_order)}
        raw_prediction = float(model["arbiter"].predict(meta_latest)[0])
        prediction = _clip_return_float(raw_prediction, final_clip)
        dispersion, confidence = _confidence_from(list(by_engine.values()), prediction, cfg)
        return {
            "ticker": normalize_ticker(ticker),
            "architecture": "XGB + RandomForest + MLP -> Ridge arbiter",
            "prediction_return": prediction,
            "by_engine": by_engine,
            "raw_by_engine": raw_by_engine,
            "abs_guarded_by_engine": abs_by_engine,
            "raw_prediction_return": raw_prediction,
            "consensus_guard": latest_consensus_meta,
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
