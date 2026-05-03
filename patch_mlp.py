from pathlib import Path
p=Path('/mnt/data/mlp_patch/app/preparation.py')
s=p.read_text()
# add helper _family_maxed and update signature/function body wholesale
start=s.index('def _greedy_low_correlation_selection(')
end=s.index('\ndef prepare_training_matrix', start)
new=r'''def _greedy_low_correlation_selection(
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
        "method": "target_relevance_low_correlation_family_balanced_greedy",
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
'''
s=s[:start]+new+s[end:]
s=s.replace('family_minimums = select_cfg.get("family_minimums", {}) or {}\n        selected, selection_meta = _greedy_low_correlation_selection(X1, y1, max_features, threshold, family_minimums)',
'''family_minimums = select_cfg.get("family_minimums", {}) or {}
        family_limits = select_cfg.get("family_limits", {}) or {}
        selected, selection_meta = _greedy_low_correlation_selection(X1, y1, max_features, threshold, family_minimums, family_limits)''')
p.write_text(s)

# Patch models.py helpers and training/predict
p=Path('/mnt/data/mlp_patch/app/models.py')
s=p.read_text()
s=s.replace('from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\n', 'from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nfrom .feature_audit import feature_family\n')
insert=r'''

def _make_named_scaler(name: str):
    scaler_name = str(name or "standard").lower()
    if scaler_name == "robust":
        return RobustScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def _mlp_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return ((cfg.get("model", {}) or {}).get("engines", {}) or {}).get("mlp", {}) or {}


def _engine_feature_columns(cfg: dict[str, Any], engine_name: str, all_features: list[str]) -> list[str]:
    """Return the feature subset used by a given engine.

    Tree engines can handle the full prepared matrix. The MLP is more sensitive to
    noisy heterogeneous inputs, so it may use a smaller family-filtered subset.
    """
    if engine_name != "mlp":
        return list(all_features)
    ecfg = _mlp_cfg(cfg)
    input_cfg = ecfg.get("input", {}) or {}
    if not bool(input_cfg.get("enabled", True)):
        return list(all_features)
    families = input_cfg.get("families", ["technical", "context"])
    families = {str(f) for f in families}
    selected = [c for c in all_features if feature_family(c) in families]
    if not selected:
        selected = list(all_features)
    max_features = int(input_cfg.get("max_features", 15) or 0)
    if max_features > 0:
        selected = selected[:max_features]
    return selected


def _make_engine_scaler(cfg: dict[str, Any], engine_name: str):
    if engine_name != "mlp":
        return _make_scaler(cfg)
    ecfg = _mlp_cfg(cfg)
    input_cfg = ecfg.get("input", {}) or {}
    scaler_name = input_cfg.get("scaler", ecfg.get("scaler", "robust"))
    return _make_named_scaler(str(scaler_name))


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

    A clipped MLP should not keep pushing the arbiter. If an engine is outside the
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
'''
# insert after _make_scaler
idx=s.index('\ndef _prediction_guard_cfg')
s=s[:idx]+insert+s[idx:]
# Replace train block from scaler comment to base_engines creation
old='''    # Model-preparation contract: all engines receive the same normalized matrix.
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
'''
new='''    # Model-preparation contract: features are selected globally, but each engine
    # may have its own input scaler/subset. This is critical for MLP stability.
    X_latest = X.tail(1)
    base_engines = _make_base_engines(cfg)
'''
s=s.replace(old,new)
# autotune replace uses X_train_s var. We'll create engine_inputs before autotune? need after base_engines
s=s.replace('''    tune_summary: dict[str, Any] = {}
    if autotune:
        base_engines, tune_summary = _tune_base_engines(cfg, base_engines, X_train_s, y_train)

    fitted: dict[str, Any] = {}
''','''    engine_inputs: dict[str, dict[str, Any]] = {}
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

    fitted: dict[str, Any] = {}
''')
# replace fit/predict loop variables
s=s.replace('''            engine.fit(X_train_s, y_train)
        fitted[name] = engine
        raw_train_pred = np.asarray(engine.predict(X_train_s), dtype=float)
        raw_test_pred = np.asarray(engine.predict(X_test_s), dtype=float)
        raw_latest_pred = float(engine.predict(X_latest_s)[0])''','''            engine.fit(engine_inputs[name]["train"], y_train)
        fitted[name] = engine
        raw_train_pred = np.asarray(engine.predict(engine_inputs[name]["train"]), dtype=float)
        raw_test_pred = np.asarray(engine.predict(engine_inputs[name]["test"]), dtype=float)
        raw_latest_pred = float(engine.predict(engine_inputs[name]["latest"])[0])''')
# insert latest guard after raw_meta_latest
s=s.replace('''    raw_pred_latest = {name: float(raw_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    pred_latest = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    abs_pred_latest = {name: float(abs_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
''','''    raw_pred_latest = {name: float(raw_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    neutralized_pred_latest, latest_discard_meta = _latest_engine_guard(raw_pred_latest, cfg)
    if latest_discard_meta.get("discarded"):
        meta_latest = np.array([[neutralized_pred_latest[name] for name in base_order]], dtype=float)
    pred_latest = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
    abs_pred_latest = {name: float(abs_meta_latest[0, idx]) for idx, name in enumerate(base_order)}
''')
# payload scaler replacement
s=s.replace('''        "features": list(X.columns),
        "scaler": scaler,
        "normalization": (cfg.get("features", {}).get("preparation", {}) or {}).get("normalization", {}),
        "base_models": fitted,''','''        "features": list(X.columns),
        "engine_inputs": {name: {"features": engine_inputs[name]["features"], "scaler": engine_inputs[name]["scaler"]} for name in base_order},
        "normalization": (cfg.get("features", {}).get("preparation", {}) or {}).get("normalization", {}),
        "base_models": fitted,''')
s=s.replace('''        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
    }''','''        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
        "latest_engine_guard": latest_discard_meta,
    }''',1)
# manifest add engine_input_features and guard
s=s.replace('''        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
        "consensus_guard": {''','''        "latest_prediction_by_engine_abs_guarded": abs_pred_latest,
        "latest_engine_guard": latest_discard_meta,
        "engine_input_features": {name: engine_inputs[name]["features"] for name in base_order},
        "consensus_guard": {''')
# predict input block replace
old='''        latest_input = model["scaler"].transform(latest_X) if model.get("scaler") is not None else latest_X.to_numpy(dtype=float)
        guards = model.get("prediction_guards") or _prediction_guard_cfg(cfg)
        engine_clip = float(guards.get("max_engine_return_abs", 0.12))
        final_clip = float(guards.get("max_final_return_abs", 0.08))
        raw_by_engine = {name: float(model["base_models"][name].predict(latest_input)[0]) for name in base_order}
        abs_values = np.array([[_clip_return_float(raw_by_engine[name], engine_clip) for name in base_order]], dtype=float)
        meta_latest, latest_consensus_meta = _apply_consensus_guard(abs_values, cfg)
        by_engine = {name: float(meta_latest[0, idx]) for idx, name in enumerate(base_order)}
        abs_by_engine = {name: float(abs_values[0, idx]) for idx, name in enumerate(base_order)}
'''
new='''        guards = model.get("prediction_guards") or _prediction_guard_cfg(cfg)
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
'''
s=s.replace(old,new)
# predict return add guard
s=s.replace('''            "raw_prediction_return": raw_prediction,
            "consensus_guard": latest_consensus_meta,''','''            "raw_prediction_return": raw_prediction,
            "consensus_guard": latest_consensus_meta,
            "latest_engine_guard": latest_engine_guard,
            "used_engines": latest_engine_guard.get("used", base_order),
            "discarded_engines": latest_engine_guard.get("discarded", []),''')
p.write_text(s)

# Patch report for used/guard wording
p=Path('/mnt/data/mlp_patch/app/report.py')
s=p.read_text()
s=s.replace('''        print(f"base engines: {line}")
        raw_engines = signal.get("prediction", {}).get("raw_by_engine", {}) or {}
        if raw_engines and any(abs(float(raw_engines.get(k, 0))) > abs(float(v)) + 1e-12 for k, v in engines.items()):
            raw_line = " | ".join([f"{k} {v*100:+.2f}%" for k, v in raw_engines.items()])
            print(f"raw engines : {raw_line}")
            print("guard       : extreme base prediction clipped before Ridge")
        print(f"arbiter     : {arbiter}")''','''        print(f"base engines: {line}")
        raw_engines = signal.get("prediction", {}).get("raw_by_engine", {}) or {}
        discarded = signal.get("prediction", {}).get("discarded_engines", []) or []
        used = signal.get("prediction", {}).get("used_engines", []) or []
        if raw_engines and any(abs(float(raw_engines.get(k, 0))) > abs(float(v)) + 1e-12 for k, v in engines.items()):
            raw_line = " | ".join([f"{k} {v*100:+.2f}%" for k, v in raw_engines.items()])
            print(f"raw engines : {raw_line}")
        if discarded:
            print(f"used engines: {', '.join(used) if used else 'none'}")
            print(f"guard       : {', '.join(discarded)} neutralized before Ridge")
        print(f"arbiter     : {arbiter}")''')
p.write_text(s)

# Patch YAML
p=Path('/mnt/data/mlp_patch/config/features.yaml')
s=p.read_text()
s=s.replace('''    method: target_relevance_low_correlation_greedy
    max_features: 20
    correlation_threshold: 0.85
    family_minimums:
      technical: 8
      context: 4
      fundamentals: 1
      sentiment: 1
    note: Ranks features by simple target relevance and rejects candidates too correlated''','''    method: target_relevance_low_correlation_family_balanced_greedy
    max_features: 20
    correlation_threshold: 0.85
    family_limits:
      technical:
        min: 8
        max: 10
      context:
        min: 3
        max: 5
      fundamentals:
        min: 0
        max: 3
      sentiment:
        min: 0
        max: 2
    note: Ranks features by simple target relevance, rejects candidates too correlated, and prevents one family from dominating''')
p.write_text(s)

# Patch config.yaml model mlp input
p=Path('/mnt/data/mlp_patch/config/config.yaml')
s=p.read_text()
needle='''      early_stopping: true
'''
rep='''      early_stopping: true
      scaler: robust
      input:
        enabled: true
        scaler: robust
        max_features: 15
        families:
        - technical
        - context
'''
s=s.replace(needle, rep)
# ensure guards maybe lower clip? inspect not needed
p.write_text(s)

# add docs/test
Path('/mnt/data/mlp_patch/docs/ENGINE_MLP_FEATURE_BALANCE.md').write_text('''# Engine and MLP feature balance\n\nThis patch keeps the original architecture `XGB + RandomForest + MLP -> Ridge`, but hardens the MLP and feature preparation.\n\n- Feature preparation is family-balanced: context can no longer dominate the selected feature set.\n- The MLP receives its own robust-scaled subset, normally technical + context features only.\n- Divergent latest engine predictions are neutralized before the Ridge arbiter and shown in the signal screen.\n- Raw predictions remain available for audit.\n''', encoding='utf-8')
Path('/mnt/data/mlp_patch/tests/test_mlp_feature_balance_static.py').write_text('''from pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\n\n\ndef test_features_yaml_has_family_limits():\n    text = (ROOT / "config" / "features.yaml").read_text(encoding="utf-8")\n    assert "family_limits:" in text\n    assert "context:" in text\n    assert "max: 5" in text\n    assert "target_relevance_low_correlation_family_balanced_greedy" in text\n\n\ndef test_mlp_has_own_scaler_and_subset():\n    text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")\n    assert "def _engine_feature_columns" in text\n    assert "def _fit_engine_input" in text\n    assert "def _latest_engine_guard" in text\n    assert "feature_family" in text\n\n\ndef test_signal_reports_used_and_discarded_engines():\n    text = (ROOT / "app" / "report.py").read_text(encoding="utf-8")\n    assert "used engines:" in text\n    assert "neutralized before Ridge" in text\n''', encoding='utf-8')
