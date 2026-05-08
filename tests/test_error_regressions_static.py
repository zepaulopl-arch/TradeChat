from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fundamentals_disabled_still_report_snapshot():
    text = (ROOT / "app" / "fundamentals.py").read_text(encoding="utf-8")
    assert "snapshot_report_only_disabled_as_feature" in text
    assert "Selection controls whether fundamentals enter the model" in text


def test_prediction_guards_exist_and_clip_outliers():
    text = (ROOT / "app" / "models.py").read_text(encoding="utf-8")
    assert "max_engine_return_abs" in text
    assert "max_final_return_abs" in text
    assert "_clip_return_float" in text
    assert "def _latest_engine_guard" in text
    assert "def _oof_valid_mask" in text


def test_policy_confidence_floor_is_conservative():
    cfg = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    assert "min_confidence_pct: 0.45" in cfg
    assert "high_confidence_pct: 0.7" in cfg
    policy = (ROOT / "app" / "policy.py").read_text(encoding="utf-8")
    assert "_confidence_floor_pct" in policy
    assert "confidence below floor" in policy


def test_external_features_do_not_shrink_rows_by_initial_nans():
    text = (ROOT / "app" / "features.py").read_text(encoding="utf-8")
    assert "external_prefixes" in text
    assert "ctx_" in text and "sent_" in text and "fund_" in text
