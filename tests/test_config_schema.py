import pytest

from app.config import load_config
from app.config_schema import assert_valid_config, normalize_config, validate_config


def test_config_schema_accepts_current_config():
    cfg = load_config()
    errors = [issue for issue in validate_config(cfg) if issue.severity == "error"]
    assert errors == []


def test_config_schema_units():
    cfg = normalize_config(
        {
            "app": {"artifact_dir": "artifacts", "data_cache_dir": "data/cache"},
            "features_file": "features.yaml",
            "simulation": {
                "mode": "invalid",
                "initial_cash": -1,
                "max_positions": 0,
                "costs": {"fee_mode": "mystery", "fee_amount": -0.1, "slippage_pct": -0.2},
            },
            "data": {"min_rows": 150, "macro_tickers": "^BVSP"},
            "batch": {"train_workers": -1},
            "trading": {"capital": 10000.0, "trade_management": {"max_hold_days": {"d99": 1}}},
            "model": {
                "confidence": {"maximum_confidence": 1.5},
                "prediction_guards": {"max_engine_return_abs": -1},
            },
        }
    )
    issues = validate_config(cfg)
    error_paths = {issue.path for issue in issues if issue.severity == "error"}
    assert {
        "simulation.mode",
        "simulation.initial_cash",
        "simulation.max_positions",
        "simulation.costs.fee_mode",
        "simulation.costs.fee_amount",
        "simulation.costs.slippage_pct",
        "data.macro_tickers",
        "batch.train_workers",
        "trading.trade_management.max_hold_days",
        "model.confidence.maximum_confidence",
        "model.prediction_guards.max_engine_return_abs",
    }.issubset(error_paths)
    with pytest.raises(ValueError, match="Invalid TradeChat config"):
        assert_valid_config(cfg)
