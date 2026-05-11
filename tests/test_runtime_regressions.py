from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import features as feature_builder
from app import models as model_module
from app import portfolio_monitor_service as monitor
from app.commands import portfolio_command
from app.evaluation_service import (
    compare_model_to_baselines,
    enrich_model_metrics_from_execution,
    evaluate_baselines,
)
from app.methodology_service import methodology_report
from app.models import _make_arbiter, _oof_valid_mask, _split_train_test_raw, _stacking_cv_splits
from app.policy import classify_signal
from app.portfolio_service import get_state_db_path, load_portfolio_state, save_portfolio_state
from app.preparation import prepare_training_matrix
from app.refine_service import (
    collect_refine_summary,
    render_refine_summary,
    render_removal_summary,
    render_removal_walkforward_summary,
    run_feature_removal,
    run_feature_removal_walkforward,
)
from app.scoring import is_actionable_signal
from app.simulation.runner import (
    _build_strategy_config,
    _execution_fn_factory,
    _normalize_signal_plan,
    _planned_entry_shares,
    _position_size_handler_factory,
    _schedule_rebalance_dates,
    _write_simulation_artifacts,
)
from app.trade_plan_service import build_trade_plan
from app.ui import model5 as ui5
from app.validation_view import render_validation_summary


def test_live_monitor_closes_short_using_signed_cash_flow(monkeypatch):
    cfg = {"trading": {"capital": 1000.0}}
    portfolio = {
        "account": {"initial_capital": 1000.0, "cash": 1100.0, "currency": "BRL"},
        "positions": {
            "VALE3.SA": {
                "ticker": "VALE3.SA",
                "entry_price": 10.0,
                "shares": -10,
                "target_partial": 9.5,
                "target_final": 9.0,
                "stop_loss": 11.0,
                "entry_date": "2026-05-06",
                "partial_executed": False,
                "status": "active",
                "side": "SHORT",
            }
        },
        "history": [],
    }
    saved = []
    monkeypatch.setattr(
        monitor,
        "load_latest_signal",
        lambda cfg, ticker: {"policy": {"target_price": 9.0, "stop_loss_price": 11.0}},
    )
    monkeypatch.setattr(monitor, "get_live_price", lambda ticker: 9.0)
    monkeypatch.setattr(monitor, "save_portfolio_state", lambda portfolio: saved.append(portfolio))

    events = monitor._close_positions_on_live_triggers(cfg, portfolio)

    assert events[-1]["action"] == "TARGET"
    assert portfolio["positions"] == {}
    assert portfolio["account"]["cash"] == pytest.approx(1010.0)
    assert portfolio["history"][-1]["pl_cash"] == pytest.approx(10.0)
    assert saved


def test_oof_valid_mask_keeps_zero_predictions_and_rejects_missing_rows():
    mask = _oof_valid_mask(
        {
            "xgb": np.array([0.0, np.nan, 0.01]),
            "catboost": np.array([0.0, 0.02, 0.03]),
            "extratrees": np.array([0.0, 0.01, np.nan]),
        }
    )
    assert mask.tolist() == [True, False, False]


def test_make_arbiter_uses_ridge_config():
    arbiter = _make_arbiter(
        {"model": {"arbiter": {"ridge": {"alpha": 2.5, "fit_intercept": False}}}}
    )
    assert arbiter.alpha == pytest.approx(2.5)
    assert arbiter.fit_intercept is False


def test_stacking_cv_uses_dedicated_config():
    assert _stacking_cv_splits({"model": {"stacking": {"cv": 4}, "autotune": {"cv": 2}}}, 100) == 4
    assert _stacking_cv_splits({"model": {"stacking": {"cv": 4}}}, 3) == 2


def test_train_holdout_uses_horizon_embargo_and_raw_target():
    index = pd.date_range("2025-01-01", periods=120, freq="B")
    X = pd.DataFrame({"ret_1": np.linspace(-0.01, 0.01, len(index))}, index=index)
    y = pd.Series(np.linspace(-0.20, 0.20, len(index)), index=index)
    cfg = {
        "data": {"min_rows": 100},
        "model": {
            "test_size": 0.20,
            "validation": {
                "embargo_by_horizon": True,
                "embargo_bars": "auto",
                "min_train_rows": 60,
            },
        },
    }

    X_train, y_train, X_test, y_test, meta = _split_train_test_raw(cfg, X, y, horizon="d20")

    assert len(X_train) == 76
    assert len(X_test) == 24
    assert meta["embargo_bars"] == 20
    assert meta["dropped_embargo_rows"] == 20
    assert meta["test_target"] == "raw_unclipped"
    assert y_train.iloc[-1] == pytest.approx(y.iloc[75])
    assert y_test.iloc[0] == pytest.approx(y.iloc[96])


def test_build_dataset_excludes_non_stationary_price_levels(monkeypatch):
    monkeypatch.setattr(
        feature_builder,
        "add_fundamental_features",
        lambda dataset, ticker, cfg: (dataset, {"enabled": False, "source": "test"}),
    )
    index = pd.date_range("2024-01-01", periods=180, freq="B")
    trend = np.linspace(10.0, 20.0, len(index))
    noise = np.sin(np.arange(len(index)) / 4.0) * 0.2
    prices = pd.DataFrame({"PETR4.SA": trend + noise}, index=index)
    cfg = {
        "features": {
            "technical": {
                "frac_diff_d": 0.5,
                "vol_windows": [5, 20, 60],
                "rsi_window": 14,
                "sma_short": 10,
                "sma_long": 50,
                "ema_short": 20,
                "ema_long": 100,
                "roc_window": 10,
            },
            "context": {"enabled": False},
            "fundamentals": {"enabled": False},
            "sentiment": {"enabled": False},
            "preparation": {
                "stationarity": {
                    "drop_raw_price": True,
                    "drop_level_features": True,
                }
            },
        }
    }

    X, _all_y, meta = feature_builder.build_dataset(cfg, prices, "PETR4.SA")

    blocked = {"PETR4.SA", "frac_mem", "sma_10", "sma_50", "ema_20", "ema_100", "bb_mid", "bb_std"}
    assert blocked.isdisjoint(set(X.columns))
    assert {
        "ret_1",
        "vol_5",
        "vol_20",
        "vol_60",
        "price_to_sma_10",
        "drawdown_20",
        "range_pos_20",
    }.issubset(set(X.columns))
    assert "PETR4.SA" in meta["excluded_training_features"]


def test_build_dataset_respects_technical_feature_switches(monkeypatch):
    monkeypatch.setattr(
        feature_builder,
        "add_fundamental_features",
        lambda dataset, ticker, cfg: (dataset, {"enabled": False, "source": "test"}),
    )
    index = pd.date_range("2024-01-01", periods=180, freq="B")
    prices = pd.DataFrame({"PETR4.SA": np.linspace(10.0, 20.0, len(index))}, index=index)
    cfg = {
        "features": {
            "technical": {
                "enabled": True,
                "features": {
                    "returns": True,
                    "volatility": False,
                    "rsi": False,
                    "moving_averages": False,
                    "ema": False,
                    "roc": False,
                    "bollinger": False,
                    "fractional_memory": False,
                },
            },
            "context": {"enabled": False},
            "fundamentals": {"enabled": False},
            "sentiment": {"enabled": False},
            "preparation": {"stationarity": {"drop_raw_price": True}},
        }
    }

    X, _all_y, _meta = feature_builder.build_dataset(cfg, prices, "PETR4.SA")

    assert {"ret_1", "ret_5", "ret_20"}.issubset(set(X.columns))
    assert "rsi" not in X.columns
    assert "vol_20" not in X.columns
    assert "sma_ratio" not in X.columns
    assert "bb_width" not in X.columns


def test_external_feature_fill_is_causal_without_backfill(monkeypatch):
    def add_late_context(dataset, prices, ticker, cfg):
        out = dataset.copy()
        out["ctx_late_ret_5"] = np.nan
        out.loc[out.index[100:], "ctx_late_ret_5"] = 0.25
        return out, {"enabled": True, "columns": ["ctx_late_ret_5"]}

    monkeypatch.setattr(feature_builder, "add_market_context_features", add_late_context)
    monkeypatch.setattr(
        feature_builder,
        "add_fundamental_features",
        lambda dataset, ticker, cfg: (dataset, {"enabled": False, "source": "test"}),
    )
    index = pd.date_range("2024-01-01", periods=180, freq="B")
    prices = pd.DataFrame({"PETR4.SA": np.linspace(10.0, 20.0, len(index))}, index=index)
    cfg = {
        "features": {
            "technical": {
                "features": {
                    "returns": True,
                    "volatility": False,
                    "rsi": False,
                    "moving_averages": False,
                    "ema": False,
                    "roc": False,
                    "bollinger": False,
                    "fractional_memory": False,
                }
            },
            "context": {"enabled": True},
            "fundamentals": {"enabled": False},
            "sentiment": {"enabled": False},
            "preparation": {"stationarity": {"drop_raw_price": True}},
        }
    }

    X, _all_y, _meta = feature_builder.build_dataset(cfg, prices, "PETR4.SA")

    assert "ctx_late_ret_5" in X.columns
    assert X["ctx_late_ret_5"].iloc[0] == pytest.approx(0.0)
    assert X["ctx_late_ret_5"].iloc[-1] == pytest.approx(0.25)


def test_preparation_defensively_drops_price_level_features():
    X = pd.DataFrame(
        {
            "PETR4.SA": [10.0, 10.5, 11.0, 11.5, 12.0],
            "sma_10": [10.0, 10.2, 10.4, 10.8, 11.0],
            "ret_1": [0.0, 0.05, 0.047, 0.045, 0.043],
            "ctx_BVSP_ret_5": [0.01, 0.02, 0.01, -0.01, 0.00],
        }
    )
    y = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01])
    cfg = {
        "features": {
            "preparation": {
                "enabled": False,
                "stationarity": {"drop_raw_price": True, "drop_level_features": True},
            }
        }
    }

    X_prepared, _y_prepared, meta = prepare_training_matrix(X, y, cfg)

    assert "PETR4.SA" not in X_prepared.columns
    assert "sma_10" not in X_prepared.columns
    assert "ret_1" in X_prepared.columns
    assert set(meta["dropped_stationarity_features"]) == {"PETR4.SA", "sma_10"}


def test_train_models_prepares_only_train_slice_with_embargo(monkeypatch, tmp_path):
    from sklearn.tree import DecisionTreeRegressor

    monkeypatch.setattr(model_module, "models_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(
        model_module,
        "_make_base_engines",
        lambda cfg, inner_threads=None: {
            "tree_a": DecisionTreeRegressor(max_depth=2, random_state=1),
            "tree_b": DecisionTreeRegressor(max_depth=3, random_state=2),
        },
    )
    calls = []

    def fake_prepare(X_train_raw, y_train_raw, cfg):
        calls.append((len(X_train_raw), X_train_raw.index[-1]))
        return (
            X_train_raw[["ret_1"]].copy(),
            y_train_raw.clip(-0.08, 0.08),
            {"selected_features": ["ret_1"], "selected_feature_count": 1},
        )

    monkeypatch.setattr(model_module, "prepare_training_matrix", fake_prepare)
    index = pd.date_range("2025-01-01", periods=140, freq="B")
    X = pd.DataFrame(
        {
            "ret_1": np.sin(np.arange(len(index)) / 7.0),
            "test_only_noise": np.arange(len(index), dtype=float),
        },
        index=index,
    )
    y = pd.Series(np.cos(np.arange(len(index)) / 9.0) / 100.0, index=index)
    cfg = {
        "data": {"min_rows": 100},
        "app": {"artifact_dir": str(tmp_path)},
        "model": {
            "test_size": 0.20,
            "stacking": {"cv": 2},
            "validation": {
                "embargo_by_horizon": True,
                "embargo_bars": "auto",
                "min_train_rows": 80,
            },
            "prediction_guards": {"max_engine_return_abs": 0.12, "max_final_return_abs": 0.08},
            "engine_safety": {"consensus_guard_enabled": True, "max_deviation_from_median": 0.025},
        },
        "features": {"preparation": {"normalization": {"enabled": False}}},
    }

    manifest = model_module.train_models(cfg, "TEST3.SA", X, y, {}, horizon="d5")

    assert calls == [(107, index[106])]
    assert manifest["features"] == ["ret_1"]
    assert manifest["validation_split"]["embargo_bars"] == 5
    assert manifest["validation_split"]["dropped_embargo_rows"] == 5


def test_policy_blocks_low_risk_reward_signal():
    cfg = {
        "policy": {
            "buy_return_pct": 0.20,
            "sell_return_pct": -0.20,
            "strong_buy_return_pct": 1.5,
            "strong_sell_return_pct": -1.5,
            "min_confidence_pct": 0.45,
            "risk_management": {"selective_multiplier": 1.8, "min_rr_threshold": 1.5},
            "max_risk_pct_for_buy": 99.0,
        },
        "trading": {"capital": 10000.0, "risk_per_trade_pct": 1.0, "allow_short": False},
    }
    results = {"d1": {"prediction_return": 0.003, "confidence": 0.80}}
    meta = {
        "latest_price": 10.0,
        "latest_risk_pct": 2.0,
        "fundamentals": {},
        "sentiment_value": 0.0,
    }

    policy = classify_signal(cfg, results, meta)

    assert policy["label"] == "NEUTRAL"
    assert policy["blocked_signal"] == "BUY"
    assert policy["actionable"] is False
    assert policy["quality_pct"] == pytest.approx(policy["confidence_pct"])
    assert "R/R" in policy["reasons"][-1]


def test_policy_keeps_sell_informational_when_short_disabled():
    cfg = {
        "policy": {
            "buy_return_pct": 0.20,
            "sell_return_pct": -0.20,
            "strong_buy_return_pct": 1.5,
            "strong_sell_return_pct": -1.5,
            "min_confidence_pct": 0.45,
            "risk_management": {"selective_multiplier": 1.8, "min_rr_threshold": 0.1},
        },
        "trading": {"capital": 10000.0, "risk_per_trade_pct": 1.0, "allow_short": False},
    }
    results = {"d1": {"prediction_return": -0.02, "confidence": 0.90}}
    meta = {
        "latest_price": 20.0,
        "latest_risk_pct": 1.0,
        "fundamentals": {},
        "sentiment_value": -0.2,
    }

    policy = classify_signal(cfg, results, meta)
    signal = {"policy": policy}

    assert policy["label"] in {"SELL", "STRONG SELL"}
    assert policy["actionable"] is False
    assert policy["quality_pct"] == pytest.approx(policy["confidence_pct"])
    assert policy["position_size"] == 0
    assert is_actionable_signal(signal) is False


def test_trade_plan_builds_partial_breakeven_and_trailing_rules():
    cfg = {
        "trading": {
            "capital": 10000.0,
            "allow_short": False,
            "partial_take_profit_pct": 50.0,
            "trade_management": {
                "partial_take_profit_pct": 40.0,
                "breakeven_after_partial": True,
                "trailing_stop_enabled": True,
                "trailing_distance_risk_multiple": 0.5,
                "max_hold_days": {"d1": 3, "d5": 9, "d20": 25},
            },
        }
    }
    policy = {
        "label": "BUY",
        "actionable": True,
        "horizon": "d5",
        "target_price": 11.0,
        "target_partial": 10.5,
        "stop_loss_price": 9.5,
        "breakeven_trigger": 10.5,
        "position_size": 123,
        "risk_reward_ratio": 2.0,
        "reasons": ["test"],
    }

    plan = build_trade_plan(
        cfg, ticker="PETR4.SA", policy=policy, latest_price=10.0, latest_risk_pct=2.0
    )

    assert plan["action"] == "ENTER"
    assert plan["side"] == "LONG"
    assert plan["target_1"] == pytest.approx(10.5)
    assert plan["target_final"] == pytest.approx(11.0)
    assert plan["stop_initial"] == pytest.approx(9.5)
    assert plan["partial_take_profit_pct"] == pytest.approx(40.0)
    assert plan["trailing_distance_pct"] == pytest.approx(2.5)
    assert plan["max_hold_days"] == 9


def test_live_monitor_executes_partial_and_moves_stop_to_breakeven(monkeypatch):
    cfg = {"trading": {"capital": 1000.0, "trade_management": {"partial_take_profit_pct": 50.0}}}
    plan = {
        "side": "LONG",
        "target_1": 11.0,
        "target_final": 12.0,
        "stop_initial": 9.0,
        "stop_current": 9.0,
        "breakeven_trigger": 11.0,
        "partial_take_profit_pct": 50.0,
        "partial_executed": False,
        "breakeven_after_partial": True,
        "trailing_enabled": True,
        "trailing_distance_pct": 5.0,
    }
    portfolio = {
        "account": {"initial_capital": 1000.0, "cash": 900.0, "currency": "BRL"},
        "positions": {
            "PETR4.SA": {
                "ticker": "PETR4.SA",
                "entry_price": 10.0,
                "shares": 10,
                "side": "LONG",
                "trade_plan": dict(plan),
            }
        },
        "history": [],
    }
    saved = []
    monkeypatch.setattr(
        monitor,
        "load_latest_signal",
        lambda cfg, ticker: {
            "ticker": ticker,
            "latest_price": 11.0,
            "policy": {"label": "BUY"},
            "trade_plan": plan,
        },
    )
    monkeypatch.setattr(monitor, "get_live_price", lambda ticker: 11.0)
    monkeypatch.setattr(
        monitor, "save_portfolio_state", lambda portfolio: saved.append(portfolio.copy())
    )

    events = monitor._close_positions_on_live_triggers(cfg, portfolio)

    assert events[-1]["action"] == "PARTIAL"
    assert events[-1]["shares"] == 5
    assert portfolio["positions"]["PETR4.SA"]["shares"] == 5
    assert portfolio["positions"]["PETR4.SA"]["partial_executed"] is True
    assert portfolio["positions"]["PETR4.SA"]["stop_current"] == pytest.approx(10.0)
    assert portfolio["account"]["cash"] == pytest.approx(955.0)
    assert saved


def test_cmd_portfolio_handles_positions_without_execution_targets(monkeypatch, capsys):
    monkeypatch.setattr("app.commands.portfolio_command.load_config", lambda path: {})
    monkeypatch.setattr(
        "app.commands.portfolio_command.load_portfolio_state",
        lambda **kwargs: {
            "account": {"initial_capital": 10000.0, "cash": 9000.0, "currency": "BRL"},
            "positions": {
                "PETR4.SA": {
                    "shares": 10,
                    "entry_price": 12.5,
                    "date": "2026-05-06",
                    "side": "LONG",
                },
            },
            "history": [],
        },
    )
    monkeypatch.setattr(
        "app.commands.portfolio_command.load_latest_signal", lambda cfg, ticker: None
    )

    class Args:
        config = None
        portfolio_action = "status"
        live = False
        rebalance = False

    portfolio_command.run(Args())
    out = capsys.readouterr().out

    assert "PETR4.SA" in out
    assert "n/a" in out.lower()


def test_state_db_path_uses_project_data_dir():
    path = get_state_db_path()
    assert str(path).endswith("data\\tradechat_state.db")


def test_portfolio_state_uses_sqlite_only(tmp_path: Path):
    db_path = tmp_path / "tradechat_state.db"
    loaded = load_portfolio_state(capital=1000.0, db_path=db_path)
    assert db_path.exists()
    assert loaded["account"]["cash"] == pytest.approx(1000.0)
    assert loaded["positions"] == {}

    loaded["account"]["cash"] = 910.0
    save_portfolio_state(loaded, db_path=db_path)
    reloaded = load_portfolio_state(capital=1000.0, db_path=db_path)
    assert reloaded["account"]["cash"] == pytest.approx(910.0)
    assert not (tmp_path / "portfolio.json").exists()


def test_schedule_rebalance_dates_respects_warmup_and_step():
    frame = pd.DataFrame(
        {
            "date": np.array(
                [
                    "2026-05-01",
                    "2026-05-02",
                    "2026-05-03",
                    "2026-05-04",
                    "2026-05-05",
                    "2026-05-06",
                ],
                dtype="datetime64[D]",
            )
        }
    )
    dates = _schedule_rebalance_dates(frame, rebalance_days=2, warmup_bars=2)
    assert [str(item.date()) for item in dates] == ["2026-05-03", "2026-05-05"]


def test_evaluation_baselines_include_trivial_comparators_without_lookahead():
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"] * 2),
            "symbol": ["AAA.SA"] * 4 + ["BBB.SA"] * 4,
            "close": [100.0, 110.0, 99.0, 108.9, 50.0, 45.0, 49.5, 44.55],
        }
    )

    baselines = evaluate_baselines(bars, ["AAA.SA", "BBB.SA"], initial_cash=10000.0, random_seed=1)

    assert set(baselines) == {
        "zero_return_no_trade",
        "buy_and_hold_equal_weight",
        "mean_return_long_flat",
        "last_return_long_flat",
        "random_long_flat",
    }
    assert baselines["zero_return_no_trade"]["metrics"]["total_return_pct"] == pytest.approx(0.0)
    assert baselines["buy_and_hold_equal_weight"]["metrics"]["trade_count"] == 2
    assert baselines["last_return_long_flat"]["metrics"]["active_exposure_pct"] < 100.0
    assert baselines["last_return_long_flat"]["metrics"]["trade_count"] > 0


def test_compare_model_to_baselines_reports_deltas_and_decision():
    comparison = compare_model_to_baselines(
        {
            "total_return_pct": 4.0,
            "max_drawdown_pct": -3.0,
            "trade_count": 7,
            "hit_rate_pct": 60.0,
            "profit_factor": 1.5,
        },
        {
            "zero_return_no_trade": {
                "metrics": {
                    "total_return_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "trade_count": 0,
                    "hit_rate_pct": 0.0,
                    "profit_factor": 0.0,
                }
            },
            "buy_and_hold_equal_weight": {
                "metrics": {
                    "total_return_pct": 5.0,
                    "max_drawdown_pct": -6.0,
                    "trade_count": 2,
                    "hit_rate_pct": 50.0,
                    "profit_factor": 1.2,
                }
            },
        },
    )

    assert comparison["beat_count"] == 1
    assert comparison["baseline_count"] == 2
    assert comparison["beat_rate_pct"] == pytest.approx(50.0)
    assert comparison["decision"] == "mixed_or_fails_baselines"
    assert comparison["rows"][0]["return_delta_pct"] == pytest.approx(4.0)
    assert comparison["rows"][1]["return_delta_pct"] == pytest.approx(-1.0)
    assert comparison["rows"][1]["hit_rate_delta_pct"] == pytest.approx(10.0)
    assert comparison["rows"][1]["profit_factor_delta"] == pytest.approx(0.3)


def test_enrich_model_metrics_from_execution_derives_trade_and_order_metrics():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 25.0], "return_pct": [2.0, -1.0, 0.5]})
    orders = pd.DataFrame(
        {"shares": [10, -5, 3], "fill_price": [20.0, 22.0, 21.0], "fees": [1.0, 1.5, 0.5]}
    )

    metrics = enrich_model_metrics_from_execution(
        {"total_return_pct": 3.0, "max_drawdown_pct": -2.0},
        trades=trades,
        orders=orders,
        initial_cash=1000.0,
    )

    assert metrics["trade_count"] == 3
    assert metrics["total_pnl"] == pytest.approx(75.0)
    assert metrics["hit_rate_pct"] == pytest.approx(200 / 3)
    assert metrics["profit_factor"] == pytest.approx(2.5)
    assert metrics["avg_trade_return_pct"] == pytest.approx(0.5)
    assert metrics["turnover_pct"] == pytest.approx(37.3)
    assert metrics["total_cost"] == pytest.approx(3.0)
    assert metrics["cost_pct_initial_cash"] == pytest.approx(0.3)


def test_methodology_report_catches_leakage_and_shadow_contracts():
    manifest = {
        "horizon": "d5",
        "features": ["ret_1", "ctx_BVSP_ret_5"],
        "train_rows": 100,
        "test_rows": 20,
        "preparation": {
            "selected_feature_count": 2,
            "selection": {"method": "target_relevance_low_correlation_greedy_family_balanced"},
        },
        "validation_split": {
            "test_target": "raw_unclipped",
            "embargo_bars": 5,
            "split_index": 90,
            "train_end_index": 85,
            "dropped_embargo_rows": 5,
        },
    }
    validation_summary = {
        "baselines": {
            "zero_return_no_trade": {},
            "buy_and_hold_equal_weight": {},
            "last_return_long_flat": {},
        },
        "baseline_comparison": {"rows": [{"baseline": "zero_return_no_trade"}]},
    }
    removal_summary = {
        "rows": [{"artifact_dir": "artifacts/refine/refine_test/full/models/PETR4_SA"}],
        "artifacts": {"results_csv": "artifacts/refine/refine_test/removal_results.csv"},
    }

    report = methodology_report(
        manifest=manifest,
        validation_summary=validation_summary,
        removal_summary=removal_summary,
    )

    assert report["passed"] is True
    assert {item["check"] for item in report["checks"]} == {
        "no_target_features",
        "temporal_split_embargo",
        "train_only_feature_selection",
        "validation_baselines_present",
        "refine_removal_shadow_artifacts",
    }


def test_methodology_report_fails_on_target_feature_or_missing_baselines():
    report = methodology_report(
        manifest={
            "horizon": "d20",
            "features": ["target_return_d1"],
            "train_rows": 100,
            "test_rows": 10,
            "preparation": {"selected_feature_count": 1, "selection": {}},
            "validation_split": {"embargo_bars": 1, "split_index": 90, "train_end_index": 89},
        },
        validation_summary={"baselines": {}, "baseline_comparison": {}},
        removal_summary={
            "rows": [{"artifact_dir": "artifacts/models/PETR4_SA"}],
            "artifacts": {"results_csv": "artifacts/refine/refine_test/removal_results.csv"},
        },
    )

    failed = {item["check"] for item in report["failed_checks"]}
    assert report["passed"] is False
    assert "no_target_features" in failed
    assert "temporal_split_embargo" in failed
    assert "validation_baselines_present" in failed
    assert "refine_removal_shadow_artifacts" in failed


def test_simulation_artifact_writes_baselines_to_summary_txt(tmp_path):
    cfg = {"app": {"artifact_dir": str(tmp_path)}}
    summary = {
        "run_id": "sim_test",
        "mode": "pybroker_artifact_replay",
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "tickers": ["PETR4.SA"],
        "rebalance_days": 5,
        "warmup_bars": 2,
        "metrics": {
            "trade_count": 1,
            "total_return_pct": 2.5,
            "win_rate": 100.0,
            "total_pnl": 250.0,
        },
        "baselines": {
            "zero_return_no_trade": {
                "metrics": {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0}
            }
        },
        "baseline_comparison": {
            "decision": "passes_baselines",
            "beat_rate_pct": 100.0,
            "rows": [
                {
                    "baseline": "zero_return_no_trade",
                    "return_delta_pct": 2.5,
                    "drawdown_delta_pct": -1.0,
                    "beat_return": True,
                }
            ],
        },
        "pybroker_execution": {"costs": {}},
    }

    artifacts = _write_simulation_artifacts(
        cfg,
        summary,
        orders=pd.DataFrame(),
        trades=pd.DataFrame(),
        signal_plan={},
    )

    text = Path(artifacts["summary_txt"]).read_text(encoding="utf-8")
    assert "BASELINES:" in text
    assert "zero_return_no_trade" in text
    assert "MODEL VS BASELINES:" in text
    assert "passes_baselines" in text


def test_refine_summary_reads_latest_manifests_and_profiles_families(monkeypatch, tmp_path):
    monkeypatch.setattr("app.refine_service.models_dir", lambda cfg: tmp_path)
    model_dir = tmp_path / "PETR4_SA"
    model_dir.mkdir(parents=True)
    (model_dir / "latest_train_d1.json").write_text(
        """
{
  "run_id": "train_test",
  "features": ["ret_1", "ctx_BVSP_ret_5", "fund_regime_score", "sent_mean_3"],
  "preparation": {
    "selection": {
      "relevance": {
        "ret_1": 0.40,
        "ctx_BVSP_ret_5": 0.30,
        "fund_regime_score": 0.20,
        "sent_mean_3": 0.10
      }
    }
  },
  "metrics": {"ridge_arbiter": {"mae_return": 0.012}},
  "latest_prediction_return": 0.018,
  "confidence": 0.60
}
""",
        encoding="utf-8",
    )

    summary = collect_refine_summary({}, ["PETR4.SA"])

    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["decision"] == "keep"
    assert row["family_counts"] == {"technical": 1, "context": 1, "fundamentals": 1, "sentiment": 1}
    assert row["family_relevance_share_pct"]["technical"] == pytest.approx(40.0)
    assert len(summary["missing"]) == 2


def test_refine_renderer_shows_family_count_and_share(monkeypatch):
    monkeypatch.setattr("app.refine_service.screen_width", lambda: 96)
    lines = render_refine_summary(
        {
            "rows": [
                {
                    "ticker": "PETR4.SA",
                    "horizon": "d1",
                    "mae_return": 0.012,
                    "latest_prediction_return": 0.018,
                    "confidence": 0.60,
                    "family_counts": {
                        "technical": 1,
                        "context": 1,
                        "fundamentals": 0,
                        "sentiment": 0,
                    },
                    "family_relevance_share_pct": {
                        "technical": 70.0,
                        "context": 30.0,
                        "fundamentals": 0.0,
                        "sentiment": 0.0,
                    },
                    "decision": "keep",
                }
            ],
            "missing": [],
        }
    )
    output = "\n".join(lines)
    assert "REFINE" in output
    assert "T1/C1/F0/S0" in output
    assert "T70/C30/F0/S0" in output


def test_feature_removal_trains_shadow_profiles(monkeypatch):
    calls = []

    def fake_load_prices(cfg, ticker, update=False):
        calls.append(("load", ticker, update, cfg.get("app", {}).get("artifact_dir")))
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        return pd.DataFrame({ticker: [10.0, 10.1, 10.2, 10.3, 10.4]}, index=idx)

    def fake_build_dataset(cfg, prices, ticker):
        calls.append(
            (
                "dataset",
                ticker,
                cfg["features"]["context"]["enabled"],
                cfg["features"]["sentiment"]["enabled"],
                cfg.get("app", {}).get("artifact_dir"),
            )
        )
        idx = prices.index
        X = pd.DataFrame(
            {"ret_1": [0.0, 0.01, 0.01, 0.01, 0.01], "ctx_x": [1, 2, 3, 4, 5]}, index=idx
        )
        y = pd.DataFrame(
            {
                "target_return_d1": [0.01, 0.01, 0.01, 0.01, None],
                "target_return_d5": [0.02, 0.02, 0.02, 0.02, None],
                "target_return_d20": [0.03, 0.03, 0.03, 0.03, None],
            },
            index=idx,
        )
        return X, y, {"latest_price": 10.4}

    def fake_train_models(cfg, ticker, X, y, meta, autotune=False, horizon="d1", inner_threads=1):
        calls.append(
            (
                "train",
                ticker,
                meta["removal_profile"],
                horizon,
                cfg.get("app", {}).get("artifact_dir"),
            )
        )
        profile = meta["removal_profile"]
        features = ["ret_1"] if profile == "technical_only" else ["ret_1", "ctx_x"]
        mae = 0.010 if profile == "full" else 0.012
        return {
            "run_id": f"{profile}_{horizon}",
            "features": features,
            "metrics": {"ridge_arbiter": {"mae_return": mae}},
            "latest_prediction_return": 0.01,
            "confidence": 0.5,
        }

    monkeypatch.setattr("app.refine_service.load_prices", fake_load_prices)
    monkeypatch.setattr("app.refine_service.build_dataset", fake_build_dataset)
    monkeypatch.setattr("app.refine_service.train_models", fake_train_models)

    cfg = {
        "app": {"artifact_dir": "artifacts"},
        "features": {
            "technical": {"enabled": True},
            "context": {"enabled": True},
            "fundamentals": {"enabled": True},
            "sentiment": {"enabled": True},
        },
    }
    summary = run_feature_removal(
        cfg,
        ["PETR4.SA"],
        horizons="d1",
        profiles="full,technical_only,no_context",
        update=True,
    )

    assert len(summary["rows"]) == 3
    assert summary["errors"] == []
    assert all(
        "artifacts/refine/" in row["artifact_dir"].replace("\\", "/") for row in summary["rows"]
    )
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["summary_txt"]).exists()
    assert Path(summary["artifacts"]["results_csv"]).exists()
    assert "technical_only" in Path(summary["artifacts"]["results_csv"]).read_text(encoding="utf-8")
    assert (
        "train",
        "PETR4.SA",
        "technical_only",
        "d1",
        summary["rows"][1]["artifact_dir"],
    ) not in calls
    dataset_calls = [call for call in calls if call[0] == "dataset"]
    assert dataset_calls[1][2] is False
    assert dataset_calls[1][3] is False
    assert dataset_calls[2][2] is False


def test_removal_renderer_compares_delta_against_full(monkeypatch):
    monkeypatch.setattr("app.refine_service.screen_width", lambda: 100)
    lines = render_removal_summary(
        {
            "run_id": "refine_test",
            "rows": [
                {
                    "ticker": "PETR4.SA",
                    "horizon": "d1",
                    "profile": "full",
                    "mae_return": 0.010,
                    "confidence": 0.50,
                    "family_counts": {
                        "technical": 1,
                        "context": 1,
                        "fundamentals": 0,
                        "sentiment": 0,
                    },
                },
                {
                    "ticker": "PETR4.SA",
                    "horizon": "d1",
                    "profile": "technical_only",
                    "mae_return": 0.012,
                    "confidence": 0.45,
                    "family_counts": {
                        "technical": 1,
                        "context": 0,
                        "fundamentals": 0,
                        "sentiment": 0,
                    },
                },
            ],
            "errors": [],
            "decisions": [
                {"profile": "full", "removed_family": "none", "decision": "reference"},
                {
                    "profile": "technical_only",
                    "removed_family": "context+fundamentals+sentiment",
                    "return_delta_pct": "",
                    "drawdown_delta_pct": "",
                    "profit_factor_delta": "",
                    "decision": "keep_family",
                },
            ],
            "artifacts": {"dir": "artifacts/refine/refine_test"},
        }
    )
    output = "\n".join(lines)
    assert "controlled removal" in output
    assert "technical_only" in output
    assert "+0.20%" in output
    assert "worse" in output
    assert "REFINE DECISION" in output


def test_feature_removal_walkforward_uses_shadow_profiles(monkeypatch):
    calls = []

    def fake_run_pybroker_replay(
        cfg,
        tickers,
        *,
        mode,
        start_date=None,
        end_date=None,
        rebalance_days=5,
        warmup_bars=150,
        initial_cash=None,
        max_positions=None,
        allow_short=False,
        walkforward_autotune=False,
        inner_threads=1,
    ):
        profile = str(cfg.get("app", {}).get("artifact_dir", "")).split("/")[-1]
        calls.append(
            (
                profile,
                mode,
                cfg["features"]["context"]["enabled"],
                cfg["features"]["sentiment"]["enabled"],
            )
        )
        return {
            "run_id": f"sim_{profile}",
            "mode": "pybroker_walkforward_shadow",
            "metrics": {
                "total_return_pct": 2.0 if profile == "full" else 1.25,
                "max_drawdown_pct": -1.0,
                "trade_count": 3,
                "hit_rate_pct": 66.0,
                "profit_factor": 1.4,
                "turnover_pct": 30.0,
                "active_exposure_pct": 40.0,
            },
            "baseline_comparison": {"decision": "mixed_or_fails_baselines", "beat_rate_pct": 50.0},
            "artifacts": {"dir": f"artifacts/refine/test/{profile}/simulations/sim_{profile}"},
        }

    monkeypatch.setattr("app.refine_service.run_pybroker_replay", fake_run_pybroker_replay)
    cfg = {
        "app": {"artifact_dir": "artifacts"},
        "features": {
            "technical": {"enabled": True},
            "context": {"enabled": True},
            "fundamentals": {"enabled": True},
            "sentiment": {"enabled": True},
        },
    }

    summary = run_feature_removal_walkforward(
        cfg,
        ["PETR4.SA"],
        profiles="full,no_context",
        start_date="2026-01-01",
        end_date="2026-04-01",
        rebalance_days=3,
        warmup_bars=10,
    )

    assert [row["profile"] for row in summary["rows"]] == ["full", "no_context"]
    assert summary["rows"][1]["return_delta_pct"] == pytest.approx(-0.75)
    assert summary["errors"] == []
    assert calls[0] == ("full", "walkforward", True, True)
    assert calls[1] == ("no_context", "walkforward", False, True)
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["summary_txt"]).exists()
    assert Path(summary["artifacts"]["results_csv"]).exists()
    assert "no_context" in Path(summary["artifacts"]["results_csv"]).read_text(encoding="utf-8")


def test_removal_walkforward_renderer_compares_return_against_full(monkeypatch):
    monkeypatch.setattr("app.refine_service.screen_width", lambda: 100)
    lines = render_removal_walkforward_summary(
        {
            "run_id": "refine_wf_test",
            "rows": [
                {
                    "profile": "full",
                    "total_return_pct": 2.0,
                    "return_delta_pct": 0.0,
                    "max_drawdown_pct": -1.0,
                    "trade_count": 3,
                    "hit_rate_pct": 66.0,
                    "profit_factor": 1.4,
                    "beat_rate_pct": 50.0,
                },
                {
                    "profile": "no_context",
                    "total_return_pct": 1.25,
                    "return_delta_pct": -0.75,
                    "max_drawdown_pct": -1.2,
                    "trade_count": 2,
                    "hit_rate_pct": 50.0,
                    "profit_factor": 1.0,
                    "beat_rate_pct": 25.0,
                },
            ],
            "errors": [],
            "decisions": [
                {"profile": "full", "removed_family": "none", "decision": "reference"},
                {
                    "profile": "no_context",
                    "removed_family": "context",
                    "return_delta_pct": -0.75,
                    "drawdown_delta_pct": -0.2,
                    "profit_factor_delta": -0.4,
                    "decision": "keep_family",
                },
            ],
            "artifacts": {"dir": "artifacts/refine/refine_wf_test"},
        }
    )
    output = "\n".join(lines)
    assert "controlled removal walk-forward" in output
    assert "no_context" in output
    assert "-0.75%" in output
    assert "worse" in output
    assert "REFINE DECISION" in output


def test_normalize_signal_plan_assigns_weights_only_to_actionable_signals():
    by_date, by_symbol = _normalize_signal_plan(
        {
            "2026-05-06": {
                "ABEV3.SA": {
                    "ticker": "ABEV3.SA",
                    "policy": {
                        "label": "BUY",
                        "score_pct": 2.0,
                        "confidence_pct": 70.0,
                        "horizon": "d5",
                    },
                    "horizons": {"d5": {"prediction_return": 0.02}},
                },
                "PETR4.SA": {
                    "ticker": "PETR4.SA",
                    "policy": {
                        "label": "SELL",
                        "score_pct": -1.0,
                        "confidence_pct": 60.0,
                        "horizon": "d1",
                    },
                    "horizons": {"d1": {"prediction_return": -0.01}},
                },
                "VALE3.SA": {
                    "ticker": "VALE3.SA",
                    "policy": {
                        "label": "NEUTRAL",
                        "score_pct": 0.0,
                        "confidence_pct": 10.0,
                        "horizon": "d1",
                    },
                    "horizons": {"d1": {"prediction_return": 0.0}},
                },
            }
        }
    )
    weights = {
        ticker: payload["target_weight"] for ticker, payload in by_date["2026-05-06"].items()
    }
    assert pytest.approx(sum(weights.values())) == 1.0
    assert weights["VALE3.SA"] == pytest.approx(0.0)
    assert by_symbol["ABEV3.SA"]["2026-05-06"]["score"] > 0


def test_pybroker_strategy_config_uses_native_execution_controls():
    pytest.importorskip("pybroker")
    cfg = {
        "trading": {"capital": 20000.0},
        "simulation": {"costs": {"fee_mode": "order_percent", "fee_amount": 0.03}},
    }

    config = _build_strategy_config(
        cfg, initial_cash=None, max_positions=3, allow_short=False, symbol_count=5
    )

    assert config.initial_cash == pytest.approx(20000.0)
    assert config.max_long_positions == 3
    assert config.max_short_positions is None
    assert config.return_stops is True
    assert config.position_mode.value == "long_only"
    assert config.fee_mode.value == "order_percent"
    assert config.fee_amount == pytest.approx(0.03)


def test_pybroker_entry_sizing_combines_risk_score_weight_and_cap():
    cfg = {
        "trading": {"risk_per_trade_pct": 1.0},
        "simulation": {"execution": {"max_position_pct": 50.0}},
    }
    signal = {
        "target_weight": 0.8,
        "trade_plan": {
            "stop_initial": 9.5,
            "position_size": 999,
            "action": "ENTER",
            "side": "LONG",
        },
    }

    shares = _planned_entry_shares(cfg, signal, price=10.0, equity=10000.0)

    assert shares == 200


def test_pybroker_position_size_handler_sets_ranked_risk_shares():
    class FakeCtx:
        total_equity = 10000.0

        def __init__(self, signals):
            self._signals = signals
            self.sized = []
            self.sessions = {
                "PETR4.SA": {
                    "pending_buy_signal": {
                        "target_weight": 0.5,
                        "trade_plan": {
                            "stop_initial": 9.5,
                            "position_size": 0,
                            "action": "ENTER",
                            "side": "LONG",
                        },
                    }
                }
            }

        def signals(self):
            return iter(self._signals)

        def set_shares(self, signal, shares):
            self.sized.append((signal.symbol, shares))

    class FakeSignal:
        symbol = "PETR4.SA"
        score = 3.0
        type = "buy"

    FakeSignal.bar_data = type(
        "BarData",
        (),
        {
            "date": np.array(["2026-05-07"], dtype="datetime64[D]"),
            "close": np.array([10.0]),
        },
    )()

    cfg = {"trading": {"risk_per_trade_pct": 1.0}}
    plan_by_symbol = {
        "PETR4.SA": {
            "2026-05-06": {
                "target_weight": 0.5,
                "trade_plan": {
                    "stop_initial": 9.5,
                    "position_size": 0,
                    "action": "ENTER",
                    "side": "LONG",
                },
            }
        }
    }
    ctx = FakeCtx([FakeSignal()])

    _position_size_handler_factory(cfg, plan_by_symbol)(ctx)

    assert ctx.sized == [("PETR4.SA", 200)]


def test_pybroker_execution_uses_native_stops_without_manual_target_sizing():
    class FakeCtx:
        dt = pd.Timestamp("2026-05-06")
        symbol = "PETR4.SA"
        close = [10.0]
        session = {}

        def long_pos(self):
            return None

        def short_pos(self):
            return None

        def cover_all_shares(self):
            raise AssertionError("unexpected cover")

        def sell_all_shares(self):
            raise AssertionError("unexpected sell all")

    signal = {
        "score": 4.2,
        "latest_price": 10.0,
        "policy": {"label": "BUY", "target_price": 12.0, "stop_loss_price": 9.0},
        "trade_plan": {
            "action": "ENTER",
            "side": "LONG",
            "stop_initial": 9.0,
            "target_final": 12.0,
            "trailing_enabled": True,
            "trailing_distance_pct": 3.0,
            "max_hold_days": 5,
        },
    }
    ctx = FakeCtx()

    _execution_fn_factory({"PETR4.SA": {"2026-05-06": signal}}, allow_short=False)(ctx)

    assert ctx.buy_shares == 1
    assert ctx.score == pytest.approx(4.2)
    assert ctx.stop_loss_pct == pytest.approx(10.0)
    assert ctx.stop_profit_pct == pytest.approx(20.0)
    assert ctx.stop_trailing_pct == pytest.approx(3.0)
    assert ctx.hold_bars == 5


def test_model5_ui_renders_clean_without_color():
    lines = []
    lines.extend(ui5.render_header("VALIDATE - PYBROKER - REPLAY", width=80, use_color=False))
    lines.extend(ui5.render_section("RESUMO", width=80, use_color=False))
    lines.extend(
        ui5.render_key_values(
            {"Experimento": "Teste", "Conclusao preliminar": "Ok"}, width=80, use_color=False
        )
    )
    lines.extend(
        ui5.render_table(
            ["Configuracao", "Erro", "Decisao"],
            [["base", "0.01", ui5.render_badge("ok", "ok", use_color=False)]],
            width=80,
            use_color=False,
        )
    )
    lines.extend(ui5.render_operational_closing(["Revisar resultado."], width=80, use_color=False))
    output = "\n".join(lines)
    assert "\x1b[" not in output
    assert "VALIDATE - PYBROKER - REPLAY" in output
    assert "[OK]" in output


def test_model5_table_separator_uses_screen_width():
    lines = ui5.render_table(
        ["Configuracao", "Erro", "Decisao"],
        [["base", "0.01", ui5.render_badge("ok", "ok", use_color=False)]],
        width=80,
        use_color=False,
    )
    separator = next(line for line in lines if set(line) == {"-"})
    assert len(separator) == 80


def test_validation_view_renders_model_vs_baselines(monkeypatch):
    monkeypatch.setattr("app.validation_view.ui5.screen_width", lambda: 96)
    lines = render_validation_summary(
        {
            "mode": "pybroker_artifact_replay",
            "start_date": "2026-01-01",
            "end_date": "2026-02-01",
            "tickers": ["PETR4.SA"],
            "metrics": {
                "total_return_pct": 2.0,
                "max_drawdown_pct": -1.0,
                "trade_count": 3,
                "hit_rate_pct": 66.0,
                "avg_trade_return_pct": 0.8,
                "profit_factor": 1.4,
                "turnover_pct": 35.0,
                "active_exposure_pct": 42.0,
                "total_cost": 3.5,
            },
            "baselines": {
                "zero_return_no_trade": {
                    "metrics": {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "trade_count": 0}
                }
            },
            "baseline_comparison": {
                "decision": "passes_baselines",
                "beat_rate_pct": 100.0,
                "rows": [
                    {
                        "baseline": "zero_return_no_trade",
                        "return_delta_pct": 2.0,
                        "drawdown_delta_pct": -1.0,
                        "hit_rate_delta_pct": 66.0,
                        "profit_factor_delta": 1.4,
                        "beat_return": True,
                    }
                ],
            },
            "validation_decision": {
                "final_decision": "observe",
                "score": 80.0,
                "checks": {
                    "beats_required_baselines": {"passed": True},
                    "enough_trades": {"passed": False},
                    "profit_factor_ok": {"passed": True},
                    "max_drawdown_ok": {"passed": True},
                    "exposure_ok": {"passed": True},
                },
                "explanation": ["Model beat 1/1 required baselines."],
            },
        },
        mode="replay",
        screen_title="VALIDATE",
    )
    output = "\n".join(lines)
    assert "ECONOMIA" in output
    assert "Profit F" in output
    assert "MODELO VS BASELINES" in output
    assert "VALIDATION DECISION" in output
    assert "passes_baselines" in output
    assert "zero_return_no_trade" in output
