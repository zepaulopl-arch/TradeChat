from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import cli
from app import batch_service
from app import features as feature_builder
from app import models as model_module
from app import portfolio_monitor_service as monitor
from app.models import _split_train_test_raw
from app.models import _stacking_cv_splits
from app.models import _make_arbiter
from app.models import _oof_valid_mask
from app.policy import classify_signal
from app.preparation import prepare_training_matrix
from app.portfolio_service import get_state_db_path, load_portfolio_state, save_portfolio_state
from app.scoring import is_actionable_signal
from app.simulator_service import (
    _build_strategy_config,
    _execution_fn_factory,
    _normalize_signal_plan,
    _planned_entry_shares,
    _position_size_handler_factory,
    _schedule_rebalance_dates,
)
from app.trade_plan_service import build_trade_plan
from app.ui import model5 as ui5


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
    monkeypatch.setattr(monitor, "load_latest_signal", lambda cfg, ticker: {"policy": {"target_price": 9.0, "stop_loss_price": 11.0}})
    monkeypatch.setattr(monitor, "get_live_price", lambda ticker: 9.0)
    monkeypatch.setattr(monitor, "save_portfolio_state", lambda portfolio: saved.append(portfolio))

    events = monitor._close_positions_on_live_triggers(cfg, portfolio)

    assert events[-1]["action"] == "TARGET"
    assert portfolio["positions"] == {}
    assert portfolio["account"]["cash"] == pytest.approx(1010.0)
    assert portfolio["history"][-1]["pl_cash"] == pytest.approx(10.0)
    assert saved


def test_oof_valid_mask_keeps_zero_predictions_and_rejects_missing_rows():
    mask = _oof_valid_mask({
        "xgb": np.array([0.0, np.nan, 0.01]),
        "catboost": np.array([0.0, 0.02, 0.03]),
        "extratrees": np.array([0.0, 0.01, np.nan]),
    })
    assert mask.tolist() == [True, False, False]


def test_make_arbiter_uses_ridge_config():
    arbiter = _make_arbiter({"model": {"arbiter": {"ridge": {"alpha": 2.5, "fit_intercept": False}}}})
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
    assert {"ret_1", "vol_5", "vol_20", "vol_60", "price_to_sma_10", "drawdown_20", "range_pos_20"}.issubset(set(X.columns))
    assert "PETR4.SA" in meta["excluded_training_features"]


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
            "validation": {"embargo_by_horizon": True, "embargo_bars": "auto", "min_train_rows": 80},
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
    meta = {"latest_price": 10.0, "latest_risk_pct": 2.0, "fundamentals": {}, "sentiment_value": 0.0}

    policy = classify_signal(cfg, results, meta)

    assert policy["label"] == "NEUTRAL"
    assert policy["blocked_signal"] == "BUY"
    assert policy["actionable"] is False
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
    meta = {"latest_price": 20.0, "latest_risk_pct": 1.0, "fundamentals": {}, "sentiment_value": -0.2}

    policy = classify_signal(cfg, results, meta)
    signal = {"policy": policy}

    assert policy["label"] in {"SELL", "STRONG SELL"}
    assert policy["actionable"] is False
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

    plan = build_trade_plan(cfg, ticker="PETR4.SA", policy=policy, latest_price=10.0, latest_risk_pct=2.0)

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
    monkeypatch.setattr(monitor, "load_latest_signal", lambda cfg, ticker: {"ticker": ticker, "latest_price": 11.0, "policy": {"label": "BUY"}, "trade_plan": plan})
    monkeypatch.setattr(monitor, "get_live_price", lambda ticker: 11.0)
    monkeypatch.setattr(monitor, "save_portfolio_state", lambda portfolio: saved.append(portfolio.copy()))

    events = monitor._close_positions_on_live_triggers(cfg, portfolio)

    assert events[-1]["action"] == "PARTIAL"
    assert events[-1]["shares"] == 5
    assert portfolio["positions"]["PETR4.SA"]["shares"] == 5
    assert portfolio["positions"]["PETR4.SA"]["partial_executed"] is True
    assert portfolio["positions"]["PETR4.SA"]["stop_current"] == pytest.approx(10.0)
    assert portfolio["account"]["cash"] == pytest.approx(955.0)
    assert saved


def test_cmd_portfolio_handles_positions_without_execution_targets(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path: {})
    monkeypatch.setattr(
        cli,
        "load_portfolio_state",
        lambda **kwargs: {
            "account": {"initial_capital": 10000.0, "cash": 9000.0, "currency": "BRL"},
            "positions": {
                "PETR4.SA": {"shares": 10, "entry_price": 12.5, "date": "2026-05-06", "side": "LONG"},
            },
            "history": [],
        },
    )
    monkeypatch.setattr(cli, "_latest_signal_for", lambda cfg, ticker: None)

    class Args:
        config = None

    cli.cmd_portfolio(Args())
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
            ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06"],
            dtype="datetime64[D]",
            )
        }
    )
    dates = _schedule_rebalance_dates(frame, rebalance_days=2, warmup_bars=2)
    assert [str(item.date()) for item in dates] == ["2026-05-03", "2026-05-05"]


def test_normalize_signal_plan_assigns_weights_only_to_actionable_signals():
    by_date, by_symbol = _normalize_signal_plan(
        {
            "2026-05-06": {
                "ABEV3.SA": {
                    "ticker": "ABEV3.SA",
                    "policy": {"label": "BUY", "score_pct": 2.0, "confidence_pct": 70.0, "horizon": "d5"},
                    "horizons": {"d5": {"prediction_return": 0.02}},
                },
                "PETR4.SA": {
                    "ticker": "PETR4.SA",
                    "policy": {"label": "SELL", "score_pct": -1.0, "confidence_pct": 60.0, "horizon": "d1"},
                    "horizons": {"d1": {"prediction_return": -0.01}},
                },
                "VALE3.SA": {
                    "ticker": "VALE3.SA",
                    "policy": {"label": "NEUTRAL", "score_pct": 0.0, "confidence_pct": 10.0, "horizon": "d1"},
                    "horizons": {"d1": {"prediction_return": 0.0}},
                },
            }
        }
    )
    weights = {ticker: payload["target_weight"] for ticker, payload in by_date["2026-05-06"].items()}
    assert pytest.approx(sum(weights.values())) == 1.0
    assert weights["VALE3.SA"] == pytest.approx(0.0)
    assert by_symbol["ABEV3.SA"]["2026-05-06"]["score"] > 0


def test_pybroker_strategy_config_uses_native_execution_controls():
    pytest.importorskip("pybroker")
    cfg = {
        "trading": {"capital": 20000.0},
        "simulation": {"costs": {"fee_mode": "order_percent", "fee_amount": 0.03}},
    }

    config = _build_strategy_config(cfg, initial_cash=None, max_positions=3, allow_short=False, symbol_count=5)

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
        "trade_plan": {"stop_initial": 9.5, "position_size": 999, "action": "ENTER", "side": "LONG"},
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
                        "trade_plan": {"stop_initial": 9.5, "position_size": 0, "action": "ENTER", "side": "LONG"},
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
                "trade_plan": {"stop_initial": 9.5, "position_size": 0, "action": "ENTER", "side": "LONG"},
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
    lines.extend(ui5.render_key_values({"Experimento": "Teste", "Conclusao preliminar": "Ok"}, width=80, use_color=False))
    lines.extend(ui5.render_table(["Configuracao", "Erro", "Decisao"], [["base", "0.01", ui5.render_badge("ok", "ok", use_color=False)]], width=80, use_color=False))
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


def test_diagnose_marks_low_history_as_skipped(monkeypatch):
    monkeypatch.setattr(
        batch_service,
        "resolve_asset",
        lambda cfg, ticker: {
            "canonical": "EMBJ3.SA",
            "changed": False,
            "profile": {"registry_status": "active", "name": "EMBRAER"},
        },
    )
    monkeypatch.setattr(
        batch_service,
        "data_status",
        lambda cfg, ticker: {
            "rows": 42,
            "start": "2026-01-01",
            "end": "2026-03-01",
            "context_tickers": [],
            "requested_context_tickers": [],
            "unavailable_context_tickers": [],
            "asset_profile": {},
        },
    )
    row = batch_service.diagnose_one_asset({"data": {"min_rows": 150}}, "EMBJ3.SA", no_data=True)
    assert row["status"] == "skipped"
    assert row["failed_stage"] == "data_rows"
    assert "insufficient rows: 42 < 150" in row["error"]
