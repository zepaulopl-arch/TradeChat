import pandas as pd

from app.evaluation_service import enrich_model_metrics_from_execution


def test_profit_factor_is_inf_when_all_detected_trades_win():
    trades = pd.DataFrame({"pnl": [10.0, 5.0]})

    metrics = enrich_model_metrics_from_execution(
        {"total_return_pct": 1.5},
        trades=trades,
        orders=pd.DataFrame(),
        initial_cash=1000.0,
    )

    assert metrics["gross_profit"] == 15.0
    assert metrics["gross_loss"] == 0.0
    assert metrics["profit_factor_display"] == "inf"
    assert metrics["profit_factor"] >= 999.0
    assert metrics["profit_factor_infinite"] is True


def test_exposure_is_estimated_from_trade_dates_when_native_metric_is_zero():
    trades = pd.DataFrame(
        {
            "entry_date": ["2026-01-02"],
            "exit_date": ["2026-01-06"],
            "pnl": [10.0],
        }
    )

    metrics = enrich_model_metrics_from_execution(
        {"total_return_pct": 1.0, "active_exposure_pct": 0.0},
        trades=trades,
        orders=pd.DataFrame(),
        initial_cash=1000.0,
        start_date="2026-01-01",
        end_date="2026-01-10",
    )

    assert metrics["active_exposure_pct"] == 50.0
    assert metrics["active_exposure_source"] == "trade_dates_estimate"


def test_positive_run_without_trade_pnl_does_not_report_zero_profit_factor():
    metrics = enrich_model_metrics_from_execution(
        {"total_return_pct": 1.2, "hit_rate_pct": 100.0},
        trades=pd.DataFrame(
            {"symbol": ["PETR4.SA"], "entry_date": ["2026-01-02"], "exit_date": ["2026-01-03"]}
        ),
        orders=pd.DataFrame(),
        initial_cash=1000.0,
        start_date="2026-01-01",
        end_date="2026-01-10",
    )

    assert metrics["profit_factor_display"] == "inf"
    assert "zero profit factor" not in " ".join(metrics["metric_warnings"])


def test_validation_summary_renders_metric_audit_lines():
    from app.validation_view import render_validation_summary

    lines = render_validation_summary(
        {
            "mode": "pybroker_artifact_replay",
            "tickers": ["PETR4.SA"],
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            "policy_profile": "balanced",
            "metrics": {
                "total_return_pct": 1.2,
                "max_drawdown_pct": -0.5,
                "trade_count": 2,
                "hit_rate_pct": 100.0,
                "avg_trade_return_pct": 0.6,
                "profit_factor_display": "inf",
                "profit_factor": 999.0,
                "active_exposure_pct": 50.0,
                "active_exposure_available": True,
                "turnover_pct": 25.0,
                "total_cost": 1.0,
                "gross_profit": 20.0,
                "gross_loss": 0.0,
                "net_profit": 19.0,
                "return_before_costs_pct": 1.3,
                "return_after_costs_pct": 1.2,
            },
            "baselines": {},
            "baseline_comparison": {},
            "validation_decision": {},
        },
        mode="replay",
    )
    text = "\n".join(lines)

    assert "inf" in text
    assert "P&L AUDIT" in text
    assert "Before Cost" in text
