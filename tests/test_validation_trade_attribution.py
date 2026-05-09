import pandas as pd

from app.evaluation_service import build_trade_attribution, enrich_model_metrics_from_execution
from app.validation_view import render_validation_summary


def test_trade_attribution_groups_by_ticker_and_side():
    trades = pd.DataFrame(
        {
            "symbol": ["PETR4.SA", "PETR4.SA", "VALE3.SA"],
            "side": ["long", "long", "short"],
            "pnl": [10.0, -4.0, 8.0],
            "return_pct": [1.0, -0.4, 0.8],
        }
    )
    orders = pd.DataFrame(
        {
            "symbol": ["PETR4.SA", "VALE3.SA"],
            "fees": [1.5, 0.5],
        }
    )

    attribution = build_trade_attribution(trades, orders)

    by_ticker = {row["group"]: row for row in attribution["by_ticker"]}
    assert by_ticker["PETR4.SA"]["trade_count"] == 2
    assert by_ticker["PETR4.SA"]["net_pnl"] == 6.0
    assert by_ticker["PETR4.SA"]["cost"] == 1.5
    assert by_ticker["VALE3.SA"]["profit_factor_display"] == "inf"

    by_side = {row["group"]: row for row in attribution["by_side"]}
    assert by_side["long"]["trade_count"] == 2
    assert by_side["short"]["trade_count"] == 1


def test_enriched_metrics_include_trade_attribution():
    trades = pd.DataFrame(
        {
            "symbol": ["ITUB4.SA", "ITUB4.SA"],
            "pnl": [12.0, -3.0],
            "entry_date": ["2026-01-02", "2026-01-05"],
            "exit_date": ["2026-01-03", "2026-01-06"],
        }
    )

    metrics = enrich_model_metrics_from_execution(
        {"total_return_pct": 0.9, "active_exposure_pct": 0.0},
        trades=trades,
        orders=pd.DataFrame(),
        initial_cash=1000.0,
        start_date="2026-01-01",
        end_date="2026-01-10",
    )

    assert metrics["trade_attribution"]["by_ticker"][0]["group"] == "ITUB4.SA"
    assert metrics["trade_attribution"]["by_ticker"][0]["net_pnl"] == 9.0


def test_validation_summary_renders_trade_attribution():
    lines = render_validation_summary(
        {
            "mode": "pybroker_artifact_replay",
            "tickers": ["PETR4.SA"],
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            "policy_profile": "active",
            "metrics": {
                "total_return_pct": -0.5,
                "max_drawdown_pct": -1.0,
                "trade_count": 2,
                "hit_rate_pct": 50.0,
                "avg_trade_return_pct": -0.2,
                "profit_factor_display": "0.80",
                "profit_factor": 0.8,
                "active_exposure_pct": 20.0,
                "active_exposure_available": True,
                "turnover_pct": 100.0,
                "total_cost": 1.0,
                "gross_profit": 10.0,
                "gross_loss": 12.0,
                "net_profit": -2.0,
                "return_before_costs_pct": -0.4,
                "return_after_costs_pct": -0.5,
                "trade_attribution": {
                    "by_ticker": [
                        {
                            "group": "PETR4.SA",
                            "trade_count": 2,
                            "hit_rate_pct": 50.0,
                            "profit_factor_display": "0.83",
                            "net_pnl": -2.0,
                            "gross_profit": 10.0,
                            "gross_loss": 12.0,
                            "avg_return_pct": -0.2,
                            "cost": 1.0,
                        }
                    ],
                    "by_side": [],
                    "by_horizon": [],
                    "warnings": [],
                },
            },
            "baselines": {},
            "baseline_comparison": {},
            "validation_decision": {},
        },
        mode="replay",
    )

    text = "\n".join(lines)
    assert "TRADE ATTRIBUTION | BY TICKER" in text
    assert "PETR4.SA" in text
