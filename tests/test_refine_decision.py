from app.refine_decision import build_refine_decision_matrix
from app.refine_service import render_refine_decision_table


def test_refine_decision_remove_candidate():
    decisions = build_refine_decision_matrix(
        [
            {
                "profile": "full",
                "total_return_pct": 2.0,
                "max_drawdown_pct": -3.0,
                "profit_factor": 1.1,
                "trade_count": 10,
                "active_exposure_pct": 30.0,
            },
            {
                "profile": "no_context",
                "total_return_pct": 3.2,
                "max_drawdown_pct": -2.2,
                "profit_factor": 1.25,
                "trade_count": 11,
                "active_exposure_pct": 32.0,
            },
        ]
    )
    no_context = next(row for row in decisions if row["profile"] == "no_context")
    assert no_context["removed_family"] == "context"
    assert no_context["decision"] == "remove_candidate"


def test_refine_decision_keep_family():
    decisions = build_refine_decision_matrix(
        [
            {
                "profile": "full",
                "total_return_pct": 2.0,
                "max_drawdown_pct": -3.0,
                "profit_factor": 1.1,
                "trade_count": 10,
            },
            {
                "profile": "no_sentiment",
                "total_return_pct": 1.6,
                "max_drawdown_pct": -3.1,
                "profit_factor": 1.0,
                "trade_count": 10,
            },
        ]
    )
    no_sentiment = next(row for row in decisions if row["profile"] == "no_sentiment")
    assert no_sentiment["decision"] == "keep_family"


def test_refine_decision_inconclusive_low_trades():
    decisions = build_refine_decision_matrix(
        [
            {
                "profile": "full",
                "total_return_pct": 2.0,
                "max_drawdown_pct": -3.0,
                "profit_factor": 1.1,
                "trade_count": 2,
            },
            {
                "profile": "no_fundamentals",
                "total_return_pct": 3.0,
                "max_drawdown_pct": -2.0,
                "profit_factor": 1.3,
                "trade_count": 2,
            },
        ]
    )
    no_fundamentals = next(row for row in decisions if row["profile"] == "no_fundamentals")
    assert no_fundamentals["decision"] == "inconclusive"


def test_refine_decision_render_consolidates_profile_and_reports_noop_family():
    decisions = build_refine_decision_matrix(
        [
            {
                "ticker": "PETR4.SA",
                "horizon": "d1",
                "profile": "full",
                "mae_return": 0.010,
                "quality": 0.50,
                "selected_feature_count": 8,
                "family_counts": {"technical": 8, "context": 1, "fundamentals": 0, "sentiment": 0},
            },
            {
                "ticker": "PETR4.SA",
                "horizon": "d1",
                "profile": "no_sentiment",
                "mae_return": 0.010,
                "quality": 0.50,
                "selected_feature_count": 8,
                "family_counts": {"technical": 8, "context": 1, "fundamentals": 0, "sentiment": 0},
            },
            {
                "ticker": "VALE3.SA",
                "horizon": "d1",
                "profile": "full",
                "mae_return": 0.010,
                "quality": 0.50,
                "selected_feature_count": 8,
                "family_counts": {"technical": 8, "context": 1, "fundamentals": 0, "sentiment": 0},
            },
            {
                "ticker": "VALE3.SA",
                "horizon": "d1",
                "profile": "no_sentiment",
                "mae_return": 0.010,
                "quality": 0.50,
                "selected_feature_count": 8,
                "family_counts": {"technical": 8, "context": 1, "fundamentals": 0, "sentiment": 0},
            },
        ]
    )
    output = "\n".join(render_refine_decision_table(decisions, width=100))
    assert output.count("no_sentiment") == 1
    assert "tickers=2" in output
    assert "sentiment has 0 selected features; removal is a no-op for this run." in output
