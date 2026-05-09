from app.refine_decision import build_refine_decision_matrix
from app.refine_service import render_refine_decision_table


def test_no_context_is_not_noop_when_baseline_has_context_without_full_row():
    decisions = build_refine_decision_matrix(
        [
            {
                "ticker": "PETR4.SA",
                "horizon": "d5",
                "profile": "no_context",
                "mae_return": 0.0353,
                "quality": 0.61,
                "selected_feature_count": 11,
                "family_counts": {"technical": 11, "context": 0, "fundamentals": 0, "sentiment": 0},
                "baseline_mae_return": 0.0352,
                "baseline_quality": 0.47,
                "baseline_selected_feature_count": 18,
                "baseline_family_counts": {
                    "technical": 11,
                    "context": 7,
                    "fundamentals": 0,
                    "sentiment": 0,
                },
            }
        ]
    )

    assert decisions[0]["no_op_notes"] == []
    rendered = "\n".join(render_refine_decision_table(decisions, width=100))
    assert "context has 0 selected features; removal is a no-op" not in rendered


def test_no_sentiment_is_noop_when_baseline_has_no_sentiment_without_full_row():
    decisions = build_refine_decision_matrix(
        [
            {
                "ticker": "PETR4.SA",
                "horizon": "d5",
                "profile": "no_sentiment",
                "mae_return": 0.0353,
                "quality": 0.61,
                "selected_feature_count": 18,
                "family_counts": {"technical": 11, "context": 7, "fundamentals": 0, "sentiment": 0},
                "baseline_mae_return": 0.0353,
                "baseline_quality": 0.61,
                "baseline_selected_feature_count": 18,
                "baseline_family_counts": {
                    "technical": 11,
                    "context": 7,
                    "fundamentals": 0,
                    "sentiment": 0,
                },
            }
        ]
    )

    assert decisions[0]["no_op_notes"] == [
        "sentiment has 0 selected features; removal is a no-op for this run."
    ]
