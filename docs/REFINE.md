# Refine

Refine is TradeChat's feature contribution analysis layer.

The public term is controlled removal. It replaces older terminology and keeps the system focused on
whether each feature family deserves to stay.

## Profiles

- `full`: all enabled families.
- `technical_only`: only technical features.
- `no_context`: context removed.
- `no_fundamentals`: fundamentals removed.
- `no_sentiment`: sentiment removed.

## Decisions

`app/refine_decision.py` compares each removal profile against `full` and returns:

- `keep_family`: removal made the result worse or less robust.
- `remove_candidate`: removal improved return, drawdown and profit factor with enough trades.
- `observe`: effect is small or mixed.
- `inconclusive`: sample is too small or metrics are insufficient.

Refine never edits the default config, never removes features automatically and never overwrites
operational model artifacts. Shadow outputs stay under `artifacts/refine/...`.
