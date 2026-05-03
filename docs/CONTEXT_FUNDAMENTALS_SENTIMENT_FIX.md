# Context, fundamentals and sentiment correction

This patch keeps the CLI unchanged and moves the new behavior into `config.yaml`.

## Context

Context is now tied to the asset data registry:

- `data.context_registry.default`
- `data.context_registry.by_asset`
- `data.context_registry.future_candidates`

The data command downloads only the context basket assigned to the ticker. Context features are temporal and window-based: returns, volatility, correlation, beta, benchmark alignment and relative strength.

## Fundamentals

Fundamentals no longer become constant training columns by default. Historical CVM-aligned data can enter as temporal features. YFinance snapshot values remain available for reports/policy context, but are not used as fixed training columns unless explicitly enabled.

## Sentiment

Sentiment now has `mode: temporal_feature`. RSS/VADER scores are cached by day in `data_cache/sentiment/<ticker>_sentiment_daily.csv` and transformed into rolling daily features such as `sent_mean_3d`, `sent_count_7d`, `sent_delta_3d` and `sent_std_7d`.

This gives sentiment a time dimension without adding commands or changing the daily flow.
