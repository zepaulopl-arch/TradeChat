# Tabular engine stack

The operational TradeGem stack now uses only tabular engines:

```text
XGBoost + CatBoost + ExtraTrees -> Ridge arbiter
```

Rationale:

- Diagnostics favor stable tabular specialists for this dataset.
- The operational system works with tabular features derived from time series, fundamentals, context and sentiment.
- Tree/boosting ensembles are more stable for heterogeneous tabular data and do not require neural-specific scaling contracts.
- Neural benchmarks should be kept outside the daily arbiter until they prove stable in a separate test bench.

The Ridge arbiter remains the stacking judge and is not a base engine.

## Engines

- `xgb`: main gradient boosting specialist.
- `catboost`: complementary boosting specialist.
- `extratrees`: stable randomized tree ensemble.

## Autotune

Autotune remains optional on `train --autotune` and now targets the three tabular engines only.
