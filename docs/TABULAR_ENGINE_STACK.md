# Tabular engine stack

The operational TradeGem stack now uses only tabular engines:

```text
XGBoost + CatBoost + ExtraTrees -> Ridge arbiter
```

Rationale:

- Diagnostics showed the removed neural engine frequently produced divergent raw predictions.
- The daily system works with tabular features derived from time series, fundamentals, context and sentiment.
- Tree/boosting ensembles are more stable for heterogeneous tabular data and do not require neural-specific scaling contracts.
- Neural benchmarks should be kept outside the daily arbiter until they prove stable in a separate test bench.

The Ridge arbiter remains the stacking judge and is not a base engine.

## Engines

- `xgb`: main gradient boosting specialist.
- `catboost`: complementary boosting specialist.
- `extratrees`: stable randomized tree ensemble.

## Removed from operational stack

- `removed_neural`: removed from production arbitration.
- `extratrees`: replaced by `extratrees` as the simpler variance-control ensemble.

## Autotune

Autotune remains optional on `train --autotune` and now targets the three tabular engines only.
