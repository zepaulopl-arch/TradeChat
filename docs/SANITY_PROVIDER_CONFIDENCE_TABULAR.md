# Sanity patch: provider registry, confidence and tabular stack

This patch keeps the CLI unchanged and focuses on operational correctness.

## Model stack

Production stack is fully tabular:

```text
XGBoost + CatBoost + ExtraTrees -> Ridge arbiter
```

The previous removed neural engine path was removed from production because diagnostics showed repeated divergent raw predictions and frequent engine guards.

## Confidence

Confidence is no longer an agreement-only score. It now combines:

- engine agreement;
- out-of-sample MAE;
- predicted-return magnitude;
- number of usable engines;
- discarded-engine penalty;
- train sample size.

This prevents near-zero D+1 predictions from showing unrealistic 98-100% confidence just because two engines agree after a third engine was discarded.

## Sentiment pipeline

Missing `sent_*` columns at prediction time are recovered as zero-valued columns. This keeps train/predict contracts stable when the RSS cache changes between training and prediction.

## Data registry

Known B3 ticker migrations / corporate actions were added as aliases and inactive legacy entries:

- ARZZ3/SOMA3 -> AZZA3
- CCRO3 -> MOTV3
- RRRP3 -> BRAV3
- BRFS3/MRFG3 -> MBRF3
- AZUL4 -> AZUL54
- GOLL4 -> GOLL54
- EMBR3 -> EMBJ3

Legacy tickers are excluded from the reference diagnostic sample. Current tickers that are still active but had provider failures are kept active and marked for provider verification instead of being removed.
