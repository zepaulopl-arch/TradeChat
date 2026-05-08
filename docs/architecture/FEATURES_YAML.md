# features.yaml

`features.yaml` is the feature contract of TradeGem. It is intentionally separate from:

- `config.yaml`: operational parameters, model, policy, simulation and UI.
- `data.yaml`: asset registry, groups, subgroups, CNPJ and context/index baskets.

## Structural rule

Feature windows are not global. Each family owns its own windows and generation attributes:

- `families.technical.presets.<name>.windows`
- `families.context.presets.<name>.windows`
- `families.fundamentals.presets.<name>.windows`
- `families.sentiment.presets.<name>.windows`

The active profile in `selection.active_profile` chooses which family preset is active.
The loader adapts this organized schema to the existing runtime keys, so the CLI and model engines remain unchanged.

## Current families

### technical
Price-derived features from the asset itself: returns, volatility, RSI, moving averages, EMA, ROC, Bollinger and fractional memory.

### context
Temporal market context linked to `data.yaml`: returns, volatility, correlation, beta, relative strength and alignment. No context snapshot should be repeated across all dates.

### fundamentals
CVM-aligned temporal fundamentals when available. Snapshot fundamentals remain report/policy context unless explicitly allowed as features.

### sentiment
Cached daily sentiment series. Training features are rolling windows from the cache: mean, count, delta and standard deviation.
