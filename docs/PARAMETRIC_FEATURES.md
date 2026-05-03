# Parametric feature update

This update keeps the working CLI and model contract intact.

Preserved:

- `data`, `train`, `predict`, `report`, `daily`
- `daily = data -> predict -> compact report`, without training
- base engines: `XGB + RandomForest + MLP`
- arbiter: `Ridge`
- original `BayesSearchCV` autotune pattern

Added behind `config.yaml`:

- `features.context`: market context from already downloaded macro tickers
- `features.fundamentals`: small valuation/quality/yield regime scores
- `sentiment.enabled`: use sentiment without adding a new CLI command
- `model.autotune.enabled_by_default`: activate autotune from YAML without changing commands

No new command was added. The normal surface remains:

```powershell
.\run data PETR4
.\run train PETR4
.\run train PETR4 --autotune
.\run predict PETR4
.\run report PETR4
.\run daily PETR4, VALE3, ITUB4
```

Recommended defaults:

- keep sentiment disabled by default because RSS/translation is slow and fragile; enable it only through `features.sentiment.enabled`;
- keep context enabled because it uses the same price cache already downloaded by `data`;
- keep fundamental regime features enabled, but treat them as context/filter, not as direct buy/sell rules.
