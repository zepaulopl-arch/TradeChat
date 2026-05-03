# data.yaml registry

`data.yaml` separates cadastral information from operational parameters.

`config.yaml` keeps runtime behavior: model, features, policy, daily/report options.

`data.yaml` keeps asset metadata:

- ticker and company name;
- group and subgroup;
- financial vs non-financial classification;
- CNPJ field for CVM matching;
- context tickers used by Yahoo/yfinance now;
- linked B3 indices for future context expansion;
- a reference sample with more than 50 Brazilian equities.

The system does not add new CLI commands. `data`, `train`, `predict`, `daily` keep the same surface.

## Conservative rule

Do not hardcode uncertain CNPJs. Assets with a trusted starter CNPJ use `cnpj_status: verified_seed`; others use `pending_lookup` and are ready for future B3/CVM hydration. This avoids silent wrong CVM joins.

## Context rule

The model downloads only `context_tickers` for the asset. `linked_indices` documents relevant B3 indices and can be used later when a reliable provider is added.
