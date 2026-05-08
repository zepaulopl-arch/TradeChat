# data.yaml registry

`data.yaml` separates cadastral information from operational parameters.

`config.yaml` keeps runtime behavior: model, features, policy, validation and UI options.

`data.yaml` keeps asset metadata:

- ticker and company name;
- group and subgroup;
- financial vs non-financial classification;
- CNPJ field for CVM matching;
- context tickers used by Yahoo/yfinance;
- linked B3 indices for context expansion;
- a reference sample with more than 50 Brazilian equities.

The public CLI is `data`, `train`, `predict`, `validate`, `report` and `portfolio`.

## Conservative rule

Do not hardcode uncertain CNPJs. Assets with a trusted starter CNPJ use `cnpj_status: verified_seed`; others use `pending_lookup` and are ready for future B3/CVM hydration. This avoids silent wrong CVM joins.

## Context rule

The model downloads only the current ticker and its configured context tickers. Old ticker aliases are not resolved.
