# Data screen

`data` now renders with the same compact screen style used by `train` and `signal`.

It updates the price cache and then displays:

- market data status, rows, period and date range;
- asset-specific context tickers from `config/data.yaml`;
- registry group, subgroup, financial class and CNPJ when registered;
- fundamental availability status;
- sentiment cache status when enabled in `config/features.yaml`.

No CLI command was added or renamed.
