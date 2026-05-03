# Ticker aliases and context registry

This patch keeps the CLI unchanged and improves data robustness.

## Ticker migrations

`data.yaml` now has an `aliases` section. Old B3 tickers can resolve to current canonical tickers before cache/download/model operations.

Example:

```text
ELET3.SA -> AXIA3.SA
ELET5.SA -> AXIA5.SA
ELET6.SA -> AXIA6.SA
```

The data screen reports the alias resolution instead of crashing silently.

## Context baskets

`data.yaml` now separates context by:

- global market context;
- group defaults;
- subgroup defaults;
- asset-level linked indices;
- asset-level context tickers.

Index codes are resolved through `indices.catalog` to fetchable Yahoo symbols only when marked available.

Available examples:

```text
IBOV   -> ^BVSP
IFNC   -> IFNC.SA
IEEX   -> ^IEE
IBRX50 -> ^IBX50
```

Future B3 sector indices remain registered but disabled until their provider symbol is verified.
