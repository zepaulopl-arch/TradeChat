# Asset diagnostic batch

Use this maintenance script when you need evidence before changing the model,
feature selection, context registry or removed neural engine guards.

It runs the current operational sequence for registered assets:

```text
data -> train -> predict
```

It does not add a new TradeGem CLI command.

## Commands

Smoke test with 5 registered assets:

```powershell
.\run_diagnostics.bat --limit 5
```

Specific assets:

```powershell
.\run_diagnostics.bat --assets PETR4,VALE3,ITUB4,AXIA3
```

All registered reference assets:

```powershell
.\run_diagnostics.bat
```

Skip data refresh and use cache:

```powershell
.\run_diagnostics.bat --no-data
```

## Outputs

Each run writes:

```text
artifacts/diagnostics/diag_YYYYMMDD_HHMMSS/assets_diagnostic.csv
artifacts/diagnostics/diag_YYYYMMDD_HHMMSS/assets_diagnostic.json
artifacts/diagnostics/diag_YYYYMMDD_HHMMSS/assets_diagnostic_summary.txt
```

The CSV is the main analysis file. It includes:

- asset registry data: group, subgroup, class, CNPJ, linked indices;
- data status: rows, date range, context available/unavailable;
- fundamentals and sentiment status;
- feature counts, top features and family mix;
- MAE, prediction and confidence;
- raw/guarded engine outputs;
- discarded or neutralized engines;
- signal and policy reasons.

## What to look for

- Many assets with context-heavy feature selection: tune `features.yaml` selection limits.
- Many assets with removed neural engine discarded: tune removed neural engine input/scaler/guards.
- One isolated asset failing: fix `data.yaml` alias/context/registry, not the engine.
- Many assets with unavailable context: validate provider tickers in `data.yaml`.
