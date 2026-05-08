# TradeChat

TradeChat is a quantitative CLI for B3 assets. It organizes data, operational model training, signal generation, validation, controlled removal and virtual portfolio review without adding extra trading engines to the core pipeline.

## Install

Use Python 3.10+.

```powershell
python -m pip install -r requirements.txt
```

Layered installs:

- `requirements-core.txt`: data, config and cache utilities.
- `requirements-ml.txt`: tabular models, optimization and PyBroker validation.
- `requirements-sentiment.txt`: RSS, sentiment and translation dependencies.
- `requirements-dev.txt`: tests and code quality tools.

## Main Commands

```powershell
python trade.py data load PETR4.SA
python trade.py train PETR4.SA
python trade.py signal generate PETR4.SA
python trade.py signal rank --list validacao
python trade.py signal report PETR4.SA
python trade.py validate --list validacao --mode walkforward
python trade.py refine --list validacao --removal --walkforward
python trade.py portfolio status
python trade.py portfolio rebalance
```

`predict` and `report` are deprecated aliases. Use `signal generate`, `signal rank` and `signal report`.

## Validation

`validate --mode replay` is an operational sanity check over saved models. `validate --mode walkforward` is the methodological validation path because it trains shadow artifacts by rebalance date.

`refine --removal` uses controlled removal and writes shadow artifacts under `artifacts/refine/...`; it does not replace operational models under `artifacts/models`.

## Test

```powershell
python -m pytest
```

See `OPERATIONAL_MANUAL.md` and the documents under `docs/` for the full workflow.
