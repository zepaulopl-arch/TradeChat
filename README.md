# TradeChat

TradeChat is a quantitative analysis CLI for B3 assets. It keeps market data, feature generation, tabular models, signal policy, reports, portfolio simulation and methodological checks in one operational pipeline.

## Install

Use Python 3.10+.

```powershell
python -m pip install -r requirements.txt
```

Layered installs are also available:

- `requirements-core.txt`: data, config and cache utilities.
- `requirements-ml.txt`: tabular models, optimization and PyBroker validation.
- `requirements-sentiment.txt`: RSS, sentiment and translation dependencies.
- `requirements-dev.txt`: tests.

## Main Commands

```powershell
python trade.py data PETR4.SA
python trade.py train PETR4.SA
python trade.py predict PETR4.SA
python trade.py predict PETR4.SA VALE3.SA --rank
python trade.py report PETR4.SA
python trade.py portfolio
python trade.py portfolio --rebalance
python trade.py validate PETR4.SA VALE3.SA --mode walkforward
python trade.py refine PETR4.SA VALE3.SA
python trade.py refine PETR4.SA VALE3.SA --removal --walkforward
```

`validate` includes economic baselines. `refine --removal` trains shadow artifacts under `artifacts/refine/...`, so removal tests do not replace operational models.

## Test

```powershell
python -m pytest
```

See `OPERATIONAL_MANUAL.md` for the practical workflow.
