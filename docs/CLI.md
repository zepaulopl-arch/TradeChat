# TradeChat CLI

The public CLI has exactly six root commands:

```text
data
train
signal
validate
refine
portfolio
```

## Data

```powershell
python trade.py data load PETR4.SA
python trade.py data load --list validacao
python trade.py data status PETR4.SA
python trade.py data audit PETR4.SA
```

Loads market data and inspects cache health. It does not train models.

## Train

```powershell
python trade.py train PETR4.SA
python trade.py train --list validacao --workers 3
python trade.py train PETR4.SA --update
python trade.py train PETR4.SA --autotune
```

Trains operational models and writes operational artifacts.

## Signal

```powershell
python trade.py signal generate PETR4.SA
python trade.py signal rank --list validacao
python trade.py signal report PETR4.SA
python trade.py signal report PETR4.SA --refresh
```

Generates signals, ranks saved/generated signals and writes signal audit reports.

## Validate

```powershell
python trade.py validate --list validacao --mode replay
python trade.py validate --list validacao --mode walkforward
python trade.py validate PETR4.SA VALE3.SA --mode walkforward
```

Runs replay or walk-forward validation, baselines and the validation decision matrix.

## Refine

```powershell
python trade.py refine --list validacao
python trade.py refine --list validacao --removal
python trade.py refine --list validacao --removal --walkforward
```

Runs feature contribution analysis and controlled removal. It only recommends changes.

## Portfolio

```powershell
python trade.py portfolio status
python trade.py portfolio plan
python trade.py portfolio rebalance
python trade.py portfolio simulate
python trade.py portfolio live
```

Inspects portfolio state, renders dry-run plans, saves explicit rebalances and monitors live prices.
