# TradeChat Operational Manual

The public CLI follows a single flow:

`data -> train -> signal -> validate -> refine -> portfolio`

## 1. Public Commands

TradeChat has six public root commands:

| Command | Role |
| --- | --- |
| `data` | Load, inspect and audit market data. |
| `train` | Train operational models and save operational artifacts. |
| `signal` | Generate signals, rankings and signal reports. |
| `validate` | Run replay or walk-forward validation with baselines and a validation decision. |
| `refine` | Run contribution analysis and controlled removal in shadow artifacts. |
| `portfolio` | Inspect and update the virtual portfolio layer. |

## 2. Recommended Workflow

1. Load data

   `python trade.py data load --list validacao`

2. Train models

   `python trade.py train --list validacao`

3. Generate signals

   `python trade.py signal rank --list validacao`

4. Validate methodology

   `python trade.py validate --list validacao --mode walkforward`

5. Refine components

   `python trade.py refine --list validacao --removal --walkforward`

6. Review portfolio

   `python trade.py portfolio status`

   `python trade.py portfolio rebalance`

## 3. Command Notes

- `data load PETR4.SA` updates the local price cache.
- `train` never validates automatically.
- `signal` does not rebalance the portfolio.
- `validate --mode replay` is a replay sanity check over saved operational models.
- `validate --mode walkforward` is the methodological validation path.
- `refine --removal` means controlled removal; it recommends, but never changes defaults automatically.
- `refine` writes shadow artifacts under `artifacts/refine/...`.
- `portfolio plan` and `portfolio simulate` render a dry-run allocation without saving state.

## 4. Interpretation

Validation decisions use:

- model return;
- baseline comparison;
- drawdown;
- profit factor;
- trade count;
- exposure;
- average return and hit rate.

Possible decisions:

- `promote`: strong enough for promotion review.
- `observe`: promising but not decisive.
- `reject`: fails economic evidence.
- `inconclusive`: insufficient sample, exposure or baseline coverage.

Operational quality is not a probability of profit. It reflects internal model agreement, historical error and stability.

## 5. Installation Layers

- `requirements-core.txt`: data, YAML, cache and utility dependencies.
- `requirements-ml.txt`: tabular models, autotune and PyBroker validation.
- `requirements-sentiment.txt`: RSS, VADER/NLTK and translation.
- `requirements-dev.txt`: pytest, pytest-cov, ruff and black.
- `requirements.txt`: complete installation.
