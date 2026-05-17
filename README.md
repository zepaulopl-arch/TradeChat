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
python trade.py signal smart PETR4.SA
python trade.py signal rank --list ibov --smart --rank-limit 20
python trade.py signal report PETR4.SA
python trade.py validate matrix --universe ibov --jobs 4
python trade.py validate report --latest
python -c "from app.commands.promote_policy import promote_policy; promote_policy('logs/policy_matrix/<run_dir>')"
python trade.py validate --list ibov --mode walkforward
python trade.py refine --list ibov --removal --walkforward
python trade.py portfolio status
python trade.py portfolio rebalance
python trade.py portfolio plan
```

The operational IBOV universe is `config/universes/ibov.yaml`. Do not use the old
validation-list examples for the Matrix -> smart rank flow.

## Validation

`validate --mode replay` is an operational sanity check over saved models. `validate --mode walkforward` is the methodological validation path because it trains shadow artifacts by rebalance date.

`refine --removal` uses controlled removal and writes shadow artifacts under `artifacts/refine/...`; it does not replace operational models under `artifacts/models`.

## Matrix -> Runtime -> Smart Rank

The intended operational flow is:

```text
Matrix -> promote_policy -> runtime/runtime_policy.json -> config/runtime_policy.yaml -> signal smart -> signal rank --smart
```

`promote_policy` reads a Matrix run directory containing `validation_summary.csv` and writes
`runtime/runtime_policy.json`. Every asset evaluated by the Matrix stays in runtime:

- `promoted: true` / `promotion_status: promoted`: passed the constraints and can generate a smart signal.
- `promoted: false` / `promotion_status: rejected_by_constraints`: evaluated by Matrix but blocked by constraints; smart rank renders `REJECTED` and does not generate a signal.
- `no_matrix`: the asset is absent from `runtime_policy.json`; smart rank renders `NO_MATRIX` / `SKIP`.

`config/runtime_policy.yaml` owns the live promotion constraints, fallback profile, Matrix decision guard, and runtime policy overrides. Stored runtime overrides are merged with live YAML overrides, with YAML taking precedence.

`python trade.py signal smart PETR4.SA` generates one smart signal using the selected runtime profile, Matrix evidence and decision guard.

`python trade.py signal rank --list ibov --smart --rank-limit 20` processes and displays only 20 assets. It never falls back to a fake balanced BUY/SELL when Matrix evidence is missing.

`python trade.py signal report PETR4.SA` prefers the latest smart signal when it exists and writes an audit report with the decision path.

SMART RANK columns:

- `TICKER`: asset ticker.
- `SIGNAL`: final smart signal, or `REJECTED`, `NO_MATRIX`, `ERROR`.
- `PROFILE`: runtime policy profile selected by Matrix/runtime.
- `MATRIX`: Matrix decision from evidence.
- `PF`: Matrix profit factor.
- `TRD`: Matrix trade count.
- `RR`: signal risk/reward after policy and guards.
- `GUARD`: `OK`, `BLOCK`, `SKIP`, or `ERROR`.
- `BLOCKER`: reason for block, rejection, skip or error.

Daily operating rhythm for a simple desktop:

- During the day: use `python trade.py signal rank --list ibov --smart --rank-limit 20` as the decision panel. Increase the limit only when you have time.
- After market close: refresh data and generate the next ranked batch.
- Overnight: run Matrix/report/promotion jobs that touch the whole IBOV.
- Weekend: retrain broader model sets and rerun detailed Matrix validation.

Keep the operating question simple: was the asset evaluated by Matrix, was it promoted or rejected, and did the operational signal confirm or block it?

## Deterministic Operating Scripts

The `scripts/` helpers automate the operating routine without adding public CLI
commands and without making buy/sell decisions.

Daily lightweight run:

```powershell
python scripts/run_daily_smart_rank.py --universe ibov --rank-limit 20
```

Default flow:

```text
data load -> runtime check -> smart rank -> smart rank output check
```

Outputs are written under `artifacts/operational/<timestamp>/`, including
`smart_rank.txt`, `runtime_check.txt`, `smart_rank_check.txt` and `summary.txt`.

Standalone checks:

```powershell
python scripts/check_runtime_policy.py
python scripts/validate_smart_rank_output.py artifacts/operational/<run>/smart_rank.txt --expected-rows 20
```

Night/weekend Matrix refresh on a modest desktop:

```powershell
python scripts/run_daily_smart_rank.py --universe ibov --rank-limit 20 --with-matrix --jobs 1
```

`--with-matrix` is explicit because it can be slow and can refresh
`runtime/runtime_policy.json` through `promote_policy`. Keep it for after-hours
or weekend work.

## Test

```powershell
python -m pytest
```

See `OPERATIONAL_MANUAL.md` and the documents under `docs/` for the full workflow.
