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

   `python trade.py data load --list ibov`

2. Train models

   `python trade.py train --list ibov`

3. Run Matrix and analyze it

   `python trade.py validate matrix --universe ibov --jobs 4`

   `python trade.py validate report --latest`

4. Promote Matrix evidence into runtime policy

   `python -c "from app.commands.promote_policy import promote_policy; promote_policy('logs/policy_matrix/<run_dir>')"`

5. Generate smart signals and smart rank

   `python trade.py signal smart PETR4.SA`

   `python trade.py signal rank --list ibov --smart --rank-limit 20`

6. Validate methodology

   `python trade.py validate --list ibov --mode walkforward`

7. Refine components

   `python trade.py refine --list ibov --removal --walkforward`

8. Review portfolio

   `python trade.py portfolio status`

   `python trade.py portfolio rebalance`

One-line operational refresh:

`python scripts/run_operational_matrix.py --universe ibov --jobs 4 --rank-limit 20`

## 3. Command Notes

- `data load PETR4.SA` updates the local price cache.
- `train` never validates automatically.
- `signal` does not rebalance the portfolio.
- `validate --mode replay` is a replay sanity check over saved operational models.
- `validate --mode walkforward` is the methodological validation path.
- `validate matrix --universe ibov` runs the Matrix over `config/universes/ibov.yaml`.
- Matrix operational uses only the `active` profile.
- `validate report --latest` analyzes the latest Matrix run and rebuilds `validation_summary.csv` if needed.
- `promote_policy` reads a Matrix run and writes `runtime/runtime_policy.json`.
- `signal smart TICKER` applies the asset-specific active runtime policy, live YAML active defaults and Matrix guard.
- `signal rank --list ibov --smart --rank-limit 20` processes all IBOV assets, sorts the complete queue, and displays the top 20 rows.
- `signal report TICKER` prefers `latest_smart_signal.json` when available and writes a decision-path audit report.
- `refine --removal` means controlled removal; it recommends, but never changes defaults automatically.
- `refine` writes shadow artifacts under `artifacts/refine/...`.
- `portfolio plan` and `portfolio simulate` render a dry-run allocation without saving state.

## 4. Daily Operating Rhythm

For a simple desktop, keep daytime commands light and leave the expensive work for quiet windows.

Morning or trading session:

`python trade.py signal rank --list ibov --smart --rank-limit 20`

or the deterministic operational wrapper:

`python scripts/run_operational_matrix.py --universe ibov --jobs 4 --rank-limit 20`

Use this as the main panel. It answers the daily triage questions:

- was the asset evaluated by Matrix?
- was it promoted or rejected?
- if promoted, did the operational signal confirm or block it?

After market close:

`python trade.py data load --list ibov`

`python trade.py signal rank --list ibov --smart --rank-limit 20`

This refreshes the local cache and rebuilds the globally sorted decision queue.

Overnight:

`python scripts/run_operational_matrix.py --universe ibov --jobs 1 --rank-limit 20`

Use `--jobs 1` on a modest machine. Raise it only if CPU, RAM and disk stay comfortable.

Weekend:

`python trade.py train --list ibov`

`python scripts/run_operational_matrix.py --universe ibov --jobs 1 --rank-limit 20`

Weekend is the right window for full retraining and detailed Matrix refreshes. Keep trading-session work focused on reading the panel, not rebuilding the world.

## 5. Deterministic Operating Scripts

These scripts are automation helpers, not new TradeChat commands. They sit above
the deterministic core and do not decide buy/sell, change constraints or operate
the portfolio.

Operational routine:

`python scripts/run_operational_matrix.py --universe ibov --jobs 4 --rank-limit 20`

This runs:

- Matrix with the `active` profile;
- `validate report --latest`;
- `promote_policy`;
- `check_runtime_policy.py`;
- `signal rank --list ibov --smart --rank-limit 20`;
- `validate_smart_rank_output.py`.

Artifacts are saved under `artifacts/operational/<timestamp>/`:

- `smart_rank.txt`: the decision panel;
- `runtime_check.txt`: runtime health audit;
- `smart_rank_check.txt`: output regression audit;
- `summary.txt`: command status and final footer.

Standalone audit commands:

`python scripts/check_runtime_policy.py`

`python scripts/validate_smart_rank_output.py artifacts/operational/<run>/smart_rank.txt --expected-rows 20`

After-hours Matrix refresh on a modest desktop:

`python scripts/run_operational_matrix.py --universe ibov --jobs 1 --rank-limit 20`

This runs Matrix, report, promotes the Matrix run that was just created, checks
`runtime/runtime_policy.json`, captures `smart_rank.txt` in UTF-8, and validates
the captured table. It does not depend on PowerShell variables or `Tee-Object`.

Use `--allow-untrained` only for smoke/dev runs. Operationally, assets with
known insufficient history should become `INELIGIBLE_DATA` / `SKIP`.

Use `--dry-run` to inspect the planned commands without executing them:

`python scripts/run_operational_matrix.py --universe ibov --rank-limit 20 --dry-run`

## 6. Matrix Runtime Policy

The operational smart flow is:

`Matrix -> promote_policy -> runtime/runtime_policy.json -> config/runtime_policy.yaml -> signal smart -> signal rank --smart`

The runtime JSON must retain every asset evaluated by Matrix:

- `evaluated: true`: the asset entered the Matrix/promotion process.
- `profile: active` and `policy_type: asset_specific_active`: the runtime policy is the final active policy calibrated for that ticker.
- `overrides`: calibrated active parameters for that ticker, including buy/sell thresholds, minimum confidence, preferred horizon, minimum RR, minimum trades, drawdown/exposure limits, risk budget and maximum position cap.
- `promoted: true` and `promotion_status: promoted`: the asset passed configured constraints.
- `actionable_candidate: true`: the selected Matrix decision is operationally allowed, normally `APPROVE`; final action still depends on the signal and guard.
- `promoted: false` and `promotion_status: rejected_by_constraints`: the asset was evaluated and rejected by constraints.
- `promoted: false` and `promotion_status: ineligible_data`: the asset was skipped by data/model eligibility, for example insufficient history.
- `rejection_reasons`: human-readable constraint failures, for example `trades 2 < 15`.
- `source: policy_matrix` or `source: data_eligibility`: confirms whether the row came from Matrix evidence or eligibility skip.
- `evidence`: Matrix row used for selection, including decision, profit factor and trade count.

An asset evaluated by Matrix must not disappear from runtime. If an asset from
the requested universe is absent from runtime after the operational Matrix, smart
rank treats it as `ERROR` because coverage failed.

`config/runtime_policy.yaml` owns:

- promotion constraints such as minimum trades, minimum Sharpe and maximum exposure;
- the fallback profile;
- live active defaults and calibration bounds;
- Matrix decision guard rules such as `OBSERVE -> NEUTRAL/BLOCK`.

The YAML active profile is the base. Stored `asset_specific_active` overrides in
`runtime/runtime_policy.json` are ticker-specific and win over the YAML base.
That is the handoff from Matrix/autotune to signal generation.

## 7. SMART RANK

Operational command:

`python trade.py signal rank --list ibov --smart --rank-limit 20`

Rules:

- promoted asset: generates or loads the smart signal flow and applies Matrix guard;
- promoted is not the same as actionable; final `ACTIONABLE` requires an allowed Matrix decision, non-ineligible data, guard `OK`, and an operational final signal;
- evaluated but rejected asset: renders `REJECTED` / `BLOCK` and does not call signal generation;
- ineligible asset: renders `INELIGIBLE_DATA` / `SKIP` and does not call signal generation;
- missing Matrix asset: renders `ERROR`;
- technical error: renders `ERROR`;
- smart rank must not render any actionable signal without Matrix evidence.

Columns:

| Column | Meaning |
| --- | --- |
| `#` | Display index after global sorting. |
| `TICKER` | Asset ticker. |
| `ACTION` | Operational state: `ACTIONABLE`, `WATCH`, `BLOCKED`, `REJECTED`, `INELIGIBLE` or `ERROR`. |
| `SIGNAL` | Final smart signal, `REJECTED`, `INELIGIBLE_DATA` or `ERROR`. |
| `MATRIX` | Matrix decision from evidence. |
| `PF` | Matrix profit factor. |
| `TRD` | Matrix trade count. |
| `RR` | Signal RR when available; otherwise Matrix/evidence RR; otherwise `n/a`. |
| `GUARD` | `OK`, `BLOCK`, `SKIP` or `ERROR`. |
| `REASON` | Reason for block, rejection, skip or error. |

The header prints a policy fingerprint with the YAML path, runtime path, universe and processed asset count. The table body is plain text without ANSI color codes so PowerShell alignment remains stable.

The footer prints:

- processed/displayed counts, for example `Processed: 79`, `Displayed: 20 of 79`, `Rows: 20 of 79`;
- row counts by action: `ACTIONABLE`, `WATCH`, `BLOCKED`, `REJECTED`, `INELIGIBLE`, `ERROR`;
- top actionable tickers, when any;
- the main blocker across blocked/rejected/skipped/error rows.

When there are no actionable rows, the footer prints a short reason breakdown,
for example rejected by constraints, blocked by Matrix guard, ineligible data,
coverage errors, and technical errors.

## 8. Interpretation

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

## 9. Installation Layers

- `requirements-core.txt`: data, YAML, cache and utility dependencies.
- `requirements-ml.txt`: tabular models, autotune and PyBroker validation.
- `requirements-sentiment.txt`: RSS, VADER/NLTK and translation.
- `requirements-dev.txt`: pytest, pytest-cov, ruff and black.
- `requirements.txt`: complete installation.
