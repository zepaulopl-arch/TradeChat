# Validation

Validation is the methodological gate of TradeChat.

## Modes

- `replay`: operational sanity check using existing operational models.
- `walkforward`: methodological validation using temporal windows and shadow artifacts.

Replay can show that the plumbing works. Walk-forward is the path for quantitative conclusions.

## Baselines

Validation must compare the model against economic baselines:

- `zero_return_no_trade`
- `buy_and_hold_equal_weight`
- `mean_return_long_flat`
- `last_return_long_flat`
- `random_long_flat`

## Decision Matrix

`app/evaluation_decision.py` returns one of:

- `promote`: strong enough for promotion review.
- `observe`: promising but incomplete.
- `reject`: clear economic failure.
- `inconclusive`: missing data, too few trades, too little exposure or insufficient baselines.

The decision uses baseline beat rate, total return, drawdown, profit factor, trade count, exposure,
hit rate and average trade return. It is conservative: low trade count or low exposure prevents
promotion even when return is positive.

Operational quality is not probability of profit. It reflects internal model agreement, historical
error and stability.
