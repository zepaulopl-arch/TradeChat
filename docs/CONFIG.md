# Configuration

TradeChat loads `config/config.yaml` and `config/features.yaml` through `app/config.py`.

The loader:

1. reads YAML;
2. normalizes runtime defaults;
3. adapts `features.yaml` family presets to runtime feature blocks;
4. validates the resulting config with `app/config_schema.py`;
5. raises on real errors and keeps replay warnings as warnings.

## Important Units

- `simulation.initial_cash`: currency amount.
- `simulation.max_positions`: positive integer.
- `simulation.costs.fee_mode`: `order_percent`, `per_order`, `per_share` or `none`.
- `simulation.costs.fee_amount`: non-negative value interpreted by `fee_mode`.
- `simulation.costs.slippage_pct`: percent value, not decimal fraction.
- `model.confidence.*`: fractions from 0 to 1.
- `model.prediction_guards.*`: non-negative return guard values.

## Validation Modes

`simulation.mode = replay` is valid, but it means operational sanity check. Use
`--mode walkforward` when the goal is methodological validation.
