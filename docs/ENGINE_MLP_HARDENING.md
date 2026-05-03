# Engine and MLP hardening

This patch keeps the original TradeGem architecture:

```text
XGB + RandomForest + MLP -> Ridge arbiter
```

The correction is conservative and focused on the engine layer.

## Main fix

The fixed MLP configuration now respects the sklearn meaning of `hidden_layer_sizes`.

```yaml
hidden_layer_sizes: [64, 32]
```

means two hidden layers, not just the last value. The previous adapter could collapse this to a single `32` unit layer, which changed the MLP materially.

## Consensus guard

A row-wise consensus guard was added before Ridge arbitration. It does not remove any engine and does not change the public architecture. It only prevents a single divergent engine prediction from dominating the stacked input.

Configured in `config/config.yaml`:

```yaml
model:
  engine_safety:
    consensus_guard_enabled: true
    max_deviation_from_median: 0.025
```

Raw predictions are still saved for audit.

## MLP defaults

MLP now has explicit defaults:

```yaml
activation: relu
solver: adam
learning_rate: adaptive
max_iter: 800
validation_fraction: 0.12
n_iter_no_change: 30
```

## Tests

Static tests verify:

- MLP keeps the configured two-layer architecture.
- consensus guard exists on train/test/latest stacking inputs.
- engine safety is YAML-configurable.
- YAML files live under `config/`.
