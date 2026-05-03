# Config organization and confidence fix

This patch is intentionally conservative.

## Preserved

- Existing CLI suextratreesace.
- Existing model architecture: XGB + CatBoost + ExtraTrees -> Ridge arbiter.
- Existing autotune command and YAML-controlled default.
- Daily routine without training.
- Report as TXT artifact.

## Changed

### 1. YAML organization

Sentiment now belongs under `features.sentiment`, together with technical, context and fundamental feature blocks.

Old shape:

```yaml
features:
  ...
sentiment:
  ...
```

New shape:

```yaml
features:
  technical:
    ...
  context:
    ...
  fundamentals:
    ...
  sentiment:
    ...
```

The shipped config uses a single sentiment switch: `features.sentiment.enabled`. There is no separate `as_feature` switch.

### 2. Confidence calculation

The previous confidence formula could collapse to 0% when the Ridge prediction was close to zero, even if base engines existed.

Confidence is now an operational agreement score based on base-engine dispersion and a configurable return scale:

```yaml
model:
  confidence:
    agreement_scale_return: 0.010
    minimum_when_engines_exist: 0.05
```

This avoids useless hard-zero confidence while still penalizing disagreement among engines.
