# TradeGem data preparation

This patch adds an internal preparation step between feature generation and model training.

No CLI command was added. The step is controlled by `features.yaml` under `preparation`.

## Contract

1. Generate all enabled feature families.
2. Remove invalid/constant columns.
3. Clip extreme target returns if configured.
4. Select up to `max_features` using simple target relevance constrained by low pairwise correlation.
5. Normalize the selected matrix before all engines.

The goal is to keep the model small, auditable and less sensitive to multicollinearity, especially after adding context, fundamentals and sentiment.
