# Engine and MLP feature balance

This patch keeps the original architecture `XGB + RandomForest + MLP -> Ridge`, but hardens the MLP and feature preparation.

- Feature preparation is family-balanced: context can no longer dominate the selected feature set.
- The MLP receives its own robust-scaled subset, normally technical + context features only.
- Divergent latest engine predictions are neutralized before the Ridge arbiter and shown in the signal screen.
- Raw predictions remain available for audit.
