# Context provider fallback

The data flow now treats the asset ticker as mandatory and context/index tickers as optional but validated at runtime.

Rules:

- `data.period` defaults to `max`.
- If the main asset supports `max`, it is downloaded normally.
- Context indices/macros that reject `max` fall back through `10y`, `5y`, `2y`, and `1y`.
- If a context provider still fails, the ticker is skipped and listed as `ctx skipped` in the data screen.
- Future B3 index candidates remain in `data.yaml`, but disabled until a provider ticker is confirmed.
- Group and subgroup baskets may include disabled B3 index codes; they do not break data loading.

This keeps the CLI simple while allowing better group/subgroup context over time.
