# Testing

The test suite is offline by default. It should not require yfinance, internet access or a real
PyBroker installation.

## Core Commands

```powershell
python -m black --check app tests
python -m ruff check app tests
python -m pytest
```

## Test Focus

- CLI contract: exactly six root commands and no legacy aliases.
- Config schema: invalid units, modes and ranges fail clearly.
- Validation decision: promote, observe, reject and inconclusive.
- Refine decision: keep, remove candidate, observe and inconclusive.
- Walk-forward: temporal windows and embargo do not leak future data.
- Shadow artifacts: validation/refine do not overwrite operational models.
- Legacy cleanup: no root launcher scripts, no compatibility handlers and no simulator facade.
