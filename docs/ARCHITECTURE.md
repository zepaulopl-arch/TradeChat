# TradeChat Architecture

TradeChat is organized around one operational flow:

`data -> train -> signal -> validate -> refine -> portfolio`

The CLI is intentionally small. Each command module in `app/commands/` interprets arguments and
delegates work to services. Quantitative logic stays in data, feature, model, validation, refine and
portfolio services. Presentation belongs to terminal UI/rendering modules.

## Main Packages

- `app/commands/`: public command orchestration.
- `app/data.py`: market data loading, cache status and ticker resolution.
- `app/features.py` and `app/preparation.py`: feature construction and train-only preparation.
- `app/models.py`: tabular model training and prediction.
- `app/pipeline_service.py`: shared operational data/train/signal helpers.
- `app/evaluation_service.py`: economic metrics and baselines.
- `app/evaluation_decision.py`: promote/observe/reject/inconclusive matrix.
- `app/refine_service.py` and `app/refine_decision.py`: controlled removal and contribution decisions.
- `app/simulation/`: replay, walk-forward, costs, PyBroker adapter and simulation artifacts.
- `app/ui/` and `app/presentation.py`: terminal rendering.

## vNext Cleanup Rules

- No `predict` or `report` root commands.
- No legacy launcher scripts.
- No compatibility facades for old command or simulation layers.
- PyBroker remains an adapter, not the TradeChat core.
- Walk-forward and refine write shadow artifacts and do not overwrite operational models.
