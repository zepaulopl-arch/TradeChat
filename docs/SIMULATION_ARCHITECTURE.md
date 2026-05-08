# Simulation Architecture

PyBroker e usado como adapter de validacao e simulacao. Ele nao e o nucleo do TradeChat.

## Pacote `app/simulation/`

- `pybroker_adapter.py`: import opcional do PyBroker e erro controlado quando ausente.
- `replay.py`: normalizacao do modo de validacao.
- `walkforward.py`: criacao de janelas temporais sem vazamento.
- `execution_costs.py`: leitura de custos e slippage.
- `artifacts.py`: utilitarios para artefatos de simulacao.
- `metrics_bridge.py`: conversao de metricas externas para dicionarios internos.
- `types.py`: dataclasses de configuracao e resultado.

## Modos

- Replay usa modelos operacionais salvos e serve para sanity check.
- Walk-forward treina modelos sombra por data de rebalanceamento e serve para validacao metodologica.

## Custos

Custos sao configurados em `simulation.costs`:

- `fee_mode`
- `fee_amount`
- `slippage_pct`

Valores negativos sao invalidos.

## Artefatos

Validacoes gravam summaries, trades, ordens, stops e signal plan em `artifacts/simulations/...`. Caminhos tecnicos aparecem na CLI apenas em modo verbose.
