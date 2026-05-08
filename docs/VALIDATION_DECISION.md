# Validation Decision

`app/evaluation_decision.py` transforma metricas economicas e baselines em decisao objetiva.

## Entradas

- `model_metrics`
- `baseline_comparison`
- `config.validation_decision`, quando existir

## Baselines Obrigatorios

- `zero_return_no_trade`
- `buy_and_hold_equal_weight`
- `mean_return_long_flat`
- `last_return_long_flat`

O baseline aleatorio continua sendo exibido, mas nao e obrigatorio por padrao.

## Checks

- baselines obrigatorios vencidos;
- retorno total positivo;
- drawdown dentro do limite;
- profit factor minimo;
- numero minimo de trades;
- exposicao ativa dentro da faixa;
- hit rate;
- retorno medio.

## Decisoes

- `promote`: checks criticos passam e a amostra e suficiente.
- `observe`: resultado promissor, mas com fragilidades.
- `reject`: falha economica clara.
- `inconclusive`: metricas, trades, exposicao ou baselines insuficientes.

## Alerta Sobre Confianca

Operational quality reflects internal model agreement, historical error and stability. It is not a statistical probability of profit.

Nenhuma decisao deve ser lida como garantia de lucro.
