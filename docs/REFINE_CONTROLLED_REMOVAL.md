# Refine Controlled Removal

O termo publico e `controlled removal`. O objetivo e medir contribuicao de familias sem contaminar artefatos operacionais.

## Perfis

- `full`: referencia completa.
- `technical_only`: remove contexto, fundamentos e sentimento.
- `no_context`: remove contexto.
- `no_fundamentals`: remove fundamentos.
- `no_sentiment`: remove sentimento.

## Decisoes

- `keep_family`: remover piorou retorno, MAE, robustez ou profit factor.
- `remove_candidate`: remover melhorou retorno, drawdown e profit factor com trades suficientes.
- `observe`: impacto pequeno ou misto.
- `inconclusive`: amostra insuficiente.

## Artefatos

`refine --removal` grava:

- `removal_results.csv`
- `summary.json`
- `summary.txt`
- `decision_matrix.csv`
- `decision_summary.txt`

`refine --removal --walkforward` grava:

- `walkforward_results.csv`
- `walkforward_summary.json`
- `walkforward_summary.txt`
- `decision_matrix.csv`
- `decision_summary.txt`

Todos os artefatos ficam em `artifacts/refine/...`. O refine nao sobrescreve `artifacts/models`.
