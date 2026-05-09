# Consolidacao Metodologica

Este patch consolida o TradeChat sem adicionar novos modelos, motores externos ou features. O objetivo e tornar a plataforma quantitativa mais governavel.

## O Que Foi Separado

- A CLI publica foi reduzida para seis comandos raiz: `data`, `train`, `signal`, `validate`, `refine` e `portfolio`.
- A orquestracao fica diretamente em `app/commands/`; a camada antiga de handlers foi removida.
- A simulacao ganhou o pacote `app/simulation/` com contratos de adapter, custos, metricas e janelas walk-forward.
- Validacao e refine agora possuem matrizes decisorias estruturadas.
- A configuracao ganhou validacao explicita em `app/config_schema.py`.

## Por Que Nao Adicionar Novos Modelos

O sistema ja tem motores tabulares suficientes para a fase atual. O gargalo nao e mais variedade de modelo; e provar, com dados e artefatos, que cada bloco existente agrega valor.

## Replay vs Walk-Forward

- `replay`: sanity check operacional com modelos salvos.
- `walk-forward`: validacao metodologica com treino em artefatos sombra por data.

Replay ajuda a encontrar falhas de execucao. Walk-forward e o caminho para conclusao metodologica.

## Decisoes

`validate` produz:

- `promote`
- `observe`
- `reject`
- `inconclusive`

`refine --removal` produz:

- `keep_family`
- `remove_candidate`
- `observe`
- `inconclusive`

Nenhuma decisao altera modelos, features ou configuracao automaticamente.

## Limitacoes Restantes

- PyBroker continua como adapter de validacao, nao como nucleo.
- A divisao de `app/simulation/` ainda preserva a fachada antiga para compatibilidade.
- A avaliacao economica depende da qualidade dos dados locais e do tamanho da janela.
