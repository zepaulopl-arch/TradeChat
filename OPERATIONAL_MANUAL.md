# TradeChat Operational Manual

Este manual descreve o pipeline real do projeto, os comandos principais e quando usar cada um.

## 1. Comandos base da CLI

Todos os comandos abaixo rodam via `trade.py`.

| Comando | O que faz | Quando usar |
| --- | --- | --- |
| `python trade.py data PETR4.SA` | Atualiza cache de precos e valida contexto, fundamentals e sentimento. | Antes de treinar ou quando quiser sincronizar dados. |
| `python trade.py train PETR4.SA` | Treina os tres horizontes (`d1`, `d5`, `d20`) e grava manifests/modelos. | Quando ainda nao ha modelo recente ou apos mudanca relevante de mercado. |
| `python trade.py predict PETR4.SA` | Gera o sinal operacional usando os modelos salvos. | Depois de treinar ou para consulta tatico-operacional. |
| `python trade.py predict --rank` | Mostra ranking consolidado dos ultimos sinais. | Triagem rapida de oportunidades. |
| `python trade.py predict ALL --rank` | Atualiza sinais e mostra ranking consolidado. | Triagem completa apos dados/modelos prontos. |
| `python trade.py report PETR4.SA` | Gera o relatorio `.txt` de auditoria do ultimo sinal. | Quando quiser registro, revisao ou rastreabilidade. |
| `python trade.py portfolio` | Mostra o estado da carteira virtual. | Consulta de exposicao e risco. |
| `python trade.py portfolio --live` | Monitora carteira com preco intraday e saida por target/stop. | Durante o pregao, com cuidado porque pode alterar estado. |
| `python trade.py portfolio --rebalance` | Recalcula a carteira virtual com os sinais atuais. | Paper trading e alocacao tatica. |
| `python trade.py validate PETR4.SA VALE3.SA` | Roda validacao PyBroker em modo replay dos modelos salvos. | Sanidade operacional, comparacao de regras e execucao. |
| `python trade.py validate PETR4.SA VALE3.SA --mode walkforward` | Roda validacao PyBroker treinando modelos sombra por data. | Validacao historica mais rigorosa, com menos vazamento temporal. |
| `python trade.py refine PETR4.SA VALE3.SA` | Audita manifests treinados por horizonte, MAE e familias de features selecionadas. | Revisao metodologica leve antes de retreinar ou remover familias. |
| `python trade.py refine PETR4.SA --removal --horizons d1` | Treina remocoes controladas em artefatos sombra (`full`, `technical_only`, `no_context`, `no_fundamentals`, `no_sentiment`). | Provar se uma familia agrega antes de manter complexidade. |
| `python trade.py refine PETR4.SA VALE3.SA --removal --walkforward` | Roda as mesmas remocoes com validacao walk-forward e metricas economicas. | Decidir remocao de familias com base em retorno, drawdown, trades e baselines, nao apenas MAE. |

O `validate` tambem grava baselines economicos no resumo da simulacao: nao operar, buy and hold igualmente ponderado, media historica long/flat, ultimo retorno long/flat e aleatorio long/flat deterministico. Ele compara o modelo contra esses baselines por delta de retorno, drawdown, hit rate e profit factor; o modelo operacional precisa justificar que bate alternativas triviais antes de virar decisao.

## 2. Pipeline recomendado

### Diario, antes do mercado

1. Atualize dados dos ativos que voce acompanha.
   `python trade.py data PETR4.SA VALE3.SA ITUB4.SA`
2. Gere sinais usando os modelos ja existentes.
   `python trade.py predict PETR4.SA VALE3.SA ITUB4.SA`
3. Se quiser registrar auditoria do que foi decidido:
   `python trade.py report PETR4.SA VALE3.SA ITUB4.SA`

### Diario, quando nao quer retreinar

1. Atualize dados.
   `python trade.py data PETR4.SA VALE3.SA ITUB4.SA`
2. Gere sinais e ranking.
   `python trade.py predict PETR4.SA VALE3.SA ITUB4.SA --rank`
3. Consulte a carteira.
   `python trade.py portfolio`
4. Durante o pregao, se quiser monitorar target/stop:
   `python trade.py portfolio --live`

### Semanal ou apos mudanca de regime

1. Atualize os dados.
   `python trade.py data ALL`
2. Retreine os ativos mais importantes.
   `python trade.py train PETR4.SA VALE3.SA ITUB4.SA`
3. Se quiser acelerar em lote por ativo:
   `python trade.py train PETR4.SA VALE3.SA ITUB4.SA --workers 3`
4. Para uma rodada pesada em lote com autotune:
   `run_weekly_training.bat`

### Fluxo de diagnostico em lote

`run_diagnostics.bat`

Esse atalho roda `scripts/diagnose_assets.py`, que atualiza dados, treina, prediz e grava artefatos de diagnostico para os ativos selecionados. Ele fica como ferramenta tecnica de manutencao, nao como rotina diaria.

Para usar paralelismo seguro por ativo:

`python scripts/diagnose_assets.py --assets PETR4.SA,VALE3.SA,ITUB4.SA --workers 3`

Para rodar todos os ativos cadastrados:

`python scripts/diagnose_assets.py --assets ALL --autotune --workers 3`

Ativos com `registry_status: inactive` em `config/data.yaml` ficam fora de `ALL`; ativos com pouca historia aparecem como `SKIP` no diagnostico.

## 3. Atalhos opcionais

| Script | Papel |
| --- | --- |
| `run_ranking.bat` | Atalho para `python trade.py predict --rank`. |
| `run_rebalance.bat` | Atalho para `python trade.py portfolio --rebalance`. |
| `run_portfolio.bat` | Atalho para `python trade.py portfolio --live`. |

## 4. Ordem operacional sugerida

### Rotina enxuta

1. `python trade.py data PETR4.SA VALE3.SA ITUB4.SA`
2. `python trade.py predict PETR4.SA VALE3.SA ITUB4.SA --rank`
3. `python trade.py portfolio`

### Rotina completa

1. `python trade.py data PETR4.SA VALE3.SA ITUB4.SA`
2. `python trade.py train PETR4.SA VALE3.SA ITUB4.SA`
3. `python trade.py predict PETR4.SA VALE3.SA ITUB4.SA`
4. `python trade.py report PETR4.SA VALE3.SA ITUB4.SA`
5. `python trade.py predict PETR4.SA VALE3.SA ITUB4.SA --rank`
6. `python trade.py portfolio --rebalance`
7. `python trade.py portfolio --live`
8. `python trade.py validate PETR4.SA VALE3.SA --start 2026-01-01 --end 2026-05-01`
9. `python trade.py validate PETR4.SA VALE3.SA --mode walkforward --start 2026-01-01 --end 2026-05-01`
10. `python trade.py refine PETR4.SA VALE3.SA`
11. `python trade.py refine PETR4.SA VALE3.SA --removal --walkforward --start 2026-01-01 --end 2026-05-01`

## 5. Regras praticas

- `data` primeiro, quando houver duvida sobre atualizacao do cache.
- `train` so quando houver motivo; ele e a etapa mais pesada.
- Paralelismo vale por ativo, nao por horizonte interno. Use `--workers` com moderacao.
- `predict` e `report` devem reutilizar modelos ja treinados.
- `predict --rank` substitui o ranking separado na rotina principal.
- `portfolio --rebalance` substitui o rebalance separado na rotina principal.
- `portfolio --live` substitui o monitor separado de carteira.
- A carteira virtual usa SQLite como estado unico em `data/tradechat_state.db`.
- `run_weekly_training.bat` deve ser usado com calma, porque faz rodada completa com autotune.
- `validate --mode replay` usa modelos ja treinados; e rapido e bom para sanidade operacional.
- `validate --mode walkforward` treina modelos sombra dentro de `artifacts/simulations` e e a opcao mais correta para validacao historica.
- Use `validate --verbose` apenas quando quiser ver caminhos tecnicos dos artefatos.
- `refine` nao treina e nao altera estado; ele le os manifests mais recentes para expor MAE, qualidade operacional e peso das familias selecionadas.
- `refine --removal` treina em `artifacts/refine/...`, nao substitui os modelos operacionais em `artifacts/models`, e grava `summary.json`, `summary.txt` e `removal_results.csv`.
- `refine --removal --walkforward` tambem usa artefatos sombra em `artifacts/refine/...`, roda validacao historica por perfil e grava `walkforward_summary.json`, `walkforward_summary.txt` e `walkforward_results.csv`.

## 6. Tutorial da fase 1

O tutorial de testes e uso esta em `docs/PHASE1_TESTING_AND_USAGE.md`.

## 7. Instalacao por camadas

- `requirements-core.txt`: dados, YAML, cache e utilitarios do pipeline base.
- `requirements-ml.txt`: modelos tabulares, autotune e validacao PyBroker.
- `requirements-sentiment.txt`: RSS, VADER/NLTK e traducao.
- `requirements-dev.txt`: ferramentas de teste.
- `requirements.txt`: instalacao completa, agregando todas as camadas acima.
