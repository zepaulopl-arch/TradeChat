# TradeChat Phase 1 - Testing and Usage Tutorial

Este tutorial fecha a fase 1 com um fluxo simples: validar ambiente, rodar testes, atualizar dados, treinar, prever, auditar, rebalancear carteira virtual e validar com PyBroker.

## 1. Ambiente

Execute tudo a partir da raiz do projeto:

```powershell
cd C:\Users\zepau\ML\projetos\TradeChat
$PY = "C:\Users\zepau\anaconda3\envs\trading\python.exe"
```

Confirme a CLI publica:

```powershell
& $PY trade.py --help
```

O help deve mostrar apenas:

```text
data
train
signal
validate
refine
portfolio
```

## 2. Testes da fase 1

Teste completo:

```powershell
& $PY -m pytest -q
```

Compilacao rapida dos modulos principais:

```powershell
& $PY -m py_compile app\cli.py app\pipeline_service.py app\batch_service.py app\portfolio_service.py app\portfolio_monitor_service.py app\ranking_service.py app\rebalance_service.py app\simulator_service.py
```

Smokes seguros, sem rebalancear carteira:

```powershell
& $PY trade.py --help
& $PY trade.py signal rank
& $PY trade.py portfolio status
& $PY trade.py validate ABEV3.SA --start 2026-03-01 --end 2026-05-05
```

## 3. Pipeline de uso

Fluxo diario enxuto:

```powershell
& $PY trade.py data load PETR4.SA VALE3.SA ITUB4.SA
& $PY trade.py signal rank PETR4.SA VALE3.SA ITUB4.SA
& $PY trade.py portfolio status
```

Fluxo com treino:

```powershell
& $PY trade.py data load PETR4.SA VALE3.SA ITUB4.SA
& $PY trade.py train PETR4.SA VALE3.SA ITUB4.SA
& $PY trade.py signal rank PETR4.SA VALE3.SA ITUB4.SA
& $PY trade.py signal report PETR4.SA VALE3.SA ITUB4.SA
```

Treino em paralelo por ativo:

```powershell
& $PY trade.py train PETR4.SA VALE3.SA ITUB4.SA --workers 3
```

Validacao PyBroker:

```powershell
& $PY trade.py validate PETR4.SA VALE3.SA --start 2026-01-01 --end 2026-05-01
& $PY trade.py validate PETR4.SA VALE3.SA --mode walkforward --start 2026-01-01 --end 2026-05-01
```

## 4. Carteira virtual

Consulta sem alterar alocacao:

```powershell
& $PY trade.py portfolio status
```

Rebalanceamento por sinais atuais:

```powershell
& $PY trade.py portfolio rebalance
```

Monitoramento intraday com target/stop:

```powershell
& $PY trade.py portfolio live
```

Importante: `portfolio rebalance` e `portfolio live` alteram o estado unico da carteira virtual em `data/tradechat_state.db`.

## 5. O que cada comando grava

| Comando | Grava artefatos? | Observacao |
| --- | --- | --- |
| `data` | Sim | Atualiza cache em `data/cache`. |
| `train` | Sim | Grava modelos e manifests em `artifacts/models`. |
| `signal generate` | Sim | Atualiza `latest_signal.json` por ativo. |
| `signal rank` | Nao, se usado sem tickers | Le sinais existentes e apresenta ranking. |
| `signal report` | Sim | Grava TXT em `artifacts/reports`. |
| `portfolio` | Nao | Apenas le estado. |
| `portfolio rebalance` | Sim | Recalcula posicoes da carteira virtual. |
| `portfolio live` | Sim, se target/stop for acionado | Pode fechar posicoes automaticamente. |
| `validate` | Sim | Grava artefatos de validacao em `artifacts/simulations`. |

## 6. Diagnostico tecnico

O diagnostico em lote continua como ferramenta de manutencao, nao como comando principal da rotina:

```powershell
& $PY scripts\diagnose_assets.py --assets PETR4.SA,VALE3.SA,ITUB4.SA --workers 3
```

Use quando quiser auditar falhas de dados, treino e predicao em muitos ativos.

Rodada pesada para todos os ativos atuais:

```powershell
& $PY scripts\diagnose_assets.py --assets ALL --autotune --workers 3
```

## 7. Criterios para considerar a fase 1 saudavel

- `pytest -q` passa sem falhas.
- `trade.py --help` mostra apenas seis comandos publicos.
- `signal rank` abre sem recalcular tudo quando usado sem tickers.
- `portfolio status` consulta estado sem alterar carteira.
- `portfolio rebalance` e `portfolio live` sao usados conscientemente, porque alteram estado.
- `validate --mode replay` roda rapido para sanidade.
- `validate --mode walkforward` e usado para validacao historica mais honesta.

## 8. Regra pratica de decisao

Nao trate uma previsao isolada como verdade. Use a ordem: dados atualizados, modelo treinado, sinal gerado, ranking, relatorio, validacao PyBroker e so entao carteira virtual. Se uma etapa parecer estranha, pare no relatorio e valide antes de rebalancear.
