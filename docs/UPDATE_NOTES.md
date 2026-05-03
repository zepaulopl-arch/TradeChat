# TradeGem update corrigido

## Superfície final da CLI

```powershell
.\run data PETR4
.\run train PETR4
.\run predict PETR4
.\run report PETR4
.\run daily PETR4,VALE3,ITUB4
```

## Decisões corrigidas

- `data` é comando único. Não há `data update`, `data check` nem `--cvm`.
- `daily` não treina. Ele executa apenas `data -> predict -> report compacto`.
- A arquitetura original de motores foi restaurada:
  - motores base/especialistas: `XGB`, `RandomForest`, `MLP`;
  - árbitro/juiz de stacking: `Ridge`.
- `Ridge` não é mais tratado como motor comum nem participa de média simples.
- `train` salva os motores base, a ordem dos motores e o árbitro em `artifacts/<ticker>/<run_id>/model.pkl`.
- `predict` carrega o modelo salvo, calcula as previsões dos motores base e passa essas previsões ao árbitro Ridge.

## Fluxo operacional recomendado

Treino quando quiser atualizar modelo:

```powershell
.\run train PETR4 --update
```

Rotina diária normal:

```powershell
.\run daily PETR4
```

O `daily` pressupõe que já existe um modelo treinado para o ativo. Se não existir, ele mostra erro orientando a rodar `train` antes.


## Correction: restore original ML contract

This update restores the original TradeGem modeling contract instead of simplifying it away.

- Base specialists: XGB, RandomForest and MLP.
- Arbiter: Ridge stacking judge, not a normal base engine.
- Default train: fixed parameters from `config.yaml`, faster and deterministic.
- Optional autotune: `run train PETR4 --autotune` uses BayesSearchCV over the three base specialists, then trains the Ridge arbiter on their predictions.
- Daily remains operational only: `data -> predict -> compact report`; it never trains and never autotunes.
- Data remains simple: `run data PETR4`, no `--cvm`, no `data update`, no `data check`.

Recommended flow:

```powershell
.\run data PETR4
.\run train PETR4 --autotune
.\run daily PETR4
```

Fast retrain when needed:

```powershell
.\run train PETR4
```
