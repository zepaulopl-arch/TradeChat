# TradeChat Tactical Operational Manual

Este manual descreve o ecossistema de scripts (BATs) do TradeChat e o protocolo recomendado para operação e simulação diária.

## 1. O Ecossistema de Comandos (BATs)

Cada arquivo `.bat` é um atalho para uma função específica do sistema:

| Comando | Função | Quando Usar |
| :--- | :--- | :--- |
| `run_diagnostics.bat` | **O Operário**: Atualiza preços e gera sinais de IA (BUY/SELL). | Diário (Antes do pregão). |
| `run_ranking.bat` | **O Analista**: Mostra o ranking das melhores oportunidades. | Diário (Após o Diagnostics). |
| `run_rebalance.bat` | **O Gerente**: Executa compras/vendas e ajusta o portfólio. | Diário (Após o Ranking). |
| `run_portfolio.bat` | **O Vigia**: Monitora P/L ao vivo e executa saídas (Target/Stop). | Durante o pregão (Monitoramento). |
| `run_tactical_cycle.bat` | **O Piloto Automático**: Roda os 3 primeiros em sequência. | Diário (Para economizar tempo). |
| `run_weekly_training.bat` | **O Mecânico**: Re-treina a IA do zero com Autotune. | Semanal (Fim de semana). |

---

## 2. Protocolo de Operação Diária (Simulação)

Para manter o sistema atualizado e simular o crescimento do seu capital de R$ 10.000,00, siga este fluxo:

1.  **Sincronização (Manhã)**:
    - Execute `run_diagnostics.bat`. Isso garante que as predições considerem os preços de fechamento mais recentes.
2.  **Tomada de Decisão (Execução)**:
    - Execute `run_rebalance.bat`. O sistema vai olhar o seu saldo, fechar o que ficou "Neutro", inverter o que mudou de sinal e abrir novas posições.
3.  **Controle de Voo (Durante o dia)**:
    - Execute `run_portfolio.bat`. Ele vai buscar os preços **ao vivo**. Se uma ação bater no seu Alvo ou no seu Stop, o sistema "venderá" virtualmente e atualizará seu saldo.

---

## 3. Inteligência de Rebalanceamento

O sistema é **Incremental**. Ele não esquece o que você fez ontem:
- **KEEP**: Se o sinal continua o mesmo, ele mantém a posição.
- **REVERSE**: Se o sinal inverteu (era Compra e virou Venda), ele vira a mão automaticamente.
- **ADJUST**: Se o sinal é o mesmo mas a confiança mudou, ele ajusta a quantidade de ações.
- **CLOSE**: Se o sinal sumiu (ficou Neutro), ele encerra a posição para proteger o capital.

## 4. Manutenção de IA

O mercado muda seus padrões. Por isso, use o `run_weekly_training.bat` uma vez por semana. Ele demora mais (pode levar horas), mas garante que o "cérebro" do TradeChat esteja calibrado com as volatilidades mais recentes.

---
*Status: Cockpit Operacional e Documentado.*
