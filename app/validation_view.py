from __future__ import annotations

from typing import Any

from .ui import model5 as ui5


def render_validation_summary(
    summary: dict[str, Any],
    *,
    mode: str,
    screen_title: str = "VALIDATE",
    verbose: bool = False,
) -> list[str]:
    metrics = summary.get("metrics", {}) or {}
    baselines = summary.get("baselines", {}) or {}
    baseline_comparison = summary.get("baseline_comparison", {}) or {}
    width = ui5.screen_width()
    total_return = float(metrics.get("total_return_pct", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    trades = int(float(metrics.get("trade_count", 0) or 0))
    hit_rate = float(metrics.get("hit_rate_pct", metrics.get("win_rate", 0.0)) or 0.0)
    avg_trade_return = float(metrics.get("avg_trade_return_pct", metrics.get("avg_return_pct", 0.0)) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    turnover = float(metrics.get("turnover_pct", 0.0) or 0.0)
    exposure = float(metrics.get("active_exposure_pct", metrics.get("avg_exposure_pct", 0.0)) or 0.0)
    total_cost = float(metrics.get("total_cost", 0.0) or 0.0)
    decision = "Sem trades na janela"
    decision_status = "warn"
    if trades > 0 and total_return >= 0:
        decision = "Manter em observacao"
        decision_status = "ok"
    elif trades > 0:
        decision = "Revisar filtros"
        decision_status = "error"
    mode_label = "walk-forward shadow" if summary.get("mode") == "pybroker_walkforward_shadow" else "replay operacional"
    conclusion = (
        "Sem entradas no periodo; aumentar janela ou reduzir filtros para investigar."
        if trades == 0
        else f"Retorno {total_return:+.2f}% com {trades} trades."
    )

    lines: list[str] = [""]
    lines.extend(ui5.render_header(f"{screen_title} - PYBROKER - {mode.upper()}", width=width))
    lines.extend(ui5.render_section("RESUMO", width=width))
    lines.extend(
        ui5.render_key_values(
            {
                "Experimento": "Simulacao PyBroker",
                "Amostra": f"{summary.get('start_date')} ate {summary.get('end_date')} | {len(summary.get('tickers', []) or [])} ativos",
                "Modo": mode_label,
                "Conclusao preliminar": conclusion,
            },
            width=width,
        )
    )
    callout_status = "warn" if mode == "replay" else "info"
    callout_text = (
        "Replay usa modelos operacionais salvos; serve para sanidade de execucao e comparacao rapida."
        if mode == "replay"
        else "Walk-forward treina em artefatos sombra por data de rebalanceamento; e mais lento, mas reduz vazamento temporal."
    )
    lines.extend(ui5.render_callout(callout_text, status=callout_status, width=width))

    lines.extend(ui5.render_section("RESULTADO", width=width))
    lines.extend(
        ui5.render_table(
            ["Configuracao", "Retorno", "Trades", "Drawdown", "Decisao"],
            [
                [
                    mode_label,
                    f"{total_return:+.2f}%",
                    str(trades),
                    f"{max_drawdown:+.2f}%",
                    ui5.render_badge(decision, decision_status),
                ]
            ],
            width=width,
            aligns=["left", "right", "right", "right", "left"],
            min_widths=[18, 8, 6, 9, 14],
        )
    )
    lines.extend(ui5.render_section("ECONOMIA", width=width))
    lines.extend(
        ui5.render_table(
            ["Hit", "Avg Trade", "Profit F", "Turnover", "Exposure", "Cost"],
            [
                [
                    f"{hit_rate:.1f}%",
                    f"{avg_trade_return:+.2f}%",
                    f"{profit_factor:.2f}",
                    f"{turnover:.1f}%",
                    f"{exposure:.1f}%",
                    f"{total_cost:+.2f}",
                ]
            ],
            width=width,
            aligns=["right", "right", "right", "right", "right", "right"],
            min_widths=[7, 9, 8, 9, 9, 8],
        )
    )

    if baselines:
        rows = []
        for name, payload in baselines.items():
            base_metrics = payload.get("metrics", {}) or {}
            rows.append(
                [
                    name,
                    f"{float(base_metrics.get('total_return_pct', 0.0) or 0.0):+.2f}%",
                    str(int(float(base_metrics.get("trade_count", 0) or 0))),
                    f"{float(base_metrics.get('max_drawdown_pct', 0.0) or 0.0):+.2f}%",
                ]
            )
        lines.extend(ui5.render_section("BASELINES", width=width))
        lines.extend(
            ui5.render_table(
                ["Baseline", "Retorno", "Trades", "Drawdown"],
                rows,
                width=width,
                aligns=["left", "right", "right", "right"],
                min_widths=[24, 8, 6, 9],
            )
        )

    if baseline_comparison:
        comparison_rows = []
        for row in baseline_comparison.get("rows", []) or []:
            comparison_rows.append(
                [
                    str(row.get("baseline", "n/a")),
                    f"{float(row.get('return_delta_pct', 0.0) or 0.0):+.2f}%",
                    f"{float(row.get('drawdown_delta_pct', 0.0) or 0.0):+.2f}%",
                    f"{float(row.get('hit_rate_delta_pct', 0.0) or 0.0):+.1f}%",
                    f"{float(row.get('profit_factor_delta', 0.0) or 0.0):+.2f}",
                    "sim" if bool(row.get("beat_return", False)) else "nao",
                ]
            )
        lines.extend(ui5.render_section("MODELO VS BASELINES", width=width))
        lines.extend(
            ui5.render_key_values(
                {
                    "Decisao": baseline_comparison.get("decision", "n/a"),
                    "Beat rate": f"{float(baseline_comparison.get('beat_rate_pct', 0.0) or 0.0):.1f}%",
                },
                width=width,
            )
        )
        lines.extend(
            ui5.render_table(
                ["Baseline", "Delta Ret", "Delta DD", "Delta Hit", "Delta PF", "Bateu"],
                comparison_rows,
                width=width,
                aligns=["left", "right", "right", "right", "right", "left"],
                min_widths=[24, 9, 9, 9, 8, 6],
            )
        )

    if verbose:
        lines.extend(ui5.render_section("ARTEFATOS", width=width))
        lines.extend(
            ui5.render_table(
                ["Arquivo", "Caminho"],
                [
                    ["Resumo", str((summary.get("artifacts", {}) or {}).get("summary_txt", "n/a"))],
                    ["Sinais", str((summary.get("artifacts", {}) or {}).get("signals_json", "n/a"))],
                    ["Trades", str((summary.get("artifacts", {}) or {}).get("trades_csv", "n/a"))],
                    ["Stops", str((summary.get("artifacts", {}) or {}).get("stops_csv", "n/a"))],
                ],
                width=width,
                aligns=["left", "left"],
                min_widths=[8, 30],
            )
        )

    closing = [
        "Revisar o resumo da simulacao antes de promover qualquer ajuste operacional.",
        "Comparar replay e walk-forward na mesma janela para medir o efeito de vazamento temporal.",
        "Abrir os artefatos apenas quando precisar auditar trades, ordens ou sinais por data.",
    ]
    lines.extend(ui5.render_section("FECHAMENTO OPERACIONAL", width=width))
    lines.extend(ui5.render_operational_closing(closing, width=width))
    return lines
