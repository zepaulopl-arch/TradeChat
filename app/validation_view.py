from __future__ import annotations

from typing import Any

from .ui import model5 as ui5


def _metric_display(metrics: dict, key: str, default: str = "n/a") -> str:
    value = metrics.get(key)
    if value is None:
        return default
    return str(value)


def _fmt_pct_or_na(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    prefix = "+" if signed else ""
    return f"{number:{prefix}.1f}%" if signed else f"{number:.1f}%"


def _fmt_cash(value: object) -> str:
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pf(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _trade_attribution_rows(rows: list[dict[str, Any]], *, max_rows: int = 12) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for row in (rows or [])[:max_rows]:
        avg_return = row.get("avg_return_pct")
        avg_return_display = "n/a" if avg_return is None else f"{float(avg_return):+.2f}%"
        table_rows.append(
            [
                str(row.get("group", "n/a")),
                str(int(float(row.get("trade_count", 0) or 0))),
                f"{float(row.get('hit_rate_pct', 0.0) or 0.0):.1f}%",
                _fmt_pf(row.get("profit_factor_display")),
                _fmt_cash(row.get("net_pnl", 0.0)),
                _fmt_cash(row.get("gross_profit", 0.0)),
                _fmt_cash(-float(row.get("gross_loss", 0.0) or 0.0)),
                avg_return_display,
                _fmt_cash(row.get("cost", 0.0)),
            ]
        )
    return table_rows


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
    validation_decision = summary.get("validation_decision", {}) or {}
    width = ui5.screen_width()
    total_return = float(metrics.get("total_return_pct", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown_pct", 0.0) or 0.0)
    trades = int(float(metrics.get("trade_count", 0) or 0))
    hit_rate = float(metrics.get("hit_rate_pct", metrics.get("win_rate", 0.0)) or 0.0)
    avg_trade_return = float(
        metrics.get("avg_trade_return_pct", metrics.get("avg_return_pct", 0.0)) or 0.0
    )
    profit_factor_display = str(
        metrics.get("profit_factor_display")
        or f"{float(metrics.get('profit_factor', 0.0) or 0.0):.2f}"
    )
    turnover = float(metrics.get("turnover_pct", 0.0) or 0.0)
    exposure_available = bool(metrics.get("active_exposure_available", True))
    exposure = float(
        metrics.get("active_exposure_pct", metrics.get("avg_exposure_pct", 0.0)) or 0.0
    )
    exposure_display = f"{exposure:.1f}%" if exposure_available else "n/a"
    total_cost = float(metrics.get("total_cost", 0.0) or 0.0)
    gross_profit = float(metrics.get("gross_profit", 0.0) or 0.0)
    gross_loss = float(metrics.get("gross_loss", 0.0) or 0.0)
    net_profit = float(metrics.get("net_profit", metrics.get("total_pnl", 0.0)) or 0.0)
    before_costs = float(metrics.get("return_before_costs_pct", total_return) or 0.0)
    after_costs = float(metrics.get("return_after_costs_pct", total_return) or 0.0)
    decision = "Sem trades na janela"
    decision_status = "warn"
    if trades > 0 and total_return >= 0:
        decision = "Manter em observacao"
        decision_status = "ok"
    elif trades > 0:
        decision = "Revisar filtros"
        decision_status = "error"
    mode_label = (
        "walk-forward shadow"
        if summary.get("mode") == "pybroker_walkforward_shadow"
        else "replay operacional"
    )
    conclusion = (
        "Sem entradas no periodo; aumentar janela ou reduzir filtros para investigar."
        if trades == 0
        else f"Retorno {total_return:+.2f}% com {trades} trades."
    )
    sample_text = (
        f"{summary.get('start_date')} ate {summary.get('end_date')} | "
        f"{len(summary.get('tickers', []) or [])} ativos"
    )

    lines: list[str] = [""]
    lines.extend(ui5.render_header(f"{screen_title} - PYBROKER - {mode.upper()}", width=width))
    lines.extend(ui5.render_section("RESUMO", width=width))
    lines.extend(
        ui5.render_key_values(
            {
                "Experimento": "Simulacao PyBroker",
                "Amostra": sample_text,
                "Modo": mode_label,
                "Policy": str(summary.get("policy_profile", "active")),
                "Conclusao preliminar": conclusion,
            },
            width=width,
        )
    )
    callout_status = "warn" if mode == "replay" else "info"
    if mode == "replay":
        callout_text = "Replay usa modelos salvos; serve para sanidade operacional rapida."
    else:
        callout_text = (
            "Walk-forward treina em artefatos sombra por rebalanceamento; "
            "e mais lento, mas reduz vazamento temporal."
        )
    lines.extend(ui5.render_callout(callout_text, status=callout_status, width=width))

    if validation_decision:
        final_decision = str(validation_decision.get("final_decision", "inconclusive")).upper()
        decision_status = {
            "PROMOTE": "ok",
            "OBSERVE": "warn",
            "REJECT": "error",
            "INCONCLUSIVE": "info",
        }.get(final_decision, "info")
        checks = validation_decision.get("checks", {}) or {}
        check_rows = []
        for name in [
            "beats_required_baselines",
            "enough_trades",
            "profit_factor_ok",
            "max_drawdown_ok",
            "exposure_ok",
        ]:
            payload = checks.get(name, {}) or {}
            check_rows.append([name, "pass" if payload.get("passed") else "fail"])
        lines.extend(ui5.render_section("VALIDATION DECISION", width=width))
        lines.extend(
            ui5.render_key_values(
                {
                    "Decision": ui5.render_badge(final_decision, decision_status),
                    "Score": f"{float(validation_decision.get('score', 0.0) or 0.0):.1f}",
                },
                width=width,
            )
        )
        lines.extend(
            ui5.render_table(
                ["Check", "Status"],
                check_rows,
                width=width,
                aligns=["left", "left"],
                min_widths=[28, 8],
            )
        )
        for explanation in validation_decision.get("explanation", [])[:3]:
            lines.extend(ui5.render_callout(str(explanation), status=decision_status, width=width))

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
                    profit_factor_display,
                    f"{turnover:.1f}%",
                    exposure_display,
                    f"{total_cost:+.2f}",
                ]
            ],
            width=width,
            aligns=["right", "right", "right", "right", "right", "right"],
            min_widths=[7, 9, 8, 9, 9, 8],
        )
    )
    if trades > 0:
        lines.extend(ui5.render_section("P&L AUDIT", width=width))
        lines.extend(
            ui5.render_table(
                ["Gross +", "Gross -", "Net", "Before Cost", "After Cost"],
                [
                    [
                        f"{gross_profit:+.2f}",
                        f"{-gross_loss:+.2f}",
                        f"{net_profit:+.2f}",
                        f"{before_costs:+.2f}%",
                        f"{after_costs:+.2f}%",
                    ]
                ],
                width=width,
                aligns=["right", "right", "right", "right", "right"],
                min_widths=[9, 9, 9, 11, 10],
            )
        )
        attribution = metrics.get("trade_attribution", {}) or {}
        by_ticker = _trade_attribution_rows(attribution.get("by_ticker", []) or [])
        if by_ticker:
            lines.extend(ui5.render_section("TRADE ATTRIBUTION | BY TICKER", width=width))
            lines.extend(
                ui5.render_table(
                    [
                        "Ticker",
                        "Trades",
                        "Hit",
                        "PF",
                        "Net",
                        "Gross +",
                        "Gross -",
                        "Avg Ret",
                        "Cost",
                    ],
                    by_ticker,
                    width=width,
                    aligns=[
                        "left",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                    ],
                    min_widths=[10, 6, 6, 6, 8, 8, 8, 8, 7],
                )
            )
        by_horizon = _trade_attribution_rows(attribution.get("by_horizon", []) or [])
        if by_horizon:
            lines.extend(ui5.render_section("TRADE ATTRIBUTION | BY HORIZON", width=width))
            lines.extend(
                ui5.render_table(
                    [
                        "Horizon",
                        "Trades",
                        "Hit",
                        "PF",
                        "Net",
                        "Gross +",
                        "Gross -",
                        "Avg Ret",
                        "Cost",
                    ],
                    by_horizon,
                    width=width,
                    aligns=[
                        "left",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                    ],
                    min_widths=[10, 6, 6, 6, 8, 8, 8, 8, 7],
                )
            )
        by_side = _trade_attribution_rows(attribution.get("by_side", []) or [])
        if by_side:
            lines.extend(ui5.render_section("TRADE ATTRIBUTION | BY SIDE", width=width))
            lines.extend(
                ui5.render_table(
                    ["Side", "Trades", "Hit", "PF", "Net", "Gross +", "Gross -", "Avg Ret", "Cost"],
                    by_side,
                    width=width,
                    aligns=[
                        "left",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                        "right",
                    ],
                    min_widths=[10, 6, 6, 6, 8, 8, 8, 8, 7],
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
                    str(
                        row.get("profit_factor_delta_display")
                        or f"{float(row.get('profit_factor_delta', 0.0) or 0.0):+.2f}"
                    ),
                    "sim" if bool(row.get("beat_return", False)) else "nao",
                ]
            )
        lines.extend(ui5.render_section("MODELO VS BASELINES", width=width))
        lines.extend(
            ui5.render_key_values(
                {
                    "Decisao": baseline_comparison.get("decision", "n/a"),
                    "Beat rate": (
                        f"{float(baseline_comparison.get('beat_rate_pct', 0.0) or 0.0):.1f}%"
                    ),
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
                    [
                        "Sinais",
                        str((summary.get("artifacts", {}) or {}).get("signals_json", "n/a")),
                    ],
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
