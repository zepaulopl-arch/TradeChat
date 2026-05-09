from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import reports_dir
from .presentation import (
    C,
    banner,
    divider,
    money_br,
    paint,
    render_facts,
    render_table,
    render_wrapped,
    screen_width,
    tone_delta,
    tone_signal,
)
from .utils import safe_ticker


def _money(value: float) -> str:
    return money_br(value)


def _clean_reasons(policy: dict[str, Any]) -> list[str]:
    return [reason for reason in policy.get("reasons", []) if "=" not in reason]


def _signal_horizon_result(signal: dict[str, Any]) -> dict[str, Any]:
    policy = signal.get("policy", {}) or {}
    horizon = str(policy.get("horizon", "d1")).lower()
    horizons = signal.get("horizons", {}) or {}
    return horizons.get(horizon, horizons.get("d1", {})) or {}


def _manifest_from_signal(signal: dict[str, Any]) -> dict[str, Any]:
    primary = signal.get("prediction", {}) or {}
    manifest = primary.get("train_manifest")
    if manifest:
        return manifest
    trigger_result = _signal_horizon_result(signal)
    return (trigger_result.get("train_manifest") or {}) if isinstance(trigger_result, dict) else {}


def _compact_items(
    items: list[str] | tuple[str, ...] | None, *, limit: int = 4, empty: str = "none"
) -> str:
    values = [str(item) for item in (items or []) if str(item).strip()]
    if not values:
        return empty
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f" (+{len(values) - limit})"


def _top_feature_summary(top_features: list[dict[str, Any]] | None, *, limit: int = 5) -> str:
    items = top_features or []
    if not items:
        return "top_5=n/a"
    chunks = []
    for item in items[:limit]:
        short_name = item.get("short") or item.get("name", "n/a")
        family = item.get("family", "n/a")
        relevance = float(item.get("relevance", 0.0) or 0.0)
        chunks.append(f"{short_name} [{family}, {relevance:.3f}]")
    return "top_5=" + "; ".join(chunks)


def _family_mix_summary(families: dict[str, Any] | None) -> str:
    profile = families or {}
    pairs = [f"{name}={count}" for name, count in profile.items() if count]
    return " | ".join(pairs) if pairs else "n/a"


def _engine_values_summary(values: dict[str, Any] | None) -> str:
    payload = values or {}
    if not payload:
        return "n/a"
    parts = []
    for name, value in payload.items():
        try:
            pct = float(value) * 100.0
            parts.append(f"{name}={pct:+.2f}%")
        except Exception:
            parts.append(f"{name}=n/a")
    return "; ".join(parts)


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def print_data_summary(status: dict[str, Any]) -> None:
    width = screen_width()
    ticker = status.get("ticker", "N/A")
    profile = status.get("asset_profile", {}) or {}
    fundamentals = status.get("fundamentals", {}) or {}
    sentiment = status.get("sentiment", {}) or {}
    sent_info = (
        "fresh" if sentiment.get("is_fresh") else f"cache_rows={sentiment.get('cache_rows', 0)}"
    )

    facts = [
        ("Status", status.get("status", "updated")),
        ("Rows", status.get("rows", 0)),
        ("Range", f"{status.get('start') or 'n/a'} -> {status.get('end') or 'n/a'}"),
        ("Period", status.get("period", "max")),
        ("Context", _compact_items(status.get("context_tickers", []) or [], limit=5)),
        (
            "Ctx Skipped",
            _compact_items(status.get("unavailable_context_tickers", []) or [], limit=5),
        ),
        (
            "Registry",
            (
                f"{profile.get('group', 'n/a')} / {profile.get('subgroup', 'n/a')} / "
                f"{profile.get('cnpj', 'n/a')}"
            ),
        ),
        (
            "Fundamentals",
            f"{fundamentals.get('status', 'n/a')} | {fundamentals.get('source', 'n/a')}",
        ),
        ("Sentiment", f"{sentiment.get('status', 'n/a')} | {sent_info}"),
    ]

    print()
    _print_lines(banner("TRADECHAT DATA", ticker, width=width))
    _print_lines(render_facts(facts, width=width, max_columns=2))
    print(divider(width))


def print_multi_horizon_train_summary(manifests: list[dict[str, Any]]) -> None:
    if not manifests:
        return

    width = screen_width()
    order = {"d1": 0, "d5": 1, "d20": 2}
    sorted_m = sorted(
        manifests, key=lambda item: order.get(str(item.get("horizon", "")).lower(), 99)
    )
    sample_manifest = sorted_m[0]
    ticker = sample_manifest.get("ticker", "N/A")
    base_engines = sample_manifest.get("base_engines", [])
    latest_manifest = sorted_m[-1]
    prep = latest_manifest.get("preparation", {}) or {}
    families = latest_manifest.get("feature_family_profile", {}) or {}

    rows: list[list[str]] = []
    for name in base_engines:
        row = [name]
        for manifest in sorted_m:
            mae = manifest.get("metrics", {}).get(name, {}).get("mae_return_raw", 0.0)
            row.append(f"{mae:.5f}")
        row.append(paint("OK", C.GREEN))
        rows.append(row)

    arb_row = [paint("ridge_arbiter", C.BOLD)]
    for manifest in sorted_m:
        mae = manifest.get("metrics", {}).get("ridge_arbiter", {}).get("mae_return", 0.0)
        arb_row.append(paint(f"{mae:.5f}", C.GREEN))
    arb_row.append(paint("FINAL", C.BOLD))
    rows.append(arb_row)

    facts = [
        ("Samples", prep.get("output_rows", 0)),
        ("Features", prep.get("selected_feature_count", 0)),
        ("Horizons", "D1 / D5 / D20"),
        ("Engines", len(base_engines)),
    ]

    print()
    _print_lines(
        banner("TRAIN REPORT", paint(ticker, C.BLUE), paint("MULTI-HORIZON", C.CYAN), width=width)
    )
    _print_lines(render_facts(facts, width=width, max_columns=2))
    _print_lines(
        render_table(
            ["ENGINE", "D1 MAE", "D5 MAE", "D20 MAE", "STATE"],
            rows,
            width=width,
            aligns=["left", "right", "right", "right", "left"],
            min_widths=[12, 8, 8, 9, 5],
        )
    )
    _print_lines(render_wrapped("Family Mix", _family_mix_summary(families), width=width))
    _print_lines(
        render_wrapped(
            "Top Feats", _top_feature_summary(latest_manifest.get("top_features", [])), width=width
        )
    )
    print(divider(width))


def _signal_facts(signal: dict[str, Any]) -> list[tuple[str, str, str] | tuple[str, str]]:
    price = float(signal["latest_price"])
    policy = signal["policy"]
    trigger_h = str(policy.get("horizon", "d1")).upper()
    label = str(policy.get("label", "NEUTRAL"))
    is_neutral = label in ["NEUTRAL", "LATERAL"]
    facts: list[tuple[str, str, str] | tuple[str, str]] = [
        ("Price", _money(price)),
        ("Signal", f"{label} ({policy.get('posture', 'n/a')})", tone_signal(label)),
        ("Quality", f"{policy.get('quality_pct', policy.get('confidence_pct', 0.0)):.0f}%"),
        ("Horizon", trigger_h),
    ]
    if is_neutral:
        facts.extend(
            [
                ("Target", "n/a"),
                ("Stop", "n/a"),
                ("Size", "0"),
                ("R/R", "n/a"),
            ]
        )
    else:
        facts.extend(
            [
                ("Target", _money(float(policy.get("target_price", 0.0) or 0.0)), C.GREEN),
                ("Stop", _money(float(policy.get("stop_loss_price", 0.0) or 0.0)), C.RED),
                ("Size", str(policy.get("position_size", 0))),
                ("R/R", f"{float(policy.get('risk_reward_ratio', 0.0) or 0.0):.2f}"),
                ("Partial", _money(float(policy.get("target_partial", 0.0) or 0.0)), C.CYAN),
                ("Breakeven", _money(float(policy.get("breakeven_trigger", 0.0) or 0.0))),
            ]
        )
    return facts


def _horizon_rows(signal: dict[str, Any]) -> list[list[str]]:
    price = float(signal["latest_price"])
    horizons = signal.get("horizons", {}) or {}
    rows: list[list[str]] = []
    for horizon in ["d1", "d5", "d20"]:
        h_data = horizons.get(horizon, {}) or {}
        if "error" in h_data:
            rows.append([horizon.upper(), "n/a", "-", "-"])
            continue
        ret = float(h_data.get("prediction_return", 0.0) or 0.0)
        quality = float(h_data.get("quality", h_data.get("confidence", 0.0)) or 0.0) * 100.0
        target_price = price * (1 + ret)
        rows.append(
            [
                horizon.upper(),
                paint(f"{ret * 100:+.2f}%", tone_delta(ret)),
                _money(target_price),
                f"{quality:.0f}%",
            ]
        )
    return rows


def _signal_context_facts(signal: dict[str, Any]) -> list[tuple[str, str, str] | tuple[str, str]]:
    fundamentals = signal.get("fundamentals", {}) or {}
    sent = float(signal.get("sentiment_value", 0.0) or 0.0)
    context_desc = (
        "aligned_with_ibov"
        if "ctx_BVSP_corr_20" in signal.get("features_used", [])
        else "neutral_context"
    )
    return [
        ("Sentiment", f"{sent:+.2f}", tone_delta(sent)),
        ("Context", context_desc),
        (
            "Fundamentals",
            (
                f"pl={float(fundamentals.get('pl', 0.0) or 0.0):.1f} | "
                f"roe={float(fundamentals.get('roe', 0.0) or 0.0) * 100:.1f}%"
            ),
        ),
    ]


def _render_signal_meta(
    signal: dict[str, Any], *, width: int, use_color: bool = True, verbose: bool = False
) -> list[str]:
    trigger_result = _signal_horizon_result(signal)
    lines: list[str] = []
    lines.extend(
        render_wrapped(
            "Used Engines",
            _compact_items(trigger_result.get("used_engines", []) or [], limit=8),
            width=width,
            use_color=use_color,
        )
    )
    lines.extend(
        render_wrapped(
            "Neutralized",
            _compact_items(trigger_result.get("discarded_engines", []) or [], limit=8),
            width=width,
            use_color=use_color,
        )
    )
    lines.extend(
        render_wrapped(
            "Reasons",
            "; ".join(_clean_reasons(signal.get("policy", {}) or {})) or "none",
            width=width,
            use_color=use_color,
        )
    )
    if verbose:
        lines.extend(
            render_wrapped(
                "Run Id", signal.get("train_run_id", "n/a"), width=width, use_color=use_color
            )
        )
    return lines


def print_signal(signal: dict[str, Any], *, verbose: bool = False) -> None:
    width = screen_width()
    ticker = signal["ticker"]
    print()
    _print_lines(
        banner("SIGNAL REPORT", paint(ticker, C.BLUE), signal.get("latest_date"), width=width)
    )
    _print_lines(render_facts(_signal_facts(signal), width=width, max_columns=2))
    _print_lines(
        render_table(
            ["H", "RETURN", "TARGET", "QUAL"],
            _horizon_rows(signal),
            width=width,
            aligns=["left", "right", "right", "right"],
            min_widths=[4, 8, 12, 6],
        )
    )
    _print_lines(render_facts(_signal_context_facts(signal), width=width, max_columns=2))
    _print_lines(_render_signal_meta(signal, width=width, verbose=verbose))
    print(divider(width))


def print_signal_brief(signal: dict[str, Any]) -> None:
    width = screen_width()
    ticker = signal["ticker"]
    brief_facts = _signal_facts(signal)[:8]
    print()
    _print_lines(
        banner("SIGNAL REPORT", paint(ticker, C.BLUE), signal.get("latest_date"), width=width)
    )
    _print_lines(render_facts(brief_facts, width=width, max_columns=2))
    print(divider(width))


def render_txt_report(signal: dict[str, Any]) -> str:
    width = 88
    ticker = signal["ticker"]
    price = float(signal["latest_price"])
    policy = signal.get("policy", {}) or {}
    manifest = _manifest_from_signal(signal)
    trigger_result = _signal_horizon_result(signal)
    top_features = manifest.get("top_features", []) or []
    family_profile = manifest.get("feature_family_profile", {}) or {}
    tune_summary = manifest.get("tune_summary", {}) or {}
    quality_pct = float(policy.get("quality_pct", policy.get("confidence_pct", 0.0)) or 0.0)

    lines: list[str] = []
    lines.extend(
        banner(
            "TRADECHAT AUDIT REPORT",
            ticker,
            signal.get("latest_date"),
            width=width,
            use_color=False,
        )
    )
    lines.extend(
        render_facts(
            [
                ("Last Price", _money(price)),
                ("Signal", f"{policy.get('label', 'N/A')} ({policy.get('posture', 'n/a')})"),
                ("Trigger", str(policy.get("horizon", "d1")).upper()),
                ("Signal Quality", f"{quality_pct:.0f}%"),
                ("Size", policy.get("position_size", 0)),
                ("R/R", f"{float(policy.get('risk_reward_ratio', 0.0) or 0.0):.2f}"),
                ("Target T1", _money(float(policy.get("target_partial", 0.0) or 0.0))),
                ("Target T2", _money(float(policy.get("target_price", 0.0) or 0.0))),
                ("Stop Loss", _money(float(policy.get("stop_loss_price", 0.0) or 0.0))),
            ],
            width=width,
            max_columns=2,
            use_color=False,
        )
    )
    lines.append(divider(width, use_color=False))
    lines.append("BASE ENGINES AND ARBITER")
    lines.extend(
        render_facts(
            [
                ("Architecture", manifest.get("architecture", "n/a")),
                ("Base Engines", _compact_items(manifest.get("base_engines", []) or [], limit=8)),
                (
                    "Used Engines",
                    _compact_items(trigger_result.get("used_engines", []) or [], limit=8),
                ),
                (
                    "Neutralized",
                    _compact_items(trigger_result.get("discarded_engines", []) or [], limit=8),
                ),
            ],
            width=width,
            max_columns=2,
            use_color=False,
        )
    )
    lines.extend(
        render_wrapped(
            "Raw Outputs",
            _engine_values_summary(trigger_result.get("raw_by_engine", {}) or {}),
            width=width,
            use_color=False,
        )
    )
    lines.extend(
        render_wrapped(
            "Guarded Outputs",
            _engine_values_summary(trigger_result.get("by_engine", {}) or {}),
            width=width,
            use_color=False,
        )
    )
    lines.extend(
        render_table(
            ["H", "RETURN", "TARGET", "QUAL"],
            [[cell for cell in row] for row in _horizon_rows(signal)],
            width=width,
            aligns=["left", "right", "right", "right"],
            min_widths=[4, 8, 12, 6],
            use_color=False,
        )
    )
    lines.append(divider(width, use_color=False))
    lines.append("FEATURE AUDIT")
    lines.extend(
        render_facts(
            [
                ("Selected", len(manifest.get("features", []) or [])),
                ("Family Mix", _family_mix_summary(family_profile)),
            ],
            width=width,
            max_columns=2,
            use_color=False,
        )
    )
    lines.extend(
        render_wrapped(
            "Top Feats", _top_feature_summary(top_features), width=width, use_color=False
        )
    )
    lines.append(divider(width, use_color=False))
    lines.append("AUTOTUNE SUMMARY")
    lines.extend(
        render_facts(
            [
                ("Enabled", _fmt_bool(manifest.get("autotune", False))),
            ],
            width=width,
            max_columns=1,
            use_color=False,
        )
    )
    lines.extend(
        render_wrapped(
            "Details", tune_summary if tune_summary else "not used", width=width, use_color=False
        )
    )
    lines.append(divider(width, use_color=False))
    lines.append("REASONS")
    reasons = _clean_reasons(policy)
    if reasons:
        for reason in reasons:
            lines.extend(render_wrapped("Reason", reason, width=width, use_color=False))
    else:
        lines.append("Reason: none")
    lines.append(divider(width, use_color=False))
    lines.append(f"RUN ID: {signal.get('train_run_id')}")
    return "\n".join(lines) + "\n"


def write_txt_report(cfg: dict[str, Any], signal: dict[str, Any]) -> Path:
    path = reports_dir(cfg) / f"{safe_ticker(signal['ticker'])}_audit.txt"
    path.write_text(render_txt_report(signal), encoding="utf-8")
    return path


def _fmt_bool(value: Any) -> str:
    return "ok" if bool(value) else "n/a"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "n/a"


def _fmt_context_missing_detail(item: dict[str, Any]) -> list[str]:
    return [
        str(item.get("ticker", "n/a")),
        f"{_fmt_count(item.get('valid_count'))}",
        f"{_fmt_count(item.get('missing_count'))} ({_fmt_pct(item.get('missing_pct', 0.0))})",
        "YES" if item.get("all_missing") else "NO",
    ]


def _fmt_count(value: Any) -> str:
    try:
        return str(int(value))
    except Exception:
        return "n/a"


def _audit_status_label(status: str) -> str:
    status = str(status or "ok").lower()
    if status == "error":
        return paint("FAIL", C.RED)
    if status == "warning":
        return paint("WARN", C.YELLOW)
    return paint("PASS", C.GREEN)


def print_data_audit(status: dict[str, Any]) -> None:
    """Render a real data-quality audit for cached market data."""
    width = screen_width()
    ticker = status.get("ticker", "N/A")
    audit = status.get("audit", {}) or {}
    fundamentals = status.get("fundamentals", {}) or {}
    sentiment = status.get("sentiment", {}) or {}

    age = audit.get("age_days")
    age_text = "n/a" if age is None else f"{age} days"
    last_valid = audit.get("effective_last_date") or audit.get("last_date") or status.get("end") or "n/a"
    raw_start = audit.get("raw_first_date") or audit.get("first_date") or status.get("start") or "n/a"
    raw_end = audit.get("raw_last_date") or status.get("end") or "n/a"
    effective_start = audit.get("effective_first_date") or "n/a"
    effective_end = audit.get("effective_last_date") or "n/a"
    gap_days = audit.get("effective_largest_gap_days", audit.get("largest_gap_days", 0))
    gap_start = audit.get("effective_largest_gap_start") or audit.get("largest_gap_start")
    gap_end = audit.get("effective_largest_gap_end") or audit.get("largest_gap_end")
    gap_span = f" ({gap_start} -> {gap_end})" if gap_start and gap_end else ""

    facts = [
        ("Status", _audit_status_label(audit.get("status", "ok"))),
        ("Raw Rows", audit.get("raw_rows", audit.get("rows", status.get("rows", 0)))),
        ("Raw Range", f"{raw_start} -> {raw_end}"),
        ("Effective Rows", audit.get("effective_rows", 0)),
        ("Asset Range", f"{effective_start} -> {effective_end}"),
        ("Age", age_text),
        ("Context", f"{audit.get('present_context_count', 0)}/{audit.get('requested_context_count', 0)}"),
        ("Coverage", _fmt_pct(audit.get("context_coverage_pct", 0.0))),
        ("Fundamentals", f"{fundamentals.get('status', 'n/a')} | {fundamentals.get('source', 'n/a')}"),
        (
            "Sentiment",
            f"{sentiment.get('status', 'n/a')} | rows={sentiment.get('cache_rows', 0)}",
        ),
    ]

    rows = [
        [
            "Minimum effective rows",
            f"{audit.get('effective_rows', 0)} / {audit.get('min_rows', 0)}",
            "PASS" if audit.get("has_min_rows", True) else "WARN",
        ],
        [
            "Price column",
            audit.get("price_column") or ("present" if audit.get("price_column_present") else "missing"),
            "PASS" if audit.get("price_column_present") else "FAIL",
        ],
        [
            "Valid price rows",
            _fmt_count(audit.get("price_valid_count")),
            "PASS" if audit.get("price_valid_count") else "FAIL",
        ],
        [
            "Pre-asset padding",
            _fmt_count(audit.get("pre_asset_padding_count")),
            "INFO" if audit.get("pre_asset_padding_count") else "PASS",
        ],
        [
            "Internal missing close",
            f"{_fmt_count(audit.get('internal_missing_close_count'))} ({_fmt_pct(audit.get('internal_missing_close_pct', 0.0))})",
            "PASS" if not audit.get("internal_missing_close_count") else "WARN",
        ],
        [
            "Post-asset missing",
            _fmt_count(audit.get("post_asset_missing_count")),
            "PASS" if not audit.get("post_asset_missing_count") else "WARN",
        ],
        [
            "Rows with missing inside",
            f"{_fmt_count(audit.get('effective_rows_with_any_missing'))} ({_fmt_pct(audit.get('effective_rows_with_any_missing_pct', 0.0))})",
            "PASS" if not audit.get("effective_rows_with_any_missing") else "WARN",
        ],
        [
            "Context missing inside",
            f"{_fmt_count(audit.get('context_missing_inside_count'))} ({_fmt_pct(audit.get('context_missing_inside_pct', 0.0))})",
            "PASS" if not audit.get("context_missing_inside_count") else "WARN",
        ],
        [
            "Context complete rows",
            f"{_fmt_count(audit.get('context_complete_rows_count'))} ({_fmt_pct(audit.get('context_complete_rows_pct', 0.0))})",
            "PASS" if audit.get("context_complete_rows_count") else "WARN",
        ],
        [
            "Duplicate dates",
            _fmt_count(audit.get("duplicate_date_count")),
            "PASS" if not audit.get("duplicate_date_count") else "WARN",
        ],
        [
            "Largest effective gap",
            f"{gap_days} days{gap_span}",
            "PASS" if int(gap_days or 0) <= 7 else "WARN",
        ],
        [
            "Freshness",
            f"last={last_valid} | age={age_text}",
            "PASS" if not audit.get("is_stale") else "WARN",
        ],
        [
            "Context coverage",
            f"{audit.get('present_context_count', 0)}/{audit.get('requested_context_count', 0)} ({_fmt_pct(audit.get('context_coverage_pct', 0.0))})",
            "PASS" if not audit.get("missing_context_count") else "WARN",
        ],
    ]

    rendered_rows = []
    for check, value, state in rows:
        tone = C.GREEN if state == "PASS" else C.RED if state == "FAIL" else C.BLUE if state == "INFO" else C.YELLOW
        rendered_rows.append([check, value, paint(state, tone)])

    print()
    _print_lines(banner("TRADECHAT DATA AUDIT", ticker, width=width))
    _print_lines(render_facts(facts, width=width, max_columns=2))
    print()
    _print_lines(render_table(["Check", "Value", "Status"], rendered_rows, width=width))

    missing_context = audit.get("missing_context_tickers", []) or []
    if missing_context:
        _print_lines(render_wrapped("Missing context", _compact_items(missing_context, limit=8), width=width))

    context_missing_by_ticker = [
        item for item in (audit.get("context_missing_by_ticker", []) or [])
        if int(item.get("missing_count", 0) or 0) > 0
    ]
    if context_missing_by_ticker:
        print()
        _print_lines(
            render_table(
                ["Context", "Valid", "Missing", "All Missing"],
                [_fmt_context_missing_detail(item) for item in context_missing_by_ticker[:8]],
                width=width,
                aligns=["left", "right", "right", "left"],
            )
        )

    all_missing = audit.get("all_missing_columns", []) or []
    if all_missing:
        _print_lines(render_wrapped("All-missing columns", _compact_items(all_missing, limit=8), width=width))

    issues = audit.get("issues", []) or []
    if issues:
        print()
        issue_rows = [
            [str(item.get("severity", "n/a")).upper(), item.get("check", "n/a"), item.get("message", "")]
            for item in issues
        ]
        _print_lines(render_table(["Severity", "Check", "Message"], issue_rows, width=width))

    print(divider(width))
