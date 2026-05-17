from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain(value: Any) -> str:
    return ANSI_RE.sub("", str(value))


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"

    if isinstance(value, dict):
        return ", ".join(f"{key}={_short_value(item)}" for key, item in value.items())

    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    return str(value)


def _print_mapping_compact(
    title: str,
    data: dict[str, Any],
    keys: tuple[str, ...] | None = None,
) -> None:
    if not data:
        return

    print(title)

    selected_keys = keys if keys is not None else tuple(data.keys())

    for key in selected_keys:
        if key not in data:
            continue

        print(f"  {key}: {_short_value(data[key])}")


def print_smart_signal_header(
    ticker: str,
    signal: dict[str, Any],
    out_path: Path,
) -> None:
    smart = signal.get("smart_signal", {}) or {}
    evidence = smart.get("evidence", {}) or {}
    live_overrides = smart.get("live_overrides", {}) or {}
    overrides = smart.get("overrides", {}) or {}
    guard_result = signal.get("matrix_decision_guard", {}) or {}

    print()
    print(f"SMART SIGNAL | {ticker}")
    print("-" * 96)
    print(
        "Profile: "
        f"{smart.get('profile')} | "
        "Source: "
        f"{smart.get('source')} | "
        "Status: "
        f"{smart.get('promotion_status', 'n/a')}"
    )

    _print_mapping_compact(
        "Matrix:",
        evidence,
        keys=(
            "decision",
            "score",
            "profit_factor",
            "trades",
            "drawdown_pct",
            "return_pct",
            "hit_pct",
            "exposure_pct",
        ),
    )

    rejection_reasons = smart.get("rejection_reasons", []) or []
    if rejection_reasons:
        print("Promotion:")
        print(f"  rejected: {_short_value(rejection_reasons)}")

    _print_mapping_compact(
        "Live YAML overrides:",
        live_overrides,
    )

    _print_mapping_compact(
        "Effective overrides:",
        overrides,
    )

    if guard_result:
        _print_mapping_compact(
            "Matrix guard:",
            guard_result,
            keys=(
                "matrix_decision",
                "max_signal",
                "blocked",
                "blocked_signal",
                "reason",
            ),
        )

    print(f"Artifact: {out_path}")
    print("-" * 96)
    print()


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _fmt_optional_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"

    text = _plain(value).strip()

    if not text or text.lower() in {"n/a", "na", "none", "nan"}:
        return "n/a"

    if text.lower() in {"inf", "+inf", "infinity"}:
        return "inf"

    return _fmt_num(value, digits=digits)


def _fmt_inf(value: Any) -> str:
    text = _plain(value).lower()

    if text in {"inf", "infinity"}:
        return "inf"

    return _fmt_num(value)


def _clip(value: Any, size: int) -> str:
    text = _plain(value)

    if len(text) <= size:
        return text

    return text[: max(0, size - 3)] + "..."


def print_smart_rank(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    fingerprint: str | None = None,
    warnings: list[str] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    width = 118
    display_limit = limit if limit > 0 else len(rows)

    print()
    print("SMART RANK")

    if fingerprint:
        print(_plain(fingerprint))

    for warning in warnings or []:
        print(f"WARNING: {_plain(warning)}")

    print("-" * width)
    print(
        f"{'#':>3} "
        f"{'TICKER':<11} "
        f"{'ACTION':<11} "
        f"{'SIGNAL':<10} "
        f"{'MATRIX':<11} "
        f"{'PF':>6} "
        f"{'TRD':>5} "
        f"{'RR':>6} "
        f"{'GUARD':<6} "
        f"REASON"
    )
    print("-" * width)

    for index, row in enumerate(rows[:display_limit], start=1):
        print(
            f"{index:>3} "
            f"{_clip(row['ticker'], 11):<11} "
            f"{_clip(row.get('action', 'n/a'), 11):<11} "
            f"{_clip(row['signal'], 10):<10} "
            f"{_clip(row['matrix'], 11):<11} "
            f"{_fmt_inf(row['pf']):>6} "
            f"{_plain(row['trades']):>5} "
            f"{_fmt_optional_num(row.get('rr')):>6} "
            f"{_clip(row['guard'], 6):<6} "
            f"{_clip(row.get('reason', row.get('blocker', 'none')), 44)}"
        )

    print("-" * width)

    total = len(rows)
    shown = min(total, display_limit)

    action_counts = (summary or {}).get("action_counts") or {}

    if not action_counts:
        action_counts = {
            "ACTIONABLE": sum(1 for row in rows if str(row.get("action", "")).upper() == "ACTIONABLE"),
            "WATCH": sum(1 for row in rows if str(row.get("action", "")).upper() == "WATCH"),
            "BLOCKED": sum(1 for row in rows if str(row.get("action", "")).upper() == "BLOCKED"),
            "REJECTED": sum(1 for row in rows if str(row.get("action", "")).upper() == "REJECTED"),
            "INELIGIBLE": sum(1 for row in rows if str(row.get("action", "")).upper() == "INELIGIBLE"),
            "ERROR": sum(1 for row in rows if str(row.get("action", "")).upper() == "ERROR"),
        }

    print(f"Processed: {total}")
    print(f"Displayed: {shown} of {total}")
    print(f"Rows: {shown} of {total}")
    print(
        f"ACTIONABLE={action_counts.get('ACTIONABLE', 0)} | "
        f"WATCH={action_counts.get('WATCH', 0)} | "
        f"BLOCKED={action_counts.get('BLOCKED', 0)} | "
        f"REJECTED={action_counts.get('REJECTED', 0)} | "
        f"INELIGIBLE={action_counts.get('INELIGIBLE', 0)} | "
        f"ERROR={action_counts.get('ERROR', 0)}"
    )

    if summary:
        top_actionable = summary.get("top_actionable") or []
        if top_actionable:
            print(f"Top actionable: {', '.join(_plain(item) for item in top_actionable)}")
        else:
            print("No actionable assets found.")
            reasons = summary.get("no_actionable_reasons") or []

            if reasons:
                print("Reasons:")

                for reason in reasons:
                    print(f"- {_plain(reason)}")

        print(f"Main blocker: {_plain(summary.get('main_blocker', 'none'))}")

    print()
