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
    width = 112
    display_limit = limit if limit > 0 else len(rows)

    print()
    print("SMART RANK")

    if fingerprint:
        print(_plain(fingerprint))

    for warning in warnings or []:
        print(f"WARNING: {_plain(warning)}")

    print("-" * width)
    print(
        f"{'TICKER':<11} "
        f"{'SIGNAL':<10} "
        f"{'PROFILE':<9} "
        f"{'MATRIX':<11} "
        f"{'PF':>6} "
        f"{'TRD':>5} "
        f"{'RR':>5} "
        f"{'GUARD':<6} "
        f"BLOCKER"
    )
    print("-" * width)

    for row in rows[:display_limit]:
        print(
            f"{_clip(row['ticker'], 11):<11} "
            f"{_clip(row['signal'], 10):<10} "
            f"{_clip(row['profile'], 9):<9} "
            f"{_clip(row['matrix'], 11):<11} "
            f"{_fmt_inf(row['pf']):>6} "
            f"{_plain(row['trades']):>5} "
            f"{_fmt_num(row['rr']):>5} "
            f"{_clip(row['guard'], 6):<6} "
            f"{_clip(row['blocker'], 45)}"
        )

    print("-" * width)

    total = len(rows)
    shown = min(total, display_limit)

    ok_count = sum(1 for row in rows if str(row.get("guard", "")).upper() == "OK")
    block_count = sum(1 for row in rows if str(row.get("guard", "")).upper() == "BLOCK")
    rejected_count = sum(1 for row in rows if str(row.get("signal", "")).upper() == "REJECTED")
    skip_count = sum(1 for row in rows if str(row.get("guard", "")).upper() == "SKIP")
    error_count = sum(1 for row in rows if str(row.get("guard", "")).upper() == "ERROR")

    print(
        f"Rows: {shown} of {total} | "
        f"OK={ok_count} | "
        f"BLOCK={block_count} | "
        f"REJECTED={rejected_count} | "
        f"SKIP={skip_count} | "
        f"ERROR={error_count}"
    )

    if summary:
        top_actionable = summary.get("top_actionable") or []
        if top_actionable:
            print(f"Top actionable: {', '.join(_plain(item) for item in top_actionable)}")
        else:
            print("Top actionable: none")

        print(f"Main blocker: {_plain(summary.get('main_blocker', 'none'))}")

    print()
