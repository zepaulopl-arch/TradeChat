from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

SUMMARY_FIELDS = [
    "phase",
    "ticker",
    "scope",
    "profile",
    "policy",
    "decision",
    "score",
    "return_pct",
    "trades",
    "drawdown_pct",
    "hit_pct",
    "avg_trade_pct",
    "profit_factor",
    "profit_factor_display",
    "turnover_pct",
    "exposure_pct",
    "cost",
    "gross_plus",
    "gross_minus",
    "net",
    "before_cost_pct",
    "after_cost_pct",
    "beat_rate_pct",
    "sample",
    "n_assets",
    "eligibility_status",
    "ineligible_reason",
    "data_rows",
    "log_path",
]


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def _to_float(value: str | int | float | None) -> float:
    if value is None or value == "":
        return math.nan
    text = str(value).strip().replace("%", "").replace("+", "").replace(",", ".")
    if text.lower() in {"inf", "+inf", "infinite"}:
        return math.inf
    if text.lower() in {"n/a", "nan", "none"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _to_int(value: str | int | float | None) -> int:
    number = _to_float(value)
    if math.isnan(number) or math.isinf(number):
        return 0
    return int(number)


def _display_pf(value: object) -> str:
    number = _to_float(value)
    if math.isinf(number):
        return "inf"
    if math.isnan(number):
        return ""
    return f"{number:.2f}"


def _pf_ok(value: str | None, minimum: float) -> bool:
    pf = _to_float(value)
    return bool(math.isinf(pf) or (not math.isnan(pf) and pf >= minimum))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _safe_name_from_log(log_path: str) -> str:
    path = Path(log_path)
    stem = path.stem
    match = re.match(r"^\d+_(.+)$", stem)
    return match.group(1) if match else stem


def _ineligible_from_log(text: str) -> tuple[str, str]:
    lowered = text.lower()
    patterns = [
        (
            "insufficient history",
            (
                "insufficient rows",
                "insufficient train rows",
                "insufficient prepared train rows",
                "not enough rows",
                "too few rows",
                "historico insuficiente",
                "histórico insuficiente",
            ),
        ),
        (
            "no market data",
            (
                "returned no bars",
                "no bars returned",
                "no market data",
                "sem dados",
            ),
        ),
        (
            "missing model artifacts",
            (
                "no trained model",
                "missing trained features",
                "model artifacts not found",
                "without model artifacts",
            ),
        ),
    ]

    for reason, needles in patterns:
        if any(needle in lowered for needle in needles):
            rows_match = re.search(
                r"(?P<rows>\d+)\s*(?:rows|linhas)?\s*<\s*(?P<minimum>\d+)",
                lowered,
            )
            detail = reason
            if rows_match:
                detail = (
                    f"{reason}: rows {rows_match.group('rows')} "
                    f"< {rows_match.group('minimum')}"
                )
            return "ineligible_data", detail

    return "", ""


def _parse_validation_log(
    log_path: Path, *, profile: str | None = None, scope: str | None = None
) -> dict[str, object]:
    try:
        txt = _clean(log_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}

    inferred_ticker = scope or _safe_name_from_log(str(log_path))
    inferred_profile = profile or log_path.parent.name
    row: dict[str, object] = {
        "phase": (
            "04_validate_per_asset" if inferred_ticker != "ALL" else "03_validate_full_universe"
        ),
        "ticker": inferred_ticker if inferred_ticker != "ALL" else "",
        "scope": inferred_ticker,
        "profile": inferred_profile,
        "policy": inferred_profile,
        "log_path": str(log_path),
    }

    match = re.search(r"Policy\s+:\s+(\w+)", txt)
    if match:
        row["policy"] = match.group(1)
        row["profile"] = match.group(1)

    match = re.search(r"Amostra\s+:\s+(.+?)\s+\|\s+(\d+)\s+ativos", txt)
    if match:
        row["sample"] = match.group(1).strip()
        row["n_assets"] = int(match.group(2))

    match = re.search(r"Decision\s+:\s+\[([A-Z]+)\]", txt)
    if match:
        row["decision"] = match.group(1)

    match = re.search(r"Score\s+:\s+([-\d.]+)", txt)
    if match:
        row["score"] = _to_float(match.group(1))

    match = re.search(
        r"replay operacional\s+([+-]?\d+\.\d+)%\s+(\d+)\s+([+-]?\d+\.\d+)%",
        txt,
    )
    if match:
        row["return_pct"] = _to_float(match.group(1))
        row["trades"] = _to_int(match.group(2))
        row["drawdown_pct"] = _to_float(match.group(3))

    match = re.search(
        r"ECONOMIA.*?\n[-\s]+\n\s*([-\d.]+)%\s+([+-]?\d+\.\d+)%\s+(\S+)\s+([-\d.]+)%\s+([-\d.]+)%\s+([+-]?\d+\.\d+)",
        txt,
        re.S,
    )
    if match:
        row["hit_pct"] = _to_float(match.group(1))
        row["avg_trade_pct"] = _to_float(match.group(2))
        row["profit_factor_display"] = match.group(3)
        row["profit_factor"] = match.group(3)
        row["turnover_pct"] = _to_float(match.group(4))
        row["exposure_pct"] = _to_float(match.group(5))
        row["cost"] = _to_float(match.group(6))

    match = re.search(
        r"P&L AUDIT.*?\n[-\s]+\n\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%",
        txt,
        re.S,
    )
    if match:
        row["gross_plus"] = _to_float(match.group(1))
        row["gross_minus"] = _to_float(match.group(2))
        row["net"] = _to_float(match.group(3))
        row["before_cost_pct"] = _to_float(match.group(4))
        row["after_cost_pct"] = _to_float(match.group(5))

    match = re.search(r"Beat rate\s+:\s+([-\d.]+)%", txt)
    if match:
        row["beat_rate_pct"] = _to_float(match.group(1))

    eligibility_status, ineligible_reason = _ineligible_from_log(txt)
    if eligibility_status:
        row["eligibility_status"] = eligibility_status
        row["ineligible_reason"] = ineligible_reason
        row.setdefault("decision", "INELIGIBLE_DATA")
        row.setdefault("trades", 0)
        row.setdefault("profit_factor", "")
        rows_match = re.search(
            r"(?P<rows>\d+)\s*(?:rows|linhas)?\s*<\s*(?P<minimum>\d+)",
            txt.lower(),
        )
        if rows_match:
            row["data_rows"] = int(rows_match.group("rows"))

    return row


def _validation_logs(log_dir: Path) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    per_asset = log_dir / "04_validate_per_asset"
    for path in sorted(per_asset.glob("*/*.log")):
        profile = path.parent.name
        ticker = _safe_name_from_log(str(path))
        items.append((path, profile, ticker))
    full = log_dir / "03_validate_full_universe"
    for path in sorted(full.glob("*.log")):
        profile = path.stem
        items.append((path, profile, "ALL"))
    return items


def rebuild_validation_summary(log_dir: Path, *, force: bool = False) -> Path:
    summary_path = log_dir / "validation_summary.csv"
    if summary_path.exists() and not force:
        return summary_path
    logs = _validation_logs(log_dir)
    if not logs:
        raise SystemExit(f"validation logs not found under: {log_dir}")
    rows = [
        _parse_validation_log(path, profile=profile, scope=scope) for path, profile, scope in logs
    ]
    rows = [row for row in rows if row]
    _write_csv(summary_path, rows, SUMMARY_FIELDS)
    return summary_path


def _normalise_validation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        scope = str(row.get("scope", "") or row.get("ticker", "") or "")
        log_path = str(row.get("log_path", "") or "")
        if not scope:
            scope = _safe_name_from_log(log_path)
        if not scope or scope.upper() == "ALL":
            continue
        ticker = scope if scope.endswith(".SA") else _safe_name_from_log(log_path)
        pf_display = str(row.get("profit_factor_display") or row.get("profit_factor") or "")
        item = {
            "ticker": ticker,
            "profile": row.get("profile") or row.get("policy") or "",
            "decision": row.get("decision", ""),
            "return_pct": _to_float(row.get("return_pct")),
            "trades": _to_int(row.get("trades")),
            "drawdown_pct": _to_float(row.get("drawdown_pct")),
            "hit_pct": _to_float(row.get("hit_pct")),
            "avg_trade_pct": _to_float(row.get("avg_trade_pct")),
            "profit_factor": _to_float(pf_display),
            "profit_factor_display": pf_display or _display_pf(row.get("profit_factor")),
            "turnover_pct": _to_float(row.get("turnover_pct")),
            "exposure_pct": _to_float(row.get("exposure_pct")),
            "cost": _to_float(row.get("cost")),
            "gross_plus": _to_float(row.get("gross_plus")),
            "gross_minus": _to_float(row.get("gross_minus")),
            "net": _to_float(row.get("net")),
            "before_cost_pct": _to_float(row.get("before_cost_pct")),
            "after_cost_pct": _to_float(row.get("after_cost_pct")),
            "beat_rate_pct": _to_float(row.get("beat_rate_pct")),
            "eligibility_status": row.get("eligibility_status", ""),
            "ineligible_reason": row.get("ineligible_reason", ""),
            "data_rows": _to_int(row.get("data_rows")),
            "log_path": log_path,
        }
        out.append(item)
    return out


def _is_candidate(
    row: dict[str, object], min_trades: int, min_pf: float, min_return_pct: float
) -> bool:
    pf = float(row.get("profit_factor", math.nan))
    return (
        int(row.get("trades", 0) or 0) >= min_trades
        and float(row.get("return_pct", math.nan)) > min_return_pct
        and (math.isinf(pf) or (not math.isnan(pf) and pf >= min_pf))
    )


def _profile_summary(
    rows: list[dict[str, object]], min_trades: int, min_pf: float, min_return_pct: float
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["profile"])].append(row)
    output: list[dict[str, object]] = []
    for profile, items in sorted(groups.items()):
        candidates = [r for r in items if _is_candidate(r, min_trades, min_pf, min_return_pct)]
        returns = [float(r["return_pct"]) for r in items if not math.isnan(float(r["return_pct"]))]
        drawdowns = [
            float(r["drawdown_pct"]) for r in items if not math.isnan(float(r["drawdown_pct"]))
        ]
        exposures = [
            float(r["exposure_pct"]) for r in items if not math.isnan(float(r["exposure_pct"]))
        ]
        output.append(
            {
                "profile": profile,
                "assets": len(items),
                "positive_assets": sum(1 for r in items if float(r["return_pct"]) > 0),
                "usable_candidates": len(candidates),
                "total_trades": sum(int(r["trades"]) for r in items),
                "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else "",
                "median_return_pct": (
                    round(sorted(returns)[len(returns) // 2], 4) if returns else ""
                ),
                "avg_drawdown_pct": round(sum(drawdowns) / len(drawdowns), 4) if drawdowns else "",
                "avg_exposure_pct": round(sum(exposures) / len(exposures), 4) if exposures else "",
            }
        )
    return output


def _eligibility(
    rows: list[dict[str, object]], min_trades: int, min_pf: float, min_return_pct: float
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["ticker"])].append(row)
    result: list[dict[str, object]] = []
    for ticker, items in sorted(groups.items()):
        good = [r for r in items if _is_candidate(r, min_trades, min_pf, min_return_pct)]
        good.sort(key=lambda r: (float(r["return_pct"]), float(r["profit_factor"])), reverse=True)
        relaxed = next((r for r in items if r.get("profile") == "relaxed"), None)
        any_trades = any(int(r.get("trades", 0) or 0) > 0 for r in items)
        if good:
            best = good[0]
            status = "observe"
            best_profile = str(best["profile"])
            reason = (
                f"best {best_profile}: return {float(best['return_pct']):+.2f}%, "
                f"PF {best['profit_factor_display']}, trades {int(best['trades'])}, "
                f"DD {float(best['drawdown_pct']):+.2f}%"
            )
        elif (
            relaxed
            and int(relaxed.get("trades", 0) or 0) >= min_trades
            and float(relaxed.get("return_pct", 0.0)) < 0
        ):
            status = "blocked_aggressive"
            best_profile = "none"
            reason = (
                f"relaxed bad: return {float(relaxed['return_pct']):+.2f}%, "
                f"PF {relaxed['profit_factor_display']}, trades {int(relaxed['trades'])}, "
                f"DD {float(relaxed['drawdown_pct']):+.2f}%"
            )
        elif not any_trades:
            status = "untested_no_trades"
            best_profile = "none"
            reason = "no trades across profiles"
        else:
            best = sorted(items, key=lambda r: float(r.get("return_pct", -9999)), reverse=True)[0]
            status = "observe_insufficient"
            best_profile = str(best["profile"])
            reason = (
                f"best weak {best_profile}: return {float(best['return_pct']):+.2f}%, "
                f"PF {best['profit_factor_display']}, trades {int(best['trades'])}, "
                f"DD {float(best['drawdown_pct']):+.2f}%"
            )
        result.append(
            {
                "ticker": ticker,
                "suggested_status": status,
                "best_profile": best_profile,
                "reason": reason,
            }
        )
    return result


def _candidates(
    rows: list[dict[str, object]], min_trades: int, min_pf: float, min_return_pct: float
) -> list[dict[str, object]]:
    candidates = [r for r in rows if _is_candidate(r, min_trades, min_pf, min_return_pct)]
    return sorted(
        candidates, key=lambda r: (float(r["return_pct"]), float(r["profit_factor"])), reverse=True
    )


def _write_yaml(path: Path, eligibility_rows: list[dict[str, object]], source: Path) -> None:
    lines = [
        "asset_eligibility_suggested:",
        f"  generated_from: {source.as_posix()}",
        "  note: replay-only evidence; require walk-forward before operational promotion",
        "  default_status: untested",
        "  assets:",
    ]
    for row in eligibility_rows:
        reason = str(row["reason"]).replace('"', "'")
        lines.extend(
            [
                f"    {row['ticker']}:",
                f"      status: {row['suggested_status']}",
                f"      best_profile: {row['best_profile']}",
                f'      reason: "{reason}"',
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matrix_health(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    if not rows:
        warnings.append("MATRIX INVALID: no validation rows were parsed.")
        return warnings
    total_trades = sum(int(row.get("trades", 0) or 0) for row in rows)
    nonzero_returns = sum(
        1
        for row in rows
        if not math.isnan(float(row.get("return_pct", math.nan)))
        and abs(float(row.get("return_pct", 0))) > 0.000001
    )
    if total_trades == 0 and nonzero_returns == 0:
        warnings.append(
            "MATRIX INVALID OR UNTRAINED: all validations have zero trades and zero returns. Rebuild artifacts/train before using this matrix."
        )
    return warnings


def _write_report(
    path: Path,
    manifest: dict,
    profile_rows: list[dict[str, object]],
    eligibility_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    warnings: list[str],
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in eligibility_rows:
        counts[str(row["suggested_status"])] += 1
    lines = [
        "# TradeChat — Policy Matrix Analysis",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Assets: {manifest.get('ticker_count', 'unknown')}",
        f"Profiles: {', '.join(manifest.get('profiles', []))}",
        "",
    ]
    if warnings:
        lines.extend(["## Health warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## Profile summary",
            "",
            "| Profile | Assets | Positive | Candidates | Trades | Avg Return | Median Return | Avg Drawdown | Avg Exposure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in profile_rows:
        lines.append(
            f"| {row['profile']} | {row['assets']} | {row['positive_assets']} | {row['usable_candidates']} | {row['total_trades']} | {row['avg_return_pct']} | {row['median_return_pct']} | {row['avg_drawdown_pct']} | {row['avg_exposure_pct']} |"
        )
    lines.extend(["", "## Main candidates", ""])
    if candidate_rows:
        lines.extend(
            [
                "| Ticker | Profile | Return | Trades | PF | Drawdown | Decision |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in candidate_rows:
            lines.append(
                f"| {row['ticker']} | {row['profile']} | {float(row['return_pct']):+.2f}% | {int(row['trades'])} | {row['profit_factor_display']} | {float(row['drawdown_pct']):+.2f}% | {row['decision']} |"
            )
    else:
        lines.append("No candidates under the selected thresholds.")
    lines.extend(["", "## Eligibility counts", ""])
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Suggested eligibility", ""])
    for row in eligibility_rows:
        lines.append(
            f"- {row['ticker']}: {row['suggested_status']} / {row['best_profile']} — {row['reason']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze TradeChat policy matrix logs.")
    parser.add_argument("log_dir", help="policy matrix run directory")
    parser.add_argument("--out-dir", default=None, help="default: <log_dir>/analysis")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-pf", type=float, default=1.0)
    parser.add_argument("--min-return-pct", type=float, default=0.0)
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="rebuild validation_summary.csv from validation logs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir or log_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "manifest.json"
    summary_path = log_dir / "validation_summary.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if args.rebuild_summary or not summary_path.exists():
        summary_path = rebuild_validation_summary(log_dir, force=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _normalise_validation_rows(_read_csv(summary_path))
    profile_rows = _profile_summary(rows, args.min_trades, args.min_pf, args.min_return_pct)
    eligibility_rows = _eligibility(rows, args.min_trades, args.min_pf, args.min_return_pct)
    candidate_rows = _candidates(rows, args.min_trades, args.min_pf, args.min_return_pct)
    warnings = _matrix_health(rows)

    _write_csv(
        out_dir / "policy_matrix_asset_results.csv",
        rows,
        [
            "ticker",
            "profile",
            "decision",
            "return_pct",
            "trades",
            "drawdown_pct",
            "hit_pct",
            "avg_trade_pct",
            "profit_factor_display",
            "turnover_pct",
            "exposure_pct",
            "cost",
            "gross_plus",
            "gross_minus",
            "net",
            "before_cost_pct",
            "after_cost_pct",
            "beat_rate_pct",
            "log_path",
        ],
    )
    _write_csv(
        out_dir / "policy_matrix_profile_summary.csv",
        profile_rows,
        [
            "profile",
            "assets",
            "positive_assets",
            "usable_candidates",
            "total_trades",
            "avg_return_pct",
            "median_return_pct",
            "avg_drawdown_pct",
            "avg_exposure_pct",
        ],
    )
    _write_csv(
        out_dir / "policy_matrix_eligibility_suggested.csv",
        eligibility_rows,
        ["ticker", "suggested_status", "best_profile", "reason"],
    )
    _write_csv(
        out_dir / "policy_matrix_candidates.csv",
        candidate_rows,
        [
            "ticker",
            "profile",
            "decision",
            "return_pct",
            "trades",
            "drawdown_pct",
            "hit_pct",
            "profit_factor_display",
            "exposure_pct",
            "beat_rate_pct",
            "log_path",
        ],
    )
    _write_yaml(out_dir / "asset_eligibility_suggested.yaml", eligibility_rows, log_dir)
    _write_report(
        out_dir / "policy_matrix_analysis_report.md",
        manifest,
        profile_rows,
        eligibility_rows,
        candidate_rows,
        warnings,
    )
    print(f"Analysis written to: {out_dir}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
