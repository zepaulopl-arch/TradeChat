from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _to_float(value: str | None) -> float:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _safe_name_from_log(log_path: str) -> str:
    path = Path(log_path)
    stem = path.stem
    match = re.match(r"^\d+_(.+)$", stem)
    return match.group(1) if match else stem


def _normalise_validation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        scope = str(row.get("scope", "") or "")
        if not scope or scope.upper() == "ALL":
            continue
        ticker = (
            scope if scope.endswith(".SA") else _safe_name_from_log(str(row.get("log_path", "")))
        )
        pf = row.get("profit_factor", "")
        item = {
            "ticker": ticker,
            "profile": row.get("profile", ""),
            "decision": row.get("decision", ""),
            "return_pct": _to_float(row.get("return_pct")),
            "trades": int(float(row.get("trades", 0) or 0)),
            "drawdown_pct": _to_float(row.get("drawdown_pct")),
            "hit_pct": _to_float(row.get("hit_pct")),
            "avg_trade_pct": _to_float(row.get("avg_trade_pct")),
            "profit_factor": _to_float(pf),
            "profit_factor_display": pf,
            "turnover_pct": _to_float(row.get("turnover_pct")),
            "exposure_pct": _to_float(row.get("exposure_pct")),
            "cost": _to_float(row.get("cost")),
            "beat_rate_pct": _to_float(row.get("beat_rate_pct")),
            "log_path": row.get("log_path", ""),
        }
        out.append(item)
    return out


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
        output.append(
            {
                "profile": profile,
                "assets": len(items),
                "positive_assets": sum(1 for r in items if float(r["return_pct"]) > 0),
                "usable_candidates": len(candidates),
                "total_trades": sum(int(r["trades"]) for r in items),
                "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else "",
                "avg_drawdown_pct": round(sum(drawdowns) / len(drawdowns), 4) if drawdowns else "",
            }
        )
    return output


def _is_candidate(
    row: dict[str, object], min_trades: int, min_pf: float, min_return_pct: float
) -> bool:
    return (
        int(row.get("trades", 0) or 0) >= min_trades
        and float(row.get("return_pct", math.nan)) > min_return_pct
        and (
            math.isinf(float(row.get("profit_factor", math.nan)))
            or float(row.get("profit_factor", math.nan)) >= min_pf
        )
    )


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
            reason = f"best {best_profile}: return {float(best['return_pct']):+.2f}%, PF {best['profit_factor_display']}, trades {int(best['trades'])}"
        elif (
            relaxed
            and int(relaxed.get("trades", 0) or 0) >= min_trades
            and float(relaxed.get("return_pct", 0.0)) < 0
        ):
            status = "blocked_aggressive"
            best_profile = "none"
            reason = f"relaxed bad: return {float(relaxed['return_pct']):+.2f}%, PF {relaxed['profit_factor_display']}, trades {int(relaxed['trades'])}"
        elif not any_trades:
            status = "untested_no_trades"
            best_profile = "none"
            reason = "no trades across profiles"
        else:
            best = sorted(items, key=lambda r: float(r.get("return_pct", -9999)), reverse=True)[0]
            status = "observe_insufficient"
            best_profile = str(best["profile"])
            reason = f"best weak {best_profile}: return {float(best['return_pct']):+.2f}%, PF {best['profit_factor_display']}, trades {int(best['trades'])}"
        result.append(
            {
                "ticker": ticker,
                "suggested_status": status,
                "best_profile": best_profile,
                "reason": reason,
            }
        )
    return result


def _write_yaml(path: Path, eligibility_rows: list[dict[str, object]], source: Path) -> None:
    lines = [
        "asset_eligibility_suggested:",
        f"  generated_from: {source.as_posix()}",
        "  note: suggested from replay only; require walk-forward before operational promotion",
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


def _write_report(
    path: Path,
    manifest: dict,
    profile_rows: list[dict[str, object]],
    eligibility_rows: list[dict[str, object]],
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
        "## Profile summary",
        "",
        "| Profile | Assets | Positive | Candidates | Trades | Avg Return | Avg Drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profile_rows:
        lines.append(
            f"| {row['profile']} | {row['assets']} | {row['positive_assets']} | {row['usable_candidates']} | {row['total_trades']} | {row['avg_return_pct']} | {row['avg_drawdown_pct']} |"
        )
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
    if not summary_path.exists():
        raise SystemExit(f"validation summary not found: {summary_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _normalise_validation_rows(_read_csv(summary_path))
    profile_rows = _profile_summary(rows, args.min_trades, args.min_pf, args.min_return_pct)
    eligibility_rows = _eligibility(rows, args.min_trades, args.min_pf, args.min_return_pct)

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
            "avg_drawdown_pct",
        ],
    )
    _write_csv(
        out_dir / "policy_matrix_eligibility_suggested.csv",
        eligibility_rows,
        ["ticker", "suggested_status", "best_profile", "reason"],
    )
    _write_yaml(out_dir / "asset_eligibility_suggested.yaml", eligibility_rows, log_dir)
    _write_report(
        out_dir / "policy_matrix_analysis_report.md", manifest, profile_rows, eligibility_rows
    )
    print(f"Analysis written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
