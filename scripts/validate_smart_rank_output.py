from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_PATH = PROJECT_ROOT / "runtime" / "runtime_policy.json"

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(
    r"Rows:\s+(?P<shown>\d+)\s+of\s+(?P<total>\d+)\s+\|\s+"
    r"OK=(?P<ok>\d+)\s+\|\s+"
    r"BLOCK=(?P<block>\d+)\s+\|\s+"
    r"REJECTED=(?P<rejected>\d+)\s+\|\s+"
    r"SKIP=(?P<skip>\d+)\s+\|\s+"
    r"ERROR=(?P<error>\d+)"
)

ACTIONABLE_SIGNALS = {
    "BUY",
    "SELL",
    "STRONG BUY",
    "STRONG SELL",
}


def _plain(text: str) -> str:
    return ANSI_RE.sub("", text)


def _load_runtime_assets(runtime_path: Path) -> tuple[dict[str, Any], str | None]:
    if not runtime_path.exists():
        return {}, f"{runtime_path.as_posix()} not found; NO_MATRIX cross-check skipped."

    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"could not read runtime policy: {exc}"

    assets = data.get("assets", {}) or {}

    if not isinstance(assets, dict):
        return {}, "runtime policy assets entry is not an object; NO_MATRIX cross-check skipped."

    return assets, None


def parse_smart_rank_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_header = False
    in_body = False

    for raw_line in text.splitlines():
        line = _plain(raw_line).rstrip()

        if not seen_header:
            if line.startswith("TICKER") and "SIGNAL" in line and "GUARD" in line:
                seen_header = True
            continue

        if line.startswith("-"):
            if not in_body:
                in_body = True
                continue
            break

        if not in_body or not line.strip():
            continue

        if len(line) >= 70:
            rows.append(
                {
                    "ticker": line[0:11].strip(),
                    "signal": line[12:22].strip(),
                    "profile": line[23:32].strip(),
                    "matrix": line[33:44].strip(),
                    "pf": line[45:51].strip(),
                    "trades": line[52:57].strip(),
                    "rr": line[58:63].strip(),
                    "guard": line[64:70].strip(),
                    "blocker": line[71:].strip(),
                }
            )
            continue

        parts = line.split(maxsplit=8)

        if len(parts) >= 8:
            rows.append(
                {
                    "ticker": parts[0],
                    "signal": parts[1],
                    "profile": parts[2],
                    "matrix": parts[3],
                    "pf": parts[4],
                    "trades": parts[5],
                    "rr": parts[6],
                    "guard": parts[7],
                    "blocker": parts[8] if len(parts) > 8 else "",
                }
            )

    return rows


def parse_smart_rank_summary(text: str) -> dict[str, int] | None:
    match = SUMMARY_RE.search(_plain(text))

    if not match:
        return None

    return {key: int(value) for key, value in match.groupdict().items()}


def _count_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "ok": sum(1 for row in rows if row["guard"].upper() == "OK"),
        "block": sum(1 for row in rows if row["guard"].upper() == "BLOCK"),
        "rejected": sum(1 for row in rows if row["signal"].upper() == "REJECTED"),
        "skip": sum(1 for row in rows if row["guard"].upper() == "SKIP"),
        "error": sum(1 for row in rows if row["guard"].upper() == "ERROR"),
    }


def validate_smart_rank_output(
    text: str,
    *,
    expected_rows: int | None = None,
    runtime_path: Path | str = DEFAULT_RUNTIME_PATH,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = parse_smart_rank_rows(text)
    summary = parse_smart_rank_summary(text)
    runtime_assets, runtime_warning = _load_runtime_assets(Path(runtime_path))

    if runtime_warning:
        warnings.append(runtime_warning)

    if ANSI_RE.search(text):
        errors.append("ANSI escape sequences found in SMART RANK output.")

    if "SMART RANK" not in _plain(text):
        errors.append("SMART RANK header not found.")

    if summary is None:
        errors.append("SMART RANK footer with row counts not found.")
    else:
        if expected_rows is not None:
            if summary["shown"] != expected_rows or summary["total"] != expected_rows:
                errors.append(
                    "rank-limit mismatch: "
                    f"expected Rows: {expected_rows} of {expected_rows}, "
                    f"got Rows: {summary['shown']} of {summary['total']}."
                )

        if summary["total"] != len(rows):
            errors.append(
                f"footer total={summary['total']} but parsed table has {len(rows)} rows."
            )

        row_counts = _count_rows(rows)

        for key, value in row_counts.items():
            if summary[key] != value:
                errors.append(
                    f"footer {key.upper()}={summary[key]} but parsed table has {value}."
                )

    for row in rows:
        ticker = row["ticker"]
        signal = row["signal"].upper()
        profile = row["profile"].lower()
        matrix = row["matrix"].lower()
        guard = row["guard"].upper()

        if signal == "REJECTED" and guard != "BLOCK":
            errors.append(f"{ticker}: REJECTED row must use guard BLOCK.")

        if signal == "NO_MATRIX" and guard != "SKIP":
            errors.append(f"{ticker}: NO_MATRIX row must use guard SKIP.")

        if signal == "ERROR" and guard != "ERROR":
            errors.append(f"{ticker}: ERROR row must use guard ERROR.")

        if signal in ACTIONABLE_SIGNALS and profile == "balanced" and matrix == "n/a":
            errors.append(f"{ticker}: actionable balanced n/a signal found in smart rank.")

        if signal == "NO_MATRIX" and ticker in runtime_assets:
            errors.append(f"{ticker}: NO_MATRIX row exists but ticker is present in runtime.")

    result = {
        "rows": rows,
        "summary": summary or {},
    }

    return errors, warnings, result


def _read_input(path_value: str | None) -> str:
    if not path_value or path_value == "-":
        return sys.stdin.read()

    return Path(path_value).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a captured SMART RANK output.")
    parser.add_argument("path", nargs="?", default="-")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = _read_input(args.path)
    errors, warnings, result = validate_smart_rank_output(
        text,
        expected_rows=args.expected_rows,
        runtime_path=args.runtime,
    )

    summary = result.get("summary", {}) or {}

    print("SMART RANK OUTPUT CHECK")
    print("-" * 80)

    if summary:
        print(
            f"Rows: {summary.get('shown', 0)} of {summary.get('total', 0)} | "
            f"OK={summary.get('ok', 0)} | "
            f"BLOCK={summary.get('block', 0)} | "
            f"REJECTED={summary.get('rejected', 0)} | "
            f"SKIP={summary.get('skip', 0)} | "
            f"ERROR={summary.get('error', 0)}"
        )
    else:
        print("Rows: n/a")

    for warning in warnings:
        print(f"WARNING: {warning}")

    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
