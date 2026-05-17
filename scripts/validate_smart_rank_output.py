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
ROWS_RE = re.compile(r"Rows:\s+(?P<shown>\d+)\s+of\s+(?P<total>\d+)")
COUNTS_RE = re.compile(
    r"ACTIONABLE=(?P<actionable>\d+)\s+\|\s+"
    r"WATCH=(?P<watch>\d+)\s+\|\s+"
    r"BLOCKED=(?P<blocked>\d+)\s+\|\s+"
    r"REJECTED=(?P<rejected>\d+)\s+\|\s+"
    r"INELIGIBLE=(?P<ineligible>\d+)\s+\|\s+"
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
        return {}, f"{runtime_path.as_posix()} not found; runtime coverage cross-check skipped."

    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"could not read runtime policy: {exc}"

    assets = data.get("assets", {}) or {}

    if not isinstance(assets, dict):
        return {}, "runtime policy assets entry is not an object; coverage cross-check skipped."

    return assets, None


def _parse_fixed_width_row(line: str) -> dict[str, str] | None:
    if len(line) < 78:
        return None

    index = line[0:3].strip()

    if not index.isdigit():
        return None

    return {
        "index": index,
        "ticker": line[4:15].strip(),
        "action": line[16:27].strip(),
        "signal": line[28:38].strip(),
        "matrix": line[39:50].strip(),
        "pf": line[51:57].strip(),
        "trades": line[58:63].strip(),
        "rr": line[64:70].strip(),
        "guard": line[71:77].strip(),
        "reason": line[78:].strip(),
    }


def _parse_split_row(line: str) -> dict[str, str] | None:
    parts = line.split(maxsplit=10)

    if len(parts) < 9 or not parts[0].isdigit():
        return None

    signal_offset = 0
    signal = parts[3]

    if len(parts) >= 11 and parts[3].upper() == "STRONG" and parts[4].upper() in {"BUY", "SELL"}:
        signal = f"{parts[3]} {parts[4]}"
        signal_offset = 1

    try:
        return {
            "index": parts[0],
            "ticker": parts[1],
            "action": parts[2],
            "signal": signal,
            "matrix": parts[4 + signal_offset],
            "pf": parts[5 + signal_offset],
            "trades": parts[6 + signal_offset],
            "rr": parts[7 + signal_offset],
            "guard": parts[8 + signal_offset],
            "reason": parts[9 + signal_offset] if len(parts) > 9 + signal_offset else "",
        }
    except IndexError:
        return None


def parse_smart_rank_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_header = False
    in_body = False

    for raw_line in text.splitlines():
        line = _plain(raw_line).rstrip()

        if not seen_header:
            if line.startswith("#") and "ACTION" in line and "REASON" in line:
                seen_header = True
            continue

        if line.startswith("-"):
            if not in_body:
                in_body = True
                continue
            break

        if not in_body or not line.strip():
            continue

        row = _parse_fixed_width_row(line) or _parse_split_row(line)

        if row:
            rows.append(row)

    return rows


def parse_smart_rank_summary(text: str) -> dict[str, int] | None:
    plain = _plain(text)
    rows_match = ROWS_RE.search(plain)
    counts_match = COUNTS_RE.search(plain)

    if not rows_match:
        return None

    summary = {key: int(value) for key, value in rows_match.groupdict().items()}

    if counts_match:
        summary.update({key: int(value) for key, value in counts_match.groupdict().items()})

    return summary


def _count_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "actionable": sum(1 for row in rows if row["action"].upper() == "ACTIONABLE"),
        "watch": sum(1 for row in rows if row["action"].upper() == "WATCH"),
        "blocked": sum(1 for row in rows if row["action"].upper() == "BLOCKED"),
        "rejected": sum(1 for row in rows if row["action"].upper() == "REJECTED"),
        "ineligible": sum(1 for row in rows if row["action"].upper() == "INELIGIBLE"),
        "error": sum(1 for row in rows if row["action"].upper() == "ERROR"),
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
            expected_shown = min(expected_rows, summary["total"])
            if summary["shown"] != expected_shown:
                errors.append(
                    "rank-limit display mismatch: "
                    f"expected Rows: {expected_shown} of {summary['total']}, "
                    f"got Rows: {summary['shown']} of {summary['total']}."
                )

        if summary["shown"] != len(rows):
            errors.append(
                f"footer shown={summary['shown']} but parsed table has {len(rows)} rows."
            )

        if "actionable" in summary:
            row_counts = _count_rows(rows)

            for key, value in row_counts.items():
                if summary[key] < value:
                    errors.append(
                        f"footer {key.upper()}={summary[key]} but displayed table has {value}."
                    )

    for row in rows:
        ticker = row["ticker"]
        action = row["action"].upper()
        signal = row["signal"].upper()
        matrix = row["matrix"].lower()
        guard = row["guard"].upper()

        if action == "REJECTED" and (signal != "REJECTED" or guard != "BLOCK"):
            errors.append(f"{ticker}: REJECTED row must use signal REJECTED and guard BLOCK.")

        if action == "INELIGIBLE" and guard != "SKIP":
            errors.append(f"{ticker}: INELIGIBLE row must use guard SKIP.")

        if action == "ERROR" and (signal != "ERROR" or guard != "ERROR"):
            errors.append(f"{ticker}: ERROR row must use signal ERROR and guard ERROR.")

        if action == "ACTIONABLE" and signal in ACTIONABLE_SIGNALS and matrix == "n/a":
            errors.append(f"{ticker}: actionable signal without Matrix decision found.")

        if action == "ERROR" and ticker in runtime_assets and "missing Matrix" in row.get("reason", ""):
            errors.append(f"{ticker}: missing Matrix row reported but ticker is present in runtime.")

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
        print(f"Rows: {summary.get('shown', 0)} of {summary.get('total', 0)}")
        if "actionable" in summary:
            print(
                f"ACTIONABLE={summary.get('actionable', 0)} | "
                f"WATCH={summary.get('watch', 0)} | "
                f"BLOCKED={summary.get('blocked', 0)} | "
                f"REJECTED={summary.get('rejected', 0)} | "
                f"INELIGIBLE={summary.get('ineligible', 0)} | "
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
