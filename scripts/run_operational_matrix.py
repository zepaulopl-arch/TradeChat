from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_PATH = Path("runtime") / "runtime_policy.json"
DEFAULT_CONFIG_PATH = Path("config") / "runtime_policy.yaml"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "operational"
DEFAULT_MATRIX_BASE = Path("logs") / "policy_matrix"
OPERATIONAL_PROFILE = "active"


@dataclass(frozen=True)
class OperationalStep:
    name: str
    command: list[str]
    output_file: str
    required: bool = True


@dataclass(frozen=True)
class OperationalResult:
    name: str
    command: list[str]
    returncode: int
    output_file: Path
    stderr_file: Path | None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _display_command(command: list[str]) -> str:
    return " ".join(command)


def _normalise_python(value: str | None) -> str:
    return value or sys.executable


def _promote_command(python: str, matrix_dir: Path) -> list[str]:
    statement = (
        "from app.commands.promote_policy import promote_policy; "
        f"promote_policy({matrix_dir.as_posix()!r})"
    )
    return [python, "-c", statement]


def build_operational_plan(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    matrix_dir: Path,
) -> list[OperationalStep]:
    python = _normalise_python(getattr(args, "python", None))
    universe = str(getattr(args, "universe", "ibov") or "ibov")
    rank_limit = int(getattr(args, "rank_limit", 20) or 20)
    jobs = int(getattr(args, "jobs", 1) or 1)
    smart_rank_path = out_dir / "smart_rank.txt"

    matrix_command = [
        python,
        "trade.py",
        "validate",
        "matrix",
        "--universe",
        universe,
        "--jobs",
        str(jobs),
        "--policy-profile",
        OPERATIONAL_PROFILE,
        "--log-dir",
        matrix_dir.as_posix(),
    ]

    if bool(getattr(args, "allow_untrained", False)):
        matrix_command.append("--allow-untrained")

    if bool(getattr(args, "skip_tests", False)):
        matrix_command.append("--skip-pytest")

    return [
        OperationalStep(
            name="matrix_active",
            command=matrix_command,
            output_file="matrix.txt",
        ),
        OperationalStep(
            name="matrix_report",
            command=[python, "trade.py", "validate", "report", "--latest"],
            output_file="matrix_report.txt",
        ),
        OperationalStep(
            name="promote_policy",
            command=_promote_command(python, matrix_dir),
            output_file="promote_policy.txt",
        ),
        OperationalStep(
            name="runtime_check",
            command=[
                python,
                "scripts/check_runtime_policy.py",
                "--runtime",
                DEFAULT_RUNTIME_PATH.as_posix(),
                "--config",
                DEFAULT_CONFIG_PATH.as_posix(),
            ],
            output_file="runtime_check.txt",
        ),
        OperationalStep(
            name="smart_rank",
            command=[
                python,
                "trade.py",
                "signal",
                "rank",
                "--list",
                universe,
                "--smart",
                "--rank-limit",
                str(rank_limit),
            ],
            output_file="smart_rank.txt",
        ),
        OperationalStep(
            name="smart_rank_check",
            command=[
                python,
                "scripts/validate_smart_rank_output.py",
                smart_rank_path.as_posix(),
                "--runtime",
                DEFAULT_RUNTIME_PATH.as_posix(),
                "--expected-rows",
                str(rank_limit),
            ],
            output_file="smart_rank_check.txt",
        ),
    ]


def _run_step(step: OperationalStep, out_dir: Path) -> OperationalResult:
    output_path = out_dir / step.output_file
    stderr_path = out_dir / f"{Path(step.output_file).stem}.stderr.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")

    print(f"[{step.name}] {_display_command(step.command)}")
    result = subprocess.run(
        step.command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )

    output_path.write_text(result.stdout, encoding="utf-8")

    stderr_file: Path | None = None

    if result.stderr:
        stderr_path.write_text(result.stderr, encoding="utf-8")
        stderr_file = stderr_path

    if result.stdout:
        print(result.stdout.rstrip())

    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)

    return OperationalResult(
        name=step.name,
        command=step.command,
        returncode=result.returncode,
        output_file=output_path,
        stderr_file=stderr_file,
    )


def _write_summary(
    out_dir: Path,
    matrix_dir: Path,
    results: list[OperationalResult],
) -> None:
    lines = [
        "OPERATIONAL MATRIX RUN",
        "-" * 80,
        f"Profile: {OPERATIONAL_PROFILE}",
        f"Run dir: {out_dir.as_posix()}",
        f"Matrix dir: {matrix_dir.as_posix()}",
        "",
    ]

    for result in results:
        status = "OK" if result.returncode == 0 else f"ERROR {result.returncode}"
        lines.append(f"{result.name}: {status}")
        lines.append(f"  command: {_display_command(result.command)}")
        lines.append(f"  output: {result.output_file.as_posix()}")

        if result.stderr_file:
            lines.append(f"  stderr: {result.stderr_file.as_posix()}")

    rank_output = out_dir / "smart_rank.txt"

    if rank_output.exists():
        lines.append("")
        lines.append("SMART RANK FOOTER")

        for line in rank_output.read_text(encoding="utf-8").splitlines():
            if (
                line.startswith("Processed:")
                or line.startswith("Displayed:")
                or line.startswith("Rows:")
                or line.startswith("ACTIONABLE=")
                or line.startswith("Top actionable:")
                or line.startswith("No actionable assets found.")
                or line.startswith("Main blocker:")
            ):
                lines.append(line)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the active-only TradeChat Matrix -> Runtime -> Smart Rank flow."
    )
    parser.add_argument("--universe", default="ibov")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rank-limit", type=int, default=20)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = _timestamp()
    out_dir = Path(args.output_dir) / stamp
    matrix_dir = DEFAULT_MATRIX_BASE / stamp

    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if not matrix_dir.is_absolute():
        matrix_dir = PROJECT_ROOT / matrix_dir

    steps = build_operational_plan(
        args,
        out_dir=out_dir,
        matrix_dir=matrix_dir,
    )

    if args.dry_run:
        print("OPERATIONAL MATRIX PLAN")
        print("-" * 80)
        print(f"Profile: {OPERATIONAL_PROFILE}")
        print(f"Output dir: {out_dir.as_posix()}")
        print(f"Matrix dir: {matrix_dir.as_posix()}")

        for step in steps:
            print(f"{step.name}: {_display_command(step.command)}")

        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.parent.mkdir(parents=True, exist_ok=True)

    results: list[OperationalResult] = []
    failed = False

    for step in steps:
        result = _run_step(step, out_dir)
        results.append(result)

        if result.returncode != 0 and step.required:
            failed = True
            break

    _write_summary(out_dir, matrix_dir, results)
    print(f"Run artifacts: {out_dir.as_posix()}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
