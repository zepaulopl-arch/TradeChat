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
DEFAULT_MATRIX_BASE = Path("logs") / "policy_matrix"
LATEST_MATRIX_PLACEHOLDER = "<latest_matrix_run>"


@dataclass(frozen=True)
class DailyStep:
    name: str
    command: list[str]
    output_file: str
    required: bool = True


@dataclass(frozen=True)
class StepResult:
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


def _promote_command(python: str, matrix_dir: str | Path) -> list[str]:
    statement = (
        "from app.commands.promote_policy import promote_policy; "
        f"promote_policy({str(matrix_dir)!r})"
    )
    return [python, "-c", statement]


def latest_matrix_run(base: Path | str = PROJECT_ROOT / DEFAULT_MATRIX_BASE) -> Path:
    base = Path(base)

    if not base.is_absolute():
        base = PROJECT_ROOT / base

    if not base.exists():
        raise FileNotFoundError(
            f"{base.as_posix()} not found. Run validate matrix before promote_policy."
        )

    candidates = [
        path for path in base.iterdir() if path.is_dir() and (path / "manifest.json").exists()
    ]

    if not candidates:
        raise FileNotFoundError(f"No Matrix run with manifest.json found under {base.as_posix()}.")

    return max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime)


def build_daily_plan(
    args: argparse.Namespace,
    *,
    out_dir: Path | None = None,
) -> list[DailyStep]:
    python = _normalise_python(getattr(args, "python", None))
    universe = str(getattr(args, "universe", "ibov") or "ibov")
    rank_limit = int(getattr(args, "rank_limit", 20) or 20)
    jobs = int(getattr(args, "jobs", 1) or 1)
    matrix_dir = getattr(args, "matrix_dir", None) or LATEST_MATRIX_PLACEHOLDER

    smart_rank_path = (
        out_dir / "smart_rank.txt" if out_dir is not None else Path("<out_dir>") / "smart_rank.txt"
    )

    steps: list[DailyStep] = []

    if not bool(getattr(args, "skip_data", False)):
        steps.append(
            DailyStep(
                name="data_load",
                command=[python, "trade.py", "data", "load", "--list", universe],
                output_file="data_load.txt",
            )
        )

    if bool(getattr(args, "with_matrix", False)):
        steps.append(
            DailyStep(
                name="matrix",
                command=[
                    python,
                    "trade.py",
                    "validate",
                    "matrix",
                    "--universe",
                    universe,
                    "--jobs",
                    str(jobs),
                ],
                output_file="matrix.txt",
            )
        )
        steps.append(
            DailyStep(
                name="matrix_report",
                command=[python, "trade.py", "validate", "report", "--latest"],
                output_file="matrix_report.txt",
            )
        )

        if not bool(getattr(args, "skip_promote", False)):
            steps.append(
                DailyStep(
                    name="promote_policy",
                    command=_promote_command(python, matrix_dir),
                    output_file="promote_policy.txt",
                )
            )

    steps.append(
        DailyStep(
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
        )
    )

    steps.append(
        DailyStep(
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
        )
    )

    validate_command = [
        python,
        "scripts/validate_smart_rank_output.py",
        str(smart_rank_path),
        "--runtime",
        DEFAULT_RUNTIME_PATH.as_posix(),
    ]

    if rank_limit > 0:
        validate_command.extend(["--expected-rows", str(rank_limit)])

    steps.append(
        DailyStep(
            name="smart_rank_check",
            command=validate_command,
            output_file="smart_rank_check.txt",
        )
    )

    return steps


def _resolve_dynamic_step(step: DailyStep) -> DailyStep:
    if step.name != "promote_policy":
        return step

    if LATEST_MATRIX_PLACEHOLDER not in step.command[-1]:
        return step

    matrix_dir = latest_matrix_run()
    return DailyStep(
        name=step.name,
        command=_promote_command(step.command[0], matrix_dir),
        output_file=step.output_file,
        required=step.required,
    )


def _run_step(step: DailyStep, out_dir: Path) -> StepResult:
    step = _resolve_dynamic_step(step)
    output_path = out_dir / step.output_file
    stderr_path = out_dir / f"{Path(step.output_file).stem}.stderr.txt"

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

    return StepResult(
        name=step.name,
        command=step.command,
        returncode=result.returncode,
        output_file=output_path,
        stderr_file=stderr_file,
    )


def _write_summary(out_dir: Path, results: list[StepResult]) -> None:
    lines = [
        "DAILY SMART RANK RUN",
        "-" * 80,
        f"Run dir: {out_dir.as_posix()}",
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
                line.startswith("Rows:")
                or line.startswith("Top actionable:")
                or line.startswith("Main blocker:")
            ):
                lines.append(line)

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic daily TradeChat Smart Rank routine."
    )
    parser.add_argument("--universe", default="ibov")
    parser.add_argument("--rank-limit", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--with-matrix", action="store_true")
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument("--matrix-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else PROJECT_ROOT / "artifacts" / "operational" / _timestamp()
    )

    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    steps = build_daily_plan(args, out_dir=out_dir)

    if args.dry_run:
        print("DAILY SMART RANK PLAN")
        print("-" * 80)

        for step in steps:
            print(f"{step.name}: {_display_command(step.command)}")

        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[StepResult] = []
    failed = False

    for step in steps:
        result = _run_step(step, out_dir)
        results.append(result)

        if result.returncode != 0 and step.required:
            failed = True
            break

    _write_summary(out_dir, results)
    print(f"Run artifacts: {out_dir.as_posix()}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
