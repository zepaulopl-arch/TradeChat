from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATUS_LOCK = Lock()
SUMMARY_LOCK = Lock()
PRINT_LOCK = Lock()

DEFAULT_PROFILES = ("strict", "balanced", "active", "relaxed")
DEFAULT_LIGHT_TESTS = (
    "tests/test_cli_contract.py",
    "tests/test_no_legacy.py",
    "tests/test_policy_profiles.py",
    "tests/test_validate_policy_profile.py",
    "tests/test_signal_policy_diagnostic.py",
    "tests/test_validation_metrics_audit.py",
    "tests/test_validation_trade_attribution.py",
)


@dataclass
class StepResult:
    phase: str
    name: str
    command: list[str]
    log_path: Path
    returncode: int
    started_at: str
    finished_at: str
    elapsed_seconds: float


@dataclass(frozen=True)
class MatrixTask:
    phase: str
    name: str
    command: list[str]
    log_path: Path
    profile: str | None = None
    scope: str | None = None
    append_validation_summary: bool = False


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _repo_root() -> Path:
    return PROJECT_ROOT


def _python() -> str:
    return sys.executable or "python"


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _load_all_tickers(config_path: str | None, asset_list: str) -> list[str]:
    # Import lazily so --help and static tests do not import the application.
    from app.config import load_config
    from app.commands._shared import registry_list_tickers

    cfg = load_config(config_path)
    return list(registry_list_tickers(cfg, asset_list))


def _artifact_profile_for_ticker(config_path: str | None, ticker: str) -> dict[str, object]:
    # Import lazily so --help/static tests remain cheap.
    from app.config import load_config, models_dir
    from app.utils import normalize_ticker, read_json, safe_ticker

    cfg = load_config(config_path)
    normalized = normalize_ticker(ticker)
    ticker_dir = models_dir(cfg) / safe_ticker(normalized)
    signal_path = ticker_dir / "latest_signal.json"
    model_files = []
    if ticker_dir.exists():
        model_files = [
            p
            for p in ticker_dir.rglob("*")
            if p.is_file() and p.name != "latest_signal.json" and not p.name.endswith(".tmp")
        ]

    has_signal = signal_path.exists()
    zero_signal = False
    quality = 0.0
    max_abs_prediction = 0.0
    if has_signal:
        try:
            signal = read_json(signal_path)
            horizons = signal.get("horizons", {}) or {}
            values = []
            for data in horizons.values():
                if isinstance(data, dict):
                    values.append(float(data.get("prediction_return", 0.0) or 0.0))
                    quality = max(
                        quality,
                        float(data.get("quality", data.get("confidence", 0.0)) or 0.0),
                    )
            policy = signal.get("policy", {}) or {}
            quality = max(
                quality,
                float(policy.get("quality_pct", policy.get("confidence_pct", 0.0)) or 0.0) / 100.0,
            )
            max_abs_prediction = max([abs(v) for v in values] or [0.0])
            zero_signal = max_abs_prediction <= 1e-12 and quality <= 1e-12
        except Exception:
            zero_signal = True

    return {
        "ticker": normalized,
        "ticker_dir": str(ticker_dir),
        "has_dir": ticker_dir.exists(),
        "has_signal": has_signal,
        "model_file_count": len(model_files),
        "has_model_artifacts": len(model_files) > 0,
        "zero_signal": zero_signal,
        "quality": quality,
        "max_abs_prediction": max_abs_prediction,
    }


def _write_preflight_report(run_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = run_dir / "preflight_artifacts.csv"
    fieldnames = (
        "ticker",
        "has_dir",
        "has_signal",
        "has_model_artifacts",
        "model_file_count",
        "zero_signal",
        "quality",
        "max_abs_prediction",
        "ticker_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_artifact_preflight(run_dir: Path, tickers: list[str], args: argparse.Namespace) -> None:
    if bool(getattr(args, "skip_preflight", False)):
        print("[SKIP] preflight artifact/model check")
        return

    rows = [_artifact_profile_for_ticker(args.config, ticker) for ticker in tickers]
    report_path = _write_preflight_report(run_dir, rows)
    missing_model = [r for r in rows if not r.get("has_model_artifacts")]
    missing_signal = [r for r in rows if not r.get("has_signal")]
    zero_signal = [r for r in rows if r.get("has_signal") and r.get("zero_signal")]

    print("Preflight artifacts/models:")
    print(f"  checked: {len(rows)}")
    print(f"  missing model artifacts: {len(missing_model)}")
    print(f"  missing latest_signal.json: {len(missing_signal)}")
    print(f"  zero latest signals: {len(zero_signal)}")
    print(f"  report: {report_path}")

    if (missing_model or missing_signal or zero_signal) and not bool(
        getattr(args, "allow_untrained", False)
    ):

        def sample(items: list[dict[str, object]]) -> str:
            cap = max(1, int(getattr(args, "preflight_sample_size", 10) or 10))
            return ", ".join(str(item.get("ticker")) for item in items[:cap])

        messages = []
        if missing_model:
            messages.append(f"assets without model artifacts: {sample(missing_model)}")
        if missing_signal:
            messages.append(f"assets without latest_signal.json: {sample(missing_signal)}")
        if zero_signal:
            messages.append(f"assets with zero signals: {sample(zero_signal)}")
        joined = "; ".join(messages)
        raise SystemExit(
            "Policy matrix preflight failed. "
            f"{joined}. "
            "Train/generate signals first, or pass --allow-untrained for a smoke test."
        )


def _iter_existing(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if Path(p).exists()]


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_status(run_dir: Path, result: StepResult) -> None:
    with STATUS_LOCK:
        status_path = run_dir / "status.csv"
        exists = status_path.exists()
        with status_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=(
                    "phase",
                    "name",
                    "returncode",
                    "started_at",
                    "finished_at",
                    "elapsed_seconds",
                    "log_path",
                    "command",
                ),
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "phase": result.phase,
                    "name": result.name,
                    "returncode": result.returncode,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "elapsed_seconds": f"{result.elapsed_seconds:.2f}",
                    "log_path": str(result.log_path),
                    "command": " ".join(result.command),
                }
            )


def _append_validation_summary(
    run_dir: Path, phase: str, profile: str, scope: str, log_path: Path
) -> None:
    metrics = _parse_validation_log(log_path)
    if not metrics:
        return
    fieldnames = (
        "phase",
        "profile",
        "scope",
        "decision",
        "return_pct",
        "trades",
        "drawdown_pct",
        "hit_pct",
        "avg_trade_pct",
        "profit_factor",
        "turnover_pct",
        "exposure_pct",
        "cost",
        "beat_rate_pct",
        "log_path",
    )
    row = {name: "" for name in fieldnames}
    row.update(metrics)
    row.update({"phase": phase, "profile": profile, "scope": scope, "log_path": str(log_path)})
    with SUMMARY_LOCK:
        path = run_dir / "validation_summary.csv"
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def _num(text: str) -> str:
    return text.replace("%", "").replace("+", "").replace("R$", "").strip()


def _parse_validation_log(log_path: Path) -> dict[str, str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    out: dict[str, str] = {}
    decision_match = re.search(r"Decision\s*:\s*\[([^\]]+)\]", text)
    if decision_match:
        out["decision"] = decision_match.group(1).strip()

    beat_match = re.search(r"Beat rate\s*:\s*([+\-]?[0-9.,]+)%", text)
    if beat_match:
        out["beat_rate_pct"] = _num(beat_match.group(1))

    result_match = re.search(
        r"replay operacional\s+([+\-]?[0-9.,]+)%\s+(\d+)\s+([+\-]?[0-9.,]+)%",
        text,
    )
    if result_match:
        out["return_pct"] = _num(result_match.group(1))
        out["trades"] = result_match.group(2).strip()
        out["drawdown_pct"] = _num(result_match.group(3))

    econ_match = re.search(
        r"\n\s*([0-9.,]+)%\s+([+\-]?[0-9.,]+)%\s+([A-Za-z0-9.]+)\s+([0-9.,]+)%\s+([0-9.,]+)%\s+\+?([0-9.,]+)",
        text,
    )
    if econ_match:
        out["hit_pct"] = _num(econ_match.group(1))
        out["avg_trade_pct"] = _num(econ_match.group(2))
        out["profit_factor"] = econ_match.group(3).strip()
        out["turnover_pct"] = _num(econ_match.group(4))
        out["exposure_pct"] = _num(econ_match.group(5))
        out["cost"] = _num(econ_match.group(6))

    return out


def _run_step(
    *,
    run_dir: Path,
    phase: str,
    name: str,
    command: list[str],
    log_path: Path,
    stop_on_error: bool,
    resume: bool,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and log_path.exists() and (log_path.with_suffix(log_path.suffix + ".ok")).exists():
        result = StepResult(
            phase=phase,
            name=name,
            command=command,
            log_path=log_path,
            returncode=0,
            started_at=_iso_now(),
            finished_at=_iso_now(),
            elapsed_seconds=0.0,
        )
        with PRINT_LOCK:
            print(f"[SKIP] {phase} | {name} -> {log_path}")
        _append_status(run_dir, result)
        return result

    started = _iso_now()
    t0 = perf_counter()
    with PRINT_LOCK:
        print(f"[RUN ] {phase} | {name}")
        print(f"       log: {log_path}")
    with log_path.open("w", encoding="utf-8", errors="ignore") as f:
        f.write(f"# phase: {phase}\n")
        f.write(f"# name: {name}\n")
        f.write(f"# started_at: {started}\n")
        f.write(f"# command: {' '.join(command)}\n")
        f.write("# " + "=" * 100 + "\n\n")
        f.flush()
        proc = subprocess.run(
            command, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT
        )
        finished = _iso_now()
        elapsed = perf_counter() - t0
        f.write("\n# " + "=" * 100 + "\n")
        f.write(f"# finished_at: {finished}\n")
        f.write(f"# elapsed_seconds: {elapsed:.2f}\n")
        f.write(f"# returncode: {proc.returncode}\n")

    result = StepResult(
        phase=phase,
        name=name,
        command=command,
        log_path=log_path,
        returncode=int(proc.returncode),
        started_at=started,
        finished_at=finished,
        elapsed_seconds=elapsed,
    )
    _append_status(run_dir, result)
    if result.returncode == 0:
        log_path.with_suffix(log_path.suffix + ".ok").write_text("ok\n", encoding="utf-8")
        with PRINT_LOCK:
            print(f"[ OK ] {phase} | {name} ({elapsed:.1f}s)")
    else:
        with PRINT_LOCK:
            print(f"[FAIL] {phase} | {name} rc={result.returncode} ({elapsed:.1f}s)")
        if stop_on_error:
            raise SystemExit(result.returncode)
    return result


def _run_task(run_dir: Path, task: MatrixTask, args: argparse.Namespace) -> StepResult:
    result = _run_step(
        run_dir=run_dir,
        phase=task.phase,
        name=task.name,
        command=task.command,
        log_path=task.log_path,
        stop_on_error=args.stop_on_error,
        resume=args.resume,
    )
    if result.returncode == 0 and task.append_validation_summary and task.profile and task.scope:
        _append_validation_summary(run_dir, task.phase, task.profile, task.scope, task.log_path)
    return result


def _run_tasks(
    run_dir: Path, tasks: list[MatrixTask], args: argparse.Namespace, *, parallel: bool
) -> list[StepResult]:
    if not tasks:
        return []
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    if not parallel or jobs <= 1 or len(tasks) == 1:
        return [_run_task(run_dir, task, args) for task in tasks]

    results: list[StepResult] = []
    with PRINT_LOCK:
        print(f"[JOBS] running {len(tasks)} tasks with jobs={jobs}")
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {executor.submit(_run_task, run_dir, task, args): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive safety for long runs
                with PRINT_LOCK:
                    print(f"[FAIL] {task.phase} | {task.name} raised {exc!r}")
                if args.stop_on_error:
                    raise
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/run_policy_matrix.py",
        description="Run long TradeChat policy validation batteries with persistent logs.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=("replay", "walkforward"), default="replay")
    parser.add_argument("--asset-list", default="all", help="asset list name resolved by TradeChat")
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="explicit tickers; overrides --asset-list"
    )
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument(
        "--jobs", type=int, default=1, help="parallel jobs for independent tasks; default 1"
    )
    parser.add_argument("--log-dir", default=None, help="default: logs/policy_matrix/<timestamp>")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance-days", type=int, default=0)
    parser.add_argument("--warmup-bars", type=int, default=0)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--walkforward-autotune", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-data-audit", action="store_true")
    parser.add_argument("--skip-signal-rank", action="store_true")
    parser.add_argument(
        "--skip-preflight", action="store_true", help="skip artifact/model preflight"
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="allow matrix to run even when preflight finds missing/zero artifacts",
    )
    parser.add_argument(
        "--preflight-sample-size",
        type=int,
        default=10,
        help="number of problematic tickers shown in preflight error",
    )
    parser.add_argument("--skip-full-universe", action="store_true")
    parser.add_argument("--skip-per-asset", action="store_true")
    parser.add_argument("--max-assets", type=int, default=0, help="optional cap for smoke tests")
    parser.add_argument("--resume", action="store_true", help="skip steps with .ok marker")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--serial-data-audit",
        action="store_true",
        help="force data audit phase to run sequentially",
    )
    return parser


def _validate_base_cmd(
    args: argparse.Namespace, tickers_or_list: list[str], profile: str
) -> list[str]:
    cmd = [_python(), "trade.py", "validate"]
    cmd.extend(tickers_or_list)
    cmd.extend(["--mode", args.mode, "--policy-profile", profile])
    if args.start:
        cmd.extend(["--start", args.start])
    if args.end:
        cmd.extend(["--end", args.end])
    if args.rebalance_days > 0:
        cmd.extend(["--rebalance-days", str(args.rebalance_days)])
    if args.warmup_bars > 0:
        cmd.extend(["--warmup-bars", str(args.warmup_bars)])
    if args.cash is not None:
        cmd.extend(["--cash", str(args.cash)])
    if args.max_positions is not None:
        cmd.extend(["--max-positions", str(args.max_positions)])
    if args.allow_short:
        cmd.append("--allow-short")
    if args.walkforward_autotune:
        cmd.append("--walkforward-autotune")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    if not (root / "trade.py").exists():
        raise SystemExit("Run this tool from the TradeChat repository root, where trade.py exists.")

    tickers = list(args.tickers or [])
    if not tickers:
        tickers = _load_all_tickers(args.config, args.asset_list)
    if args.max_assets and args.max_assets > 0:
        tickers = tickers[: args.max_assets]
    if not tickers:
        raise SystemExit("No tickers resolved for the policy matrix run.")

    run_dir = Path(args.log_dir or Path("logs") / "policy_matrix" / _timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": _iso_now(),
        "repo_root": str(root),
        "mode": args.mode,
        "asset_list": args.asset_list,
        "profiles": list(args.profiles),
        "tickers": tickers,
        "ticker_count": len(tickers),
        "options": {
            "start": args.start,
            "end": args.end,
            "rebalance_days": args.rebalance_days,
            "warmup_bars": args.warmup_bars,
            "cash": args.cash,
            "max_positions": args.max_positions,
            "allow_short": args.allow_short,
            "walkforward_autotune": args.walkforward_autotune,
            "jobs": max(1, int(args.jobs or 1)),
            "skip_preflight": bool(args.skip_preflight),
            "allow_untrained": bool(args.allow_untrained),
        },
    }
    _write_manifest(run_dir / "manifest.json", manifest)
    print(f"Policy matrix run directory: {run_dir}")
    print(
        f"Tickers: {len(tickers)} | Profiles: {', '.join(args.profiles)} | Mode: {args.mode} | Jobs: {max(1, int(args.jobs or 1))}"
    )

    _run_artifact_preflight(run_dir, tickers, args)

    if not args.skip_pytest:
        tests = _iter_existing(DEFAULT_LIGHT_TESTS)
        if tests:
            _run_step(
                run_dir=run_dir,
                phase="00_pytest",
                name="light_suite",
                command=[_python(), "-m", "pytest", *tests],
                log_path=run_dir / "00_pytest" / "light_suite.log",
                stop_on_error=args.stop_on_error,
                resume=args.resume,
            )

    if not args.skip_data_audit:
        data_tasks = [
            MatrixTask(
                phase="01_data_audit",
                name=f"{i:04d}_{ticker}",
                command=[_python(), "trade.py", "data", "audit", ticker],
                log_path=run_dir / "01_data_audit" / f"{i:04d}_{_sanitize(ticker)}.log",
            )
            for i, ticker in enumerate(tickers, start=1)
        ]
        _run_tasks(run_dir, data_tasks, args, parallel=not args.serial_data_audit)

    if not args.skip_signal_rank:
        for profile in args.profiles:
            _run_step(
                run_dir=run_dir,
                phase="02_signal_rank",
                name=profile,
                command=[
                    _python(),
                    "trade.py",
                    "signal",
                    "rank",
                    *tickers,
                    "--diagnostic",
                    "--policy-profile",
                    profile,
                    "--rank-limit",
                    str(max(40, len(tickers))),
                ],
                log_path=run_dir / "02_signal_rank" / f"{_sanitize(profile)}.log",
                stop_on_error=args.stop_on_error,
                resume=args.resume,
            )

    if not args.skip_full_universe:
        for profile in args.profiles:
            log_path = run_dir / "03_validate_full_universe" / f"{_sanitize(profile)}.log"
            result = _run_step(
                run_dir=run_dir,
                phase="03_validate_full_universe",
                name=profile,
                command=_validate_base_cmd(args, tickers, profile),
                log_path=log_path,
                stop_on_error=args.stop_on_error,
                resume=args.resume,
            )
            if result.returncode == 0:
                _append_validation_summary(run_dir, "full_universe", profile, "ALL", log_path)

    if not args.skip_per_asset:
        validate_tasks: list[MatrixTask] = []
        for profile in args.profiles:
            for i, ticker in enumerate(tickers, start=1):
                log_path = (
                    run_dir
                    / "04_validate_per_asset"
                    / _sanitize(profile)
                    / f"{i:04d}_{_sanitize(ticker)}.log"
                )
                validate_tasks.append(
                    MatrixTask(
                        phase="04_validate_per_asset",
                        name=f"{profile}:{ticker}",
                        command=_validate_base_cmd(args, [ticker], profile),
                        log_path=log_path,
                        profile=profile,
                        scope=ticker,
                        append_validation_summary=True,
                    )
                )
        _run_tasks(run_dir, validate_tasks, args, parallel=True)

    # Always rebuild the summary at the end. This makes --resume robust and
    # avoids empty/stale summaries when all validation tasks were skipped.
    try:
        from tools.analyze_policy_matrix import rebuild_validation_summary

        summary = rebuild_validation_summary(run_dir, force=True)
    except Exception as exc:  # pragma: no cover - defensive safety for long runs
        summary = run_dir / "validation_summary.csv"
        print(f"[WARN] could not rebuild validation summary: {exc}")

    print("\nDone.")
    print(f"Run directory: {run_dir}")
    print(f"Status CSV: {run_dir / 'status.csv'}")
    if summary.exists():
        print(f"Validation summary CSV: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
