from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..policy import apply_policy_profile
from ..simulation.runner import run_pybroker_replay
from ..validation_view import render_validation_summary
from ._shared import resolve_cli_tickers

_MATRIX_ACTIONS = {"matrix", "report"}


def _validate_action(args: argparse.Namespace) -> str | None:
    tickers = list(getattr(args, "tickers", []) or [])
    if tickers and str(tickers[0]).lower() in _MATRIX_ACTIONS:
        return str(tickers[0]).lower()
    return None


def _append_optional(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _append_positive_int(argv: list[str], flag: str, value: object | None) -> None:
    try:
        intval = int(value or 0)
    except (TypeError, ValueError):
        intval = 0
    if intval > 0:
        argv.extend([flag, str(intval)])


def _latest_matrix_run(base: Path = Path("logs") / "policy_matrix") -> Path:
    if not base.exists():
        raise SystemExit(
            "No policy matrix logs found. Run: python trade.py validate matrix --universe all"
        )
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not candidates:
        raise SystemExit("No policy matrix run with manifest.json found.")
    return max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime)


def _run_matrix(args: argparse.Namespace) -> None:
    from tools import run_policy_matrix

    mode = str(args.mode or "replay")
    tickers = list(getattr(args, "tickers", []) or [])[1:]
    argv: list[str] = ["--mode", mode]

    if args.config:
        argv.extend(["--config", str(args.config)])
    if tickers:
        argv.append("--tickers")
        argv.extend(tickers)
    else:
        universe = args.universe or getattr(args, "asset_list", None) or "all"
        argv.extend(["--asset-list", str(universe)])

    profiles = list(getattr(args, "profiles", None) or [])
    if profiles:
        argv.append("--profiles")
        argv.extend(profiles)

    _append_positive_int(argv, "--jobs", getattr(args, "jobs", 1))
    _append_optional(argv, "--log-dir", getattr(args, "log_dir", None))
    _append_optional(argv, "--start", getattr(args, "start", None))
    _append_optional(argv, "--end", getattr(args, "end", None))
    _append_positive_int(argv, "--rebalance-days", getattr(args, "rebalance_days", 0))
    _append_positive_int(argv, "--warmup-bars", getattr(args, "warmup_bars", 0))
    _append_optional(argv, "--cash", getattr(args, "cash", None))
    _append_optional(argv, "--max-positions", getattr(args, "max_positions", None))
    _append_positive_int(argv, "--max-assets", getattr(args, "max_assets", 0))
    _append_positive_int(
        argv, "--preflight-sample-size", getattr(args, "preflight_sample_size", 10)
    )

    for attr, flag in (
        ("allow_short", "--allow-short"),
        ("walkforward_autotune", "--walkforward-autotune"),
        ("skip_pytest", "--skip-pytest"),
        ("skip_data_audit", "--skip-data-audit"),
        ("skip_signal_rank", "--skip-signal-rank"),
        ("skip_preflight", "--skip-preflight"),
        ("allow_untrained", "--allow-untrained"),
        ("skip_per_asset", "--skip-per-asset"),
        ("resume", "--resume"),
        ("stop_on_error", "--stop-on-error"),
        ("serial_data_audit", "--serial-data-audit"),
    ):
        if bool(getattr(args, attr, False)):
            argv.append(flag)

    # Human default: matrix runs focus on per-asset validation; full-universe validation
    # currently remains optional because a single bad ticker/window can break the full batch.
    if bool(getattr(args, "skip_full_universe", False)) or not bool(
        getattr(args, "include_full_universe", False)
    ):
        argv.append("--skip-full-universe")

    raise SystemExit(run_policy_matrix.main(argv))


def _run_report(args: argparse.Namespace) -> None:
    from tools import analyze_policy_matrix

    tickers = list(getattr(args, "tickers", []) or [])
    requested_path = tickers[1] if len(tickers) > 1 else None
    log_dir = Path(requested_path) if requested_path else _latest_matrix_run()

    argv: list[str] = [str(log_dir)]
    _append_optional(argv, "--out-dir", getattr(args, "out_dir", None))
    _append_positive_int(argv, "--min-trades", getattr(args, "min_trades", 5))
    _append_optional(argv, "--min-pf", getattr(args, "min_pf", 1.0))
    _append_optional(argv, "--min-return-pct", getattr(args, "min_return_pct", 0.0))
    raise SystemExit(analyze_policy_matrix.main(argv))


def _run_validation(args: argparse.Namespace) -> None:
    if not args.mode:
        raise SystemExit(
            "tradechat: error: validate requires --mode replay|walkforward. "
            "For long batteries use: python trade.py validate matrix --universe all --jobs 4"
        )

    cfg = load_config(args.config)
    policy_profile = getattr(args, "policy_profile", None)
    if policy_profile:
        cfg = apply_policy_profile(cfg, policy_profile)
    tickers = resolve_cli_tickers(cfg, args)
    sim_cfg = cfg.get("simulation", {}) or {}
    mode = str(args.mode or sim_cfg.get("mode", "replay") or "replay").lower()
    summary = run_pybroker_replay(
        cfg,
        tickers,
        mode=mode,
        start_date=args.start,
        end_date=args.end,
        rebalance_days=(
            args.rebalance_days
            if args.rebalance_days > 0
            else int(sim_cfg.get("rebalance_days", 5) or 5)
        ),
        warmup_bars=(
            args.warmup_bars
            if args.warmup_bars > 0
            else int(sim_cfg.get("warmup_bars", 150) or 150)
        ),
        initial_cash=args.cash,
        max_positions=args.max_positions,
        allow_short=bool(args.allow_short or sim_cfg.get("allow_short", False)),
        walkforward_autotune=bool(args.walkforward_autotune),
        inner_threads=1,
    )
    for line in render_validation_summary(
        summary,
        mode=mode,
        screen_title=str(getattr(args, "screen_title", "VALIDATE")),
        verbose=bool(args.verbose),
    ):
        print(line)


def run(args: argparse.Namespace) -> None:
    action = _validate_action(args)
    if action == "matrix":
        _run_matrix(args)
        return
    if action == "report":
        _run_report(args)
        return
    _run_validation(args)

# =========================================================
# promote-policy imports
# =========================================================

try:

    from app.commands.promote_policy import (
        promote_policy,
    )

except Exception:

    promote_policy = None


# =========================================================
# promote-policy runtime dispatch
# =========================================================

try:

    if getattr(
        args,
        "validate_command",
        None,
    ) == "promote-policy":

        promote_policy(
            matrix_dir=args.matrix_dir,
        )

except Exception:
    pass

