from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config import load_config
from ..pipeline_service import latest_signal_path, make_signal
from ..policy import apply_policy_profile
from ..ranking_service import render_ranking
from ..report import print_signal, write_txt_report
from ..runtime_policy import (
    apply_matrix_decision_guard,
    apply_runtime_policy_overrides,
    load_runtime_policy,
    merge_runtime_overrides,
    resolve_policy_profile,
    resolve_policy_selection,
    runtime_decision_guard_config,
    runtime_policy_overrides_for_profile,
)
from ..smart_rank_service import (
    SmartRankDependencies,
    build_smart_rank_rows,
    rank_limit_from_args,
    smart_rank_fingerprint,
    smart_rank_preflight_warnings,
    smart_rank_summary,
)
from ..smart_report import print_smart_rank, print_smart_signal_header
from ..smart_signal_service import (
    SmartSignalDependencies,
    build_smart_signal,
)
from ..utils import read_json
from ._shared import resolve_cli_tickers


def _policy_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    profile = getattr(args, "policy_profile", None)

    if not profile:
        tickers = []

        if getattr(args, "tickers", None):
            tickers = list(args.tickers)

        elif getattr(args, "asset_list", None):
            try:
                from ..tickers import load_asset_list

                tickers = load_asset_list(cfg, args.asset_list)

            except Exception:
                tickers = []

        if len(tickers) == 1:
            profile = resolve_policy_profile(str(tickers[0]))

    return apply_policy_profile(cfg, profile) if profile else cfg


def _generate(
    cfg: dict,
    args: argparse.Namespace,
    *,
    print_output: bool = True,
) -> None:
    cfg = _policy_cfg(cfg, args)
    tickers = resolve_cli_tickers(cfg, args, required=True)

    for ticker in tickers:
        signal = make_signal(
            cfg,
            ticker,
            update=bool(getattr(args, "update", False)),
        )

        if print_output:
            print_signal(
                signal,
                verbose=bool(getattr(args, "verbose", False)),
                diagnostic=bool(getattr(args, "diagnostic", False)),
                cfg=cfg,
            )


def _smart_signal_deps() -> SmartSignalDependencies:
    return SmartSignalDependencies(
        resolve_policy_selection=resolve_policy_selection,
        apply_policy_profile=apply_policy_profile,
        runtime_policy_overrides_for_profile=runtime_policy_overrides_for_profile,
        merge_runtime_overrides=merge_runtime_overrides,
        apply_runtime_policy_overrides=apply_runtime_policy_overrides,
        make_signal=make_signal,
        runtime_decision_guard_config=runtime_decision_guard_config,
        apply_matrix_decision_guard=apply_matrix_decision_guard,
        latest_signal_path=latest_signal_path,
    )


def _build_smart_signal(
    cfg: dict,
    args: argparse.Namespace,
    ticker: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    return build_smart_signal(
        cfg,
        args,
        ticker,
        deps=_smart_signal_deps(),
    )


def _smart(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=True)

    for ticker in tickers:
        signal, smart_cfg, out_path = _build_smart_signal(
            cfg,
            args,
            ticker,
        )

        print_smart_signal_header(
            ticker,
            signal,
            out_path,
        )

        print_signal(
            signal,
            verbose=bool(getattr(args, "verbose", False)),
            diagnostic=bool(getattr(args, "diagnostic", False)),
            cfg=smart_cfg,
        )


def _rank(cfg: dict, args: argparse.Namespace) -> None:
    if bool(getattr(args, "smart", False)):
        _smart_rank(cfg, args)
        return

    cfg = _policy_cfg(cfg, args)
    tickers = resolve_cli_tickers(cfg, args, required=False)

    if tickers:
        for ticker in tickers:
            make_signal(
                cfg,
                ticker,
                update=bool(getattr(args, "update", False)),
            )

    for line in render_ranking(
        cfg,
        limit=int(getattr(args, "rank_limit", 40) or 40),
        tickers=tickers or None,
        diagnostic=bool(getattr(args, "diagnostic", False)),
    ):
        print(line)


def _smart_rank_deps() -> SmartRankDependencies:
    return SmartRankDependencies(
        resolve_policy_selection=resolve_policy_selection,
        build_smart_signal=_build_smart_signal,
    )


def _smart_rank(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=True,
    )

    runtime_data = load_runtime_policy()
    previous_prefer_latest = getattr(args, "_prefer_latest_signal", None)
    setattr(args, "_prefer_latest_signal", True)

    try:
        rows = build_smart_rank_rows(
            cfg,
            args,
            tickers,
            deps=_smart_rank_deps(),
            runtime_data=runtime_data,
            show_progress=False,
        )
    finally:
        if previous_prefer_latest is None:
            try:
                delattr(args, "_prefer_latest_signal")
            except AttributeError:
                pass
        else:
            setattr(args, "_prefer_latest_signal", previous_prefer_latest)

    print_smart_rank(
        rows,
        limit=rank_limit_from_args(args),
        fingerprint=smart_rank_fingerprint(
            args,
            total_tickers=len(tickers),
            processed_tickers=len(rows),
            runtime_data=runtime_data,
        ),
        warnings=smart_rank_preflight_warnings(
            runtime_data=runtime_data,
            tickers=tickers,
        ),
        summary=smart_rank_summary(rows),
    )


def _report(cfg: dict, args: argparse.Namespace) -> None:
    for ticker in resolve_cli_tickers(cfg, args, required=True):
        path = latest_signal_path(cfg, ticker)
        smart_path = path.with_name("latest_smart_signal.json")

        if bool(getattr(args, "refresh", False)):
            signal = make_signal(
                cfg,
                ticker,
                update=bool(getattr(args, "update", False)),
            )
        elif smart_path.exists():
            signal = read_json(smart_path)
        elif path.exists():
            signal = read_json(path)
        else:
            signal = make_signal(
                cfg,
                ticker,
                update=bool(getattr(args, "update", False)),
            )

        write_txt_report(cfg, signal)
        print("report written")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    action = str(getattr(args, "signal_action", "generate"))

    if action == "generate":
        _generate(cfg, args, print_output=True)

    elif action == "smart":
        _smart(cfg, args)

    elif action == "rank":
        _rank(cfg, args)

    elif action == "report":
        _report(cfg, args)

    else:
        raise SystemExit(f"unknown signal action: {action}")
