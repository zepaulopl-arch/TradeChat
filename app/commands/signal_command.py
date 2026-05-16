from __future__ import annotations

import argparse
import json
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
    merge_runtime_overrides,
    resolve_policy_profile,
    resolve_policy_selection,
    runtime_decision_guard_config,
    runtime_policy_overrides_for_profile,
)
from ..utils import read_json
from ._shared import resolve_cli_tickers


def _policy_cfg(
    cfg: dict,
    args: argparse.Namespace,
) -> dict:
    profile = getattr(
        args,
        "policy_profile",
        None,
    )

    if not profile:
        tickers = []

        if getattr(args, "tickers", None):
            tickers = list(args.tickers)

        elif getattr(args, "asset_list", None):
            try:
                from ..tickers import load_asset_list

                tickers = load_asset_list(
                    cfg,
                    args.asset_list,
                )

            except Exception:
                tickers = []

        if len(tickers) == 1:
            profile = resolve_policy_profile(
                str(tickers[0]),
            )

    return (
        apply_policy_profile(
            cfg,
            profile,
        )
        if profile
        else cfg
    )


def _generate(
    cfg: dict,
    args: argparse.Namespace,
    *,
    print_output: bool = True,
) -> None:
    cfg = _policy_cfg(
        cfg,
        args,
    )

    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=True,
    )

    for ticker in tickers:
        signal = make_signal(
            cfg,
            ticker,
            update=bool(
                getattr(
                    args,
                    "update",
                    False,
                )
            ),
        )

        if print_output:
            print_signal(
                signal,
                verbose=bool(
                    getattr(
                        args,
                        "verbose",
                        False,
                    )
                ),
                diagnostic=bool(
                    getattr(
                        args,
                        "diagnostic",
                        False,
                    )
                ),
                cfg=cfg,
            )


def _smart_signal_path(
    cfg: dict,
    ticker: str,
) -> Path:
    base_path = latest_signal_path(
        cfg,
        ticker,
    )

    return base_path.with_name("latest_smart_signal.json")


def _write_smart_signal(
    cfg: dict,
    ticker: str,
    signal: dict[str, Any],
) -> Path:
    path = _smart_signal_path(
        cfg,
        ticker,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            signal,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def _build_smart_signal(
    cfg: dict,
    args: argparse.Namespace,
    ticker: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    selection = resolve_policy_selection(
        str(ticker),
        fallback=getattr(
            args,
            "policy_profile",
            None,
        ),
    )

    profile = str(
        selection.get(
            "profile",
            "",
        )
    )

    if not profile:
        raise SystemExit(f"no runtime policy profile available for {ticker}")

    smart_cfg = apply_policy_profile(
        cfg,
        profile,
    )

    stored_overrides = (
        selection.get(
            "overrides",
            {},
        )
        or {}
    )

    live_overrides = runtime_policy_overrides_for_profile(profile)

    overrides = merge_runtime_overrides(
        stored_overrides,
        live_overrides,
    )

    smart_cfg = apply_runtime_policy_overrides(
        smart_cfg,
        overrides,
    )

    evidence = (
        selection.get(
            "evidence",
            {},
        )
        or {}
    )

    raw_signal = make_signal(
        smart_cfg,
        ticker,
        update=bool(
            getattr(
                args,
                "update",
                False,
            )
        ),
    )

    guard_cfg = runtime_decision_guard_config()

    signal = apply_matrix_decision_guard(
        raw_signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    signal["smart_signal"] = {
        "enabled": True,
        "source": selection.get(
            "source",
            "runtime_policy",
        ),
        "profile": profile,
        "overrides": overrides,
        "stored_overrides": stored_overrides,
        "live_overrides": live_overrides,
        "evidence": evidence,
        "selection": selection.get(
            "selection",
            {},
        ),
        "matrix_decision_guard": signal.get(
            "matrix_decision_guard",
            {},
        ),
    }

    out_path = _write_smart_signal(
        smart_cfg,
        ticker,
        signal,
    )

    return signal, smart_cfg, out_path


def _print_smart_header(
    ticker: str,
    signal: dict[str, Any],
    out_path: Path,
) -> None:
    smart = (
        signal.get(
            "smart_signal",
            {},
        )
        or {}
    )

    evidence = (
        smart.get(
            "evidence",
            {},
        )
        or {}
    )

    live_overrides = (
        smart.get(
            "live_overrides",
            {},
        )
        or {}
    )

    overrides = (
        smart.get(
            "overrides",
            {},
        )
        or {}
    )

    print()
    print(f"SMART SIGNAL — {ticker}")
    print(f"Runtime policy profile: {smart.get('profile')}")
    print(f"Source: {smart.get('source')}")

    if evidence:
        print("Matrix evidence:")

        for key in (
            "profit_factor",
            "sharpe",
            "drawdown_pct",
            "trades",
            "return_pct",
            "hit_pct",
            "avg_trade_pct",
            "exposure_pct",
            "score",
            "decision",
        ):
            if key in evidence:
                print(f"  {key}: {evidence[key]}")

    if live_overrides:
        print("Live YAML overrides:")

        for key, value in live_overrides.items():
            print(f"  {key}: {value}")

    if overrides:
        print("Effective runtime overrides:")

        for key, value in overrides.items():
            print(f"  {key}: {value}")

    guard_result = (
        signal.get(
            "matrix_decision_guard",
            {},
        )
        or {}
    )

    if guard_result:
        print("Matrix decision guard:")

        print(f"  decision: {guard_result.get('matrix_decision')}")
        print(f"  max_signal: {guard_result.get('max_signal')}")
        print(f"  blocked: {guard_result.get('blocked')}")

        if guard_result.get("reason"):
            print(f"  reason: {guard_result.get('reason')}")

    print(f"Smart artifact: {out_path}")
    print()


def _smart(
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=True,
    )

    for ticker in tickers:
        signal, smart_cfg, out_path = _build_smart_signal(
            cfg,
            args,
            ticker,
        )

        _print_smart_header(
            ticker,
            signal,
            out_path,
        )

        print_signal(
            signal,
            verbose=bool(
                getattr(
                    args,
                    "verbose",
                    False,
                )
            ),
            diagnostic=bool(
                getattr(
                    args,
                    "diagnostic",
                    False,
                )
            ),
            cfg=smart_cfg,
        )


def _rank(
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    if bool(
        getattr(
            args,
            "smart",
            False,
        )
    ):
        _smart_rank(
            cfg,
            args,
        )
        return

    cfg = _policy_cfg(
        cfg,
        args,
    )

    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=False,
    )

    if tickers:
        for ticker in tickers:
            make_signal(
                cfg,
                ticker,
                update=bool(
                    getattr(
                        args,
                        "update",
                        False,
                    )
                ),
            )

    for line in render_ranking(
        cfg,
        limit=int(
            getattr(
                args,
                "rank_limit",
                40,
            )
            or 40
        ),
        tickers=tickers or None,
        diagnostic=bool(
            getattr(
                args,
                "diagnostic",
                False,
            )
        ),
    ):
        print(line)


def _policy_dict(
    signal: dict[str, Any],
) -> dict[str, Any]:
    return (
        signal.get(
            "policy",
            {},
        )
        or {}
    )


def _smart_dict(
    signal: dict[str, Any],
) -> dict[str, Any]:
    return (
        signal.get(
            "smart_signal",
            {},
        )
        or {}
    )


def _smart_rank_score(
    signal: dict[str, Any],
) -> tuple[int, float, float]:
    policy = _policy_dict(signal)

    smart = _smart_dict(signal)

    evidence = (
        smart.get(
            "evidence",
            {},
        )
        or {}
    )

    guard = (
        smart.get(
            "matrix_decision_guard",
            {},
        )
        or {}
    )

    label = str(
        policy.get(
            "label",
            "NEUTRAL",
        )
    ).upper()

    label_score = {
        "STRONG BUY": 5,
        "BUY": 4,
        "NEUTRAL": 2,
        "SELL": 1,
        "STRONG SELL": 1,
    }.get(
        label,
        0,
    )

    if bool(
        guard.get(
            "blocked",
            False,
        )
    ):
        label_score -= 2

    matrix_score = float(
        evidence.get(
            "score",
            0.0,
        )
        or 0.0
    )

    rr = float(
        policy.get(
            "risk_reward_ratio",
            0.0,
        )
        or 0.0
    )

    return (
        label_score,
        matrix_score,
        rr,
    )


def _fmt_pct(
    value: Any,
) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def _smart_rank_row(
    signal: dict[str, Any],
) -> dict[str, Any]:
    policy = _policy_dict(signal)

    smart = _smart_dict(signal)

    evidence = (
        smart.get(
            "evidence",
            {},
        )
        or {}
    )

    guard = (
        smart.get(
            "matrix_decision_guard",
            {},
        )
        or {}
    )

    reasons = list(
        policy.get(
            "reasons",
            [],
        )
        or []
    )

    return {
        "ticker": signal.get(
            "ticker",
            evidence.get(
                "ticker",
                "N/A",
            ),
        ),
        "signal": str(
            policy.get(
                "label",
                signal.get(
                    "label",
                    "NEUTRAL",
                ),
            )
        ),
        "posture": str(
            policy.get(
                "posture",
                "n/a",
            )
        ),
        "profile": str(
            smart.get(
                "profile",
                "n/a",
            )
        ),
        "matrix": str(
            evidence.get(
                "decision",
                "n/a",
            )
        ),
        "pf": evidence.get(
            "profit_factor",
            "n/a",
        ),
        "trades": evidence.get(
            "trades",
            "n/a",
        ),
        "rr": policy.get(
            "risk_reward_ratio",
            0.0,
        ),
        "guard": (
            "BLOCK"
            if bool(
                guard.get(
                    "blocked",
                    False,
                )
            )
            else "OK"
        ),
        "blocker": "; ".join(str(item) for item in reasons[:2]) or "none",
    }


def _print_smart_rank(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    print()
    print("SMART RANK")
    print("-" * 132)
    print(
        f"{'TICKER':<12} {'SIGNAL':<12} {'PROFILE':<10} {'MATRIX':<12} "
        f"{'PF':>7} {'TRADES':>7} {'R/R':>7} {'GUARD':<7} BLOCKER"
    )
    print("-" * 132)

    for row in rows[:limit]:
        print(
            f"{str(row['ticker']):<12} "
            f"{str(row['signal']):<12} "
            f"{str(row['profile']):<10} "
            f"{str(row['matrix']):<12} "
            f"{_fmt_pct(row['pf']):>7} "
            f"{str(row['trades']):>7} "
            f"{_fmt_pct(row['rr']):>7} "
            f"{str(row['guard']):<7} "
            f"{str(row['blocker'])[:70]}"
        )

    print("-" * 132)
    print(f"Rows: {min(len(rows), limit)} of {len(rows)}")
    print()


def _smart_rank(
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=True,
    )

    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            signal, _, _ = _build_smart_signal(
                cfg,
                args,
                ticker,
            )

            rows.append(_smart_rank_row(signal))

        except Exception as exc:
            rows.append(
                {
                    "ticker": ticker,
                    "signal": "ERROR",
                    "posture": "error",
                    "profile": "n/a",
                    "matrix": "n/a",
                    "pf": "n/a",
                    "trades": "n/a",
                    "rr": 0.0,
                    "guard": "ERROR",
                    "blocker": str(exc),
                }
            )

    rows.sort(
        key=lambda row: _smart_rank_score(
            {
                "policy": {
                    "label": row.get(
                        "signal",
                        "NEUTRAL",
                    ),
                    "risk_reward_ratio": row.get(
                        "rr",
                        0.0,
                    ),
                },
                "smart_signal": {
                    "evidence": {
                        "score": 0.0,
                    },
                    "matrix_decision_guard": {
                        "blocked": row.get(
                            "guard",
                            "",
                        )
                        == "BLOCK",
                    },
                },
            }
        ),
        reverse=True,
    )

    _print_smart_rank(
        rows,
        limit=int(
            getattr(
                args,
                "rank_limit",
                40,
            )
            or 40
        ),
    )


def _report(
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    for ticker in resolve_cli_tickers(
        cfg,
        args,
        required=True,
    ):
        path = latest_signal_path(
            cfg,
            ticker,
        )

        if not path.exists() or bool(
            getattr(
                args,
                "refresh",
                False,
            )
        ):
            signal = make_signal(
                cfg,
                ticker,
                update=bool(
                    getattr(
                        args,
                        "update",
                        False,
                    )
                ),
            )
        else:
            signal = read_json(path)

        write_txt_report(
            cfg,
            signal,
        )
        print("report written")


def run(
    args: argparse.Namespace,
) -> None:
    cfg = load_config(args.config)

    action = str(
        getattr(
            args,
            "signal_action",
            "generate",
        )
    )

    if action == "generate":
        _generate(
            cfg,
            args,
            print_output=True,
        )

    elif action == "smart":
        _smart(
            cfg,
            args,
        )

    elif action == "rank":
        _rank(
            cfg,
            args,
        )

    elif action == "report":
        _report(
            cfg,
            args,
        )

    else:
        raise SystemExit(f"unknown signal action: {action}")
