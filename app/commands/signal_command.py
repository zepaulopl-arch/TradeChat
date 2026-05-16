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


def _smart_signal_path(cfg: dict, ticker: str) -> Path:
    base_path = latest_signal_path(cfg, ticker)
    return base_path.with_name("latest_smart_signal.json")


def _write_smart_signal(
    cfg: dict,
    ticker: str,
    signal: dict[str, Any],
) -> Path:
    path = _smart_signal_path(cfg, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(signal, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


def _selection_is_policy_matrix(selection: dict[str, Any]) -> bool:
    return str(selection.get("source", "")).lower() == "policy_matrix"


def _selection_is_promoted(selection: dict[str, Any]) -> bool:
    return bool(selection.get("promoted", True))


def _build_smart_signal(
    cfg: dict,
    args: argparse.Namespace,
    ticker: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    selection = resolve_policy_selection(
        str(ticker),
        fallback=getattr(args, "policy_profile", None),
    )

    profile = str(selection.get("profile", ""))

    if not profile:
        raise SystemExit(f"no runtime policy profile available for {ticker}")

    smart_cfg = apply_policy_profile(cfg, profile)

    stored_overrides = selection.get("overrides", {}) or {}
    live_overrides = runtime_policy_overrides_for_profile(profile)

    overrides = merge_runtime_overrides(
        stored_overrides,
        live_overrides,
    )

    smart_cfg = apply_runtime_policy_overrides(
        smart_cfg,
        overrides,
    )

    evidence = selection.get("evidence", {}) or {}

    raw_signal = make_signal(
        smart_cfg,
        ticker,
        update=bool(getattr(args, "update", False)),
    )

    guard_cfg = runtime_decision_guard_config()

    signal = apply_matrix_decision_guard(
        raw_signal,
        evidence=evidence,
        guard_cfg=guard_cfg,
    )

    signal["smart_signal"] = {
        "enabled": True,
        "source": selection.get("source", "runtime_policy"),
        "profile": profile,
        "promoted": selection.get("promoted", True),
        "promotion_status": selection.get("promotion_status", "promoted"),
        "rejection_reasons": selection.get("rejection_reasons", []) or [],
        "overrides": overrides,
        "stored_overrides": stored_overrides,
        "live_overrides": live_overrides,
        "evidence": evidence,
        "selection": selection.get("selection", {}),
        "matrix_decision_guard": signal.get("matrix_decision_guard", {}),
    }

    out_path = _write_smart_signal(
        smart_cfg,
        ticker,
        signal,
    )

    return signal, smart_cfg, out_path


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"

    if isinstance(value, dict):
        return ", ".join(f"{key}={_short_value(item)}" for key, item in value.items())

    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    return str(value)


def _print_mapping_compact(
    title: str,
    data: dict[str, Any],
    keys: tuple[str, ...] | None = None,
) -> None:
    if not data:
        return

    print(title)

    selected_keys = keys if keys is not None else tuple(data.keys())

    for key in selected_keys:
        if key not in data:
            continue

        print(f"  {key}: {_short_value(data[key])}")


def _print_smart_header(
    ticker: str,
    signal: dict[str, Any],
    out_path: Path,
) -> None:
    smart = signal.get("smart_signal", {}) or {}
    evidence = smart.get("evidence", {}) or {}
    live_overrides = smart.get("live_overrides", {}) or {}
    overrides = smart.get("overrides", {}) or {}
    guard_result = signal.get("matrix_decision_guard", {}) or {}

    print()
    print(f"SMART SIGNAL | {ticker}")
    print("-" * 96)
    print(
        "Profile: "
        f"{smart.get('profile')} | "
        "Source: "
        f"{smart.get('source')} | "
        "Status: "
        f"{smart.get('promotion_status', 'n/a')}"
    )

    _print_mapping_compact(
        "Matrix:",
        evidence,
        keys=(
            "decision",
            "score",
            "profit_factor",
            "trades",
            "drawdown_pct",
            "return_pct",
            "hit_pct",
            "exposure_pct",
        ),
    )

    rejection_reasons = smart.get("rejection_reasons", []) or []
    if rejection_reasons:
        print("Promotion:")
        print(f"  rejected: {_short_value(rejection_reasons)}")

    _print_mapping_compact(
        "Live YAML overrides:",
        live_overrides,
    )

    _print_mapping_compact(
        "Effective overrides:",
        overrides,
    )

    if guard_result:
        _print_mapping_compact(
            "Matrix guard:",
            guard_result,
            keys=(
                "matrix_decision",
                "max_signal",
                "blocked",
                "blocked_signal",
                "reason",
            ),
        )

    print(f"Artifact: {out_path}")
    print("-" * 96)
    print()


def _smart(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(cfg, args, required=True)

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


def _policy_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("policy", {}) or {}


def _smart_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("smart_signal", {}) or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _clip(value: Any, size: int) -> str:
    text = str(value)

    if len(text) <= size:
        return text

    return text[: max(0, size - 3)] + "..."


def _smart_rank_score_from_signal(signal: dict[str, Any]) -> tuple[int, float, float, float]:
    policy = _policy_dict(signal)
    smart = _smart_dict(signal)
    evidence = smart.get("evidence", {}) or {}
    guard = smart.get("matrix_decision_guard", {}) or {}

    label = str(policy.get("label", "NEUTRAL")).upper()

    label_score = {
        "STRONG BUY": 5,
        "BUY": 4,
        "NEUTRAL": 2,
        "SELL": 1,
        "STRONG SELL": 1,
        "REJECTED": -1,
        "NO_MATRIX": -2,
        "ERROR": -5,
    }.get(label, 0)

    if bool(guard.get("blocked", False)):
        label_score -= 2

    matrix_score = _safe_float(evidence.get("score", 0.0))
    profit_factor = _safe_float(evidence.get("profit_factor", 0.0))
    rr = _safe_float(policy.get("risk_reward_ratio", 0.0))

    return (
        label_score,
        matrix_score,
        profit_factor,
        rr,
    )


def _final_blocker(
    policy: dict[str, Any],
    guard: dict[str, Any],
    matrix_decision: str,
    smart: dict[str, Any],
) -> str:
    rejection_reasons = smart.get("rejection_reasons", []) or []
    promotion_status = str(smart.get("promotion_status", ""))

    if promotion_status and promotion_status != "promoted":
        if rejection_reasons:
            return "; ".join(str(item) for item in rejection_reasons[:2])
        return promotion_status

    if guard.get("reason"):
        return str(guard.get("reason"))

    reasons = [str(item) for item in (policy.get("reasons", []) or []) if str(item).strip()]

    rr_reasons = [reason for reason in reasons if "R/R" in reason or "blocked" in reason]

    if rr_reasons:
        return rr_reasons[0]

    if matrix_decision and matrix_decision not in {"APPROVE", "N/A", "NA", "NONE"}:
        return f"Matrix decision is {matrix_decision}"

    if reasons:
        return reasons[0]

    return "none"


def _smart_rank_row(signal: dict[str, Any]) -> dict[str, Any]:
    policy = _policy_dict(signal)
    smart = _smart_dict(signal)
    evidence = smart.get("evidence", {}) or {}
    guard = smart.get("matrix_decision_guard", {}) or {}

    matrix_decision = str(evidence.get("decision", "n/a")).upper()

    return {
        "ticker": signal.get("ticker", evidence.get("ticker", "N/A")),
        "signal": str(policy.get("label", signal.get("label", "NEUTRAL"))),
        "profile": str(smart.get("profile", "n/a")),
        "matrix": str(evidence.get("decision", "n/a")),
        "pf": evidence.get("profit_factor", "n/a"),
        "trades": evidence.get("trades", "n/a"),
        "rr": policy.get("risk_reward_ratio", 0.0),
        "guard": "BLOCK" if bool(guard.get("blocked", False)) else "OK",
        "blocker": _final_blocker(
            policy,
            guard,
            matrix_decision,
            smart,
        ),
        "_sort": _smart_rank_score_from_signal(signal),
    }


def _no_matrix_rank_row(ticker: str, source: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "signal": "NO_MATRIX",
        "profile": "n/a",
        "matrix": "n/a",
        "pf": "n/a",
        "trades": "n/a",
        "rr": 0.0,
        "guard": "SKIP",
        "blocker": f"not found in policy_matrix runtime; source={source}",
        "_sort": (-8, 0.0, 0.0, 0.0),
    }


def _rejected_matrix_rank_row(
    ticker: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    evidence = selection.get("evidence", {}) or {}
    rejection_reasons = selection.get("rejection_reasons", []) or []

    blocker = (
        "; ".join(str(item) for item in rejection_reasons[:2])
        if rejection_reasons
        else str(selection.get("promotion_status", "rejected_by_constraints"))
    )

    return {
        "ticker": ticker,
        "signal": "REJECTED",
        "profile": str(selection.get("profile", "n/a")),
        "matrix": str(evidence.get("decision", "n/a")),
        "pf": evidence.get("profit_factor", "n/a"),
        "trades": evidence.get("trades", "n/a"),
        "rr": 0.0,
        "guard": "BLOCK",
        "blocker": blocker,
        "_sort": (
            -1,
            _safe_float(evidence.get("score", 0.0)),
            _safe_float(evidence.get("profit_factor", 0.0)),
            0.0,
        ),
    }


def _error_rank_row(ticker: str, exc: Exception) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "signal": "ERROR",
        "profile": "n/a",
        "matrix": "n/a",
        "pf": "n/a",
        "trades": "n/a",
        "rr": 0.0,
        "guard": "ERROR",
        "blocker": str(exc),
        "_sort": (-9, 0.0, 0.0, 0.0),
    }


def _print_smart_rank(
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> None:
    width = 112

    print()
    print("SMART RANK")
    print("-" * width)
    print(
        f"{'TICKER':<11} "
        f"{'SIGNAL':<10} "
        f"{'PROFILE':<9} "
        f"{'MATRIX':<11} "
        f"{'PF':>6} "
        f"{'TRD':>5} "
        f"{'RR':>5} "
        f"{'GUARD':<6} "
        f"BLOCKER"
    )
    print("-" * width)

    for row in rows[:limit]:
        print(
            f"{_clip(row['ticker'], 11):<11} "
            f"{_clip(row['signal'], 10):<10} "
            f"{_clip(row['profile'], 9):<9} "
            f"{_clip(row['matrix'], 11):<11} "
            f"{_fmt_num(row['pf']):>6} "
            f"{str(row['trades']):>5} "
            f"{_fmt_num(row['rr']):>5} "
            f"{_clip(row['guard'], 6):<6} "
            f"{_clip(row['blocker'], 45)}"
        )

    print("-" * width)
    print(f"Rows: {min(len(rows), limit)} of {len(rows)}")
    print()


def _smart_rank(cfg: dict, args: argparse.Namespace) -> None:
    tickers = resolve_cli_tickers(
        cfg,
        args,
        required=True,
    )

    limit = int(getattr(args, "rank_limit", 40) or 40)

    if limit > 0:
        tickers = tickers[:limit]

    rows: list[dict[str, Any]] = []

    for index, ticker in enumerate(tickers, start=1):
        try:
            selection = resolve_policy_selection(
                str(ticker),
                fallback=None,
            )

            source = str(selection.get("source", "fallback"))

            if not _selection_is_policy_matrix(selection):
                rows.append(
                    _no_matrix_rank_row(
                        str(ticker),
                        source,
                    )
                )

            elif not _selection_is_promoted(selection):
                rows.append(
                    _rejected_matrix_rank_row(
                        str(ticker),
                        selection,
                    )
                )

            else:
                signal, _, _ = _build_smart_signal(
                    cfg,
                    args,
                    ticker,
                )

                rows.append(_smart_rank_row(signal))

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            rows.append(
                _error_rank_row(
                    str(ticker),
                    exc,
                )
            )

        if index % 5 == 0:
            print(
                f"processed {index}/{len(tickers)}...",
                flush=True,
            )

    rows.sort(
        key=lambda row: row.get("_sort", (-9, 0.0, 0.0, 0.0)),
        reverse=True,
    )

    _print_smart_rank(
        rows,
        limit=limit,
    )


def _report(cfg: dict, args: argparse.Namespace) -> None:
    for ticker in resolve_cli_tickers(cfg, args, required=True):
        path = latest_signal_path(cfg, ticker)

        if not path.exists() or bool(getattr(args, "refresh", False)):
            signal = make_signal(
                cfg,
                ticker,
                update=bool(getattr(args, "update", False)),
            )
        else:
            signal = read_json(path)

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
