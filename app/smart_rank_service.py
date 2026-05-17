from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .runtime_policy import (
    POLICY_PATH,
    default_policy_profile,
    load_runtime_policy,
    resolve_policy_selection,
    resolve_policy_selection_from_data,
)
from .runtime_policy_config import CONFIG_PATH
from .smart_signal_service import build_smart_signal


@dataclass(frozen=True)
class SmartRankDependencies:
    resolve_policy_selection: Callable[..., dict[str, Any]]
    build_smart_signal: Callable[..., tuple[dict[str, Any], dict[str, Any], Path]]


def default_smart_rank_dependencies() -> SmartRankDependencies:
    return SmartRankDependencies(
        resolve_policy_selection=resolve_policy_selection,
        build_smart_signal=build_smart_signal,
    )


def rank_limit_from_args(
    args: argparse.Namespace,
    *,
    default: int = 40,
) -> int:
    try:
        return int(getattr(args, "rank_limit", default) or default)
    except Exception:
        return default


def selection_is_policy_matrix(selection: dict[str, Any]) -> bool:
    return str(selection.get("source", "")).lower() == "policy_matrix"


def selection_is_promoted(selection: dict[str, Any]) -> bool:
    return bool(selection.get("promoted", True))


def _policy_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("policy", {}) or {}


def _smart_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("smart_signal", {}) or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_actionable_signal(value: Any) -> bool:
    label = str(value).upper()
    return label not in {
        "",
        "NEUTRAL",
        "WAIT",
        "WATCH",
        "AVOID",
        "REJECTED",
        "NO_MATRIX",
        "ERROR",
    }


def _runtime_counts(runtime_data: dict[str, Any]) -> tuple[int, int, int]:
    assets = runtime_data.get("assets", {}) or {}
    total = int(runtime_data.get("assets_total", len(assets)) or len(assets))
    promoted = int(
        runtime_data.get(
            "assets_promoted",
            sum(1 for item in assets.values() if bool(item.get("promoted", False))),
        )
        or 0
    )
    rejected = int(
        runtime_data.get(
            "assets_rejected",
            sum(1 for item in assets.values() if not bool(item.get("promoted", False))),
        )
        or 0
    )
    return total, promoted, rejected


def _runtime_mtime_label() -> str:
    if not POLICY_PATH.exists():
        return "missing"

    timestamp = datetime.fromtimestamp(POLICY_PATH.stat().st_mtime)
    return timestamp.strftime("%Y-%m-%d %H:%M")


def smart_rank_fingerprint(
    args: argparse.Namespace,
    *,
    total_tickers: int,
    processed_tickers: int,
    runtime_data: dict[str, Any],
) -> str:
    universe = str(getattr(args, "asset_list", None) or "explicit")
    runtime_total, _promoted, _rejected = _runtime_counts(runtime_data)
    runtime_assets = runtime_total if runtime_total > 0 else "n/a"

    return (
        f"Policy: {CONFIG_PATH.as_posix()} | "
        f"Runtime: {POLICY_PATH.as_posix()} ({_runtime_mtime_label()}) | "
        f"Universe: {universe} | "
        f"Assets: {processed_tickers}/{total_tickers} | "
        f"Runtime assets: {runtime_assets}"
    )


def smart_rank_preflight_warnings(
    *,
    runtime_data: dict[str, Any],
    tickers: list[str],
) -> list[str]:
    warnings: list[str] = []

    if not CONFIG_PATH.exists():
        warnings.append(f"{CONFIG_PATH.as_posix()} not found.")

    if not POLICY_PATH.exists():
        warnings.append(
            f"{POLICY_PATH.as_posix()} not found. Run promote_policy before smart rank."
        )

    assets = runtime_data.get("assets", {}) or {}

    if not assets:
        warnings.append("runtime_policy has 0 assets. Check promote_policy output.")
        return warnings

    total, promoted, rejected = _runtime_counts(runtime_data)

    if total != len(assets):
        warnings.append(
            f"runtime_policy assets_total={total} but assets map has {len(assets)} entries."
        )

    if promoted <= 0:
        warnings.append("runtime_policy has 0 promoted assets. Check promotion constraints.")

    if rejected <= 0:
        warnings.append("runtime_policy has 0 rejected assets. Check promote_policy output.")

    if tickers and not any(str(ticker) in assets for ticker in tickers):
        warnings.append("requested universe has no overlap with policy_matrix runtime.")

    return warnings


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


def smart_rank_row(signal: dict[str, Any]) -> dict[str, Any]:
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


def no_matrix_rank_row(ticker: str, source: str) -> dict[str, Any]:
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


def rejected_matrix_rank_row(
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


def error_rank_row(ticker: str, exc: Exception) -> dict[str, Any]:
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


def _selection_resolver(
    deps: SmartRankDependencies,
    runtime_data: dict[str, Any] | None,
) -> Callable[[str], dict[str, Any]]:
    if deps.resolve_policy_selection is resolve_policy_selection:
        data = runtime_data if runtime_data is not None else load_runtime_policy()
        fallback = default_policy_profile()

        def resolve_from_loaded(ticker: str) -> dict[str, Any]:
            return resolve_policy_selection_from_data(
                data,
                ticker,
                fallback=fallback,
            )

        return resolve_from_loaded

    def resolve_from_dependency(ticker: str) -> dict[str, Any]:
        return deps.resolve_policy_selection(
            ticker,
            fallback=None,
        )

    return resolve_from_dependency


def _operational_priority(row: dict[str, Any]) -> int:
    signal = str(row.get("signal", "")).upper()
    guard = str(row.get("guard", "")).upper()

    if guard == "ERROR" or signal == "ERROR":
        return 0

    if guard == "SKIP" or signal == "NO_MATRIX":
        return 1

    if signal == "REJECTED":
        return 2

    if guard == "BLOCK":
        return 3

    if guard == "OK":
        return 4

    return 0


def build_smart_rank_rows(
    cfg: dict,
    args: argparse.Namespace,
    tickers: list[str],
    *,
    deps: SmartRankDependencies | None = None,
    runtime_data: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    deps = deps or default_smart_rank_dependencies()
    selected_tickers = list(tickers)
    limit = rank_limit_from_args(args)

    if limit > 0:
        selected_tickers = selected_tickers[:limit]

    rows: list[dict[str, Any]] = []
    resolve_selection = _selection_resolver(
        deps,
        runtime_data,
    )

    for index, ticker in enumerate(selected_tickers, start=1):
        try:
            selection = resolve_selection(str(ticker))

            source = str(selection.get("source", "fallback"))

            if not selection_is_policy_matrix(selection):
                rows.append(
                    no_matrix_rank_row(
                        str(ticker),
                        source,
                    )
                )

            elif not selection_is_promoted(selection):
                rows.append(
                    rejected_matrix_rank_row(
                        str(ticker),
                        selection,
                    )
                )

            else:
                signal, _, _ = deps.build_smart_signal(
                    cfg,
                    args,
                    ticker,
                )

                rows.append(smart_rank_row(signal))

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            rows.append(
                error_rank_row(
                    str(ticker),
                    exc,
                )
            )

        if show_progress and index % 5 == 0:
            print(
                f"processed {index}/{len(selected_tickers)}...",
                flush=True,
            )

    rows.sort(
        key=lambda row: (
            _operational_priority(row),
            *row.get("_sort", (-9, 0.0, 0.0, 0.0)),
        ),
        reverse=True,
    )

    return rows


def _normalise_blocker(value: Any) -> str:
    text = str(value or "").strip()

    if not text or text == "none":
        return "none"

    if "trades " in text and " < " in text:
        return "trades < minimum"

    if "Matrix decision is" in text:
        return text

    if "not found in policy_matrix runtime" in text:
        return "not found in runtime"

    return text


def smart_rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actionable = [
        str(row.get("ticker"))
        for row in rows
        if str(row.get("guard", "")).upper() == "OK"
        and _is_actionable_signal(row.get("signal"))
    ]

    blockers: dict[str, int] = {}

    for row in rows:
        if str(row.get("guard", "")).upper() == "OK":
            continue

        blocker = _normalise_blocker(row.get("blocker"))
        blockers[blocker] = blockers.get(blocker, 0) + 1

    main_blocker = "none"

    if blockers:
        main_blocker = sorted(
            blockers.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[0][0]

    return {
        "top_actionable": actionable[:3],
        "main_blocker": main_blocker,
    }
