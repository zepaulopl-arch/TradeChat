from __future__ import annotations

import argparse
import math
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


ACTIONABLE_DECISIONS = {"APPROVE"}
WATCH_DECISIONS = {"OBSERVE", "INCONCLUSIVE", "WATCH"}
REJECT_DECISIONS = {"REJECT", "DENY", "AVOID"}


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


def selection_is_ineligible(selection: dict[str, Any]) -> bool:
    status = str(selection.get("promotion_status", "")).lower()
    return bool(selection.get("ineligible_data", False)) or status in {
        "ineligible_data",
        "skip_data",
    }


def _policy_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("policy", {}) or {}


def _smart_dict(signal: dict[str, Any]) -> dict[str, Any]:
    return signal.get("smart_signal", {}) or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default

    if math.isnan(number):
        return default

    return number


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"n/a", "na", "none", "nan"}:
        return None

    if text.lower() in {"inf", "+inf", "infinity"}:
        return math.inf

    try:
        number = float(text)
    except Exception:
        return None

    if math.isnan(number):
        return None

    return number


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in mapping:
            continue

        value = mapping.get(key)

        if value is None or value == "":
            continue

        return value

    return None


def _matrix_decision(evidence: dict[str, Any] | None) -> str:
    return str((evidence or {}).get("decision", "n/a") or "n/a").upper()


def _matrix_rr(evidence: dict[str, Any] | None) -> float | None:
    value = _first_present(
        evidence or {},
        (
            "matrix_rr",
            "risk_reward_ratio",
            "rr",
            "reward_risk_ratio",
            "risk_reward",
        ),
    )
    return _optional_float(value)


def _signal_rr(policy: dict[str, Any], guard: dict[str, Any]) -> float | None:
    if bool(guard.get("blocked", False)):
        return None

    if "risk_reward_ratio" not in policy:
        return None

    return _optional_float(policy.get("risk_reward_ratio"))


def _is_operational_signal(value: Any) -> bool:
    label = str(value).upper()
    return label not in {
        "",
        "REJECTED",
        "INELIGIBLE_DATA",
        "ERROR",
    }


def _runtime_counts(runtime_data: dict[str, Any]) -> tuple[int, int, int, int]:
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
    ineligible = int(
        runtime_data.get(
            "assets_ineligible",
            sum(1 for item in assets.values() if bool(item.get("ineligible_data", False))),
        )
        or 0
    )
    return total, promoted, rejected, ineligible


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
    runtime_total, _promoted, _rejected, _ineligible = _runtime_counts(runtime_data)
    runtime_assets = runtime_total if runtime_total > 0 else "n/a"

    return (
        f"Policy: {CONFIG_PATH.as_posix()} | "
        f"Runtime: {POLICY_PATH.as_posix()} ({_runtime_mtime_label()}) | "
        f"Universe: {universe} | "
        f"Processed: {processed_tickers}/{total_tickers} | "
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

    total, promoted, rejected, _ineligible = _runtime_counts(runtime_data)

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


def _base_row(
    *,
    ticker: str,
    action: str,
    signal: str,
    profile: Any,
    matrix: Any,
    pf: Any,
    trades: Any,
    matrix_rr: float | None = None,
    signal_rr: float | None = None,
    guard: str,
    blocker: str,
    score: Any = 0.0,
) -> dict[str, Any]:
    action = str(action).upper()
    priority = {
        "ACTIONABLE": 60,
        "WATCH": 50,
        "BLOCKED": 40,
        "REJECTED": 30,
        "INELIGIBLE": 20,
        "ERROR": 0,
    }.get(action, 0)

    rr_sort = signal_rr if signal_rr is not None else matrix_rr
    rr = signal_rr if signal_rr is not None else matrix_rr

    return {
        "ticker": ticker,
        "action": action,
        "signal": signal,
        "profile": str(profile),
        "matrix": str(matrix),
        "pf": pf,
        "trades": trades,
        "rr": rr,
        "matrix_rr": matrix_rr,
        "signal_rr": signal_rr,
        "guard": guard,
        "blocker": blocker or "none",
        "reason": blocker or "none",
        "_sort": (
            priority,
            _safe_float(score),
            _safe_float(pf),
            rr_sort if rr_sort is not None else -999.0,
        ),
    }


def _final_blocker(
    policy: dict[str, Any],
    guard: dict[str, Any],
    matrix_decision: str,
    smart: dict[str, Any],
) -> str:
    rejection_reasons = smart.get("rejection_reasons", []) or []
    promotion_status = str(smart.get("promotion_status", ""))

    if promotion_status and promotion_status not in {"promoted", "legacy"}:
        if rejection_reasons:
            return "; ".join(str(item) for item in rejection_reasons[:2])
        return promotion_status

    if guard.get("reason"):
        return str(guard.get("reason"))

    reasons = [str(item) for item in (policy.get("reasons", []) or []) if str(item).strip()]
    rr_reasons = [reason for reason in reasons if "R/R" in reason or "blocked" in reason]

    if rr_reasons:
        return rr_reasons[0]

    if matrix_decision in WATCH_DECISIONS | REJECT_DECISIONS:
        return f"Matrix decision is {matrix_decision}"

    if reasons:
        return reasons[0]

    return "none"


def _action_for_promoted_signal(
    *,
    signal_label: str,
    matrix_decision: str,
    guard: dict[str, Any],
    selection: dict[str, Any],
) -> str:
    if bool(guard.get("blocked", False)):
        if matrix_decision in WATCH_DECISIONS:
            return "WATCH"
        return "BLOCKED"

    if matrix_decision in WATCH_DECISIONS:
        return "WATCH"

    if matrix_decision in REJECT_DECISIONS:
        return "BLOCKED"

    actionable_candidate = bool(selection.get("actionable_candidate", True))

    if (
        actionable_candidate
        and matrix_decision in ACTIONABLE_DECISIONS
        and _is_operational_signal(signal_label)
    ):
        return "ACTIONABLE"

    return "WATCH"


def smart_rank_row(
    signal: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection = selection or {}
    policy = _policy_dict(signal)
    smart = _smart_dict(signal)
    evidence = smart.get("evidence", {}) or selection.get("evidence", {}) or {}
    guard = smart.get("matrix_decision_guard", {}) or signal.get("matrix_decision_guard", {}) or {}

    matrix_decision = _matrix_decision(evidence)
    signal_label = str(policy.get("label", signal.get("label", "NEUTRAL"))).upper()
    display_guard = "BLOCK" if bool(guard.get("blocked", False)) else "OK"
    action = _action_for_promoted_signal(
        signal_label=signal_label,
        matrix_decision=matrix_decision,
        guard=guard,
        selection=selection or smart,
    )

    if action in {"WATCH", "BLOCKED"} and matrix_decision not in ACTIONABLE_DECISIONS:
        display_guard = "BLOCK"

    return _base_row(
        ticker=str(signal.get("ticker", evidence.get("ticker", "N/A"))),
        action=action,
        signal=signal_label,
        profile=smart.get("profile", selection.get("profile", "n/a")),
        matrix=evidence.get("decision", "n/a"),
        pf=evidence.get("profit_factor", "n/a"),
        trades=evidence.get("trades", "n/a"),
        matrix_rr=_matrix_rr(evidence),
        signal_rr=_signal_rr(policy, guard),
        guard=display_guard,
        blocker=_final_blocker(
            policy,
            guard,
            matrix_decision,
            smart,
        ),
        score=evidence.get("score", 0.0),
    )


def missing_matrix_rank_row(ticker: str, source: str) -> dict[str, Any]:
    return _base_row(
        ticker=ticker,
        action="ERROR",
        signal="ERROR",
        profile="n/a",
        matrix="n/a",
        pf="n/a",
        trades="n/a",
        guard="ERROR",
        blocker=f"missing Matrix runtime row; source={source}",
    )


def ineligible_rank_row(
    ticker: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    evidence = selection.get("evidence", {}) or {}
    reasons = selection.get("rejection_reasons", []) or []
    blocker = (
        str(selection.get("blocker"))
        if selection.get("blocker")
        else "; ".join(str(item) for item in reasons[:2]) if reasons else "insufficient history"
    )

    return _base_row(
        ticker=ticker,
        action="INELIGIBLE",
        signal="INELIGIBLE_DATA",
        profile=selection.get("profile", "n/a"),
        matrix=evidence.get("decision", "INELIGIBLE_DATA"),
        pf=evidence.get("profit_factor", "n/a"),
        trades=evidence.get("trades", "n/a"),
        matrix_rr=_matrix_rr(evidence),
        guard="SKIP",
        blocker=blocker,
        score=evidence.get("score", 0.0),
    )


def rejected_matrix_rank_row(
    ticker: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    evidence = selection.get("evidence", {}) or {}
    rejection_reasons = selection.get("rejection_reasons", []) or []

    blocker = (
        str(selection.get("blocker"))
        if selection.get("blocker")
        else "; ".join(str(item) for item in rejection_reasons[:2])
        if rejection_reasons
        else str(selection.get("promotion_status", "rejected_by_constraints"))
    )

    return _base_row(
        ticker=ticker,
        action="REJECTED",
        signal="REJECTED",
        profile=selection.get("profile", "n/a"),
        matrix=evidence.get("decision", "n/a"),
        pf=evidence.get("profit_factor", "n/a"),
        trades=evidence.get("trades", "n/a"),
        matrix_rr=_matrix_rr(evidence),
        guard="BLOCK",
        blocker=blocker,
        score=evidence.get("score", 0.0),
    )


def error_rank_row(ticker: str, exc: Exception) -> dict[str, Any]:
    return _base_row(
        ticker=ticker,
        action="ERROR",
        signal="ERROR",
        profile="n/a",
        matrix="n/a",
        pf="n/a",
        trades="n/a",
        guard="ERROR",
        blocker=str(exc),
    )


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


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = row.get("_sort", (0.0, 0.0, 0.0, 0.0))
    return tuple(float(item) if isinstance(item, (int, float)) else _safe_float(item) for item in raw)  # type: ignore[return-value]


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

    rows: list[dict[str, Any]] = []
    resolve_selection = _selection_resolver(
        deps,
        runtime_data,
    )

    for index, ticker in enumerate(selected_tickers, start=1):
        try:
            selection = resolve_selection(str(ticker))
            source = str(selection.get("source", "fallback"))

            if selection_is_ineligible(selection):
                rows.append(
                    ineligible_rank_row(
                        str(ticker),
                        selection,
                    )
                )

            elif not selection_is_policy_matrix(selection):
                rows.append(
                    missing_matrix_rank_row(
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

                rows.append(
                    smart_rank_row(
                        signal,
                        selection,
                    )
                )

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
        key=_row_sort_key,
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

    if "missing Matrix runtime row" in text:
        return "missing Matrix runtime row"

    if "insufficient history" in text.lower():
        return "insufficient history"

    return text


def smart_rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = {
        "ACTIONABLE": 0,
        "WATCH": 0,
        "BLOCKED": 0,
        "REJECTED": 0,
        "INELIGIBLE": 0,
        "ERROR": 0,
    }

    for row in rows:
        action = str(row.get("action", "")).upper()
        if action in action_counts:
            action_counts[action] += 1

    actionable = [
        str(row.get("ticker"))
        for row in rows
        if str(row.get("action", "")).upper() == "ACTIONABLE"
    ]

    blockers: dict[str, int] = {}

    for row in rows:
        if str(row.get("action", "")).upper() == "ACTIONABLE":
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

    no_actionable_reasons = [
        (
            f"{action_counts['REJECTED']} rejected by constraints, "
            f"mainly {main_blocker if action_counts['REJECTED'] else 'n/a'}"
        ),
        (
            f"{action_counts['WATCH'] + action_counts['BLOCKED']} blocked by Matrix guard, "
            "mainly matrix_decision=REJECT/OBSERVE"
        ),
        f"{action_counts['INELIGIBLE']} ineligible by data quality",
        f"{action_counts['ERROR']} errors",
    ]

    return {
        "top_actionable": actionable[:3],
        "main_blocker": main_blocker,
        "action_counts": action_counts,
        "no_actionable_reasons": no_actionable_reasons,
    }
