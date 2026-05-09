from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ContextPolicy:
    min_valid_count: int = 150
    pass_coverage_pct: float = 90.0
    warn_coverage_pct: float = 70.0
    review_coverage_pct: float = 40.0
    use_min_coverage_pct: float = 70.0


@dataclass(frozen=True)
class ContextDecision:
    ticker: str
    valid_count: int
    total_count: int
    coverage_pct: float
    status: str
    action: str
    reason: str

    @property
    def use_for_features(self) -> bool:
        return self.action == "use"


def load_context_policy(cfg: dict[str, Any] | None = None) -> ContextPolicy:
    raw = ((cfg or {}).get("data", {}) or {}).get("context_policy", {}) or {}
    return ContextPolicy(
        min_valid_count=int(raw.get("min_valid_count", 150)),
        pass_coverage_pct=float(raw.get("pass_coverage_pct", 90.0)),
        warn_coverage_pct=float(raw.get("warn_coverage_pct", 70.0)),
        review_coverage_pct=float(raw.get("review_coverage_pct", 40.0)),
        use_min_coverage_pct=float(raw.get("use_min_coverage_pct", 70.0)),
    )


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(part) / float(total) * 100.0


def classify_context_coverage(
    *,
    ticker: str,
    valid_count: int,
    total_count: int,
    policy: ContextPolicy | None = None,
) -> ContextDecision:
    policy = policy or ContextPolicy()
    coverage_pct = _pct(valid_count, total_count)

    if total_count <= 0:
        return ContextDecision(
            ticker=ticker,
            valid_count=valid_count,
            total_count=total_count,
            coverage_pct=coverage_pct,
            status="drop",
            action="drop",
            reason="asset effective range is empty",
        )

    if valid_count < policy.min_valid_count:
        return ContextDecision(
            ticker=ticker,
            valid_count=valid_count,
            total_count=total_count,
            coverage_pct=coverage_pct,
            status="drop",
            action="drop",
            reason=f"valid_count {valid_count} < {policy.min_valid_count}",
        )

    if coverage_pct >= policy.pass_coverage_pct:
        status = "pass"
    elif coverage_pct >= policy.warn_coverage_pct:
        status = "warn"
    elif coverage_pct >= policy.review_coverage_pct:
        status = "review"
    else:
        status = "drop"

    action = "use" if coverage_pct >= policy.use_min_coverage_pct else "drop"
    if action == "use":
        reason = f"coverage {coverage_pct:.1f}% >= {policy.use_min_coverage_pct:.1f}%"
    else:
        reason = f"coverage {coverage_pct:.1f}% < {policy.use_min_coverage_pct:.1f}%"

    return ContextDecision(
        ticker=ticker,
        valid_count=valid_count,
        total_count=total_count,
        coverage_pct=coverage_pct,
        status=status,
        action=action,
        reason=reason,
    )


def context_coverage_decisions(
    df: pd.DataFrame,
    *,
    asset_column: str,
    context_columns: list[str],
    policy: ContextPolicy | None = None,
) -> list[ContextDecision]:
    policy = policy or ContextPolicy()
    if df.empty or asset_column not in df.columns:
        return [
            classify_context_coverage(ticker=ticker, valid_count=0, total_count=0, policy=policy)
            for ticker in context_columns
        ]

    price = df[asset_column]
    valid_asset = price.notna()
    if not bool(valid_asset.any()):
        effective = df.iloc[0:0]
    else:
        first = price[valid_asset].index.min()
        last = price[valid_asset].index.max()
        effective = df.loc[(df.index >= first) & (df.index <= last)]

    total_count = int(len(effective))
    decisions: list[ContextDecision] = []
    for ticker in context_columns:
        if ticker not in df.columns:
            decisions.append(
                classify_context_coverage(
                    ticker=ticker, valid_count=0, total_count=total_count, policy=policy
                )
            )
            continue
        valid_count = int(effective[ticker].notna().sum()) if total_count else 0
        decisions.append(
            classify_context_coverage(
                ticker=ticker,
                valid_count=valid_count,
                total_count=total_count,
                policy=policy,
            )
        )
    return decisions


def filter_context_columns(
    df: pd.DataFrame,
    *,
    asset_column: str,
    context_columns: list[str],
    policy: ContextPolicy | None = None,
) -> tuple[pd.DataFrame, list[ContextDecision]]:
    decisions = context_coverage_decisions(
        df,
        asset_column=asset_column,
        context_columns=context_columns,
        policy=policy,
    )
    drop_columns = [
        d.ticker for d in decisions if not d.use_for_features and d.ticker in df.columns
    ]
    if not drop_columns:
        return df, decisions
    return df.drop(columns=drop_columns), decisions
