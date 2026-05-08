from __future__ import annotations

from typing import Any


def feature_family(feature: str) -> str:
    if feature.startswith("ctx_") or feature.startswith("macro_"):
        return "context"
    if feature.startswith("sent_"):
        return "sentiment"
    if feature.startswith("fund_") or feature in {"pl", "pvp", "roe", "dy"}:
        return "fundamentals"
    return "technical"


_ABBREVIATIONS = [
    ("ctx_benchmark_", "ctx_bmk_"),
    ("ctx_relative_strength_", "ctx_rel_"),
    ("ctx_", "ctx_"),
    ("sentiment_", "sent_"),
    ("sent_", "sent_"),
    ("fundamental_", "fund_"),
    ("fund_", "fund_"),
    ("moving_average", "ma"),
    ("volatility", "vol"),
    ("correlation", "corr"),
    ("alignment", "align"),
    ("relative", "rel"),
    ("strength", "str"),
    ("benchmark", "bmk"),
]


def abbreviate_feature_name(feature: str, max_len: int = 22) -> str:
    """Return a compact feature label for one-line CLI summaries."""
    name = str(feature)
    for old, new in _ABBREVIATIONS:
        name = name.replace(old, new)
    name = name.replace("return", "ret")
    name = name.replace("ratio", "rat")
    name = name.replace("window", "win")
    if len(name) <= max_len:
        return name
    return name[: max(3, max_len - 1)] + "…"


def selected_feature_scores(
    prep_meta: dict[str, Any], features: list[str] | None = None
) -> dict[str, float]:
    sel = (prep_meta or {}).get("selection", {}) or {}
    rel = sel.get("relevance", {}) or {}
    out: dict[str, float] = {}
    source = features or list(rel.keys())
    for feature in source:
        try:
            out[str(feature)] = float(rel.get(feature, 0.0) or 0.0)
        except Exception:
            out[str(feature)] = 0.0
    return out


def top_selected_features(
    prep_meta: dict[str, Any], features: list[str], n: int = 5
) -> list[dict[str, Any]]:
    scores = selected_feature_scores(prep_meta, features)
    ordered = sorted([str(f) for f in features], key=lambda f: scores.get(f, 0.0), reverse=True)
    result: list[dict[str, Any]] = []
    for feature in ordered[: max(0, int(n))]:
        result.append(
            {
                "name": feature,
                "short": abbreviate_feature_name(feature),
                "family": feature_family(feature),
                "score": float(scores.get(feature, 0.0)),
            }
        )
    return result


def feature_family_profile(features: list[str]) -> dict[str, int]:
    profile = {"technical": 0, "context": 0, "fundamentals": 0, "sentiment": 0}
    for feature in features:
        fam = feature_family(str(feature))
        profile[fam] = profile.get(fam, 0) + 1
    return profile
