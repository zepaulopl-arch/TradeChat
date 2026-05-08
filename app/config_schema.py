from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    severity: str
    message: str
    value: Any = None


def _issue(path: str, severity: str, message: str, value: Any = None) -> ConfigIssue:
    return ConfigIssue(path=path, severity=severity, message=message, value=value)


def _get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _require_positive(config: dict[str, Any], path: str, issues: list[ConfigIssue]) -> None:
    value = _get(config, path)
    if value is None:
        return
    if not _is_number(value) or float(value) <= 0:
        issues.append(_issue(path, "error", "must be a positive number", value))


def _require_non_negative(config: dict[str, Any], path: str, issues: list[ConfigIssue]) -> None:
    value = _get(config, path)
    if value is None:
        return
    if not _is_number(value) or float(value) < 0:
        issues.append(_issue(path, "error", "must be a non-negative number", value))


def validate_config(config: dict[str, Any]) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    simulation = config.get("simulation", {}) or {}

    _require_positive(config, "simulation.initial_cash", issues)
    _require_positive(config, "simulation.max_positions", issues)
    _require_positive(config, "trading.capital", issues)
    _require_positive(config, "data.min_rows", issues)
    _require_non_negative(config, "batch.train_workers", issues)
    _require_non_negative(config, "batch.diagnose_workers", issues)
    _require_non_negative(config, "simulation.costs.fee_amount", issues)
    _require_non_negative(config, "simulation.costs.slippage_pct", issues)
    _require_non_negative(config, "model.prediction_guards.max_engine_return_abs", issues)
    _require_non_negative(config, "model.prediction_guards.max_final_return_abs", issues)

    mode = str(simulation.get("mode", "replay") or "replay").lower()
    if mode not in {"replay", "walkforward"}:
        issues.append(_issue("simulation.mode", "error", "must be replay or walkforward", mode))
    elif mode == "replay":
        issues.append(
            _issue(
                "simulation.mode",
                "warning",
                "replay is operational sanity, not methodological validation",
                mode,
            )
        )

    fee_mode = _get(config, "simulation.costs.fee_mode")
    if fee_mode is not None:
        raw_fee = str(fee_mode).strip().lower().replace("-", "_")
        if raw_fee not in {
            "order_percent",
            "percent",
            "per_order",
            "order",
            "per_share",
            "share",
            "none",
            "off",
        }:
            issues.append(
                _issue(
                    "simulation.costs.fee_mode",
                    "error",
                    "must be order_percent, per_order, per_share or none",
                    fee_mode,
                )
            )

    slippage = _get(config, "simulation.costs.slippage_pct")
    if slippage is not None and _is_number(slippage) and float(slippage) > 20:
        issues.append(
            _issue(
                "simulation.costs.slippage_pct",
                "warning",
                "slippage_pct is interpreted as a percent, not a decimal fraction",
                slippage,
            )
        )

    confidence = config.get("model", {}).get("confidence", {}) or {}
    for key in ("minimum_when_engines_exist", "maximum_confidence"):
        value = confidence.get(key)
        if value is not None and (not _is_number(value) or not 0 <= float(value) <= 1):
            issues.append(
                _issue(f"model.confidence.{key}", "error", "must be between 0 and 1", value)
            )

    macro_tickers = _get(config, "data.macro_tickers", [])
    if macro_tickers is not None and not isinstance(macro_tickers, list):
        issues.append(_issue("data.macro_tickers", "error", "must be a list", macro_tickers))

    for path in ("app.artifact_dir", "app.data_cache_dir", "features_file"):
        value = _get(config, path)
        if value is not None and not isinstance(value, str):
            issues.append(_issue(path, "error", "must be a string path", value))

    horizons = _get(config, "trading.trade_management.max_hold_days", {}) or {}
    if isinstance(horizons, dict):
        invalid = [key for key in horizons if key not in {"d1", "d5", "d20"}]
        if invalid:
            issues.append(
                _issue(
                    "trading.trade_management.max_hold_days",
                    "error",
                    "unknown horizon key",
                    invalid,
                )
            )

    return issues


def assert_valid_config(config: dict[str, Any]) -> None:
    errors = [issue for issue in validate_config(config) if issue.severity == "error"]
    if not errors:
        return
    rendered = "; ".join(
        f"{issue.path}: {issue.message} (value={issue.value!r})" for issue in errors
    )
    raise ValueError(f"Invalid TradeChat config: {rendered}")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config or {})
    simulation = dict(normalized.get("simulation", {}) or {})
    trading = normalized.get("trading", {}) or {}
    if "initial_cash" not in simulation and "capital" in trading:
        simulation["initial_cash"] = trading["capital"]
    if "max_positions" not in simulation:
        simulation["max_positions"] = 1
    normalized["simulation"] = simulation
    return normalized
