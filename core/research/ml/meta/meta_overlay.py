from __future__ import annotations

import math
from statistics import mean
from typing import Any

from core.research.ml.metrics.evaluation import classification_metrics
from core.research.ml.overlays.overlay import should_reduce_exposure


_MAX_ABS_PERIOD_RETURN = 5.0


def _threshold_sweep(
    rows: list[dict[str, str]],
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
    reduced_exposures: list[float],
    reduce_when: str,
) -> dict[str, Any]:
    scenarios = []
    for threshold in thresholds:
        for reduced_exposure in reduced_exposures:
            threshold_value = float(threshold)
            predictions = [int(value >= threshold_value) for value in probabilities]
            overlay = _overlay_summary(
                rows,
                probabilities,
                threshold_value,
                float(reduced_exposure),
                reduce_when=reduce_when,
            )
            scenarios.append({
                "decision_threshold": threshold_value,
                "reduced_exposure": float(reduced_exposure),
                "metrics": classification_metrics(labels, predictions),
                "overlay": overlay,
                "finite_sanity_check": _finite_sanity_check(overlay),
            })
    ranked = sorted(
        scenarios,
        key=lambda row: (
            -float(row["metrics"].get("balanced_accuracy") or 0.0),
            -float(row["overlay"].get("return_delta") or 0.0),
        ),
    )
    return {
        "mode": "meta_ensemble_threshold_sweep_research_only",
        "scenarios": scenarios,
        "best": ranked[0] if ranked else None,
        "research_only": True,
        "trading_impact": "none",
    }


def _promotion_gate_report(
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    overlay: dict[str, Any],
    walk_forward: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = walk_forward.get("summary", {})
    thresholds = {
        "min_walk_forward_balanced_accuracy": float(
            config.get("promotion_min_walk_forward_balanced_accuracy", 0.50)
        ),
        "max_brier_score": float(config.get("promotion_max_brier_score", 0.25)),
        "max_expected_calibration_error": float(
            config.get("promotion_max_expected_calibration_error", 0.10)
        ),
        "min_overlay_return_delta": float(
            config.get("promotion_min_overlay_return_delta", 0.0)
        ),
        "min_overlay_sample_count": int(
            config.get("promotion_min_overlay_sample_count", 50)
        ),
        "min_max_drawdown_delta": float(
            config.get("promotion_min_max_drawdown_delta", 0.0)
        ),
    }
    checks = {
        "finite_sanity_check": _finite_sanity_check(overlay),
        "walk_forward_balanced_accuracy": _passes_minimum(
            summary.get("balanced_accuracy"),
            thresholds["min_walk_forward_balanced_accuracy"],
        ),
        "brier_score": _passes_maximum(
            calibration.get("brier_score"),
            thresholds["max_brier_score"],
        ),
        "expected_calibration_error": _passes_maximum(
            calibration.get("expected_calibration_error"),
            thresholds["max_expected_calibration_error"],
        ),
        "overlay_return_delta": _passes_minimum(
            overlay.get("return_delta"),
            thresholds["min_overlay_return_delta"],
        ),
        "overlay_sample_count": _passes_minimum(
            overlay.get("overlay_sample_count"),
            thresholds["min_overlay_sample_count"],
        ),
        "max_drawdown_delta": _passes_minimum(
            overlay.get("max_drawdown_delta"),
            thresholds["min_max_drawdown_delta"],
        ),
    }
    passed = all(item.get("passed") for item in checks.values())
    return {
        "promotion_candidate": passed,
        "checks": checks,
        "thresholds": thresholds,
        "observed": {
            "holdout_balanced_accuracy": metrics.get("balanced_accuracy"),
            "walk_forward_balanced_accuracy": summary.get("balanced_accuracy"),
            "brier_score": calibration.get("brier_score"),
            "expected_calibration_error": calibration.get(
                "expected_calibration_error"
            ),
            "overlay_return_delta": overlay.get("return_delta"),
            "overlay_max_drawdown_improvement": overlay.get("max_drawdown_delta"),
            "turnover": overlay.get("overlay_turnover"),
            "reduced_exposure_days": overlay.get("reduced_exposure_days"),
        },
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }


def _passes_minimum(value: Any, minimum: float) -> dict[str, Any]:
    numeric = _finite_or_none(value)
    return {
        "value": numeric,
        "minimum": minimum,
        "passed": numeric is not None and numeric >= minimum,
    }


def _passes_maximum(value: Any, maximum: float) -> dict[str, Any]:
    numeric = _finite_or_none(value)
    return {
        "value": numeric,
        "maximum": maximum,
        "passed": numeric is not None and numeric <= maximum,
    }


def _finite_sanity_check(payload: dict[str, Any]) -> dict[str, Any]:
    checked_fields = [
        "overlay_baseline_return",
        "overlay_adjusted_return",
        "return_delta",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "max_drawdown_delta",
        "overlay_turnover",
    ]
    invalid = [
        name for name in checked_fields
        if payload.get(name) is not None and _finite_or_none(payload.get(name)) is None
    ]
    return {"passed": not invalid, "invalid_fields": invalid}


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _with_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [{**row, "split": split} for row in rows]


def _overlay_summary(
    rows: list[dict[str, str]],
    probabilities: list[float],
    threshold: float,
    reduced_exposure: float,
    reduce_when: str = "above_or_equal_threshold",
) -> dict[str, float | int]:
    if len(rows) != len(probabilities):
        raise ValueError("Overlay rows and probabilities must have the same length")
    if not math.isfinite(float(threshold)):
        raise ValueError("Overlay threshold must be finite")
    if not math.isfinite(float(reduced_exposure)) or not 0 <= reduced_exposure <= 1:
        raise ValueError("Reduced exposure must be a finite decimal between 0 and 1")

    pairs = _holdout_overlay_pairs(rows, probabilities)
    if not pairs:
        return {
            "overlay_start_date": None,
            "overlay_end_date": None,
            "overlay_sample_count": 0,
            "overlay_evaluated_dates": 0,
            "base_total_return": 0.0,
            "overlay_total_return": 0.0,
            "overlay_baseline_return": 0.0,
            "overlay_adjusted_return": 0.0,
            "return_delta": 0.0,
            "base_compounded_return": 0.0,
            "overlay_compounded_return": 0.0,
            "base_max_drawdown": 0.0,
            "overlay_max_drawdown": 0.0,
            "max_drawdown_delta": 0.0,
            "reduced_exposure_days": 0,
            "overlay_turnover": 0.0,
            "aggregation": "mean_by_rebalance_date_not_compounded",
        }

    by_date: dict[str, list[tuple[dict[str, str], float]]] = {}
    for row, probability in pairs:
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Overlay rows must include rebalance_date or date")
        by_date.setdefault(date, []).append((row, probability))

    date_baseline_returns = []
    date_adjusted_returns = []
    reduced_days = 0
    overlay_turnover = 0.0
    active_reduced = False
    for date in sorted(by_date):
        base_returns = []
        adjusted_returns = []
        date_reduced = False
        for row, probability in by_date[date]:
            _validate_probability(probability)
            base_return = _period_return(row)
            multiplier = (
                reduced_exposure
                if should_reduce_exposure(probability, threshold, reduce_when)
                else 1.0
            )
            date_reduced = date_reduced or multiplier < 1.0
            base_returns.append(base_return)
            adjusted_returns.append(base_return * multiplier)
        date_baseline_returns.append(mean(base_returns))
        date_adjusted_returns.append(mean(adjusted_returns))
        reduced_days += int(date_reduced)
        if date_reduced != active_reduced:
            overlay_turnover += abs((reduced_exposure if date_reduced else 1.0) - (reduced_exposure if active_reduced else 1.0))
            active_reduced = date_reduced

    baseline_return = mean(date_baseline_returns)
    adjusted_return = mean(date_adjusted_returns)
    _validate_finite("overlay_baseline_return", baseline_return)
    _validate_finite("overlay_adjusted_return", adjusted_return)
    return_delta = adjusted_return - baseline_return
    _validate_finite("overlay_return_delta", return_delta)
    base_curve = _equity_curve(date_baseline_returns)
    overlay_curve = _equity_curve(date_adjusted_returns)
    base_max_drawdown = _max_drawdown(base_curve)
    overlay_max_drawdown = _max_drawdown(overlay_curve)
    max_drawdown_delta = overlay_max_drawdown - base_max_drawdown
    for name, value in (
        ("base_compounded_return", base_curve[-1] - 1.0),
        ("overlay_compounded_return", overlay_curve[-1] - 1.0),
        ("base_max_drawdown", base_max_drawdown),
        ("overlay_max_drawdown", overlay_max_drawdown),
        ("max_drawdown_delta", max_drawdown_delta),
    ):
        _validate_finite(name, value)

    return {
        "overlay_start_date": min(by_date),
        "overlay_end_date": max(by_date),
        "overlay_sample_count": len(pairs),
        "overlay_evaluated_dates": len(by_date),
        "base_total_return": baseline_return,
        "overlay_total_return": adjusted_return,
        "overlay_baseline_return": baseline_return,
        "overlay_adjusted_return": adjusted_return,
        "return_delta": return_delta,
        "base_compounded_return": base_curve[-1] - 1.0,
        "overlay_compounded_return": overlay_curve[-1] - 1.0,
        "base_max_drawdown": base_max_drawdown,
        "overlay_max_drawdown": overlay_max_drawdown,
        "max_drawdown_delta": max_drawdown_delta,
        "reduced_exposure_days": reduced_days,
        "overlay_turnover": float(overlay_turnover),
        "aggregation": "mean_by_rebalance_date_not_compounded",
    }


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    values = [equity]
    for value in returns:
        _validate_finite("period return", value)
        if value <= -1.0:
            raise ValueError("period return would zero or invert equity")
        equity *= 1.0 + value
        _validate_finite("overlay equity", equity)
        values.append(equity)
    return values


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        if peak <= 0:
            raise ValueError("baseline equity denominator must be positive")
        drawdown = min(drawdown, (value / peak) - 1.0)
    return drawdown


def _holdout_overlay_pairs(
    rows: list[dict[str, str]],
    probabilities: list[float],
) -> list[tuple[dict[str, str], float]]:
    pairs = list(zip(rows, probabilities))
    has_split = any(row.get("split") for row, _ in pairs)
    if not has_split:
        return pairs
    return [
        (row, probability)
        for row, probability in pairs
        if row.get("split") in {"holdout", "test"}
    ]


def _period_return(row: dict[str, str]) -> float:
    value = float(row.get("champion_return_next_period", 0.0) or 0.0)
    _validate_finite("champion_return_next_period", value)
    if abs(value) > _MAX_ABS_PERIOD_RETURN:
        raise ValueError(
            "champion_return_next_period must be a decimal return, not a percent"
        )
    if value <= -1.0:
        raise ValueError("champion_return_next_period would zero or invert equity")
    return value


def _validate_probability(probability: float) -> None:
    value = float(probability)
    _validate_finite("overlay probability", value)
    if value < 0 or value > 1:
        raise ValueError("Overlay probabilities must be between 0 and 1")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
