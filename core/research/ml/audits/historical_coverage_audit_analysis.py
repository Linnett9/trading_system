from __future__ import annotations

from datetime import timedelta
from typing import Any

from core.research.ml.audits.historical_coverage_audit_math import _date


def _required_history(
    *,
    latest: str | None,
    target_count: int,
    label_window_days: int,
) -> dict[str, Any]:
    latest_date = _date(latest)
    days = max(0, (target_count - 1) * max(1, label_window_days))
    required_start = latest_date - timedelta(days=days) if latest_date else None
    return {
        "target_independent_periods": target_count,
        "estimated_required_calendar_days": days,
        "estimated_required_start_date": (
            required_start.date().isoformat() if required_start else None
        ),
    }
def _bottleneck(
    *,
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    canonical: dict[str, Any],
    minimum: int,
) -> dict[str, Any]:
    raw_count = int(canonical.get("raw_independent_periods_exact") or 0)
    reasons = []
    if raw_count < minimum:
        reasons.append("too_few_canonical_independent_periods")
    if _date(raw.get("earliest_date")) and _date(source_predictions.get("earliest_date")):
        if _date(raw["earliest_date"]) < _date(source_predictions["earliest_date"]):
            reasons.append("prediction_artifacts_start_after_price_history")
    if _date(source_predictions.get("earliest_date")) and _date(meta_predictions.get("earliest_date")):
        if _date(source_predictions["earliest_date"]) < _date(meta_predictions["earliest_date"]):
            reasons.append("meta_or_canonical_artifacts_start_after_source_predictions")
    if not adjusted.get("available"):
        reasons.append("adjusted_prices_missing")
    limiting_layer = "canonical_or_meta_artifacts"
    if reasons == ["too_few_canonical_independent_periods"]:
        limiting_layer = "rebalance_cadence_or_label_window"
    if "meta_or_canonical_artifacts_start_after_source_predictions" in reasons:
        limiting_layer = "meta_or_canonical_artifacts"
    elif "prediction_artifacts_start_after_price_history" in reasons:
        limiting_layer = "prediction_artifacts"
    if not raw.get("available") or not adjusted.get("available"):
        limiting_layer = "price_data"
    return {
        "limiting_layer": limiting_layer,
        "reasons": reasons,
        "current_raw_independent_periods": raw_count,
        "minimum_independent_periods": minimum,
    }
def _blockers(
    *,
    bottleneck: dict[str, Any],
    canonical: dict[str, Any],
    adjusted_replay: dict[str, Any],
    minimum: int,
) -> list[str]:
    blockers = list(bottleneck.get("reasons", []))
    exact_adjusted = adjusted_replay.get("exact_champion_replay", {})
    if int(exact_adjusted.get("valid_adjusted_independent_period_count") or 0) < minimum:
        blockers.append("too_few_valid_adjusted_independent_periods")
    if int(canonical.get("diagnostic_periods_exact") or 0) > int(
        canonical.get("raw_independent_periods_exact") or 0
    ):
        blockers.append("overlapping_label_windows_reduce_independent_count")
    return sorted(set(blockers))
def _recommendations(
    *,
    bottleneck: dict[str, Any],
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    needed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    full_rerun = bottleneck.get("limiting_layer") in {
        "prediction_artifacts",
        "meta_or_canonical_artifacts",
    }
    return {
        "extend_benchmark_start_date": (
            "raw_adjusted_data_support_earlier_history"
            if raw.get("earliest_date") and adjusted.get("earliest_date")
            else "price_data_unavailable"
        ),
        "regenerate_base_artifacts": bool(full_rerun),
        "regenerate_reason": bottleneck.get("limiting_layer"),
        "minimum_history_needed": needed,
        "full_model_rerun_required": bool(full_rerun),
        "price_data_supports_earlier_than_predictions": bool(
            _date(raw.get("earliest_date"))
            and _date(source_predictions.get("earliest_date"))
            and _date(raw["earliest_date"]) < _date(source_predictions["earliest_date"])
        ),
        "source_predictions_support_earlier_than_meta": bool(
            _date(source_predictions.get("earliest_date"))
            and _date(meta_predictions.get("earliest_date"))
            and _date(source_predictions["earliest_date"])
            < _date(meta_predictions["earliest_date"])
        ),
    }
def _rows(
    raw: dict[str, Any],
    adjusted: dict[str, Any],
    source_predictions: dict[str, Any],
    meta_predictions: dict[str, Any],
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _row("raw_stooq_parquet", raw),
        _row("adjusted_yahoo_reference", adjusted),
        _row("source_prediction_artifacts", source_predictions),
        _row("meta_auxiliary_predictions", meta_predictions),
        {
            "layer": "canonical_replay",
            "earliest_date": canonical.get("earliest_canonical_replay_date"),
            "latest_date": canonical.get("latest_canonical_replay_date"),
            "rebalance_date_count": canonical.get("rebalance_date_count"),
            "independent_period_count": canonical.get("raw_independent_periods_exact"),
        },
    ]
def _row(layer: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": layer,
        "earliest_date": summary.get("earliest_date"),
        "latest_date": summary.get("latest_date"),
        "rebalance_date_count": summary.get("unique_rebalance_dates"),
        "independent_period_count": None,
    }
