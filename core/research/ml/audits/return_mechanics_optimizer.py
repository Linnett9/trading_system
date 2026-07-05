from __future__ import annotations

from typing import Any

from core.research.ml.allocation.allocation_v2_variants import grid_variant
from core.research.ml.allocation.exposures import _variant_exposures
from core.research.ml.audits.return_mechanics_math import _number


def _load_selected_optimizer_series(
    payload: dict[str, Any],
    optimizer: dict[str, Any],
) -> dict[str, Any] | None:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return None
    selected = optimizer.get("selected_policy")
    metrics = {}
    if isinstance(selected, dict):
        raw_metrics = selected.get("frozen_holdout_metrics") or selected.get(
            "holdout_metrics"
        )
        if isinstance(raw_metrics, dict):
            metrics = raw_metrics
    candidate_name = str(
        metrics.get("policy_name")
        or f"selected_{payload.get('sampler_used', 'unknown')}_optimizer_diagnostic_policy"
    )
    return {
        "candidate_name": candidate_name,
        "policy_kind": "optimizer_diagnostic",
        "forecast_source": metrics.get("forecast_source"),
        "required_prediction_columns": metrics.get("required_prediction_columns", []),
        "transaction_cost_bps": _number(
            rows[0].get("transaction_cost_bps")
        ) or _number(metrics.get("transaction_cost_bps")) or 5.0,
        "period_source": "selected_optimizer_exposure_path_exact",
        "exact_period_path": True,
        "rows": [
            {
                "date": row.get("rebalance_date"),
                "baseline_return": row.get("period_return"),
                "exposure": row.get("exposure"),
            }
            for row in rows
        ],
        "reported_metrics": metrics,
    }


def _reconstruct_optimizer_series(
    *,
    config: dict[str, Any],
    optimizer: dict[str, Any],
    meta_rows: list[dict[str, str]],
    auxiliary_rows: list[dict[str, str]],
) -> dict[str, Any] | None:
    selected = optimizer.get("selected_policy")
    if not isinstance(selected, dict):
        return None
    metrics = selected.get("frozen_holdout_metrics") or selected.get("holdout_metrics")
    if not isinstance(metrics, dict):
        return None
    candidate_name = str(
        metrics.get("policy_name")
        or f"selected_{optimizer.get('sampler_used', 'unknown')}_optimizer_diagnostic_policy"
    )
    parameters = selected.get("parameters") or selected.get("selected_params")
    if not isinstance(parameters, dict) or not meta_rows or not auxiliary_rows:
        return {
            "candidate_name": candidate_name,
            "policy_kind": "optimizer_diagnostic",
            "forecast_source": metrics.get("forecast_source"),
            "required_prediction_columns": metrics.get("required_prediction_columns", []),
            "transaction_cost_bps": float(metrics.get("transaction_cost_bps", 5.0)),
            "period_source": "allocation_optimizer_scalar_metrics_only",
            "exact_period_path": False,
            "rows": [],
            "reconstruction_warning": (
                "Selected optimizer period path is not persisted in existing artifacts."
            ),
            "reported_metrics": metrics,
        }
    holdout_by_feature_id = {
        row.get("feature_id"): row
        for row in meta_rows
        if row.get("feature_id") and row.get("split") == "holdout"
    }
    joined_rows = []
    for row in auxiliary_rows:
        feature_id = row.get("feature_id")
        source = holdout_by_feature_id.get(feature_id)
        if not source:
            continue
        joined = dict(source)
        joined.update(row)
        joined_rows.append(joined)
    if not joined_rows:
        return None
    variant = grid_variant(parameters)
    exposures = _variant_exposures(
        joined_rows,
        [],
        config.get("ml", {}),
        variant=variant,
        fit_rows=None,
    )
    rows = [
        {
            "date": row.get("rebalance_date", ""),
            "baseline_return": row.get("champion_return_next_period", 0.0),
            "exposure": exposure,
        }
        for row, exposure in zip(joined_rows, exposures)
    ]
    return {
        "candidate_name": candidate_name,
        "policy_kind": "optimizer_diagnostic",
        "forecast_source": metrics.get("forecast_source"),
        "required_prediction_columns": metrics.get("required_prediction_columns", []),
        "transaction_cost_bps": float(metrics.get("transaction_cost_bps", 5.0)),
        "period_source": "reconstructed_from_saved_holdout_auxiliary_predictions",
        "exact_period_path": False,
        "rows": rows,
        "reconstruction_warning": (
            "Exact optimizer fit rows/cross-fitted auxiliary forecasts are not "
            "persisted, so quantile thresholds are refit on holdout for the "
            "per-date audit path. Saved optimizer scalar holdout metrics remain "
            "the authoritative optimizer result."
        ),
        "reported_metrics": metrics,
    }
