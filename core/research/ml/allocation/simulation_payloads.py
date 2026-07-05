from __future__ import annotations

from typing import Any

from core.research.ml.allocation.types import AllocationPolicyResult, RESEARCH_METADATA


def _trading_rank_key(result: AllocationPolicyResult) -> tuple[float, ...]:
    return (
        -result.total_return,
        result.max_drawdown,
        -result.sharpe,
        -result.sortino,
        -result.calmar,
        result.turnover,
        result.estimated_transaction_costs,
    )
def _comparison_winners(ranked_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranked_payloads:
        return {"selected_policy": None}
    selected = ranked_payloads[0]
    return {
        "selected_policy": selected.get("policy_name"),
        "selected_rank": selected.get("rank"),
        "selected_policy_kind": selected.get("policy_kind"),
    }
def _result_payload(result: AllocationPolicyResult) -> dict[str, Any]:
    payload = {
        "policy_name": result.policy_name,
        "policy_version": result.policy_version,
        "policy_kind": result.policy_kind,
        "mapping_method": result.mapping_method,
        "threshold_fit_scope": result.threshold_fit_scope,
        "overfit_warning": result.overfit_warning,
        "transaction_cost_bps": result.transaction_cost_bps,
        "required_prediction_columns": list(result.required_prediction_columns),
        "exposure_min": result.exposure_min,
        "exposure_max": result.exposure_max,
        "trading_impact": result.trading_impact,
        "research_only": result.research_only,
        "production_validated": result.production_validated,
        "available": result.available,
        "skip_reason": result.skip_reason,
        "forecast_source": result.forecast_source,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "max_drawdown": result.max_drawdown,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "calmar": result.calmar,
        "turnover": result.turnover,
        "estimated_transaction_costs": result.estimated_transaction_costs,
        "return_per_unit_drawdown": result.return_per_unit_drawdown,
        "mean_exposure": result.mean_exposure,
        "median_exposure": result.median_exposure,
        "min_exposure": result.min_exposure,
        "max_exposure": result.max_exposure,
        "exposure_std": result.exposure_std,
        "days_at_0_exposure": result.days_at_0_exposure,
        "days_at_full_exposure": result.days_at_full_exposure,
        "number_of_exposure_changes": result.number_of_exposure_changes,
        "average_exposure_change": result.average_exposure_change,
        "maximum_one_period_exposure_change": result.maximum_one_period_exposure_change,
        "pct_periods_at_0_exposure": result.pct_periods_at_0_exposure,
        "pct_periods_at_20_exposure": result.pct_periods_at_20_exposure,
        "pct_periods_at_50_exposure": result.pct_periods_at_50_exposure,
        "pct_periods_at_80_exposure": result.pct_periods_at_80_exposure,
        "pct_periods_at_100_exposure": result.pct_periods_at_100_exposure,
        "evaluated_periods": result.evaluated_periods,
        "performance_when_exposure_reduced": result.performance_when_exposure_reduced,
        "performance_when_exposure_high": result.performance_when_exposure_high,
        "performance_during_worst_drawdown_windows": result.performance_during_worst_drawdown_windows,
        "drawdown_impact": result.drawdown_impact,
        "prediction_to_exposure_diagnostics": result.prediction_to_exposure_diagnostics,
        "balanced_accuracy": result.balanced_accuracy,
        "brier_score": result.brier_score,
        "expected_calibration_error": result.expected_calibration_error,
    }
    payload.update(RESEARCH_METADATA)
    return payload
def _add_robustness_flags(
    ranked_payloads: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    del config
    for row in ranked_payloads:
        exposure_min = row.get("min_exposure")
        exposure_max = row.get("max_exposure")
        row["robustness_flags"] = {
            "exposure_is_constant": (
                exposure_min is not None
                and exposure_max is not None
                and float(exposure_min) == float(exposure_max)
            )
        }
