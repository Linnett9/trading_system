from __future__ import annotations

import math
from statistics import mean, median

from core.research.ml.allocation.exposures import _forecast_source
from core.research.ml.allocation.simulation_diagnostics import _prediction_to_exposure_diagnostics
from core.research.ml.allocation.simulation_math import (
    _annualized_return,
    _compound_returns,
    _current_drawdown,
    _max_drawdown,
    _percentage_at_exposure,
    _population_std,
    _sharpe_ratio,
    _sortino_ratio,
)
from core.research.ml.allocation.types import AllocationPolicyDefinition, AllocationPolicyResult, RESEARCH_METADATA
from core.research.ml.allocation.utils import _finite_float
from core.research.performance_metrics import calmar_ratio


def _net_period_returns(
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
) -> list[float]:
    if not exposures:
        return []
    previous_exposure = 1.0
    returns = []
    for _, period_return, exposure in _aggregate_periods(rows, exposures):
        turnover = abs(exposure - previous_exposure)
        cost = turnover * transaction_cost_bps / 10_000.0
        returns.append((period_return * exposure) - cost)
        previous_exposure = exposure
    return returns
def _simulate_policy(
    definition: AllocationPolicyDefinition,
    rows: list[dict[str, str]],
    exposures: list[float],
    transaction_cost_bps: float,
    diagnostics: dict[str, float | None],
) -> AllocationPolicyResult:
    periods = _aggregate_periods(rows, exposures)
    equity = 1.0
    baseline_equity = 1.0
    curve = [equity]
    baseline_curve = [baseline_equity]
    net_returns = []
    turnover = 0.0
    estimated_costs = 0.0
    previous_exposure = 1.0
    records: list[dict[str, float | str]] = []
    for date, period_return, exposure in periods:
        change = abs(exposure - previous_exposure)
        cost = change * transaction_cost_bps / 10_000.0
        net_return = (period_return * exposure) - cost
        if net_return <= -1.0 or period_return <= -1.0:
            raise ValueError("Allocation return would zero or invert equity")
        equity *= 1.0 + net_return
        baseline_equity *= 1.0 + period_return
        curve.append(equity)
        baseline_curve.append(baseline_equity)
        net_returns.append(net_return)
        turnover += change
        estimated_costs += cost
        records.append({
            "date": date,
            "baseline_return": period_return,
            "allocated_return": net_return,
            "exposure": exposure,
            "baseline_drawdown": _current_drawdown(baseline_curve),
        })
        previous_exposure = exposure

    total_return = equity - 1.0
    annualized_return = _annualized_return(total_return, periods)
    drawdown = _max_drawdown(curve)
    baseline_drawdown = _max_drawdown(baseline_curve)
    exposure_values = [period[2] for period in periods]
    changes = [
        abs(current - previous)
        for previous, current in zip(exposure_values, exposure_values[1:])
        if not math.isclose(current, previous)
    ]
    all_changes = [
        abs(current - previous)
        for previous, current in zip(exposure_values, exposure_values[1:])
    ]
    worst_count = max(1, math.ceil(len(records) * 0.20)) if records else 0
    worst_records = sorted(
        records,
        key=lambda row: float(row["baseline_drawdown"]),
    )[:worst_count]
    drawdown_delta = baseline_drawdown - drawdown
    if drawdown_delta > 1e-12:
        drawdown_effect = "avoided"
    elif drawdown_delta < -1e-12:
        drawdown_effect = "worsened"
    else:
        drawdown_effect = "unchanged"

    return AllocationPolicyResult(
        policy_name=definition.policy_name,
        policy_version=definition.policy_version,
        policy_kind=definition.policy_kind,
        mapping_method=definition.mapping_method,
        threshold_fit_scope=definition.threshold_fit_scope,
        overfit_warning=definition.overfit_warning,
        transaction_cost_bps=transaction_cost_bps,
        required_prediction_columns=definition.required_prediction_columns,
        exposure_min=definition.exposure_min,
        exposure_max=definition.exposure_max,
        available=True,
        skip_reason=None,
        forecast_source=_forecast_source(definition, rows),
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=drawdown,
        sharpe=_sharpe_ratio(net_returns, periods),
        sortino=_sortino_ratio(net_returns, periods),
        calmar=calmar_ratio(
            annualized_return if annualized_return is not None else total_return,
            drawdown,
        ),
        turnover=turnover,
        estimated_transaction_costs=estimated_costs,
        return_per_unit_drawdown=(total_return / drawdown if drawdown else None),
        mean_exposure=mean(exposure_values) if exposure_values else 0.0,
        median_exposure=median(exposure_values) if exposure_values else 0.0,
        min_exposure=min(exposure_values, default=0.0),
        max_exposure=max(exposure_values, default=0.0),
        exposure_std=_population_std(exposure_values),
        days_at_0_exposure=sum(math.isclose(value, 0.0) for value in exposure_values),
        days_at_full_exposure=sum(math.isclose(value, 1.0) for value in exposure_values),
        number_of_exposure_changes=len(changes),
        average_exposure_change=mean(changes) if changes else 0.0,
        maximum_one_period_exposure_change=max(all_changes, default=0.0),
        pct_periods_at_0_exposure=_percentage_at_exposure(exposure_values, 0.0),
        pct_periods_at_20_exposure=_percentage_at_exposure(exposure_values, 0.2),
        pct_periods_at_50_exposure=_percentage_at_exposure(exposure_values, 0.5),
        pct_periods_at_80_exposure=_percentage_at_exposure(exposure_values, 0.8),
        pct_periods_at_100_exposure=_percentage_at_exposure(exposure_values, 1.0),
        evaluated_periods=len(periods),
        performance_when_exposure_reduced=_performance_summary(
            [row for row in records if float(row["exposure"]) < 0.5]
        ),
        performance_when_exposure_high=_performance_summary(
            [row for row in records if float(row["exposure"]) >= 0.8]
        ),
        performance_during_worst_drawdown_windows=_performance_summary(worst_records),
        drawdown_impact={
            "baseline_max_drawdown": baseline_drawdown,
            "policy_max_drawdown": drawdown,
            "drawdown_improvement": drawdown_delta,
            "effect": drawdown_effect,
        },
        prediction_to_exposure_diagnostics=(
            _prediction_to_exposure_diagnostics(rows, exposures)
        ),
        balanced_accuracy=diagnostics.get("balanced_accuracy"),
        brier_score=diagnostics.get("brier_score"),
        expected_calibration_error=diagnostics.get("expected_calibration_error"),
        **RESEARCH_METADATA,
    )
def _aggregate_periods(
    rows: list[dict[str, str]],
    exposures: list[float],
) -> list[tuple[str, float, float]]:
    if len(rows) != len(exposures):
        raise ValueError("Allocation rows and exposures must have equal length")
    by_date: dict[str, list[tuple[float, float]]] = {}
    for row, exposure in zip(rows, exposures):
        date = str(row.get("rebalance_date") or row.get("date") or "")
        if not date:
            raise ValueError("Allocation row is missing rebalance_date")
        period_return = _finite_float(
            row.get("champion_return_next_period", 0.0) or 0.0
        )
        by_date.setdefault(date, []).append((period_return, _finite_float(exposure)))
    return [
        (date, mean(value[0] for value in values), mean(value[1] for value in values))
        for date, values in sorted(by_date.items())
    ]
def _performance_summary(
    records: list[dict[str, float | str]],
) -> dict[str, float | int]:
    baseline_returns = [float(row["baseline_return"]) for row in records]
    allocated_returns = [float(row["allocated_return"]) for row in records]
    return {
        "period_count": len(records),
        "baseline_total_return": _compound_returns(baseline_returns),
        "allocated_total_return": _compound_returns(allocated_returns),
        "return_difference": (
            _compound_returns(allocated_returns) - _compound_returns(baseline_returns)
        ),
        "mean_exposure": (
            mean(float(row["exposure"]) for row in records) if records else 0.0
        ),
    }
