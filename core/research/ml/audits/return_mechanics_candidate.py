from __future__ import annotations

import math
from statistics import mean, median
from typing import Any

from core.research.performance_metrics import calmar_ratio
from core.research.ml.audits.return_mechanics_math import (
    _annualized_return,
    _compound_returns,
    _equity_curve,
    _finite_float,
    _geometric_mean_return,
    _max_drawdown,
    _metric_delta,
    _number,
    _sharpe_ratio,
    _sortino_ratio,
)
from core.research.ml.audits.return_mechanics_types import (
    CAP_SCENARIOS,
    COST_SENSITIVITY_BPS,
    RESEARCH_METADATA,
)


def _candidate_audit(
    candidate_name: str,
    series: dict[str, Any],
    reported_metrics: dict[str, Any],
    *,
    default_cost_bps: float,
) -> dict[str, Any]:
    cost_bps = _number(series.get("transaction_cost_bps"))
    if cost_bps is None:
        cost_bps = default_cost_bps
    period_rows = _aggregate_return_rows(series.get("rows", []))
    records = _equity_records(period_rows, cost_bps=float(cost_bps))
    summary = _records_summary(records)
    reported = {**series.get("reported_metrics", {}), **reported_metrics}
    total_delta = _metric_delta(summary.get("total_return"), reported.get("total_return"))
    red_flags = []
    if summary["largest_positive_period_contribution"] > 0.50:
        red_flags.append("period_return_above_50pct")
    if summary["largest_negative_period_contribution"] < -0.50:
        red_flags.append("period_return_below_-50pct")
    if total_delta is not None and abs(total_delta) > 1e-6:
        red_flags.append("recomputed_total_return_differs_from_reported")
    exposure_sanity = _exposure_sanity(records)
    if exposure_sanity["out_of_range_exposure_dates"]:
        red_flags.append("exposure_outside_0_1")
    return {
        "candidate_name": candidate_name,
        "available": bool(records) or bool(reported),
        "policy_kind": series.get("policy_kind"),
        "period_source": series.get("period_source"),
        "exact_period_path": bool(series.get("exact_period_path")),
        "reconstruction_warning": series.get("reconstruction_warning"),
        "forecast_source": series.get("forecast_source"),
        "required_prediction_columns": series.get("required_prediction_columns", []),
        "forecast_inputs_use_actual_columns": any(
            column.startswith("actual_")
            for column in _requirement_columns(
                series.get("required_prediction_columns", [])
            )
        ),
        "transaction_cost_bps": float(cost_bps),
        "reported_total_return": _number(reported.get("total_return")),
        "reported_max_drawdown": _number(reported.get("max_drawdown")),
        "reported_sharpe": _number(reported.get("sharpe")),
        "reported_turnover": _number(reported.get("turnover")),
        "reported_estimated_transaction_costs": _number(
            reported.get("estimated_transaction_costs")
        ),
        "total_return_delta_vs_reported": total_delta,
        "max_drawdown_delta_vs_reported": _metric_delta(
            summary.get("max_drawdown"),
            reported.get("max_drawdown"),
        ),
        **summary,
        "top_20_contributing_rebalance_dates": _top_records(records, reverse=True),
        "worst_20_contributing_rebalance_dates": _top_records(records, reverse=False),
        "return_concentration": _return_concentration(records),
        "capped_return_sensitivity": _capped_return_sensitivity(records),
        "cost_sensitivity": _cost_sensitivity(period_rows),
        "exposure_sanity_checks": exposure_sanity,
        "red_flags": red_flags,
        **RESEARCH_METADATA,
    }


def _aggregate_return_rows(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    by_date: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        date = str(row.get("date") or row.get("rebalance_date") or "")
        if not date:
            continue
        period_return = _finite_float(row.get("baseline_return", 0.0))
        exposure = _finite_float(row.get("exposure", 1.0))
        by_date.setdefault(date, []).append((period_return, exposure))
    return [
        {
            "date": date,
            "baseline_return": mean(value[0] for value in values),
            "exposure": mean(value[1] for value in values),
            "source_row_count": len(values),
        }
        for date, values in sorted(by_date.items())
    ]


def _equity_records(
    periods: list[dict[str, float | str]],
    *,
    cost_bps: float,
) -> list[dict[str, float | str]]:
    equity = 1.0
    previous_exposure = 1.0
    records = []
    for period in periods:
        exposure = float(period["exposure"])
        baseline_return = float(period["baseline_return"])
        turnover = abs(exposure - previous_exposure)
        cost = turnover * cost_bps / 10_000.0
        net_return = (baseline_return * exposure) - cost
        equity *= 1.0 + net_return
        records.append({
            "date": str(period["date"]),
            "baseline_return": baseline_return,
            "exposure": exposure,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
            "source_row_count": int(period.get("source_row_count", 1)),
        })
        previous_exposure = exposure
    return records


def _records_summary(records: list[dict[str, float | str]]) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    exposures = [float(row["exposure"]) for row in records]
    equity_curve = [1.0] + [float(row["equity"]) for row in records]
    total = _compound_returns(returns)
    annualized = _annualized_return(total, records)
    drawdown = _max_drawdown(equity_curve)
    return {
        "start_date": records[0]["date"] if records else None,
        "end_date": records[-1]["date"] if records else None,
        "number_of_periods": len(records),
        "total_return": total,
        "compounded_return": total,
        "arithmetic_mean_period_return": mean(returns) if returns else None,
        "geometric_mean_period_return": _geometric_mean_return(total, len(returns)),
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "sharpe": _sharpe_ratio(returns, records),
        "sortino": _sortino_ratio(returns, records),
        "calmar": calmar_ratio(annualized if annualized is not None else total, drawdown),
        "turnover": sum(float(row["turnover"]) for row in records),
        "costs": sum(float(row["cost"]) for row in records),
        "exposure_mean": mean(exposures) if exposures else None,
        "exposure_median": median(exposures) if exposures else None,
        "exposure_min": min(exposures) if exposures else None,
        "exposure_max": max(exposures) if exposures else None,
        "largest_positive_period_contribution": max(returns, default=0.0),
        "largest_negative_period_contribution": min(returns, default=0.0),
    }


def _top_records(
    records: list[dict[str, float | str]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda row: float(row["net_return"]),
        reverse=reverse,
    )[:20]
    return [
        {
            "date": row["date"],
            "net_return": row["net_return"],
            "baseline_return": row["baseline_return"],
            "exposure": row["exposure"],
            "turnover": row["turnover"],
            "cost": row["cost"],
            "equity": row["equity"],
            "source_row_count": row["source_row_count"],
        }
        for row in sorted_records
    ]


def _return_concentration(records: list[dict[str, float | str]]) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    positives = sorted([value for value in returns if value > 0.0], reverse=True)
    total_positive = sum(positives)
    total_return = _compound_returns(returns)
    output = {}
    for count in (1, 3, 5, 10, 20):
        top = positives[:count]
        compounded = _compound_returns(top)
        output[f"top_{count}"] = {
            "sum_net_return": sum(top),
            "share_of_positive_period_returns": (
                sum(top) / total_positive if total_positive else None
            ),
            "compounded_return_from_top_periods": compounded,
            "compounded_share_of_total_return": (
                compounded / total_return if total_return else None
            ),
        }
    return output


def _capped_return_sensitivity(
    records: list[dict[str, float | str]],
) -> dict[str, Any]:
    returns = [float(row["net_return"]) for row in records]
    output = {}
    for scenario, (minimum, maximum) in CAP_SCENARIOS.items():
        capped = [min(maximum, max(minimum, value)) for value in returns]
        output[scenario] = {
            "minimum_period_return": minimum,
            "maximum_period_return": maximum,
            "total_return": _compound_returns(capped),
            "max_drawdown": _max_drawdown([1.0] + _equity_curve(capped)),
            "largest_positive_period_contribution": max(capped, default=0.0),
            "largest_negative_period_contribution": min(capped, default=0.0),
        }
    return output


def _cost_sensitivity(
    periods: list[dict[str, float | str]],
) -> dict[str, Any]:
    output = {}
    for cost_bps in COST_SENSITIVITY_BPS:
        records = _equity_records(periods, cost_bps=cost_bps)
        output[f"{cost_bps:g}_bps"] = {
            "transaction_cost_bps": cost_bps,
            "total_return": _compound_returns(
                [float(row["net_return"]) for row in records]
            ),
            "max_drawdown": _max_drawdown(
                [1.0] + [float(row["equity"]) for row in records]
            ),
            "turnover": sum(float(row["turnover"]) for row in records),
            "costs": sum(float(row["cost"]) for row in records),
        }
    return output


def _exposure_sanity(records: list[dict[str, float | str]]) -> dict[str, Any]:
    exposures = [float(row["exposure"]) for row in records]
    changes = [
        abs(current - previous)
        for previous, current in zip(exposures, exposures[1:])
    ]
    return {
        "average_exposure": mean(exposures) if exposures else None,
        "number_of_exposure_changes": sum(
            not math.isclose(current, previous)
            for previous, current in zip(exposures, exposures[1:])
        ),
        "largest_exposure_jump": max(changes, default=0.0),
        "out_of_range_exposure_dates": [
            row["date"]
            for row in records
            if float(row["exposure"]) < 0.0 or float(row["exposure"]) > 1.0
        ],
        "exposure_below_zero": any(value < 0.0 for value in exposures),
        "exposure_above_one": any(value > 1.0 for value in exposures),
    }


def _requirement_columns(requirements: Any) -> list[str]:
    output = []
    for requirement in requirements or []:
        output.extend(
            column
            for column in str(requirement).split("|")
            if column
        )
    return output


def _missing_candidate(candidate_name: str) -> dict[str, Any]:
    return {
        "candidate_name": candidate_name,
        "available": False,
        "skip_reason": "candidate not found in allocation audit artifacts",
        "red_flags": ["missing_candidate_period_path"],
        **RESEARCH_METADATA,
    }
