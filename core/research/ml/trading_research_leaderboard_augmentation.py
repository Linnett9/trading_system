from __future__ import annotations

from typing import Any

from core.research.ml.trading_research_leaderboard_math import (
    _drawdown_magnitude,
    _number,
)


def _augment_with_benchmark_validation(
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    by_name = {
        str(candidate.get("candidate_name")): candidate
        for candidate in validation.get("candidates", []) or []
        if isinstance(candidate, dict)
    }
    for row in rows:
        candidate = by_name.get(str(row.get("entity_name")))
        if not candidate:
            continue
        row.update({
            "benchmark_relative_pass": bool(
                candidate.get("benchmark_relative_pass", False)
            ),
            "tradability_validation_pass": bool(
                candidate.get("tradability_validation_pass", False)
            ),
            "promotion_candidate_status": candidate.get(
                "promotion_candidate_status"
            ),
        })


def _augment_with_canonical_metrics(
    rows: list[dict[str, Any]],
    canonical: dict[str, Any],
    concentration: dict[str, Any],
) -> None:
    canonical_by_name = canonical.get("candidates", {})
    concentration_by_name = concentration.get("candidates", {})
    for row in rows:
        name = str(row.get("entity_name"))
        candidate = canonical_by_name.get(name)
        if not isinstance(candidate, dict):
            continue
        diagnostic = candidate.get("diagnostic_period_grid", {})
        canonical_metrics = candidate.get("canonical_continuous_equity", {})
        row["diagnostic_period_grid_return"] = _number(
            diagnostic.get("total_return")
        )
        row["canonical_continuous_return"] = _number(
            canonical_metrics.get("total_return")
        )
        row["canonical_non_overlap_return"] = row[
            "canonical_continuous_return"
        ]
        row["canonical_tradable_total_return"] = _number(
            canonical_metrics.get("canonical_tradable_total_return")
        )
        row["max_drawdown"] = _drawdown_magnitude(
            canonical_metrics.get("max_drawdown")
        )
        row["sharpe"] = _number(canonical_metrics.get("sharpe"))
        row["sortino"] = _number(canonical_metrics.get("sortino"))
        row["calmar"] = _number(canonical_metrics.get("calmar"))
        row["turnover"] = _number(canonical_metrics.get("turnover"))
        row["estimated_transaction_costs"] = _number(
            canonical_metrics.get("estimated_transaction_costs")
        )
        concentration_candidate = concentration_by_name.get(name, {})
        row["profit_concentration_ratio"] = _number(
            concentration_candidate.get("profit_concentration", {}).get(
                "top_5_date_positive_return_share"
            )
        )
        row["anomaly_adjusted_return"] = _scenario_return(
            concentration_candidate,
            "remove_anomaly_dates",
        )
        row["anomaly_adjusted_canonical_return"] = row[
            "anomaly_adjusted_return"
        ]
        _refresh_robustness_metrics(row)


def _refresh_robustness_metrics(row: dict[str, Any]) -> None:
    if row.get("optimizer_objective_mode") != (
        "robustness_adjusted_canonical_score"
    ):
        return
    metrics = row.get("detail", {}).get("objective_metrics", {})
    canonical_return = _number(row.get("canonical_non_overlap_return"))
    anomaly_return = _number(row.get("anomaly_adjusted_canonical_return"))
    if canonical_return is None or anomaly_return is None:
        return
    ratio = max(
        0.0,
        (canonical_return - anomaly_return)
        / max(abs(canonical_return), 1e-12),
    )
    maximum = _number(metrics.get("max_allowed_anomaly_dependency_ratio"))
    dependency_penalty = max(0.0, ratio - (maximum if maximum is not None else 0.25))
    weights = metrics.get("robustness_weights", {})
    row["anomaly_dependency_ratio"] = ratio
    row["robustness_adjusted_score"] = (
        anomaly_return
        - float(weights.get("drawdown", 0.50))
        * float(row.get("max_drawdown") or 0.0)
        - float(weights.get("turnover", 0.25))
        * float(metrics.get("turnover_penalty") or 0.0)
        - float(weights.get("concentration", 0.25))
        * float(row.get("profit_concentration_ratio") or 0.0)
        - float(weights.get("anomaly_dependency", 0.50))
        * dependency_penalty
        - float(weights.get("cost_stress", 1.0))
        * float(metrics.get("cost_stress_penalty") or 0.0)
    )


def _scenario_return(candidate: dict[str, Any], scenario_name: str) -> float | None:
    for scenario in candidate.get("scenarios", []) or []:
        if scenario.get("scenario_name") == scenario_name:
            return _number(scenario.get("summary", {}).get("total_return"))
    return None
