from __future__ import annotations

from typing import Any

from core.research.ml.replay.canonical_continuous_equity_replay import score_candidate_exposure_path
from core.research.ml.data.data_anomaly_quarantine import detect_period_anomalies
from core.research.ml.audits.profit_concentration_audit import score_candidate_concentration
from core.research.ml.allocation.allocation_optimizer_types import OBJECTIVE_MODES


def optimizer_objective_mode(config: dict[str, Any]) -> str:
    mode = str(
        config.get("allocation_optimizer", {}).get(
            "objective_mode",
            "diagnostic_period_grid_return",
        )
    )
    if mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"Unsupported allocation optimizer objective_mode '{mode}'"
        )
    return mode

def score_optimizer_candidate(
    *,
    diagnostic_objective: float,
    exposure_path: list[dict[str, Any]],
    config: dict[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    """Return one optimizer objective and its pure research diagnostics."""
    mode = optimizer_objective_mode(config)
    if mode == "diagnostic_period_grid_return":
        return {
            "objective_mode": mode,
            "objective_value": float(diagnostic_objective),
            "diagnostic_period_grid_objective": float(diagnostic_objective),
            "canonical_non_overlap_return": None,
            "anomaly_adjusted_canonical_return": None,
            "anomaly_dependency_ratio": None,
            "robustness_adjusted_score": None,
            "selected_by_robustness_objective": False,
        }

    optimizer = config.get("allocation_optimizer", {})
    anomaly_policy = str(optimizer.get("anomaly_policy", "penalize"))
    if anomaly_policy not in {"ignore", "penalize"}:
        raise ValueError(
            "allocation_optimizer.anomaly_policy must be 'ignore' or 'penalize'"
        )
    canonical = score_candidate_exposure_path(
        exposure_path,
        candidate_name=candidate_name,
    )
    canonical_metrics = canonical.get("canonical_continuous_equity", {})
    canonical_return = float(canonical_metrics.get("total_return", 0.0) or 0.0)
    canonical_drawdown = abs(float(canonical_metrics.get("max_drawdown", 0.0) or 0.0))
    canonical_turnover = float(canonical_metrics.get("turnover", 0.0) or 0.0)
    canonical_row_count = max(int(canonical_metrics.get("row_count", 0) or 0), 1)

    anomalies = detect_period_anomalies(
        exposure_path,
        large_symbol_return_abs=float(
            optimizer.get("large_symbol_return_abs", 1.0)
        ),
        large_portfolio_return_abs=float(
            optimizer.get("large_portfolio_return_abs", 0.50)
        ),
    )
    flagged_dates = (
        {
            str(row["rebalance_date"])
            for row in anomalies
            if row.get("rebalance_date")
        }
        if anomaly_policy == "penalize"
        else set()
    )
    concentration = score_candidate_concentration(
        canonical,
        flagged_dates=flagged_dates,
    )
    anomaly_adjusted_return = _scenario_return(
        concentration,
        "remove_anomaly_dates",
        fallback=canonical_return,
    )
    concentration_penalty = float(
        concentration.get("profit_concentration", {}).get(
            "top_5_date_positive_return_share",
            0.0,
        )
        or 0.0
    )
    anomaly_dependency_ratio = max(
        0.0,
        (canonical_return - anomaly_adjusted_return)
        / max(abs(canonical_return), 1e-12),
    )
    maximum_dependency = float(
        optimizer.get("max_allowed_anomaly_dependency_ratio", 0.25)
    )
    anomaly_dependency_penalty = max(
        0.0,
        anomaly_dependency_ratio - maximum_dependency,
    )
    turnover_penalty = canonical_turnover / canonical_row_count
    cost_stress_multiplier = float(optimizer.get("cost_stress_multiplier", 2.0))
    stressed = score_candidate_exposure_path(
        exposure_path,
        candidate_name=candidate_name,
        cost_multiplier=cost_stress_multiplier,
    )
    stressed_return = float(
        stressed.get("canonical_continuous_equity", {}).get("total_return", 0.0)
        or 0.0
    )
    cost_stress_penalty = max(0.0, canonical_return - stressed_return)
    weights = optimizer.get("robustness_weights", {})
    resolved_weights = {
        "drawdown": float(weights.get("drawdown", 0.50)),
        "turnover": float(weights.get("turnover", 0.25)),
        "concentration": float(weights.get("concentration", 0.25)),
        "anomaly_dependency": float(
            weights.get("anomaly_dependency", 0.50)
        ),
        "cost_stress": float(weights.get("cost_stress", 1.0)),
    }
    robustness_score = (
        anomaly_adjusted_return
        - resolved_weights["drawdown"] * canonical_drawdown
        - resolved_weights["turnover"] * turnover_penalty
        - resolved_weights["concentration"] * concentration_penalty
        - resolved_weights["anomaly_dependency"]
        * anomaly_dependency_penalty
        - resolved_weights["cost_stress"] * cost_stress_penalty
    )
    objective_values = {
        "canonical_non_overlap_return": canonical_return,
        "anomaly_adjusted_canonical_return": anomaly_adjusted_return,
        "robustness_adjusted_canonical_score": robustness_score,
    }
    return {
        "objective_mode": mode,
        "objective_value": objective_values[mode],
        "diagnostic_period_grid_objective": float(diagnostic_objective),
        "canonical_non_overlap_return": canonical_return,
        "anomaly_adjusted_canonical_return": anomaly_adjusted_return,
        "anomaly_dependency_ratio": anomaly_dependency_ratio,
        "robustness_adjusted_score": robustness_score,
        "selected_by_robustness_objective": (
            mode == "robustness_adjusted_canonical_score"
        ),
        "canonical_max_drawdown": canonical_drawdown,
        "canonical_turnover": canonical_turnover,
        "canonical_row_count": canonical_row_count,
        "turnover_penalty": turnover_penalty,
        "concentration_penalty": concentration_penalty,
        "anomaly_dependency_penalty": anomaly_dependency_penalty,
        "cost_stress_penalty": cost_stress_penalty,
        "cost_stressed_canonical_return": stressed_return,
        "cost_stress_multiplier": cost_stress_multiplier,
        "flagged_anomaly_dates": sorted(flagged_dates),
        "anomaly_policy": anomaly_policy,
        "max_allowed_anomaly_dependency_ratio": maximum_dependency,
        "robustness_weights": resolved_weights,
    }

def _scenario_return(
    candidate: dict[str, Any],
    scenario_name: str,
    *,
    fallback: float,
) -> float:
    for scenario in candidate.get("scenarios", []) or []:
        if scenario.get("scenario_name") != scenario_name:
            continue
        value = scenario.get("summary", {}).get("total_return")
        if value is not None:
            return float(value)
    return fallback
