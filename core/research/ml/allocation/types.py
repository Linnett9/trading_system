from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


POLICY_VERSION = "2.0.0"
RESEARCH_METADATA = {
    "trading_impact": "none",
    "research_only": True,
    "production_validated": False,
}


@dataclass(frozen=True)
class AllocationPolicyDefinition:
    policy_name: str
    required_prediction_columns: tuple[str, ...]
    exposure_builder: Callable[
        [list[dict[str, str]], list[float], dict[str, Any]],
        list[float],
    ]
    policy_version: str = POLICY_VERSION
    policy_kind: str = "allocation_policy"
    mapping_method: str = "fixed"
    threshold_fit_scope: str = "fixed_configured_thresholds"
    overfit_warning: str | None = None
    transaction_cost_bps: float | None = None
    exposure_min: float = 0.0
    exposure_max: float = 1.0


@dataclass(frozen=True)
class AllocationPolicyResult:
    policy_name: str
    policy_version: str
    policy_kind: str
    mapping_method: str
    threshold_fit_scope: str
    overfit_warning: str | None
    transaction_cost_bps: float
    required_prediction_columns: tuple[str, ...]
    exposure_min: float
    exposure_max: float
    trading_impact: str
    research_only: bool
    production_validated: bool
    available: bool
    skip_reason: str | None
    forecast_source: str
    total_return: float
    annualized_return: float | None
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float
    turnover: float
    estimated_transaction_costs: float
    return_per_unit_drawdown: float | None
    mean_exposure: float
    median_exposure: float
    min_exposure: float
    max_exposure: float
    exposure_std: float
    days_at_0_exposure: int
    days_at_full_exposure: int
    number_of_exposure_changes: int
    average_exposure_change: float
    maximum_one_period_exposure_change: float
    pct_periods_at_0_exposure: float
    pct_periods_at_20_exposure: float
    pct_periods_at_50_exposure: float
    pct_periods_at_80_exposure: float
    pct_periods_at_100_exposure: float
    evaluated_periods: int
    performance_when_exposure_reduced: dict[str, float | int]
    performance_when_exposure_high: dict[str, float | int]
    performance_during_worst_drawdown_windows: dict[str, float | int]
    drawdown_impact: dict[str, float | str]
    prediction_to_exposure_diagnostics: dict[str, float | None]
    balanced_accuracy: float | None
    brier_score: float | None
    expected_calibration_error: float | None


@dataclass(frozen=True)
class AllocationV2Paths:
    comparison_json: Path
    comparison_csv: Path
    leaderboard_markdown: Path
    shadow_overlay_json: Path
    diagnostics_json: Path
    diagnostics_markdown: Path
    grid_search_csv: Path
    grid_search_json: Path
    grid_search_markdown: Path
    optimizer_candidates_csv: Path
    optimizer_results_json: Path
    optimizer_report_markdown: Path
    selected_optimizer_exposure_path_csv: Path
    selected_optimizer_exposure_path_json: Path

