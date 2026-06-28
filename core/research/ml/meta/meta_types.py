from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetaEnsembleResult:
    output_dir: Path
    meta_dataset_path: Path
    audit_path: Path
    metrics_path: Path
    walk_forward_metrics_path: Path
    probability_calibration_path: Path
    calibrated_probability_calibration_path: Path
    holdout_shadow_overlay_path: Path
    threshold_sweep_path: Path
    meta_model_comparison_path: Path
    promotion_gates_path: Path
    overlay_model_comparison_path: Path
    leaderboard_path: Path
    leaderboard_markdown_path: Path
    allocation_policy_comparison_json_path: Path
    allocation_policy_comparison_csv_path: Path
    allocation_policy_leaderboard_path: Path
    allocation_shadow_overlay_path: Path
    allocation_policy_diagnostics_json_path: Path
    allocation_policy_diagnostics_markdown_path: Path
    allocation_policy_grid_search_csv_path: Path
    allocation_policy_grid_search_json_path: Path
    allocation_policy_grid_search_markdown_path: Path
    meta_auxiliary_predictions_path: Path
    meta_auxiliary_metrics_json_path: Path
    meta_auxiliary_metrics_markdown_path: Path
    allocation_optimizer_candidates_path: Path
    allocation_optimizer_results_path: Path
    allocation_optimizer_report_path: Path
    selected_optimizer_exposure_path_csv: Path
    selected_optimizer_exposure_path_json: Path
    trading_research_leaderboard_csv_path: Path
    trading_research_leaderboard_json_path: Path
    trading_research_leaderboard_markdown_path: Path
