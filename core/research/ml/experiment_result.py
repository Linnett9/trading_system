from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLExperimentResult:
    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    confusion_matrix_path: Path
    metadata_path: Path
    model_path: Path
    features_path: Path
    feature_summary_path: Path
    labels_path: Path
    dataset_path: Path
    dataset_audit_path: Path
    walk_forward_metrics_path: Path
    threshold_sweep_path: Path
    model_comparison_path: Path
    shadow_overlay_path: Path
    holdout_shadow_overlay_path: Path
    rebalance_dataset_path: Path
    rebalance_dataset_audit_path: Path
    history_coverage_path: Path
    drawdown_event_review_path: Path
    rule_exposure_study_path: Path
    probability_calibration_path: Path
    walk_forward_probability_calibration_path: Path
    baseline_model_comparison_path: Path
    ranking_diagnostics_path: Path
    calibrated_probability_calibration_path: Path
    overlay_model_comparison_path: Path
    prediction_artifacts_path: Path
    prediction_artifacts_metadata_path: Path
    html_report_path: Path
