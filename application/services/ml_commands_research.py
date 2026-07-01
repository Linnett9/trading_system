from core.research.ml.experiment_runner import MLExperimentRunner
from core.research.ml.runtime_parallelism import apply_runtime_parallelism, format_runtime_settings
from application.services.ml_commands_artifacts import _update_source_leaderboard


def run_ml_research(config, feed=None):
    runtime_settings = apply_runtime_parallelism(config)
    result = MLExperimentRunner(config, feed=feed).run()
    leaderboard_markdown_path, leaderboard_json_path = _update_source_leaderboard(
        config,
        result.output_dir,
    )
    print("\nML RESEARCH")
    print(
        "mode=research | "
        f"label={config.get('ml', {}).get('research_label', 'UNSPECIFIED_RESEARCH')} | "
        "trading_impact=none"
    )
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Output dir: {result.output_dir}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Feature importance: {result.feature_importance_path}")
    print(f"Confusion matrix: {result.confusion_matrix_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Model: {result.model_path}")
    print(f"Features: {result.features_path}")
    print(f"Feature summary: {result.feature_summary_path}")
    print(f"Labels: {result.labels_path}")
    print(f"Dataset: {result.dataset_path}")
    print(f"Dataset audit: {result.dataset_audit_path}")
    print(f"Walk-forward metrics: {result.walk_forward_metrics_path}")
    print(f"Threshold sweep: {result.threshold_sweep_path}")
    print(f"Model comparison: {result.model_comparison_path}")
    print(f"Shadow overlay: {result.shadow_overlay_path}")
    print(f"Holdout shadow overlay: {result.holdout_shadow_overlay_path}")
    print(f"Champion rebalance dataset: {result.rebalance_dataset_path}")
    print(f"Champion rebalance audit: {result.rebalance_dataset_audit_path}")
    print(f"History coverage: {result.history_coverage_path}")
    print(f"Drawdown event review: {result.drawdown_event_review_path}")
    print(f"Rule exposure study: {result.rule_exposure_study_path}")
    print(f"Probability calibration: {result.probability_calibration_path}")
    print(
        "Walk-forward probability calibration: "
        f"{result.walk_forward_probability_calibration_path}"
    )
    print(f"Baseline model comparison: {result.baseline_model_comparison_path}")
    print(f"Ranking diagnostics: {result.ranking_diagnostics_path}")
    print(f"Leaderboard: {leaderboard_markdown_path}")
    print(f"Leaderboard JSON: {leaderboard_json_path}")

def run_ml_expanded_rebalance_dataset(config, feed):
    runtime_settings = apply_runtime_parallelism(config)
    dataset_path, audit_path, row_count = MLExperimentRunner(
        config,
        feed=feed,
    ).build_expanded_rebalance_dataset()
    print("\nML EXPANDED REBALANCE DATASET")
    print("mode=research | trading_impact=none")
    print(f"Runtime: {format_runtime_settings(runtime_settings)}")
    print(f"Rows: {row_count}")
    print(f"Dataset: {dataset_path}")
    print(f"Audit: {audit_path}")
