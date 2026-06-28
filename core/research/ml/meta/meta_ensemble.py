from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.allocation.allocation_v2 import write_allocation_v2_reports
from core.research.ml.metrics.calibration import (
    build_probability_calibration,
    compare_calibration_methods,
)
from core.research.ml.metrics.evaluation import classification_metrics
from core.research.ml.metrics.leaderboard import write_leaderboard
from core.research.ml.meta.meta_auxiliary import (
    actual_auxiliary_values,
    namespaced_auxiliary_features,
    run_meta_auxiliary_ensemble,
)
from core.research.ml.overlays.overlay import overlay_decision_rule
from core.research.ml.stock_level.trading_research_leaderboard import (
    write_trading_research_leaderboard,
)
from core.research.ml.meta.meta_dataset import (
    _feature_values,
    _load_source_predictions,
    build_meta_dataset_rows,
)
from core.research.ml.meta.meta_comparison import (
    _best_calibration_summary,
    _compare_meta_learners,
)
from core.research.ml.meta.meta_evaluation import (
    _chronological_meta_probabilities,
    _walk_forward_meta_evaluation,
)
from core.research.ml.meta.meta_horizon import (
    _extended_horizon_rows,
)
from core.research.ml.meta.meta_io import _read_csv, _write_csv
from core.research.ml.meta.meta_models import (
    _fit_meta_model,
    _meta_ensemble_name,
    _meta_model_types,
)
from core.research.ml.meta.meta_overlay import (
    _overlay_summary,
    _promotion_gate_report,
    _threshold_sweep,
)
from core.research.ml.meta.meta_types import MetaEnsembleResult


def run_meta_ensemble(config: dict[str, Any]) -> MetaEnsembleResult:
    ml_config = config.get("ml", {})
    output_dir = Path(ml_config.get("output_dir", "reports/ml/regime_transformer_meta_ensemble_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(ml_config.get("meta_dataset_path", "cache/ml/meta_ensemble_dataset.csv"))
    expanded_dataset_path = Path(
        ml_config.get("expanded_rebalance_dataset_path", "cache/ml/expanded_rebalance_dataset.csv")
    )
    source_dirs = [Path(path) for path in ml_config.get("source_prediction_dirs", [])]
    source_predictions, warnings = _load_source_predictions(source_dirs)
    source_models = sorted(source_predictions)
    if len(source_models) < 2:
        raise RuntimeError(
            "Meta ensemble requires at least two available source prediction artifacts"
        )

    expanded_rows = _read_csv(expanded_dataset_path)
    meta_rows, audit = build_meta_dataset_rows(expanded_rows, source_predictions)
    audit.update({
        "warnings": warnings,
        "source_prediction_dirs": [str(path) for path in source_dirs],
        "research_only": True,
        "trading_impact": "none",
    })
    _write_csv(dataset_path, meta_rows)
    audit_path = output_dir / "meta_dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    train_rows = [row for row in meta_rows if row["split"] == "out_of_fold"]
    holdout_rows = [row for row in meta_rows if row["split"] == "holdout"]
    if not train_rows or not holdout_rows:
        raise RuntimeError("Meta ensemble requires out-of-fold and holdout rows")

    features = [_feature_values(row) for row in train_rows]
    labels = [int(row["actual_label"]) for row in train_rows]
    holdout_features = [_feature_values(row) for row in holdout_rows]
    holdout_labels = [int(row["actual_label"]) for row in holdout_rows]

    random_seed = int(ml_config.get("random_seed", 42))
    selected_meta_model_type = str(
        ml_config.get("meta_model_type", "logistic_regression")
    )
    model = _fit_meta_model(
        selected_meta_model_type,
        features,
        labels,
        random_seed=random_seed,
        sklearn_n_jobs=int(ml_config.get("sklearn_n_jobs", 1)),
    )
    train_probabilities = model.predict_proba(features)
    holdout_probabilities = model.predict_proba(holdout_features)
    auxiliary_result = run_meta_auxiliary_ensemble(
        train_rows,
        holdout_rows,
        output_dir,
        walk_forward_folds=int(
            ml_config.get("meta_auxiliary_walk_forward_folds", 3)
        ),
        embargo_rebalance_dates=int(
            ml_config.get("meta_auxiliary_embargo_rebalance_dates", 1)
        ),
        purge_overlapping_labels=bool(
            ml_config.get("meta_auxiliary_purge_overlapping_labels", True)
        ),
    )
    horizon = _extended_horizon_rows(
        train_rows=auxiliary_result.train_rows,
        holdout_rows=auxiliary_result.holdout_rows,
        holdout_probabilities=holdout_probabilities,
        model_type=selected_meta_model_type,
        config=ml_config,
        random_seed=random_seed,
        sklearn_n_jobs=int(ml_config.get("sklearn_n_jobs", 1)),
    )
    if horizon["enabled"] and horizon["available"]:
        allocation_rows = horizon["evaluation_rows"]
        allocation_probabilities = horizon["evaluation_probabilities"]
        allocation_selection_rows = horizon["selection_rows"]
        allocation_selection_probabilities = horizon["selection_probabilities"]
        _write_csv(auxiliary_result.predictions_path, allocation_rows)
        auxiliary_result.metrics["extended_meta_canonical_horizon"] = (
            horizon["audit"]
        )
        auxiliary_result.metrics_json_path.write_text(
            json.dumps(auxiliary_result.metrics, indent=2),
            encoding="utf-8",
        )
    else:
        allocation_rows = auxiliary_result.holdout_rows
        allocation_probabilities = holdout_probabilities
        allocation_selection_rows = [
            auxiliary_result.train_rows[index]
            for index in auxiliary_result.selection_train_indexes
        ]
        allocation_selection_probabilities = [
            train_probabilities[index]
            for index in auxiliary_result.selection_train_indexes
        ]
    threshold = float(ml_config.get("decision_threshold", 0.5))
    predictions = [int(probability >= threshold) for probability in holdout_probabilities]
    label_type = ml_config.get("label_type", "should_reduce_exposure")
    reduce_when, decision_rule = overlay_decision_rule(str(label_type))
    reduced_exposure = float(ml_config.get("promotion_reduced_exposure", 0.7))
    metrics_path = output_dir / "metrics.json"
    metrics = classification_metrics(holdout_labels, predictions)
    metrics_path.write_text(json.dumps({
        "mode": "research",
        "ensemble_name": ml_config.get("ensemble_name", "regime_transformer_meta_ensemble_v1"),
        "model_type": _meta_ensemble_name(selected_meta_model_type),
        "meta_model_type": selected_meta_model_type,
        "label_type": label_type,
        "feature_set": ml_config.get("feature_set", "expanded_rebalance_v1"),
        "train_sample_count": len(train_rows),
        "test_sample_count": len(holdout_rows),
        "metrics": metrics,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }, indent=2), encoding="utf-8")

    calibration_path = output_dir / "probability_calibration.json"
    calibration = build_probability_calibration(holdout_labels, holdout_probabilities)
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    calibrated_probability_calibration_path = (
        output_dir / "calibrated_probability_calibration.json"
    )
    calibration_comparison = compare_calibration_methods(
        labels,
        train_probabilities,
        holdout_labels,
        holdout_probabilities,
        bin_count=int(ml_config.get("calibration_bin_count", 10)),
    )
    calibrated_probability_calibration_path.write_text(
        json.dumps(calibration_comparison, indent=2),
        encoding="utf-8",
    )
    leaderboard_calibration = _best_calibration_summary(calibration_comparison)

    walk_forward_path = output_dir / "walk_forward_metrics.json"
    walk_forward = _walk_forward_meta_evaluation(
        meta_rows,
        model_type=selected_meta_model_type,
        fold_count=int(ml_config.get("walk_forward_folds", 3)),
        threshold=threshold,
        reduced_exposure=reduced_exposure,
        reduce_when=reduce_when,
        random_seed=random_seed,
        sklearn_n_jobs=int(ml_config.get("sklearn_n_jobs", 1)),
        calibration_bin_count=int(ml_config.get("calibration_bin_count", 10)),
    )
    walk_forward_path.write_text(
        json.dumps(walk_forward, indent=2),
        encoding="utf-8",
    )

    holdout_overlay_path = output_dir / "holdout_shadow_overlay.json"
    overlay = _overlay_summary(
        holdout_rows,
        holdout_probabilities,
        threshold,
        reduced_exposure,
        reduce_when=reduce_when,
    )
    holdout_overlay_path.write_text(json.dumps({
        "mode": "meta_ensemble_holdout_shadow_research_only",
        "decision_threshold": threshold,
        "reduced_exposure": reduced_exposure,
        "overlay_decision_rule": decision_rule,
        "result": overlay,
        "research_only": True,
        "trading_impact": "none",
    }, indent=2), encoding="utf-8")

    threshold_sweep_path = output_dir / "threshold_sweep.json"
    threshold_sweep = _threshold_sweep(
        holdout_rows,
        holdout_labels,
        holdout_probabilities,
        thresholds=ml_config.get("decision_thresholds", [0.5, 0.6, 0.7]),
        reduced_exposures=ml_config.get("reduced_exposures", [0.7, 0.8, 0.9]),
        reduce_when=reduce_when,
    )
    threshold_sweep_path.write_text(
        json.dumps(threshold_sweep, indent=2),
        encoding="utf-8",
    )

    meta_model_comparison_path = output_dir / "meta_model_comparison.json"
    meta_model_comparison = _compare_meta_learners(
        train_rows,
        holdout_rows,
        model_types=_meta_model_types(ml_config, selected_meta_model_type),
        threshold=threshold,
        reduced_exposure=reduced_exposure,
        reduce_when=reduce_when,
        random_seed=random_seed,
        calibration_bin_count=int(ml_config.get("calibration_bin_count", 10)),
        all_rows=meta_rows,
        walk_forward_folds=int(ml_config.get("walk_forward_folds", 3)),
        promotion_config=ml_config,
    )
    meta_model_comparison_path.write_text(
        json.dumps(meta_model_comparison, indent=2),
        encoding="utf-8",
    )

    promotion_gates_path = output_dir / "promotion_gates.json"
    promotion_gates = _promotion_gate_report(
        metrics=metrics,
        calibration=leaderboard_calibration,
        overlay=overlay,
        walk_forward=walk_forward,
        config=ml_config,
    )
    promotion_gates_path.write_text(
        json.dumps(promotion_gates, indent=2),
        encoding="utf-8",
    )

    overlay_comparison_path = output_dir / "overlay_model_comparison.json"
    scenarios = []
    for decision_threshold in ml_config.get("decision_thresholds", [0.5, 0.6, 0.7]):
        for scenario_reduced_exposure in ml_config.get("reduced_exposures", [0.7, 0.8, 0.9]):
            scenarios.append({
                "decision_threshold": float(decision_threshold),
                "reduced_exposure": float(scenario_reduced_exposure),
                "summary": _overlay_summary(
                    holdout_rows,
                    holdout_probabilities,
                    float(decision_threshold),
                    float(scenario_reduced_exposure),
                    reduce_when=reduce_when,
                ),
            })
    overlay_comparison_path.write_text(json.dumps({
        "mode": "meta_ensemble_overlay_model_comparison_research_only",
        "overlay_decision_rule": decision_rule,
        "models": [
            {
                "model_type": _meta_ensemble_name(selected_meta_model_type),
                "scenarios": scenarios,
            }
        ],
        "research_only": True,
        "trading_impact": "none",
    }, indent=2), encoding="utf-8")

    leaderboard_path = output_dir / "leaderboard.json"
    leaderboard_markdown_path = output_dir / "leaderboard.md"
    write_leaderboard(
        leaderboard_path,
        leaderboard_markdown_path,
        source_dirs,
        metrics,
        leaderboard_calibration,
        overlay,
        meta_model_name=_meta_ensemble_name(selected_meta_model_type),
        meta_walk_forward=walk_forward.get("summary", {}),
        promotion_gates=promotion_gates,
        meta_selections=meta_model_comparison.get("selections", {}),
    )
    allocation_paths = write_allocation_v2_reports(
        output_dir=output_dir,
        rows=allocation_rows,
        meta_probabilities=allocation_probabilities,
        diagnostics={
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "brier_score": leaderboard_calibration.get("brier_score"),
            "expected_calibration_error": leaderboard_calibration.get(
                "expected_calibration_error"
            ),
        },
        config=ml_config,
        selection_rows=allocation_selection_rows,
        selection_meta_probabilities=allocation_selection_probabilities,
    )
    trading_leaderboard_paths = write_trading_research_leaderboard(
        output_dir=output_dir,
        classification_leaderboard_path=leaderboard_path,
        allocation_comparison_path=allocation_paths.comparison_json,
        optimizer_results_path=allocation_paths.optimizer_results_json,
        auxiliary_metrics_path=auxiliary_result.metrics_json_path,
    )
    return MetaEnsembleResult(
        output_dir=output_dir,
        meta_dataset_path=dataset_path,
        audit_path=audit_path,
        metrics_path=metrics_path,
        walk_forward_metrics_path=walk_forward_path,
        probability_calibration_path=calibration_path,
        calibrated_probability_calibration_path=calibrated_probability_calibration_path,
        holdout_shadow_overlay_path=holdout_overlay_path,
        threshold_sweep_path=threshold_sweep_path,
        meta_model_comparison_path=meta_model_comparison_path,
        promotion_gates_path=promotion_gates_path,
        overlay_model_comparison_path=overlay_comparison_path,
        leaderboard_path=leaderboard_path,
        leaderboard_markdown_path=leaderboard_markdown_path,
        allocation_policy_comparison_json_path=allocation_paths.comparison_json,
        allocation_policy_comparison_csv_path=allocation_paths.comparison_csv,
        allocation_policy_leaderboard_path=allocation_paths.leaderboard_markdown,
        allocation_shadow_overlay_path=allocation_paths.shadow_overlay_json,
        allocation_policy_diagnostics_json_path=allocation_paths.diagnostics_json,
        allocation_policy_diagnostics_markdown_path=(
            allocation_paths.diagnostics_markdown
        ),
        allocation_policy_grid_search_csv_path=allocation_paths.grid_search_csv,
        allocation_policy_grid_search_json_path=allocation_paths.grid_search_json,
        allocation_policy_grid_search_markdown_path=(
            allocation_paths.grid_search_markdown
        ),
        meta_auxiliary_predictions_path=auxiliary_result.predictions_path,
        meta_auxiliary_metrics_json_path=auxiliary_result.metrics_json_path,
        meta_auxiliary_metrics_markdown_path=auxiliary_result.metrics_markdown_path,
        allocation_optimizer_candidates_path=allocation_paths.optimizer_candidates_csv,
        allocation_optimizer_results_path=allocation_paths.optimizer_results_json,
        allocation_optimizer_report_path=allocation_paths.optimizer_report_markdown,
        selected_optimizer_exposure_path_csv=(
            allocation_paths.selected_optimizer_exposure_path_csv
        ),
        selected_optimizer_exposure_path_json=(
            allocation_paths.selected_optimizer_exposure_path_json
        ),
        trading_research_leaderboard_csv_path=(
            trading_leaderboard_paths.csv_path
        ),
        trading_research_leaderboard_json_path=(
            trading_leaderboard_paths.json_path
        ),
        trading_research_leaderboard_markdown_path=(
            trading_leaderboard_paths.markdown_path
        ),
    )
