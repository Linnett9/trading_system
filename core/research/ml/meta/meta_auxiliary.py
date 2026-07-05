from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.research.ml.meta.meta_auxiliary_cv import (
    _chronological_cross_fitted_predictions,
    _contiguous_blocks,
    _label_window_ends_before,
    _minimum_rebalance_date,
    _purged_training_rows,
    _rebalance_date,
)
from core.research.ml.meta.meta_auxiliary_features import (
    _auxiliary_feature_names,
    _feature_matrix,
    actual_auxiliary_values,
    namespaced_auxiliary_features,
)
from core.research.ml.meta.meta_auxiliary_io import (
    _write_metrics_markdown,
    _write_predictions,
)
from core.research.ml.meta.meta_auxiliary_math import (
    _finite_value,
    _format_metric,
    _pearson,
    _quantile,
    _ranks,
    _regression_metrics,
)
from core.research.ml.meta.meta_auxiliary_model import _AuxiliaryRegressor, _fit_regressor
from core.research.ml.meta.meta_auxiliary_types import (
    AUXILIARY_PREDICTION_COLUMNS,
    AUXILIARY_TARGETS,
    MetaAuxiliaryResult,
)


def run_meta_auxiliary_ensemble(
    train_rows: list[dict[str, str]],
    holdout_rows: list[dict[str, str]],
    output_dir: Path,
    *,
    walk_forward_folds: int = 3,
    embargo_rebalance_dates: int = 1,
    purge_overlapping_labels: bool = True,
) -> MetaAuxiliaryResult:
    if walk_forward_folds < 1:
        raise ValueError("walk_forward_folds must be at least one")
    if embargo_rebalance_dates < 0:
        raise ValueError("embargo_rebalance_dates must be non-negative")
    augmented_train = [dict(row) for row in train_rows]
    augmented_holdout = [dict(row) for row in holdout_rows]
    feature_names = _auxiliary_feature_names(train_rows)
    target_metrics: dict[str, dict[str, Any]] = {}
    fold_audits: dict[str, list[dict[str, Any]]] = {}
    predicted_indexes_by_target: list[set[int]] = []
    holdout_start = _minimum_rebalance_date(holdout_rows)

    for actual_name, prediction_name in AUXILIARY_TARGETS.items():
        usable_train = [row for row in train_rows if _finite_value(row.get(actual_name))]
        if not usable_train or not feature_names:
            target_metrics[actual_name] = {
                "available": False,
                "reason": "missing training targets or auxiliary source features",
                "prediction_column": prediction_name,
                "sample_count": 0,
            }
            continue
        holdout_training_rows, holdout_training_audit = _purged_training_rows(
            usable_train,
            validation_start=holdout_start,
            embargo_rebalance_dates=embargo_rebalance_dates,
            purge_overlapping_labels=purge_overlapping_labels,
        )
        if not holdout_training_rows:
            target_metrics[actual_name] = {
                "available": False,
                "reason": "no eligible training rows before purged holdout",
                "prediction_column": prediction_name,
                "sample_count": 0,
            }
            continue
        model = _fit_regressor(holdout_training_rows, actual_name, feature_names)
        train_predictions, target_fold_audits = _chronological_cross_fitted_predictions(
            augmented_train,
            actual_name,
            feature_names,
            fold_count=walk_forward_folds,
            embargo_rebalance_dates=embargo_rebalance_dates,
            purge_overlapping_labels=purge_overlapping_labels,
        )
        holdout_predictions = model.predict(augmented_holdout)
        for row, prediction in zip(augmented_train, train_predictions):
            if prediction is not None:
                row[prediction_name] = str(prediction)
        for row, prediction in zip(augmented_holdout, holdout_predictions):
            row[prediction_name] = str(prediction)
        predicted_indexes_by_target.append({
            index
            for index, prediction in enumerate(train_predictions)
            if prediction is not None
        })
        fold_audits[actual_name] = target_fold_audits
        target_metrics[actual_name] = _regression_metrics(
            augmented_holdout,
            actual_name,
            prediction_name,
        )
        target_metrics[actual_name]["holdout_training_audit"] = holdout_training_audit

    selection_train_indexes = tuple(sorted(
        set.intersection(*predicted_indexes_by_target)
        if predicted_indexes_by_target
        else set()
    ))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "meta_auxiliary_predictions.csv"
    metrics_json_path = output_dir / "meta_auxiliary_metrics.json"
    metrics_markdown_path = output_dir / "meta_auxiliary_metrics.md"
    _write_predictions(predictions_path, augmented_holdout)
    metrics = {
        "mode": "meta_auxiliary_ensemble_research_only",
        "feature_columns": feature_names,
        "targets": target_metrics,
        "available_targets": [
            name for name, payload in target_metrics.items() if payload.get("available")
        ],
        "train_prediction_method": "purged_chronological_walk_forward",
        "holdout_prediction_method": (
            "refit_purged_out_of_fold_rows_then_predict_frozen_holdout"
        ),
        "fold_design": {
            "walk_forward_folds": walk_forward_folds,
            "embargo_rebalance_dates": embargo_rebalance_dates,
            "purge_overlapping_labels": purge_overlapping_labels,
            "date_grouping": "rebalance_date",
            "training_window": "expanding",
            "validation_window": "contiguous_future_date_blocks",
            "warmup_rows_are_forecasted": False,
            "selection_train_row_count": len(selection_train_indexes),
        },
        "fold_audits": fold_audits,
        "research_only": True,
        "trading_impact": "none",
        "production_validated": False,
    }
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_metrics_markdown(metrics_markdown_path, metrics)
    return MetaAuxiliaryResult(
        train_rows=augmented_train,
        holdout_rows=augmented_holdout,
        selection_train_indexes=selection_train_indexes,
        predictions_path=predictions_path,
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metrics=metrics,
    )
