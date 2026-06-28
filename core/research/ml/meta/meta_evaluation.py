from __future__ import annotations

import math
from typing import Any

from core.research.ml.metrics.calibration import build_probability_calibration
from core.research.ml.metrics.evaluation import classification_metrics
from core.research.ml.meta.meta_dataset import _feature_values
from core.research.ml.meta.meta_models import (
    _fit_meta_model,
    _normalize_meta_model_type,
)
from core.research.ml.meta.meta_overlay import _overlay_summary, _with_split


def _chronological_meta_probabilities(
    rows: list[dict[str, str]],
    *,
    model_type: str,
    fold_count: int,
    embargo_rebalance_dates: int,
    purge_overlapping_labels: bool,
    random_seed: int,
    sklearn_n_jobs: int,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    probabilities: list[float | None] = [None] * len(rows)
    unique_dates = sorted({
        row.get("rebalance_date", "")
        for row in rows
        if row.get("rebalance_date")
    })
    if len(unique_dates) < 2:
        return probabilities, []
    date_positions = {date: index for index, date in enumerate(unique_dates)}
    effective_fold_count = min(max(1, int(fold_count)), len(unique_dates) - 1)
    candidate_test_dates = unique_dates[1:]
    chunk_size = max(1, math.ceil(len(candidate_test_dates) / effective_fold_count))
    audits: list[dict[str, Any]] = []
    for fold_index in range(effective_fold_count):
        test_dates = candidate_test_dates[
            fold_index * chunk_size : (fold_index + 1) * chunk_size
        ]
        if not test_dates:
            continue
        first_test_date = test_dates[0]
        first_position = date_positions[first_test_date]
        embargo_start = max(0, first_position - embargo_rebalance_dates)
        embargoed_dates = set(unique_dates[embargo_start:first_position])
        train_dates = [
            date
            for date in unique_dates
            if date < first_test_date and date not in embargoed_dates
        ]
        train_fold_rows = [
            row
            for row in rows
            if row.get("rebalance_date") in train_dates
            and (
                not purge_overlapping_labels
                or not _label_overlaps_validation(row, first_test_date)
            )
        ]
        test_indexes = [
            index
            for index, row in enumerate(rows)
            if row.get("rebalance_date") in set(test_dates)
        ]
        test_rows = [rows[index] for index in test_indexes]
        audit = {
            "fold": fold_index + 1,
            "validation_start": first_test_date,
            "validation_end": test_dates[-1],
            "train_sample_count": len(train_fold_rows),
            "test_sample_count": len(test_rows),
            "embargoed_rebalance_date_count": len(embargoed_dates),
            "purged_label_overlap_count": len([
                row
                for row in rows
                if row.get("rebalance_date") in train_dates
                and _label_overlaps_validation(row, first_test_date)
            ])
            if purge_overlapping_labels
            else 0,
            "prediction_generated": False,
        }
        if not train_fold_rows or not test_rows:
            audits.append(audit)
            continue
        model = _fit_meta_model(
            model_type,
            [_feature_values(row) for row in train_fold_rows],
            [int(row["actual_label"]) for row in train_fold_rows],
            random_seed=random_seed + fold_index,
            sklearn_n_jobs=sklearn_n_jobs,
        )
        predicted = model.predict_proba([_feature_values(row) for row in test_rows])
        for index, probability in zip(test_indexes, predicted):
            probabilities[index] = float(probability)
        audit["prediction_generated"] = True
        audit["max_training_rebalance_date"] = max(
            row.get("rebalance_date", "") for row in train_fold_rows
        )
        audits.append(audit)
    return probabilities, audits


def _label_overlaps_validation(row: dict[str, str], validation_start: str) -> bool:
    label_end = row.get("label_end_date") or row.get("outcome_end_date") or ""
    return bool(label_end and label_end >= validation_start)


def _walk_forward_meta_evaluation(
    rows: list[dict[str, str]],
    model_type: str,
    fold_count: int,
    threshold: float,
    reduced_exposure: float,
    reduce_when: str,
    random_seed: int,
    calibration_bin_count: int,
    sklearn_n_jobs: int = 1,
) -> dict[str, Any]:
    unique_dates = sorted({row["rebalance_date"] for row in rows if row.get("rebalance_date")})
    if len(unique_dates) < 2:
        return {
            "validation": "chronological_meta_walk_forward_grouped_by_rebalance_date",
            "fold_count": 0,
            "folds": [],
            "summary": {},
            "research_only": True,
            "trading_impact": "none",
        }
    effective_fold_count = min(max(1, int(fold_count)), len(unique_dates) - 1)
    candidate_test_dates = unique_dates[1:]
    chunk_size = max(1, math.ceil(len(candidate_test_dates) / effective_fold_count))
    folds = []
    for fold_index in range(effective_fold_count):
        test_dates = candidate_test_dates[
            fold_index * chunk_size : (fold_index + 1) * chunk_size
        ]
        if not test_dates:
            continue
        first_test_date = test_dates[0]
        train_dates = [date for date in unique_dates if date < first_test_date]
        train_rows = [row for row in rows if row.get("rebalance_date") in train_dates]
        test_rows = [row for row in rows if row.get("rebalance_date") in set(test_dates)]
        if not train_rows or not test_rows:
            continue
        train_features = [_feature_values(row) for row in train_rows]
        train_labels = [int(row["actual_label"]) for row in train_rows]
        test_features = [_feature_values(row) for row in test_rows]
        test_labels = [int(row["actual_label"]) for row in test_rows]
        model = _fit_meta_model(
            model_type,
            train_features,
            train_labels,
            random_seed=random_seed + fold_index,
            sklearn_n_jobs=sklearn_n_jobs,
        )
        probabilities = model.predict_proba(test_features)
        predictions = [int(value >= threshold) for value in probabilities]
        calibration = build_probability_calibration(
            test_labels,
            probabilities,
            bin_count=calibration_bin_count,
        )
        overlay = _overlay_summary(
            _with_split(test_rows, "test"),
            probabilities,
            threshold,
            reduced_exposure,
            reduce_when=reduce_when,
        )
        folds.append({
            "fold": fold_index + 1,
            "train_start_date": min(train_dates),
            "train_end_date": max(train_dates),
            "test_start_date": min(test_dates),
            "test_end_date": max(test_dates),
            "train_sample_count": len(train_rows),
            "test_sample_count": len(test_rows),
            "metrics": classification_metrics(test_labels, predictions),
            "calibration": calibration,
            "overlay": overlay,
        })
    summary = _walk_forward_summary(folds)
    return {
        "validation": "chronological_meta_walk_forward_grouped_by_rebalance_date",
        "model_type": _normalize_meta_model_type(model_type),
        "fold_count": len(folds),
        "folds": folds,
        "summary": summary,
        "research_only": True,
        "trading_impact": "none",
    }


def _walk_forward_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    def average(path: tuple[str, ...]) -> float | None:
        values = []
        for fold in folds:
            value: Any = fold
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return sum(values) / len(values) if values else None

    return {
        "fold_count": len(folds),
        "balanced_accuracy": average(("metrics", "balanced_accuracy")),
        "accuracy": average(("metrics", "accuracy")),
        "brier_score": average(("calibration", "brier_score")),
        "expected_calibration_error": average(
            ("calibration", "expected_calibration_error")
        ),
        "overlay_return_delta": average(("overlay", "return_delta")),
        "overlay_max_drawdown_improvement": average(("overlay", "max_drawdown_delta")),
        "overlay_turnover": average(("overlay", "overlay_turnover")),
        "reduced_exposure_days": average(("overlay", "reduced_exposure_days")),
    }
