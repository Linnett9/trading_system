from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from core.research.ml.meta_ensemble import (
    build_meta_dataset_rows,
    _compare_meta_learners,
    _feature_values,
    _load_source_predictions,
    _overlay_summary,
    _promotion_gate_report,
    _threshold_sweep,
    _walk_forward_meta_evaluation,
)


def test_meta_ensemble_joins_predictions_by_feature_id_and_audits_leakage():
    expanded_rows = [
        _expanded("a", "2024-01-01"),
        _expanded("b", "2024-01-01"),
        _expanded("c", "2024-01-08"),
    ]
    sources = {
        "dlinear": {
            "a": _prediction("a", "2024-01-01", "out_of_fold", 1),
            "b": _prediction("b", "2024-01-01", "out_of_fold", 0),
            "c": _prediction("c", "2024-01-08", "holdout", 1),
        },
        "patchtst": {
            "a": _prediction("a", "2024-01-01", "out_of_fold", 1),
            "b": _prediction("b", "2024-01-01", "out_of_fold", 0),
            "c": _prediction("c", "2024-01-08", "holdout", 1),
        },
    }

    rows, audit = build_meta_dataset_rows(expanded_rows, sources)

    assert len(rows) == 3
    assert audit["source_model_count"] == 2
    assert audit["same_date_leakage_check"]["passed"]
    assert not audit["meta_training_uses_in_sample_base_predictions"]
    assert "dlinear_raw_probability" in rows[0]
    assert "patchtst_raw_probability" in rows[0]


def test_meta_ensemble_ingests_predicted_auxiliary_columns_without_actual_leakage():
    expanded_rows = [_expanded("a", "2024-01-01")]
    prediction = _prediction("a", "2024-01-01", "holdout", 1)
    prediction.update({
        "predicted_forward_return_5d": "0.012",
        "predicted_future_volatility": "0.18",
        "actual_forward_return_5d": "0.99",
        "actual_future_drawdown": "-0.45",
        "future_drawdown": "-0.45",
    })
    sources = {
        "multitask_transformer": {"a": prediction},
        "dlinear": {"a": _prediction("a", "2024-01-01", "holdout", 1)},
    }

    rows, audit = build_meta_dataset_rows(expanded_rows, sources)
    features = _feature_values(rows[0])

    assert rows[0]["multitask_transformer_predicted_forward_return_5d"] == "0.012"
    assert rows[0]["multitask_transformer_predicted_future_volatility"] == "0.18"
    assert "multitask_transformer_predicted_forward_return_5d" in features
    assert "multitask_transformer_predicted_future_volatility" in features
    assert all("actual_" not in name for name in features)
    assert "future_drawdown" not in features
    assert audit["auxiliary_prediction_columns_by_model"]["multitask_transformer"] == [
        "multitask_transformer_predicted_forward_return_5d",
        "multitask_transformer_predicted_future_volatility",
    ]
    assert "actual_forward_return_5d" in audit[
        "ignored_leakage_columns_by_model"
    ]["multitask_transformer"]


def test_meta_ensemble_refuses_mixed_prediction_artifact_dataset_hashes(tmp_path):
    first_dir = tmp_path / "dlinear"
    second_dir = tmp_path / "patchtst"
    _write_prediction_artifact_dir(first_dir, "dlinear", "dataset-hash-a")
    _write_prediction_artifact_dir(second_dir, "patchtst", "dataset-hash-b")

    with pytest.raises(RuntimeError, match="different dataset hashes"):
        _load_source_predictions([first_dir, second_dir])


def test_meta_ensemble_refuses_csv_metadata_dataset_hash_mismatch(tmp_path):
    source_dir = tmp_path / "dlinear"
    _write_prediction_artifact_dir(
        source_dir,
        "dlinear",
        metadata_dataset_hash="dataset-hash-a",
        csv_dataset_hash="dataset-hash-b",
    )

    with pytest.raises(RuntimeError, match="does not match metadata"):
        _load_source_predictions([source_dir])


def test_meta_ensemble_refuses_legacy_prediction_artifacts_without_csv_dataset_hash(tmp_path):
    source_dir = tmp_path / "dlinear"
    _write_prediction_artifact_dir(
        source_dir,
        "dlinear",
        metadata_dataset_hash="dataset-hash-a",
        csv_dataset_hash="",
    )

    with pytest.raises(RuntimeError, match="missing dataset_hash"):
        _load_source_predictions([source_dir])


def test_meta_ensemble_detects_same_date_split_leakage():
    expanded_rows = [_expanded("a", "2024-01-01"), _expanded("b", "2024-01-01")]
    sources = {
        "dlinear": {
            "a": _prediction("a", "2024-01-01", "out_of_fold", 1),
            "b": _prediction("b", "2024-01-01", "holdout", 0),
        },
        "patchtst": {
            "a": _prediction("a", "2024-01-01", "out_of_fold", 1),
            "b": _prediction("b", "2024-01-01", "holdout", 0),
        },
    }

    _, audit = build_meta_dataset_rows(expanded_rows, sources)

    assert not audit["same_date_leakage_check"]["passed"]


def test_meta_overlay_does_not_compound_overlapping_variant_rows():
    rows = [
        {
            "rebalance_date": f"2024-01-{(index % 10) + 1:02d}",
            "split": "holdout",
            "champion_return_next_period": "0.05",
        }
        for index in range(1_000)
    ]
    probabilities = [0.9 for _ in rows]

    summary = _overlay_summary(
        rows,
        probabilities,
        threshold=0.5,
        reduced_exposure=0.7,
        reduce_when="above_or_equal_threshold",
    )

    assert math.isfinite(summary["return_delta"])
    assert abs(summary["return_delta"]) < 1.0
    assert summary["overlay_sample_count"] == 1_000
    assert summary["reduced_exposure_days"] == 10
    assert summary["aggregation"] == "mean_by_rebalance_date_not_compounded"


def test_meta_overlay_uses_only_holdout_or_test_rows_for_reduced_days():
    rows = [
        {
            "rebalance_date": "2024-01-01",
            "split": "out_of_fold",
            "champion_return_next_period": "0.10",
        },
        {
            "rebalance_date": "2024-01-02",
            "split": "out_of_fold",
            "champion_return_next_period": "0.10",
        },
        {
            "rebalance_date": "2024-02-01",
            "split": "holdout",
            "champion_return_next_period": "0.10",
        },
        {
            "rebalance_date": "2024-02-01",
            "split": "holdout",
            "champion_return_next_period": "0.20",
        },
        {
            "rebalance_date": "2024-02-08",
            "split": "test",
            "champion_return_next_period": "-0.05",
        },
    ]
    probabilities = [0.9, 0.9, 0.9, 0.2, 0.2]

    summary = _overlay_summary(
        rows,
        probabilities,
        threshold=0.5,
        reduced_exposure=0.7,
        reduce_when="above_or_equal_threshold",
    )

    assert summary["overlay_sample_count"] == 3
    assert summary["overlay_evaluated_dates"] == 2
    assert summary["reduced_exposure_days"] == 1
    assert summary["overlay_start_date"] == "2024-02-01"
    assert summary["overlay_end_date"] == "2024-02-08"


def test_meta_overlay_uses_champion_success_direction():
    rows = [
        {
            "rebalance_date": "2024-01-01",
            "split": "holdout",
            "champion_return_next_period": "0.10",
        },
        {
            "rebalance_date": "2024-01-08",
            "split": "holdout",
            "champion_return_next_period": "0.10",
        },
    ]

    summary = _overlay_summary(
        rows,
        probabilities=[0.40, 0.60],
        threshold=0.5,
        reduced_exposure=0.7,
        reduce_when="below_threshold",
    )

    assert summary["reduced_exposure_days"] == 1
    assert summary["overlay_adjusted_return"] < summary["overlay_baseline_return"]


def test_meta_overlay_rejects_percent_style_returns():
    rows = [{
        "rebalance_date": "2024-01-01",
        "split": "holdout",
        "champion_return_next_period": "10.0",
    }]

    with pytest.raises(ValueError, match="decimal return"):
        _overlay_summary(
            rows,
            probabilities=[0.9],
            threshold=0.5,
            reduced_exposure=0.7,
            reduce_when="above_or_equal_threshold",
        )


def test_meta_walk_forward_uses_prior_dates_and_reports_summary():
    rows = [
        _meta_row("a", "2024-01-01", 0, 0.2),
        _meta_row("b", "2024-01-08", 1, 0.8),
        _meta_row("c", "2024-01-15", 0, 0.3),
        _meta_row("d", "2024-01-22", 1, 0.7),
        _meta_row("e", "2024-01-29", 1, 0.9),
    ]

    result = _walk_forward_meta_evaluation(
        rows,
        model_type="logistic_regression",
        fold_count=2,
        threshold=0.5,
        reduced_exposure=0.7,
        reduce_when="above_or_equal_threshold",
        random_seed=42,
        calibration_bin_count=5,
    )

    assert result["fold_count"] == 2
    assert result["folds"][0]["train_end_date"] < result["folds"][0]["test_start_date"]
    assert result["summary"]["balanced_accuracy"] is not None
    assert result["summary"]["overlay_return_delta"] is not None


def test_meta_learner_comparison_includes_requested_models_and_optional_lightgbm():
    train_rows = [
        _meta_row("a", "2024-01-01", 0, 0.2),
        _meta_row("b", "2024-01-08", 1, 0.8),
        _meta_row("c", "2024-01-15", 0, 0.3),
        _meta_row("d", "2024-01-22", 1, 0.7),
    ]
    holdout_rows = [
        _meta_row("e", "2024-02-01", 0, 0.4, split="holdout"),
        _meta_row("f", "2024-02-08", 1, 0.6, split="holdout"),
    ]

    result = _compare_meta_learners(
        train_rows,
        holdout_rows,
        model_types=[
            "logistic_regression",
            "ridge_logistic",
            "random_forest",
            "gradient_boosting",
            "lightgbm",
        ],
        threshold=0.5,
        reduced_exposure=0.7,
        reduce_when="above_or_equal_threshold",
        random_seed=42,
        calibration_bin_count=5,
    )

    model_types = {row["model_type"] for row in result["models"]}
    assert {
        "logistic_regression",
        "ridge_logistic",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
    } <= model_types
    assert any(row["available"] for row in result["models"])
    lightgbm = [row for row in result["models"] if row["model_type"] == "lightgbm"][0]
    assert lightgbm["available"] in {True, False}
    assert result["selections"]["selected_classifier"]["selected_model"]
    assert result["selections"]["selected_calibrated"]["selected_model"]
    assert result["selections"]["selected_overlay"]["selected_model"]
    assert "selection_reason" in result["selections"]["selected_overlay"]
    assert result["promotion_gate_ranking"][0]["promotion_gate_score"] is not None


def test_meta_threshold_sweep_and_promotion_gates_report_sanity_fields():
    rows = [
        _meta_row("a", "2024-01-01", 0, 0.2, split="holdout"),
        _meta_row("b", "2024-01-08", 1, 0.8, split="holdout"),
    ]
    probabilities = [0.2, 0.8]

    sweep = _threshold_sweep(
        rows,
        labels=[0, 1],
        probabilities=probabilities,
        thresholds=[0.5],
        reduced_exposures=[0.7],
        reduce_when="above_or_equal_threshold",
    )
    overlay = sweep["scenarios"][0]["overlay"]
    gates = _promotion_gate_report(
        metrics={"balanced_accuracy": 1.0},
        calibration={"brier_score": 0.1, "expected_calibration_error": 0.05},
        overlay=overlay,
        walk_forward={"summary": {"balanced_accuracy": 0.6}},
        config={"promotion_min_overlay_sample_count": 2},
    )

    assert sweep["best"]["decision_threshold"] == 0.5
    assert sweep["scenarios"][0]["finite_sanity_check"]["passed"]
    assert "turnover" in gates["observed"]
    assert "reduced_exposure_days" in gates["observed"]
    assert gates["checks"]["finite_sanity_check"]["passed"]


def _expanded(feature_id: str, date: str) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "feature_date": date,
        "rebalance_date": date,
        "variant_id": "variant",
        "variant_top_n": "3",
        "variant_universe_symbol_count": "32",
        "breadth_above_sma_200": "0.5",
        "spy_realized_volatility_21d": "0.2",
        "spy_max_drawdown_63d": "-0.1",
        "recent_champion_excess_return": "0.01",
        "replacements": "1",
    }


def _meta_row(
    feature_id: str,
    date: str,
    label: int,
    probability: float,
    split: str = "out_of_fold",
) -> dict[str, str]:
    row = _expanded(feature_id, date)
    return {
        **row,
        "split": split,
        "fold": "1",
        "actual_label": str(label),
        "dlinear_raw_probability": str(probability),
        "dlinear_calibrated_probability": str(probability),
        "patchtst_raw_probability": str(probability),
        "patchtst_calibrated_probability": str(probability),
        "champion_return_next_period": "0.02" if label else "-0.01",
    }


def _prediction(feature_id: str, date: str, split: str, label: int) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "date": date,
        "rebalance_date": date,
        "variant_id": "variant",
        "split": split,
        "fold": "1",
        "actual_label": str(label),
        "raw_probability": "0.6",
        "calibrated_probability": "",
    }


def _write_prediction_artifact_dir(
    path: Path,
    model_type: str,
    metadata_dataset_hash: str,
    csv_dataset_hash: str | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dataset_hash = metadata_dataset_hash if csv_dataset_hash is None else csv_dataset_hash
    rows = [
        {
            "date": "2024-01-01",
            "rebalance_date": "2024-01-01",
            "feature_id": "feature-a",
            "variant_id": "variant",
            "model_type": model_type,
            "label_type": "should_reduce_exposure",
            "split": "out_of_fold",
            "fold": "1",
            "actual_label": "0",
            "raw_probability": "0.4",
            "calibrated_probability": "",
            "prediction": "0",
            "decision_threshold": "0.5",
            "source_dataset_row_count": "100",
            "train_sample_count": "80",
            "test_sample_count": "20",
            "generated_at": "2026-06-25T00:00:00Z",
            "dataset_hash": dataset_hash,
            "research_label": "should_reduce_exposure",
        },
        {
            "date": "2024-01-08",
            "rebalance_date": "2024-01-08",
            "feature_id": "feature-b",
            "variant_id": "variant",
            "model_type": model_type,
            "label_type": "should_reduce_exposure",
            "split": "holdout",
            "fold": "holdout",
            "actual_label": "1",
            "raw_probability": "0.7",
            "calibrated_probability": "",
            "prediction": "1",
            "decision_threshold": "0.5",
            "source_dataset_row_count": "100",
            "train_sample_count": "80",
            "test_sample_count": "20",
            "generated_at": "2026-06-25T00:00:00Z",
            "dataset_hash": dataset_hash,
            "research_label": "should_reduce_exposure",
        },
    ]
    with (path / "prediction_artifacts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "prediction_artifacts.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "label_type": "should_reduce_exposure",
                "dataset_hash": metadata_dataset_hash,
                "data_hash": metadata_dataset_hash,
                "source_dataset_row_count": 100,
                "train_sample_count": 80,
                "test_sample_count": 20,
                "generated_at": "2026-06-25T00:00:00Z",
                "research_only": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
