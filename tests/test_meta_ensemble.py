from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from core.research.ml.artifacts.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.meta_ensemble import (
    build_meta_dataset_rows,
    _compare_meta_learners,
    _chronological_meta_probabilities,
    _extended_horizon_rows,
    _feature_values,
    _load_source_predictions,
    _overlay_summary,
    _promotion_gate_report,
    _threshold_sweep,
    _walk_forward_meta_evaluation,
    run_meta_ensemble,
)
from core.research.ml.meta_auxiliary import (
    _chronological_cross_fitted_predictions,
    run_meta_auxiliary_ensemble,
)


EXPECTED_META_SOURCE_DIRS = [
    "reports/ml/logistic_regression_should_reduce_exposure",
    "reports/ml/random_forest_should_reduce_exposure",
    "reports/ml/gradient_boosting_should_reduce_exposure",
    "reports/ml/dlinear_should_reduce_exposure",
    "reports/ml/patchtst_should_reduce_exposure",
    "reports/ml/transformer_should_reduce_exposure",
    "reports/ml/itransformer_should_reduce_exposure",
    "reports/ml/momentum_transformer_should_reduce_exposure",
    "reports/ml/multitask_transformer_should_reduce_exposure",
    "reports/ml/market_context_encoder_should_reduce_exposure",
    "reports/ml/news_analysis_transformer_should_reduce_exposure",
    "reports/ml/tft_should_reduce_exposure",
]


def test_meta_ensemble_config_includes_all_v1_model_source_dirs():
    payload = yaml.safe_load(
        Path("configs/research/regime_transformer_meta_ensemble_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_dirs = payload["ml"]["source_prediction_dirs"]

    assert source_dirs == EXPECTED_META_SOURCE_DIRS
    assert len(source_dirs) == 12
    assert all("champion" not in source_dir for source_dir in source_dirs)


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
    expanded_rows[0]["label_start_date"] = "2024-01-02"
    expanded_rows[0]["label_end_date"] = "2024-01-31"
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
    assert rows[0][
        "multitask_transformer__predicted_forward_return_5d"
    ] == "0.012"
    assert rows[0]["multitask_transformer__predicted_future_volatility"] == "0.18"
    assert "multitask_transformer_predicted_forward_return_5d" in features
    assert "multitask_transformer_predicted_future_volatility" in features
    assert "multitask_transformer__predicted_forward_return_5d" not in features
    assert "multitask_transformer__predicted_future_volatility" not in features
    assert all("actual_" not in name for name in features)
    assert "future_drawdown" not in features
    assert rows[0]["label_end_date"] == "2024-01-31"
    assert "label_start_date" not in features
    assert "label_end_date" not in features
    assert audit["auxiliary_prediction_columns_by_model"]["multitask_transformer"] == [
        "multitask_transformer_predicted_forward_return_5d",
        "multitask_transformer_predicted_future_volatility",
    ]
    assert "actual_forward_return_5d" in audit[
        "ignored_leakage_columns_by_model"
    ]["multitask_transformer"]


def test_meta_auxiliary_predictions_and_metrics_are_generated(tmp_path):
    train_rows = [
        _auxiliary_meta_row(index, "out_of_fold")
        for index in range(8)
    ]
    holdout_rows = [
        _auxiliary_meta_row(index + 8, "holdout")
        for index in range(4)
    ]

    result = run_meta_auxiliary_ensemble(train_rows, holdout_rows, tmp_path)

    assert result.predictions_path.exists()
    assert result.metrics_json_path.exists()
    assert result.metrics_markdown_path.exists()
    assert result.metrics["train_prediction_method"] == (
        "purged_chronological_walk_forward"
    )
    assert result.metrics["fold_design"] == {
        "walk_forward_folds": 3,
        "embargo_rebalance_dates": 1,
        "purge_overlapping_labels": True,
        "date_grouping": "rebalance_date",
        "training_window": "expanding",
        "validation_window": "contiguous_future_date_blocks",
        "warmup_rows_are_forecasted": False,
        "selection_train_row_count": len(result.selection_train_indexes),
    }
    assert result.selection_train_indexes
    assert 0 not in result.selection_train_indexes
    for actual_name, metrics in result.metrics["targets"].items():
        assert metrics["available"] is True
        assert metrics["sample_count"] == 4
        assert math.isfinite(metrics["mae"])
        assert math.isfinite(metrics["rmse"])
        assert math.isfinite(metrics["pearson_correlation"])
        assert math.isfinite(metrics["spearman_correlation"])
        if "forward_return" in actual_name:
            assert math.isfinite(metrics["directional_accuracy"])
    assert all(
        "meta_predicted_forward_return_10d" in row
        for row in result.holdout_rows
    )


def test_meta_auxiliary_walk_forward_never_uses_future_targets():
    rows = [_auxiliary_meta_row(index, "out_of_fold") for index in range(12)]
    feature_names = ["transformer_raw_probability"]

    baseline, audits = _chronological_cross_fitted_predictions(
        rows,
        "actual_forward_return_5d",
        feature_names,
        fold_count=3,
        embargo_rebalance_dates=1,
        purge_overlapping_labels=False,
    )
    changed_future = [dict(row) for row in rows]
    for row in changed_future[6:]:
        row["actual_forward_return_5d"] = "1000000"
    changed, _ = _chronological_cross_fitted_predictions(
        changed_future,
        "actual_forward_return_5d",
        feature_names,
        fold_count=3,
        embargo_rebalance_dates=1,
        purge_overlapping_labels=False,
    )

    assert baseline[3:6] == pytest.approx(changed[3:6])
    assert all(
        audit["max_training_rebalance_date"] < audit["validation_start"]
        for audit in audits
        if audit["prediction_generated"]
    )
    assert all(
        audit["embargoed_rebalance_date_count"] == 1
        for audit in audits
        if audit["prediction_generated"]
    )


def test_meta_auxiliary_walk_forward_purges_overlapping_label_windows():
    rows = [_auxiliary_meta_row(index, "out_of_fold") for index in range(12)]
    for index, row in enumerate(rows):
        row["label_end_date"] = (
            f"2024-01-{min(index + 4, 28):02d}"
        )

    _, audits = _chronological_cross_fitted_predictions(
        rows,
        "actual_forward_return_5d",
        ["transformer_raw_probability"],
        fold_count=3,
        embargo_rebalance_dates=1,
        purge_overlapping_labels=True,
    )

    assert any(audit["purged_label_overlap_count"] > 0 for audit in audits)
    assert all(
        audit["max_training_rebalance_date"] < audit["validation_start"]
        for audit in audits
        if audit["prediction_generated"]
    )


def test_meta_auxiliary_missing_targets_do_not_break_classification_rows(tmp_path):
    train_rows = [_meta_row("a", "2024-01-01", 0, 0.4)]
    holdout_rows = [_meta_row("b", "2024-01-08", 1, 0.6, split="holdout")]
    features_before = _feature_values(train_rows[0])

    result = run_meta_auxiliary_ensemble(train_rows, holdout_rows, tmp_path)

    assert result.metrics["available_targets"] == []
    assert _feature_values(train_rows[0]) == features_before


def test_extended_meta_canonical_horizon_uses_existing_source_predictions_only(tmp_path):
    train_rows = [_auxiliary_meta_row(index, "out_of_fold") for index in range(12)]
    holdout_rows = [
        _auxiliary_meta_row(index + 12, "holdout")
        for index in range(2)
    ]
    auxiliary = run_meta_auxiliary_ensemble(
        train_rows,
        holdout_rows,
        tmp_path,
        walk_forward_folds=3,
    )

    horizon = _extended_horizon_rows(
        train_rows=auxiliary.train_rows,
        holdout_rows=auxiliary.holdout_rows,
        holdout_probabilities=[0.4, 0.6],
        model_type="logistic_regression",
        config={
            "meta_canonical_horizon": {
                "expand_from_source_predictions": True,
                "minimum_selection_rebalance_dates": 2,
                "walk_forward_folds": 3,
                "embargo_rebalance_dates": 1,
                "purge_overlapping_labels": True,
            }
        },
        random_seed=42,
        sklearn_n_jobs=1,
    )

    assert horizon["available"] is True
    assert horizon["audit"]["source"] == "existing_prediction_artifacts_only"
    assert horizon["audit"]["base_models_rerun"] is False
    assert horizon["audit"]["in_sample_meta_predictions"] is False
    assert min(row["rebalance_date"] for row in horizon["evaluation_rows"]) < (
        min(row["rebalance_date"] for row in holdout_rows)
    )
    assert len(horizon["evaluation_rows"]) == len(horizon["evaluation_probabilities"])
    assert len(horizon["selection_rows"]) == len(horizon["selection_probabilities"])


def test_extended_meta_probabilities_preserve_chronological_leakage_safety():
    rows = [_auxiliary_meta_row(index, "out_of_fold") for index in range(15)]

    probabilities, audits = _chronological_meta_probabilities(
        rows,
        model_type="logistic_regression",
        fold_count=3,
        embargo_rebalance_dates=1,
        purge_overlapping_labels=True,
        random_seed=42,
        sklearn_n_jobs=1,
    )

    assert any(value is not None for value in probabilities)
    assert all(
        audit["max_training_rebalance_date"] < audit["validation_start"]
        for audit in audits
        if audit["prediction_generated"]
    )
    assert all(
        audit["embargoed_rebalance_date_count"] == 1
        for audit in audits
        if audit["prediction_generated"]
    )


def test_extended_meta_horizon_has_no_operational_imports():
    source = Path("core/research/ml/meta_ensemble.py").read_text(encoding="utf-8")
    assert "infrastructure.broker" not in source
    assert "paper_trading" not in source
    assert "paper_commands" not in source
    assert "live_trading" not in source
    assert "core.entities.order" not in source
    assert "order_execution" not in source


def test_meta_ensemble_refuses_mixed_prediction_artifact_dataset_hashes(tmp_path):
    first_dir = tmp_path / "dlinear"
    second_dir = tmp_path / "patchtst"
    _write_prediction_artifact_dir(first_dir, "dlinear", "dataset-hash-a")
    _write_prediction_artifact_dir(second_dir, "patchtst", "dataset-hash-b")

    with pytest.raises(RuntimeError, match="different dataset hashes"):
        _load_source_predictions([first_dir, second_dir])


def test_meta_ensemble_refuses_missing_prediction_artifact_source(tmp_path):
    source_dir = tmp_path / "missing_model"
    source_dir.mkdir()

    with pytest.raises(RuntimeError, match="Missing prediction artifact CSV"):
        _load_source_predictions([source_dir])


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


def test_run_meta_ensemble_writes_trading_research_leaderboard_files(tmp_path):
    expanded_path = tmp_path / "expanded_rebalance_dataset.csv"
    meta_dataset_path = tmp_path / "meta_ensemble_dataset.csv"
    output_dir = tmp_path / "meta_output"
    dlinear_dir = tmp_path / "dlinear"
    patchtst_dir = tmp_path / "patchtst"
    _write_prediction_artifact_dir(dlinear_dir, "dlinear", "dataset-hash")
    _write_prediction_artifact_dir(patchtst_dir, "patchtst", "dataset-hash")
    _write_expanded_rows(expanded_path)

    result = run_meta_ensemble({
        "ml": {
            "output_dir": str(output_dir),
            "meta_dataset_path": str(meta_dataset_path),
            "expanded_rebalance_dataset_path": str(expanded_path),
            "source_prediction_dirs": [str(dlinear_dir), str(patchtst_dir)],
            "meta_model_type": "logistic_regression",
            "decision_threshold": 0.5,
            "decision_thresholds": [0.5],
            "reduced_exposures": [0.7],
            "allocation_optimizer": {"enabled": False},
        }
    })

    assert result.trading_research_leaderboard_csv_path.exists()
    assert result.trading_research_leaderboard_json_path.exists()
    assert result.trading_research_leaderboard_markdown_path.exists()
    payload = json.loads(
        result.trading_research_leaderboard_json_path.read_text(encoding="utf-8")
    )
    assert payload["research_only"] is True
    assert payload["classification_metrics_role"] == "diagnostics_only"
    assert (
        "Research only. Trading impact: none. Production validated: false."
        in result.trading_research_leaderboard_markdown_path.read_text(encoding="utf-8")
    )


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


def _write_expanded_rows(path: Path) -> None:
    rows = [
        {
            **_expanded("feature-a", "2024-01-01"),
            "champion_return_next_period": "-0.01",
            "actual_forward_return_5d": "-0.004",
            "actual_forward_return_10d": "-0.01",
            "actual_future_volatility": "0.12",
            "actual_future_drawdown": "-0.03",
            "actual_max_adverse_excursion": "-0.04",
            "actual_max_favourable_excursion": "0.01",
        },
        {
            **_expanded("feature-b", "2024-01-08"),
            "champion_return_next_period": "0.02",
            "actual_forward_return_5d": "0.01",
            "actual_forward_return_10d": "0.02",
            "actual_future_volatility": "0.10",
            "actual_future_drawdown": "-0.01",
            "actual_max_adverse_excursion": "-0.02",
            "actual_max_favourable_excursion": "0.03",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _auxiliary_meta_row(index: int, split: str) -> dict[str, str]:
    value = (index - 4) / 100.0
    return {
        "feature_id": f"aux-{index}",
        "rebalance_date": f"2024-{(index // 28) + 1:02d}-{(index % 28) + 1:02d}",
        "variant_id": "variant",
        "split": split,
        "actual_label": str(int(value < 0.0)),
        "transformer_raw_probability": str(0.4 + (index * 0.01)),
        "transformer_calibrated_probability": str(0.4 + (index * 0.01)),
        "transformer__predicted_forward_return_5d": str(value * 0.8),
        "transformer__predicted_forward_return_10d": str(value * 1.1),
        "transformer__predicted_future_volatility": str(0.10 + abs(value)),
        "transformer__predicted_future_drawdown": str(-abs(value) * 0.8),
        "transformer__predicted_max_adverse_excursion": str(-abs(value)),
        "transformer__predicted_max_favourable_excursion": str(abs(value) * 1.2),
        "actual_forward_return_5d": str(value),
        "actual_forward_return_10d": str(value * 1.2),
        "actual_future_volatility": str(0.12 + abs(value)),
        "actual_future_drawdown": str(-abs(value)),
        "actual_max_adverse_excursion": str(-abs(value) * 1.1),
        "actual_max_favourable_excursion": str(abs(value) * 1.3),
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
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "profile": "test",
            "model_name": model_type,
            "date": "2024-01-01",
            "prediction_date": "2024-01-01",
            "symbol": "",
            "rebalance_date": "2024-01-01",
            "feature_id": "feature-a",
            "variant_id": "variant",
            "config_path": "configs/research/test.yaml",
            "model_type": model_type,
            "label_type": "should_reduce_exposure",
            "split": "out_of_fold",
            "fold": "1",
            "actual_label": "0",
            "predicted_probability": "0.4",
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
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "profile": "test",
            "model_name": model_type,
            "date": "2024-01-08",
            "prediction_date": "2024-01-08",
            "symbol": "",
            "rebalance_date": "2024-01-08",
            "feature_id": "feature-b",
            "variant_id": "variant",
            "config_path": "configs/research/test.yaml",
            "model_type": model_type,
            "label_type": "should_reduce_exposure",
            "split": "holdout",
            "fold": "holdout",
            "actual_label": "1",
            "predicted_probability": "0.7",
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
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "profile": "test",
                "model_name": model_type,
                "model_type": model_type,
                "label_type": "should_reduce_exposure",
                "config_path": "configs/research/test.yaml",
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
