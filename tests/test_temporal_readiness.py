from __future__ import annotations

import json
from pathlib import Path

from core.research.ml.temporal_readiness import (
    READINESS_NOT_READY,
    audit_exposure_csv,
    audit_meta_inputs,
    audit_selector_csv,
    build_readiness_report,
    same_day_timing_convention,
    validate_immutable_output,
    validate_temporal_artifacts,
)


def test_date_only_same_day_outcome_availability_fails_closed(tmp_path):
    path = tmp_path / "selector.csv"
    _write_csv(
        path,
        [
            {
                "rebalance_date": "2024-01-05",
                "symbol": "AAPL",
                "outcome_end_date": "2024-01-05",
                "actual_forward_return_10d": "0.01",
            }
        ],
    )

    audit = audit_selector_csv(path)

    assert same_day_timing_convention()["same_day_label_eligible"] is False
    assert audit["rows_where_label_availability_equals_decision_time"] == 1
    assert audit["rows_rejected_by_fail_closed_validation"] == 1
    assert audit["all_availability_safe"] is False


def test_label_ending_on_t_is_eligible_only_after_t(tmp_path):
    path = tmp_path / "exposure.csv"
    _write_csv(
        path,
        [
            _exposure_row("a", "2024-01-05", "2024-01-05", 0),
            _exposure_row("b", "2024-01-06", "2024-01-07", 1),
        ],
    )

    audit = audit_exposure_csv(path)

    assert audit["rows_eligible_by_decision"]["2024-01-05"][
        "eligible_training_rows"
    ] == 0
    assert audit["rows_eligible_by_decision"]["2024-01-06"][
        "eligible_training_rows"
    ] == 1


def test_historical_audit_detects_missing_label_availability(tmp_path):
    path = tmp_path / "selector.csv"
    _write_csv(
        path,
        [
            {
                "rebalance_date": "2024-01-05",
                "symbol": "AAPL",
                "actual_forward_return_10d": "0.01",
            }
        ],
    )

    audit = audit_selector_csv(path)

    assert audit["rows_missing_label_availability"] == 1
    assert audit["all_availability_safe"] is False


def test_historical_audit_detects_invalid_temporal_ordering(tmp_path):
    path = tmp_path / "selector.csv"
    _write_csv(
        path,
        [
            {
                "rebalance_date": "2024-01-05",
                "feature_timestamp": "2024-01-05",
                "symbol": "AAPL",
                "label_available_timestamp": "2024-01-04",
                "outcome_end_date": "2024-01-06",
                "actual_forward_return_10d": "0.01",
            }
        ],
    )

    audit = audit_selector_csv(path)

    assert audit["rows_where_label_availability_before_feature_time"] == 1
    assert audit["rows_rejected_by_fail_closed_validation"] == 1


def test_historical_audit_detects_duplicate_decision_keys(tmp_path):
    path = tmp_path / "selector.csv"
    row = {
        "rebalance_date": "2024-01-05",
        "symbol": "AAPL",
        "outcome_end_date": "2024-01-06",
        "actual_forward_return_10d": "0.01",
    }
    _write_csv(path, [row, row])

    audit = audit_selector_csv(path)

    assert audit["duplicate_decision_key_count"] == 1


def test_exposure_audit_detects_target_derived_predictors(tmp_path):
    path = tmp_path / "exposure.csv"
    row = _exposure_row("a", "2024-01-05", "2024-01-06", 1)
    row["future_custom_outcome"] = "5.0"
    _write_csv(path, [row])

    audit = audit_exposure_csv(path)

    assert "future_custom_outcome" in audit["target_derived_columns_present_in_source"]
    assert audit["target_derived_columns_present_in_predictors"] == []


def test_meta_audit_detects_in_sample_base_predictions(tmp_path):
    expanded = tmp_path / "expanded.csv"
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    _write_csv(expanded, [_expanded_row("x", "2024-01-05", 1)])
    _write_source(source_a, "a", "x", "train")
    _write_source(source_b, "b", "x", "out_of_fold")

    audit = audit_meta_inputs(
        expanded_dataset_path=expanded,
        source_prediction_dirs=[source_a, source_b],
    )

    assert audit["all_training_source_predictions_out_of_fold"] is False
    assert audit["in_sample_base_prediction_rows"] == 1


def test_meta_audit_detects_mixed_temporal_policy_identities(tmp_path):
    expanded = tmp_path / "expanded.csv"
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    _write_csv(expanded, [_expanded_row("x", "2024-01-05", 1)])
    _write_source(source_a, "a", "x", "out_of_fold", temporal_policy={"version": 1})
    _write_source(source_b, "b", "x", "out_of_fold", temporal_policy={"version": 2})

    audit = audit_meta_inputs(
        expanded_dataset_path=expanded,
        source_prediction_dirs=[source_a, source_b],
    )

    assert audit["mixed_temporal_identities"] is True


def test_smoke_temporal_audits_match_oos_prediction_coverage(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "exposure_temporal_audit.json").write_text(
        json.dumps({"leakage_checks_passed": True, "folds": [{"fold": 1}]}),
        encoding="utf-8",
    )
    _write_csv(output / "exposure_temporal_folds.csv", [{"fold": "1"}])

    audit = validate_temporal_artifacts(output)

    assert audit["audit_exists"] is True
    assert audit["fold_count"] == 1


def test_incomplete_smoke_run_does_not_update_latest_completed(tmp_path):
    output = tmp_path / "output"
    output.mkdir()

    audit = validate_immutable_output(output)

    assert audit["latest_completed_exists"] is False
    assert audit["manifest_exists"] is False


def test_completed_smoke_run_preserves_immutable_artifacts(tmp_path):
    output = tmp_path / "output"
    run_dir = output / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (output / "latest_completed.json").write_text(
        json.dumps({"run_id": "run-1"}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_status": "complete", "artifacts": [{"path": "x"}]}),
        encoding="utf-8",
    )

    audit = validate_immutable_output(output)

    assert audit["latest_completed_exists"] is True
    assert audit["manifest_status"] == "complete"
    assert audit["artifact_count"] == 1


def test_readiness_cannot_be_ready_when_mandatory_check_fails(tmp_path):
    selector = tmp_path / "selector.csv"
    exposure = tmp_path / "exposure.csv"
    expanded = tmp_path / "expanded.csv"
    _write_csv(selector, [])
    _write_csv(exposure, [])
    _write_csv(expanded, [])

    class Paths:
        selector_config_path = Path("selector.yaml")
        exposure_config_path = Path("exposure.yaml")
        meta_config_path = Path("meta.yaml")
        selector_input_path = selector
        selector_output_dir = tmp_path / "selector_out"
        exposure_dataset_path = exposure
        exposure_output_dir = tmp_path / "exposure_out"
        meta_expanded_dataset_path = expanded
        meta_dataset_path = tmp_path / "meta.csv"
        meta_output_dir = tmp_path / "meta_out"
        meta_source_prediction_dirs: list[Path] = []

    report = build_readiness_report(
        output_dir=tmp_path / "report",
        full_suite_status={"completed": False, "passed": False},
        smoke_status={"completed": False},
        paths=Paths(),  # type: ignore[arg-type]
    )

    assert report["final_readiness_decision"] == READINESS_NOT_READY
    assert report["remaining_blockers"]


def _exposure_row(
    feature_id: str,
    decision: str,
    label_available: str,
    label: int,
) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "feature_date": decision,
        "rebalance_date": decision,
        "label_start_date": "2024-01-06",
        "label_end_date": label_available,
        "label_available_timestamp": label_available,
        "safe_feature": "1.0",
        "should_reduce_exposure": str(label),
    }


def _expanded_row(feature_id: str, date: str, label: int) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "rebalance_date": date,
        "label_end_date": "2024-01-06",
        "label_available_timestamp": "2024-01-06",
        "actual_label": str(label),
        "champion_return_next_period": "0.01",
    }


def _write_source(
    path: Path,
    model: str,
    feature_id: str,
    split: str,
    *,
    temporal_policy: dict[str, int] | None = None,
) -> None:
    path.mkdir(parents=True)
    _write_csv(
        path / "prediction_artifacts.csv",
        [
            {
                "feature_id": feature_id,
                "model_type": model,
                "split": split,
                "actual_label": "1",
                "raw_probability": "0.6",
                "calibrated_probability": "0.6",
                "dataset_hash": "hash",
            }
        ],
    )
    (path / "prediction_artifacts.json").write_text(
        json.dumps({
            "model_type": model,
            "dataset_hash": "hash",
            "validation_method": "rolling_walk_forward_out_of_fold_plus_holdout",
            "temporal_policy": temporal_policy,
        }),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["feature_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
