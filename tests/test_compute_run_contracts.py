from __future__ import annotations

import pytest

from core.research.compute.run_contracts import (
    build_item_status,
    build_result_record,
    build_run_manifest,
    derive_run_status,
    metric_value,
)


def manifest(run_id: str = "run-1", **overrides):
    values = {
        "run_id": run_id, "pipeline": "selector", "stage": "campaign",
        "run_purpose": "synthetic contract test", "source_git_commit": "commit",
        "configuration_identity": "cfg", "configuration_checksum": "c",
        "machine_profile_identity": "machine",
        "requested_resource_profile_identity": "resource-profile",
        "parent_input_artifacts": [{"identity": "dataset", "checksum": "d"}],
        "expected_inventory": [
            {"item_id": "one", "ordered_position": 0},
            {"item_id": "two", "ordered_position": 1},
        ],
    }
    values.update(overrides)
    return build_run_manifest(**values)


def item(status: str, item_id: str = "one", **evidence):
    return build_item_status(
        run_identity="run-identity", item_id=item_id, ordered_position=0,
        pipeline="selector", stage="campaign", attempt_identity="attempt",
        status=status, **evidence,
    )


def test_manifest_identity_inventory_and_root_contract() -> None:
    first = manifest()
    second = manifest()
    assert first["compatibility_identity"] == second["compatibility_identity"]
    assert first["run_root_relative_path"] == "selector/campaign/run-1"
    with pytest.raises(ValueError, match="Duplicate"):
        manifest(expected_inventory=[
            {"item_id": "one", "ordered_position": 0},
            {"item_id": "one", "ordered_position": 1},
        ])
    with pytest.raises(ValueError, match="ancestry"):
        manifest(parent_input_artifacts=[])


def test_status_precedence_and_completion_evidence() -> None:
    assert derive_run_status([item("PLANNED")], inputs_valid=False) == "PLANNED"
    assert derive_run_status([item("INPUTS_READY")], inputs_valid=True) == "INPUTS_READY"
    assert derive_run_status([item("WAITING_FOR_RESOURCES")], inputs_valid=True) == "WAITING_FOR_RESOURCES"
    assert derive_run_status([item("RUNNING")], inputs_valid=True) == "RUNNING"
    complete = item(
        "COMPLETE", required_artifact_kind="STAGE",
        artifact_validation={"stage_artifact_valid": True},
    )
    assert derive_run_status(
        [complete, item("RUNNING", "two")], inputs_valid=True
    ) == "PARTIALLY_COMPLETE"
    assert derive_run_status([complete], inputs_valid=True) == "COMPONENTS_COMPLETE"
    assert derive_run_status(
        [complete], inputs_valid=True, evaluation_required=True,
        evaluation_artifacts_valid=True,
    ) == "EVALUATION_COMPLETE"
    assert derive_run_status([item("BLOCKED")], inputs_valid=True) == "BLOCKED"
    assert derive_run_status([item("FAILED")], inputs_valid=True) == "FAILED"
    with pytest.raises(ValueError, match="model/prediction"):
        item(
            "COMPLETE", required_artifact_kind="MODEL",
            predictions_required=True,
            artifact_validation={"fitted_model_valid": True},
        )


def test_metrics_and_result_records_preserve_missing_semantics() -> None:
    missing = metric_value(
        "sharpe", None, unit="ratio", population_identity="population",
        direction="HIGHER_IS_BETTER", availability="NOT_COMPUTED",
        source_artifact_identity=None,
    )
    assert missing["value"] is None
    not_applicable = metric_value(
        "trees", None, unit="count", population_identity="population",
        direction="INFORMATIONAL", availability="NOT_APPLICABLE",
        source_artifact_identity=None,
    )
    assert not_applicable["value"] is None
    with pytest.raises(ValueError, match="remain null"):
        metric_value(
            "bad", 0, unit="count", population_identity="population",
            direction="INFORMATIONAL", availability="NOT_APPLICABLE",
            source_artifact_identity=None,
        )
    result = build_result_record(
        result_identity="result", run_identity="run", item_identity="one",
        result_kind="DATA_STAGE", pipeline="data", stage="finalise",
        status="COMPLETE", artifact_identities=["artifact"],
        metrics={"sharpe": missing, "trees": not_applicable},
        counts={"rows": 10},
    )
    assert result["counts"]["rows"] == 10
