from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.compute.artifact_contracts import (
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
    build_stage_artifact_manifest,
)
from core.research.compute.artifact_storage import publish_artifact_package
from core.research.compute.run_contracts import (
    build_item_status,
    build_result_record,
    build_run_manifest,
    metric_value,
)
from core.research.compute.run_results import build_leaderboard, results_payload
from core.research.compute.run_storage import (
    StaleRunRevision,
    build_artifact_inventory,
    initialise_run,
    publish_item_status,
    publish_failure_and_blocker_records,
    publish_artifact_inventory,
    publish_results_snapshot,
    publish_summary,
    update_run_status,
    update_global_registry_snapshot,
    validate_run_compatibility,
)


def run_manifest(run_id: str = "run"):
    return build_run_manifest(
        run_id=run_id, pipeline="data", stage="build",
        run_purpose="synthetic", source_git_commit="commit",
        configuration_identity="cfg", configuration_checksum="c",
        machine_profile_identity="machine",
        requested_resource_profile_identity="resources",
        parent_input_artifacts=[{"identity": "source", "checksum": "s"}],
        expected_inventory=[{"item_id": "item", "ordered_position": 0}],
    )


def stage_package(path: Path):
    manifest = build_stage_artifact_manifest(
        stage_owner="test.stage",
        output_counts={"rows": 1},
        schema_identity="rows-v1",
        coverage_evidence={"population": "synthetic"},
        resumability_evidence={"resume": "checksum"},
        artifact_id="stage-artifact", artifact_type=ArtifactType.DATASET_ARTIFACT.value,
        artifact_subtype="ROWS", artifact_role=ArtifactRole.REFERENCE_DATA.value,
        pipeline="data", stage="build", run_id="run", attempt_id="attempt",
        dataset_input_ancestry=[{"identity": "source", "checksum": "s"}],
        source_artifacts=[], configuration_identity="cfg",
        configuration_checksum="c", source_git_commit="commit",
    )
    publish_artifact_package(path, manifest, {"files/rows.csv": b"a,b\n1,2\n"})


def test_run_initialise_resume_revision_results_and_summary(tmp_path: Path) -> None:
    manifest = run_manifest()
    root = initialise_run(manifest, runs_root=tmp_path)
    assert initialise_run(run_manifest(), runs_root=tmp_path) == root
    assert validate_run_compatibility(root, manifest)
    incompatible = run_manifest()
    incompatible["configuration_checksum"] = "changed"
    # A correctly rebuilt incompatible identity must not attach to this root.
    rebuilt = build_run_manifest(
        **{
            "run_id": "run", "pipeline": "data", "stage": "build",
            "run_purpose": "synthetic", "source_git_commit": "commit",
            "configuration_identity": "cfg", "configuration_checksum": "changed",
            "machine_profile_identity": "machine",
            "requested_resource_profile_identity": "resources",
            "parent_input_artifacts": [{"identity": "source", "checksum": "s"}],
            "expected_inventory": [{"item_id": "item", "ordered_position": 0}],
        }
    )
    with pytest.raises(ValueError, match="INCOMPATIBLE"):
        initialise_run(rebuilt, runs_root=tmp_path)

    planned = update_run_status(root, expected_revision=-1, inputs_valid=False)
    assert planned["current_status"] == "PLANNED"
    ready = update_run_status(root, expected_revision=0, inputs_valid=True)
    assert ready["state_revision"] == 1
    with pytest.raises(StaleRunRevision):
        update_run_status(root, expected_revision=0, inputs_valid=True)

    package = tmp_path / "package"
    stage_package(package)
    complete = build_item_status(
        run_identity=manifest["run_identity"], item_id="item", ordered_position=0,
        pipeline="data", stage="build", attempt_identity="attempt",
        status="COMPLETE", required_artifact_kind="STAGE",
        stage_artifact_identity="stage-artifact",
        stage_artifact_package_path=str(package),
        artifact_validation={"stage_artifact_valid": True},
    )
    publish_item_status(root, complete)
    status = update_run_status(
        root, expected_revision=1, inputs_valid=True,
        resource_evidence={
            "reserved_ram_bytes": 100, "measured_peak_ram_bytes": 80,
            "resource_wait_seconds": 2.5, "estimate_exceeded": False,
        },
    )
    assert status["current_status"] == "COMPONENTS_COMPLETE"

    metric = metric_value(
        "rows", 10, unit="count", population_identity="rows",
        direction="INFORMATIONAL", availability="AVAILABLE",
        source_artifact_identity="stage-artifact",
    )
    result = build_result_record(
        result_identity="result", run_identity=manifest["run_identity"],
        item_identity="item", result_kind="DATA_STAGE", pipeline="data",
        stage="build", status="COMPLETE", artifact_identities=["stage-artifact"],
        metrics={"rows": metric},
    )
    publish_results_snapshot(root, [result])
    assert json.loads((root / "results.json").read_text())["records"][0]["item_identity"] == "item"
    assert "item_identity" in (root / "results.csv").read_text()

    inventory, inventory_identity = publish_artifact_inventory(root, [{
        "package_root": package, "owning_run_identity": manifest["run_identity"],
        "owning_item_identity": "item",
    }])
    assert inventory[0]["package_checksum"]
    assert inventory_identity and "stage-artifact" in (
        root / "artifact_inventory.csv"
    ).read_text()
    summary = publish_summary(root, artifact_inventory=inventory)
    assert "Components Complete" not in summary  # status is rendered as exact contract value
    assert "COMPONENTS_COMPLETE" in summary
    assert "No eligible winner exists" in summary
    assert str(package / "manifest.json") in summary

    from core.research.compute.run_contracts import blocker_record, failure_record

    failure = failure_record(
        run_identity=manifest["run_identity"], item_identity="item",
        failure_code="SYNTHETIC", failure_category="TEST", message="failed",
        phase="write", retryable=True, first_occurrence="t1",
        last_occurrence="t2",
    )
    blocker = blocker_record(
        run_identity=manifest["run_identity"], blocker_code="MISSING_INPUT",
        blocker_category="DEPENDENCY", dependency="input-x",
        affected_jobs=["item"], operator_action_required="supply input",
        automatically_resolvable=False, evidence_artifact_identity=None,
    )
    publish_failure_and_blocker_records(
        root, failures=[failure], blockers=[blocker]
    )
    assert "SYNTHETIC" in (root / "failures.csv").read_text()
    assert "MISSING_INPUT" in (root / "blockers.csv").read_text()

    corrupt_registry = tmp_path / "run_registry.json"
    corrupt_registry.write_text("{bad", encoding="utf-8")
    registry_result = update_global_registry_snapshot(
        root, registry_path=corrupt_registry
    )
    assert registry_result["health"] == "DEGRADED_REGISTRY"
    assert json.loads((root / "run_status.json").read_text())[
        "logical_checksum"
    ] == status["logical_checksum"]


def entry(identity: str, value: float, **flags):
    values = {
        "model_component_identity": identity,
        "fitted_model_valid": True, "prediction_valid": True,
        "matched_evaluation_valid": True, "safeguards_passed": True,
        "population_compatible": True, "promotion_gates_complete": True,
        "metrics": {"score": {"availability": "AVAILABLE", "value": value}},
    }
    values.update(flags)
    return values


def test_leaderboard_exclusions_ties_and_deterministic_results() -> None:
    leaderboard = build_leaderboard(
        run_identity="run", campaign_identity="campaign",
        population_identity="population", ranking_metric="score",
        ranking_direction="HIGHER_IS_BETTER",
        entries=[
            entry("b", 1.0), entry("a", 1.0),
            entry("missing-model", 9.0, fitted_model_valid=False),
            entry("missing-eval", 8.0, matched_evaluation_valid=False),
            entry("failed-safe", 7.0, safeguards_passed=False),
        ],
    )
    assert [row["model_component_identity"] for row in leaderboard["ordered_entries"]] == ["a", "b"]
    reasons = {
        row["model_component_identity"]: row["exclusion_reasons"]
        for row in leaderboard["excluded_entries"]
    }
    assert "FITTED_MODEL_MISSING_OR_INVALID" in reasons["missing-model"]
    assert "MATCHED_EVALUATION_MISSING" in reasons["missing-eval"]
    assert "SAFEGUARDS_FAILED" in reasons["failed-safe"]

    rows = [
        {"item_identity": "b", "result_identity": "2", "result_kind": "GENERIC_STAGE",
         "pipeline": "x", "stage": "y", "status": "COMPLETE", "metrics": {}},
        {"item_identity": "a", "result_identity": "1", "result_kind": "GENERIC_STAGE",
         "pipeline": "x", "stage": "y", "status": "COMPLETE", "metrics": {}},
    ]
    first, csv_first = results_payload("run", rows)
    second, csv_second = results_payload("run", list(reversed(rows)))
    assert first == second and csv_first == csv_second
    assert first["records"][0]["item_identity"] == "a"
