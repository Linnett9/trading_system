from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.research.compute.artifact_storage import validate_artifact_package
from core.research.ml.stock_level.multi_horizon_linear_selector import (
    FittedMultiHorizonMember,
    HORIZON_IDS,
)
from core.research.ml.stock_level.selector_multihorizon_model_artifacts import (
    publish_selector_multihorizon_package,
    validate_multihorizon_artifacts,
)
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    resolve_selector_model_package,
)


class FakeEstimator:
    def __init__(self, marker: str):
        self.marker = marker

    def get_params(self, deep=True):
        return {"marker": self.marker}


def _members(family: str) -> list[FittedMultiHorizonMember]:
    return [
        FittedMultiHorizonMember(
            estimator=FakeEstimator(f"{family}-{horizon}"),
            model_family=family,
            horizon_id=horizon,
            horizon_order=index,
            family_order=0,
            member_order=index,
            ordered_feature_ids=("feature_a", "feature_b"),
            preprocessing={
                "contract_version": "multi_horizon_preprocessing.v1",
                "horizon_id": horizon,
                "ordered_feature_ids": ["feature_a", "feature_b"],
                "location": [float(index), 1.0],
                "scale": [1.0, 2.0],
            },
            estimator_configuration={"alpha": 1.0 + index},
            random_state_identity=(
                0 if family == "elastic_net"
                else "NOT_APPLICABLE_DETERMINISTIC"
            ),
            target_identity=f"target-{horizon}",
            training_population={
                "training_checksum": f"population-{horizon}",
                "maximum_label_maturity_timestamp": "2024-12-30",
            },
            fold_identity="fold-1",
            training_cutoff="2024-12-31",
            input_identity="input-checksum",
            configuration_identity="configuration-checksum",
        )
        for index, horizon in enumerate(HORIZON_IDS)
    ]


def _publish(tmp_path: Path, family: str = "ridge", **changes):
    rows = [
        {
            "row_id": "row-1",
            "asset_id": "asset-1",
            "symbol": "AAA",
            "prediction_date": "2025-01-02",
        },
        {
            "row_id": "row-2",
            "asset_id": "asset-2",
            "symbol": "BBB",
            "prediction_date": "2025-01-02",
        },
    ]
    predictions = [
        {
            "row_id": row["row_id"],
            "model_family": family,
            "horizon_id": horizon,
            "predicted_return": index + horizon_index / 10,
        }
        for index, row in enumerate(rows)
        for horizon_index, horizon in enumerate(HORIZON_IDS)
    ]
    values = {
        "component_root": tmp_path / "component",
        "published_component_root": tmp_path / "component",
        "fitted_members": _members(family),
        "fit_result": {
            "predictions": predictions,
            "configuration": {"model_families": [family]},
            "configuration_checksum": "fit-configuration",
        },
        "selected_component_rows": rows,
        "model_id": f"multi_horizon_{family}",
        "campaign_identity": "campaign",
        "plan_job_identity": "job",
        "component_identity": "component",
        "component_runner": "runner",
        "runtime_owner": "runtime",
        "decision_date": "2025-01-02",
        "training_row_artifact_identity": "training-rows",
        "prediction_row_artifact_identity": "prediction-rows",
        "input_package_identity": "input-package",
        "source_schema_guarantee_identity": "schema",
        "input_population_checksum": "input-checksum",
        "source_git_commit": "commit",
    }
    values.update(changes)
    return publish_selector_multihorizon_package(**values)


@pytest.mark.parametrize("family", ["ridge", "elastic_net"])
def test_publishes_ordered_members_ensemble_and_wide_predictions(
    tmp_path, family
):
    result = _publish(tmp_path, family)
    validated = validate_multihorizon_artifacts(
        component_root=tmp_path / "component",
        expected_horizons=HORIZON_IDS,
    )
    assert result["ordered_horizons"] == list(HORIZON_IDS)
    assert len(result["member_package_paths"]) == 4
    assert len(validated["members"]) == 4
    metadata = validated["ensemble"]["model_metadata"]
    assert metadata["ordered_horizons"] == list(HORIZON_IDS)
    assert metadata["member_count"] == metadata["expected_member_count"] == 4
    assert metadata["combination_rule"] == (
        "ordered_horizon_vector_no_scalar_aggregation"
    )
    assert all(
        "model/estimator.joblib" not in row["relative_path"]
        for row in validated["ensemble"]["file_inventory"]
    )
    prediction = (
        tmp_path / "component" / "shared_model_artifact" / "prediction"
        / "predictions" / "multi_horizon_predictions.csv"
    )
    with prediction.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert list(reader.fieldnames or ()) == [
            "row_id", "asset_id", "symbol", "prediction_date",
            *[f"selector_score_{horizon}" for horizon in HORIZON_IDS],
        ]
        assert len(list(reader)) == 2


def test_rejects_missing_duplicate_and_unexpected_members(tmp_path):
    members = _members("ridge")
    with pytest.raises(ValueError, match="ENSEMBLE_MEMBER_COUNT_MISMATCH"):
        _publish(tmp_path / "missing", fitted_members=members[:-1])
    with pytest.raises(ValueError, match="MEMBER_HORIZON_DUPLICATE"):
        _publish(
            tmp_path / "duplicate",
            fitted_members=[*members[:-1], replace(members[-1], horizon_id="return_10s")],
        )
    with pytest.raises(ValueError, match="MEMBER_HORIZON_UNEXPECTED"):
        _publish(
            tmp_path / "unexpected",
            fitted_members=[*members[:-1], replace(members[-1], horizon_id="return_99s")],
        )
    reversed_order = [
        replace(row, horizon_order=len(HORIZON_IDS) - 1 - row.horizon_order)
        for row in members
    ]
    with pytest.raises(ValueError, match="ENSEMBLE_MEMBER_COUNT_MISMATCH"):
        _publish(tmp_path / "reversed", fitted_members=reversed_order)


def test_exact_republish_skips_and_changed_estimator_is_incompatible(tmp_path):
    first = _publish(tmp_path)
    second = _publish(tmp_path)
    assert first["compatible_skip_status"] == "COMPLETE"
    assert second["compatible_skip_status"] == "SKIPPED_COMPATIBLE"
    members = _members("ridge")
    members[0] = replace(members[0], estimator=FakeEstimator("changed"))
    with pytest.raises(FileExistsError):
        _publish(tmp_path, fitted_members=members)


def test_validation_fails_closed_for_tampered_member_and_prediction(tmp_path):
    _publish(tmp_path)
    member_manifest = (
        tmp_path / "component" / "shared_model_artifact" / "members"
        / HORIZON_IDS[0] / "manifest.json"
    )
    payload = json.loads(member_manifest.read_text(encoding="utf-8"))
    payload["package_checksum"] = "bad"
    member_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="MEMBER_MODEL_PACKAGE_CORRUPT"):
        validate_multihorizon_artifacts(
            component_root=tmp_path / "component",
            expected_horizons=HORIZON_IDS,
        )


def test_metadata_validation_does_not_deserialize_estimators(tmp_path, monkeypatch):
    _publish(tmp_path)
    monkeypatch.setattr(
        "joblib.load",
        lambda *args, **kwargs: pytest.fail("must not deserialize"),
    )
    result = validate_multihorizon_artifacts(
        component_root=tmp_path / "component",
        expected_horizons=HORIZON_IDS,
    )
    assert result["ensemble"]["completion_status"] == "COMPLETE"
    assert validate_artifact_package(
        tmp_path / "component" / "shared_model_artifact" / "prediction"
    )["completion_status"] == "COMPLETE"


def test_shared_hook_resolves_complete_evidence_and_rejects_partial(tmp_path):
    published = _publish(tmp_path)
    component = tmp_path / "component"
    component_prediction = component / "predictions.csv"
    component_prediction.write_text(
        "row_id,selector_score\nrow-1,0.1\n", encoding="utf-8"
    )
    job = {
        "model_id": "multi_horizon_ridge",
        "campaign_identity": "campaign",
        "job_id": "job",
        "prediction_date": "2025-01-02",
        "logical_checksum": "job-checksum",
    }
    manifest = {
        "publication_status": "complete",
        "production_plan_job_checksum": "job-checksum",
        "selector_model_identity": "multi_horizon_ridge",
        "prediction_date": "2025-01-02",
        "prediction_artifact_path": str(component_prediction),
        "prediction_checksum": hashlib.sha256(
            component_prediction.read_bytes()
        ).hexdigest(),
        "experiment_run_id": "component",
    }
    manifest_path = component / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved = resolve_selector_model_package(
        job=job,
        component_result={"manifest_path": str(manifest_path)},
        run_identity="shared-run",
    )
    assert resolved is not None
    assert resolved["artifact_identity"] == published["artifact_identity"]
    assert resolved["ordered_horizons"] == list(HORIZON_IDS)
    assert len(resolved["ordered_member_checksums"]) == 4

    missing = (
        component / "shared_model_artifact" / "members" / HORIZON_IDS[-1]
    )
    missing.rename(missing.with_name("missing-member"))
    with pytest.raises(ValueError, match="MEMBER_MODEL_PACKAGE_MISSING"):
        resolve_selector_model_package(
            job=job,
            component_result={"manifest_path": str(manifest_path)},
            run_identity="shared-run",
        )
