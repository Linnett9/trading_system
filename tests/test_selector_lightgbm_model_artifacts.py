from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from core.research.compute.artifact_storage import validate_artifact_package
from core.research.ml.stock_level.selector_lightgbm_model_artifacts import (
    publish_selector_lightgbm_model_package,
)
from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
    resolve_selector_model_package,
)


class FakeBooster:
    def __init__(self, objective: str):
        self.objective = objective

    def model_to_string(self):
        return f"tree\nobjective={self.objective}\n"

    def num_trees(self):
        return 3


def _booster(objective: str):
    return FakeBooster(objective), [[0.0], [1.0], [2.0], [3.0], [4.0]]


def _publish(tmp_path: Path, objective: str, **changes):
    booster, matrix = _booster(objective)
    prediction = tmp_path / "component" / "predictions.csv"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    schema = [
        "row_id", "asset_id", "symbol", "prediction_date", "model_id",
        "horizon_id", "selector_score", "deterministic_rank",
    ]
    with prediction.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        for index, score in enumerate(range(len(matrix))):
            writer.writerow({
                "row_id": str(index), "asset_id": f"A{index}",
                "symbol": f"A{index}", "prediction_date": "2025-01-02",
                "model_id": f"lightgbm_{objective}",
                "horizon_id": "return_10s", "selector_score": score,
                "deterministic_rank": index + 1,
            })
    gain_logical = {
        "contract_version": "lightgbm_lambdarank_gain_policy_v1",
        "gain_policy_id": "exponential_gain_quintile_0_4_v1",
        "label_contract": "within_date_quintile_relevance_v1",
        "ordered_relevance_levels": list(range(5)),
        "gain_values": [0, 1, 3, 7, 15],
        "maximum_supported_relevance": 4,
    }
    gain = {
        **gain_logical,
        "gain_checksum": hashlib.sha256(json.dumps(
            gain_logical, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest().upper(),
    }
    configuration = {
        "contract_version": f"lightgbm_{objective}_configuration_v1",
        "parameters": {
            "objective": objective, "metric": "ndcg", "n_jobs": 1,
            "n_estimators": 24,
            **({"label_gain": gain["gain_values"]}
               if objective == "lambdarank" else {}),
        },
        **({"gain_policy": gain} if objective == "lambdarank" else {}),
    }
    values = {
        "component_root": tmp_path / "component",
        "published_component_root": tmp_path / "component",
        "estimator": booster,
        "feature_order": ["signal"],
        "feature_schema_identity": "tree-schema",
        "feature_schema_checksum": "feature-checksum",
        "source_schema_guarantee_identity": "source-schema",
        "configuration": configuration,
        "input_contract": {
            "training_cutoff": "2024-12-31",
            "maximum_training_label_maturity_timestamp": "2024-12-30",
            "split_identity": "fold-1",
            "training_population_checksum": "training-population",
            "target_contract_identity": "forward_return_10d",
        },
        "group_evidence": {
            "source_group_contract_identity": "grouped_ranking_dataset_v1",
            "grouped_query_contract": "grouped_ranking_dataset_v1",
            "deterministic_group_ordering": "date_asset_row",
            "training_group_dates": ["2024-01-02"],
            "training_group_sizes": [5],
            "training_group_row_count": 5,
            "ordered_training_membership_checksum": "membership",
        },
        "ranking_label_evidence": {
            "raw_ranking_outcome_identity": "forward_return_10d",
            "relevance_label_contract": "within_date_quintile_relevance_v1",
            "ranking_label_contract_checksum": "labels",
            "ordered_training_label_checksum": "ordered-labels",
            "ordered_relevance_levels": [0, 1, 2, 3, 4],
            "label_distribution": {str(value): 1 for value in range(5)},
            "label_count": 5,
            "training_only_label_claim": True,
            "published_prediction_rows_unlabeled": True,
        },
        "model_id": f"lightgbm_{objective}",
        "prediction_path": prediction,
        "prediction_schema": schema,
        "prediction_count": 5,
        "output_population_checksum": hashlib.sha256(json.dumps(
            [str(index) for index in range(5)],
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "campaign_identity": "campaign",
        "plan_job_identity": "job",
        "component_identity": "component",
        "component_runner": "runner",
        "runtime_owner": "runtime",
        "decision_date": "2025-01-02",
        "horizon_identity": "return_10s",
        "training_row_artifact_identity": "training-rows",
        "prediction_row_artifact_identity": "prediction-rows",
        "input_package_identity": "input-package",
        "input_population_checksum": "input-population",
        "source_git_commit": "commit",
        "lightgbm_version": "4.6.0",
    }
    values.update(changes)
    return publish_selector_lightgbm_model_package(**values), booster, matrix


@pytest.mark.parametrize("objective", ["rank_xendcg", "lambdarank"])
def test_native_package_binding_and_compatible_skip(tmp_path, objective):
    result, booster, matrix = _publish(tmp_path, objective)
    model = validate_artifact_package(Path(result["model_package_path"]))
    assert model["serialization_handler"] == "LIGHTGBM_NATIVE"
    assert model["model_metadata"]["objective"] == objective
    assert model["model_metadata"]["feature_order"] == ["signal"]
    assert model["model_metadata"]["grouped_ranking_configuration"][
        "training_group_sizes"
    ] == [5]
    second, _, _ = _publish(tmp_path, objective)
    assert second["compatible_skip_status"] == "SKIPPED_COMPATIBLE"


@pytest.mark.parametrize(
    "change",
    [
        {"feature_order": ["changed"]},
        {"input_population_checksum": "changed-input"},
        {"plan_job_identity": "changed-job"},
    ],
)
def test_native_package_incompatibility_is_not_overwritten(tmp_path, change):
    _publish(tmp_path, "rank_xendcg")
    with pytest.raises(FileExistsError):
        _publish(tmp_path, "rank_xendcg", **change)


def test_native_serialization_failure_is_atomic(tmp_path):
    class Broken:
        def model_to_string(self):
            raise RuntimeError("broken")

    with pytest.raises(ValueError, match="SERIALIZATION"):
        _publish(tmp_path, "rank_xendcg", estimator=Broken())
    root = tmp_path / "component" / "shared_model_artifact" / "model"
    assert not (root / "manifest.json").exists()
    assert not (root / "completion.json").exists()


def test_group_gain_prediction_and_native_changes_fail_closed(tmp_path):
    _publish(tmp_path / "groups", "rank_xendcg")
    changed_groups = {
        "source_group_contract_identity": "grouped_ranking_dataset_v1",
        "grouped_query_contract": "grouped_ranking_dataset_v1",
        "deterministic_group_ordering": "date_asset_row",
        "training_group_dates": ["2024-01-01", "2024-01-02"],
        "training_group_sizes": [2, 3],
        "training_group_row_count": 5,
        "ordered_training_membership_checksum": "changed-membership",
    }
    with pytest.raises(FileExistsError, match="INCOMPATIBLE_EXISTING"):
        _publish(
            tmp_path / "groups", "rank_xendcg",
            group_evidence=changed_groups,
        )

    _publish(tmp_path / "native", "rank_xendcg")
    with pytest.raises(FileExistsError, match="INCOMPATIBLE_EXISTING"):
        _publish(
            tmp_path / "native", "rank_xendcg",
            estimator=FakeBooster("changed"),
        )

    with pytest.raises(ValueError, match="PREDICTION_SCHEMA_MISMATCH"):
        _publish(
            tmp_path / "schema", "rank_xendcg",
            prediction_schema=["row_id", "selector_score"],
        )
    with pytest.raises(ValueError, match="PREDICTION_COUNT_MISMATCH"):
        _publish(tmp_path / "count", "rank_xendcg", prediction_count=4)

    gain_logical = {
        "contract_version": "lightgbm_lambdarank_gain_policy_v1",
        "gain_policy_id": "exponential_gain_quintile_0_4_v1",
        "label_contract": "within_date_quintile_relevance_v1",
        "ordered_relevance_levels": list(range(5)),
        "gain_values": [0, 1, 3, 8, 15],
        "maximum_supported_relevance": 4,
    }
    changed_gain = {
        **gain_logical,
        "gain_checksum": hashlib.sha256(json.dumps(
            gain_logical, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest().upper(),
    }
    changed_configuration = {
        "contract_version": "lightgbm_lambdarank_configuration_v1",
        "parameters": {
            "objective": "lambdarank", "metric": "ndcg", "n_jobs": 1,
            "n_estimators": 24, "label_gain": changed_gain["gain_values"],
        },
        "gain_policy": changed_gain,
    }
    _publish(tmp_path / "gain", "lambdarank")
    with pytest.raises(FileExistsError, match="INCOMPATIBLE_EXISTING"):
        _publish(
            tmp_path / "gain", "lambdarank",
            configuration=changed_configuration,
        )


def test_shared_hook_resolves_complete_and_prediction_only_is_incomplete(
    tmp_path,
):
    published, _, _ = _publish(tmp_path, "rank_xendcg")
    component = tmp_path / "component"
    prediction = component / "predictions.csv"
    job = {
        "model_id": "lightgbm_rank_xendcg",
        "campaign_identity": "campaign",
        "job_id": "job",
        "prediction_date": "2025-01-02",
        "logical_checksum": "job-checksum",
    }
    manifest = {
        "publication_status": "complete",
        "production_plan_job_checksum": "job-checksum",
        "selector_model_identity": job["model_id"],
        "prediction_date": job["prediction_date"],
        "prediction_artifact_path": str(prediction),
        "prediction_checksum": hashlib.sha256(prediction.read_bytes()).hexdigest(),
    }
    manifest_path = component / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved = resolve_selector_model_package(
        job=job,
        component_result={"manifest_path": str(manifest_path)},
        run_identity="run",
    )
    assert resolved is not None
    assert resolved["artifact_identity"] == published["artifact_identity"]
    assert resolved["group_query_identity"]
    assert resolved["ranking_label_identity"]
    assert resolved["prediction_binding_identity"]

    prediction_only = tmp_path / "prediction-only"
    prediction_only.mkdir()
    copied = prediction_only / "predictions.csv"
    copied.write_bytes(prediction.read_bytes())
    prediction_only_manifest = {
        **manifest,
        "prediction_artifact_path": str(copied),
    }
    prediction_only_path = prediction_only / "manifest.json"
    prediction_only_path.write_text(
        json.dumps(prediction_only_manifest), encoding="utf-8"
    )
    assert resolve_selector_model_package(
        job=job,
        component_result={"manifest_path": str(prediction_only_path)},
        run_identity="run",
    ) is None
