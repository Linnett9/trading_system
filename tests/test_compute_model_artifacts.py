from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from core.research.compute.artifact_contracts import (
    FITTED_MODEL_CONTRACT,
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
    canonical_checksum,
    manifest_logical_checksum,
)
from core.research.compute.artifact_storage import publish_artifact_package
from core.research.compute.model_artifacts import (
    inspect_model_metadata,
    read_trusted_model_bytes,
    validate_model_artifact_manifest,
)


def model(handler: str, metadata: dict, **overrides):
    feature_order = metadata.setdefault("feature_order", ["a", "b"])
    metadata.setdefault("feature_order_checksum", canonical_checksum(feature_order))
    base = {
        "model_id": "model", "implementation_owner": "owner",
        "model_family": "family", "model_configuration": {"alpha": 1},
        "model_configuration_checksum": canonical_checksum({"alpha": 1}),
        "random_seed": 1, "training_boundary": {"end": "2025-01-01"},
        "training_population_checksum": "population",
        "target_horizon_identity": "target:1d",
        "preprocessing_identity": "prep-v1",
    }
    base.update(metadata)
    values = {
        "artifact_id": "model-artifact",
        "artifact_type": ArtifactType.FITTED_MODEL.value,
        "artifact_subtype": "MODEL",
        "artifact_role": ArtifactRole.RESEARCH_FOLD_MODEL.value,
        "pipeline": "selector", "stage": "fold", "run_id": "run",
        "attempt_id": "attempt",
        "dataset_input_ancestry": [{"identity": "dataset", "checksum": "d"}],
        "source_artifacts": [], "configuration_identity": "cfg",
        "configuration_checksum": "c", "source_git_commit": "commit",
        "serialization_handler": handler,
        "feature_schema_identity": "schema",
        "fitted_model_contract_version": FITTED_MODEL_CONTRACT,
        "model_metadata": base,
    }
    values.update(overrides)
    return build_artifact_manifest(**values)


def test_sklearn_lightgbm_and_pytorch_metadata_contracts() -> None:
    sklearn = model("SKLEARN_PIPELINE", {
        "sklearn_version": "1.7", "preprocessing_embedded": True,
        "fitted_feature_count": 2,
    })
    validate_model_artifact_manifest(sklearn)
    broken = deepcopy(sklearn)
    broken["model_metadata"]["preprocessing_embedded"] = False
    broken["logical_checksum"] = manifest_logical_checksum(broken)
    with pytest.raises(ValueError, match="preprocessing"):
        validate_model_artifact_manifest(broken)

    lightgbm = model("LIGHTGBM_NATIVE", {
        "lightgbm_version": "4.6", "native_format": "text",
        "objective": "lambdarank", "metrics": ["ndcg"], "num_trees": 10,
        "label_gain_policy": "gain-v1",
        "grouped_ranking_configuration": {"grouped": True},
    })
    validate_model_artifact_manifest(lightgbm)

    pytorch = model("PYTORCH_STATE_DICT", {
        "pytorch_version": "2", "architecture_owner": "module.Class",
        "architecture_configuration": {"hidden": 8},
        "state_dict_file": "model/state_dict.pt", "dtype": "float32",
        "device_at_save": "cpu", "channel_order": ["a", "b"],
        "lookback": 20, "tensor_schema": "tensor-v1",
        "target_heads": ["return"],
    })
    validate_model_artifact_manifest(pytorch)
    bad_torch = deepcopy(pytorch)
    del bad_torch["model_metadata"]["architecture_owner"]
    bad_torch["logical_checksum"] = manifest_logical_checksum(bad_torch)
    with pytest.raises(ValueError, match="PyTorch"):
        validate_model_artifact_manifest(bad_torch)


def test_external_reference_and_ensemble() -> None:
    external = build_artifact_manifest(
        artifact_id="finbert", artifact_type="EXTERNAL_MODEL_REFERENCE",
        artifact_subtype="FINBERT", artifact_role="REFERENCE_DATA",
        pipeline="news", stage="scoring", run_id="run", attempt_id="attempt",
        dataset_input_ancestry=[],
        source_artifacts=[{"identity": "huggingface/repo", "checksum": "revision"}],
        configuration_identity="cfg", configuration_checksum="c",
        source_git_commit="commit",
        serialization_handler="EXTERNAL_PINNED_MODEL_REFERENCE",
        model_metadata={
            "provider": "huggingface", "repository": "ProsusAI/finbert",
            "revision": "abc", "tokenizer_revision": "abc",
            "configuration_checksum": "config",
            "scoring_configuration": {"max_length": 512},
        },
    )
    validate_model_artifact_manifest(external)

    ensemble = model("ENSEMBLE_MANIFEST", {
        "ordered_members": [
            {"artifact_identity": "a", "artifact_checksum": "1"},
            {"artifact_identity": "b", "artifact_checksum": "2"},
        ],
        "weights": [0.4, 0.6], "combination_rule": "weighted_mean",
        "required_prediction_schema": "pred-v1",
    }, artifact_type=ArtifactType.ENSEMBLE_ARTIFACT.value)
    validate_model_artifact_manifest(ensemble)


def test_metadata_inspection_does_not_load_and_trust_is_explicit(tmp_path: Path) -> None:
    manifest = model("SKLEARN_PIPELINE", {
        "sklearn_version": "1.7", "preprocessing_embedded": True,
        "fitted_feature_count": 2,
    })
    root = tmp_path / "model"
    publish_artifact_package(root, manifest, {"model/pipeline.joblib": b"not-a-pickle"})
    metadata = inspect_model_metadata(root)
    assert metadata["serialization_handler"] == "SKLEARN_PIPELINE"
    with pytest.raises(PermissionError, match="trusted"):
        read_trusted_model_bytes(
            root, "model/pipeline.joblib", trusted_artifact=False
        )
    assert read_trusted_model_bytes(
        root, "model/pipeline.joblib", trusted_artifact=True
    ) == b"not-a-pickle"
    with pytest.raises(ValueError, match="escapes"):
        read_trusted_model_bytes(root, "../outside", trusted_artifact=True)
