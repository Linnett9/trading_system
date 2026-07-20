from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.compute.artifact_contracts import (
    ArtifactRole,
    ArtifactType,
    build_artifact_manifest,
)
from core.research.compute.artifact_storage import (
    publish_artifact_package,
    quarantine_incomplete_artifact,
    validate_artifact_package,
)


def template(*, feature: str = "schema-v1"):
    return build_artifact_manifest(
        artifact_id="stage-files-1",
        artifact_type=ArtifactType.DATASET_ARTIFACT.value,
        artifact_subtype="ROW_PACKAGE",
        artifact_role=ArtifactRole.REFERENCE_DATA.value,
        pipeline="selector", stage="fold", run_id="run", attempt_id="attempt",
        dataset_input_ancestry=[{"identity": "dataset", "checksum": "d"}],
        source_artifacts=[],
        configuration_identity="cfg", configuration_checksum="c",
        source_git_commit="commit", serialization_handler="GENERIC_STAGE_FILES",
        feature_schema_identity=feature,
    )


def test_atomic_publication_hashes_resume_and_incompatibility(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    status, manifest = publish_artifact_package(
        root, template(),
        {"model/model.bin": b"model", "model/preprocessing.bin": b"prep"},
    )
    assert status == "COMPLETE"
    assert (root / "manifest.json").exists()
    assert (root / "completion.json").exists()
    assert validate_artifact_package(root)["package_checksum"] == manifest["package_checksum"]
    assert manifest["atomic_publication_evidence"]["manifest_written_after_owned_files"]

    status, existing = publish_artifact_package(
        root, template(),
        {"model/model.bin": b"model", "model/preprocessing.bin": b"prep"},
    )
    assert status == "SKIPPED_COMPATIBLE"
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="INCOMPATIBLE_EXISTING"):
        publish_artifact_package(
            root, template(feature="changed"),
            {"model/model.bin": b"changed", "model/preprocessing.bin": b"prep"},
        )
    assert (root / "manifest.json").read_bytes() == before


def test_tamper_missing_partial_and_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    publish_artifact_package(
        root, template(),
        {"model/model.bin": b"model", "model/preprocessing.bin": b"prep"},
    )
    (root / "model/model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        validate_artifact_package(root)

    root2 = tmp_path / "artifact2"
    publish_artifact_package(
        root2, template(),
        {"model/model.bin": b"model", "model/preprocessing.bin": b"prep"},
    )
    (root2 / "model/preprocessing.bin").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_artifact_package(root2)

    partial = tmp_path / ".abandoned.writing-123"
    partial.mkdir()
    (partial / "files").mkdir()
    quarantined = quarantine_incomplete_artifact(
        partial, quarantine_root=tmp_path / "quarantine", reason="abandoned"
    )
    assert quarantined.exists() and not partial.exists()
    with pytest.raises(ValueError):
        quarantine_incomplete_artifact(
            root2, quarantine_root=tmp_path / "quarantine", reason="unsafe"
        )


def test_completion_is_required_and_control_paths_are_owned(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(template()))
    with pytest.raises(ValueError, match="partial"):
        validate_artifact_package(root)
    with pytest.raises(ValueError, match="control files"):
        publish_artifact_package(
            tmp_path / "bad", template(), {"manifest.json": b"overwrite"}
        )


def test_temporary_package_names_preserve_windows_path_budget(tmp_path: Path) -> None:
    root = (
        tmp_path
        / "components"
        / "model=ordered_logit_ranker"
        / "date=2024-03-15"
        / "shared_model_artifact"
        / "prediction"
    )

    status, manifest = publish_artifact_package(
        root,
        template(),
        {"metadata/prediction_binding.json": b"{}"},
    )

    assert status == "COMPLETE"
    assert validate_artifact_package(root)["package_checksum"] == manifest["package_checksum"]
    assert not list(root.parent.glob("*.writing-*"))
