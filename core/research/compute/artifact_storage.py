from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .artifact_contracts import (
    ArtifactStatus,
    canonical_checksum,
    manifest_logical_checksum,
    validate_artifact_manifest,
)

MANIFEST_NAME = "manifest.json"
COMPLETION_NAME = "completion.json"


def publish_artifact_package(
    artifact_root: Path,
    manifest_template: Mapping[str, Any],
    owned_files: Mapping[str, bytes],
) -> tuple[str, dict[str, Any]]:
    root = artifact_root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}.writing-{uuid.uuid4().hex}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        inventory = []
        for relative_path in sorted(owned_files):
            destination = _owned_path(temporary, relative_path)
            if relative_path in {MANIFEST_NAME, COMPLETION_NAME}:
                raise ValueError("Owned files cannot replace package control files")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes(destination, owned_files[relative_path])
            inventory.append(_file_record(temporary, destination))
        manifest = _complete_manifest(manifest_template, inventory)
        _write_json(temporary / MANIFEST_NAME, manifest)
        completion = {
            "artifact_id": manifest["artifact_id"],
            "completion_status": ArtifactStatus.COMPLETE.value,
            "manifest_logical_checksum": manifest["logical_checksum"],
            "package_checksum": manifest["package_checksum"],
        }
        _write_json(temporary / COMPLETION_NAME, completion)
        validate_artifact_package(temporary)
        if root.exists():
            existing = validate_artifact_package(root)
            if (
                existing.get("logical_checksum") == manifest["logical_checksum"]
                and existing.get("package_checksum") == manifest["package_checksum"]
            ):
                return ArtifactStatus.SKIPPED_COMPATIBLE.value, existing
            raise FileExistsError(
                "INCOMPATIBLE_EXISTING: existing artifact ownership differs"
            )
        os.replace(temporary, root)
        _fsync_directory(root.parent)
        return ArtifactStatus.COMPLETE.value, manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_artifact_package(package_root: Path) -> dict[str, Any]:
    root = package_root.resolve()
    manifest_path = root / MANIFEST_NAME
    completion_path = root / COMPLETION_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Artifact package is partial or corrupt: {exc}") from exc
    validate_artifact_manifest(manifest)
    if manifest.get("artifact_type") in {
        "FITTED_MODEL", "TRAINING_CHECKPOINT", "ENSEMBLE_ARTIFACT",
        "EXTERNAL_MODEL_REFERENCE",
    }:
        from .model_artifacts import validate_model_artifact_manifest

        validate_model_artifact_manifest(manifest)
    if manifest.get("stage_contract_version"):
        from .artifact_contracts import validate_stage_artifact_manifest

        validate_stage_artifact_manifest(manifest)
    inventory = manifest.get("file_inventory")
    if not isinstance(inventory, list) or (
        manifest.get("artifact_type") != "EXTERNAL_MODEL_REFERENCE" and not inventory
    ):
        raise ValueError("Artifact file inventory is incomplete")
    for record in inventory:
        path = _owned_path(root, str(record.get("relative_path") or ""))
        if not path.is_file():
            raise ValueError(f"Owned artifact file missing: {record.get('relative_path')}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Owned artifact size mismatch: {record.get('relative_path')}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"Owned artifact checksum mismatch: {record.get('relative_path')}")
    if manifest.get("package_checksum") != _package_checksum(manifest):
        raise ValueError("Artifact package checksum mismatch")
    if (
        completion.get("completion_status") != ArtifactStatus.COMPLETE.value
        or completion.get("manifest_logical_checksum") != manifest.get("logical_checksum")
        or completion.get("package_checksum") != manifest.get("package_checksum")
    ):
        raise ValueError("Artifact completion evidence mismatch")
    return manifest


def resolve_compatible_artifact(
    package_root: Path, expected_manifest: Mapping[str, Any]
) -> str:
    actual = validate_artifact_package(package_root)
    if (
        actual.get("logical_checksum") == expected_manifest.get("logical_checksum")
        and actual.get("package_checksum") == expected_manifest.get("package_checksum")
    ):
        return ArtifactStatus.SKIPPED_COMPATIBLE.value
    return ArtifactStatus.INCOMPATIBLE_EXISTING.value


def quarantine_incomplete_artifact(
    path: Path, *, quarantine_root: Path, reason: str
) -> Path:
    if not reason.strip() or not path.name.startswith(".") or ".writing-" not in path.name:
        raise ValueError("Only an explicitly identified temporary package may be quarantined")
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{path.name}.{canonical_checksum(reason)[:12]}.partial"
    if target.exists():
        raise FileExistsError(target)
    os.replace(path, target)
    return target


def _complete_manifest(
    template: Mapping[str, Any], inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest = deepcopy(dict(template))
    manifest["file_inventory"] = inventory
    manifest["completion_status"] = ArtifactStatus.COMPLETE.value
    manifest["atomic_publication_evidence"] = {
        "applicable": True,
        "published": True,
        "manifest_written_after_owned_files": True,
        "completion_written_last": True,
        "temporary_sibling_directory": True,
    }
    manifest["logical_checksum"] = manifest_logical_checksum(manifest)
    manifest["compatibility_identity"] = manifest["logical_checksum"]
    manifest["package_checksum"] = _package_checksum(manifest)
    validate_artifact_manifest(manifest)
    return manifest


def _package_checksum(manifest: Mapping[str, Any]) -> str:
    payload = {
        "logical_checksum": manifest.get("logical_checksum"),
        "file_inventory": manifest.get("file_inventory", []),
    }
    return canonical_checksum(payload)


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes(
        path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _owned_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("Artifact relative path is required")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path escapes owned directory") from exc
    return path


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
