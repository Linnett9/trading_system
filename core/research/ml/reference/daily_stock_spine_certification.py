from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import yaml

from core.research.ml.reference.canonical_assets import file_sha256


VALIDATOR_VERSION = "daily_stock_spine_validator.v5a"
VALIDATION_CONTRACT_VERSION = "daily_stock_spine_verification.v1"
CERTIFICATION_CONTRACT_VERSION = "daily_stock_spine_certification.v2"


def build_certification_identity(
    *,
    base_path: Path,
    enriched_path: Path,
    registry_path: Path,
    registry_content_hash: str,
    aliases_path: Path,
    archive_manifest: Path,
    expected_config: Path,
) -> dict[str, Any]:
    parents = {
        "base_artifact": _artifact_identity(base_path),
        "enriched_artifact": _artifact_identity(enriched_path),
        "registry": {
            "content_hash": str(registry_content_hash or ""),
            "registry_csv": _content_identity(registry_path),
            "alias_csv": _content_identity(aliases_path),
        },
        "daily_archive": _archive_semantic_identity(archive_manifest),
        "expected_config": _config_semantic_identity(expected_config),
        "validator_version": VALIDATOR_VERSION,
        "validation_contract_version": VALIDATION_CONTRACT_VERSION,
    }
    canonical = _canonical_json(parents)
    return {
        "certification_id": hashlib.sha256(canonical).hexdigest().upper(),
        "parents": parents,
    }


def load_ready_certification(
    root: Path,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    path = certification_path(root, str(identity["certification_id"]))
    if not path.is_file():
        return None, "not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "corrupt"
    checksum = str(payload.pop("certification_payload_checksum", ""))
    if checksum != _payload_checksum(payload):
        return None, "payload_checksum_mismatch"
    required = {
        "certification_id", "parents", "validation_contract_version",
        "validator_version", "status", "verification",
    }
    if not required.issubset(payload):
        return None, "incomplete"
    if payload["status"] != "READY":
        return None, "not_ready"
    if payload["certification_id"] != identity["certification_id"]:
        return None, "certification_id_mismatch"
    if payload["parents"] != identity["parents"]:
        return None, "parent_identity_mismatch"
    if payload["validator_version"] != VALIDATOR_VERSION:
        return None, "validator_version_mismatch"
    if payload["validation_contract_version"] != VALIDATION_CONTRACT_VERSION:
        return None, "contract_version_mismatch"
    verification = payload["verification"]
    verification_required = {
        "status", "base_artifact", "enriched_artifact", "symbol_resolution",
        "alignment", "target_alignment", "spine_dataset_id",
        "price_feature_dataset_id", "logical_output_checksum",
    }
    if not isinstance(verification, dict) or not verification_required.issubset(verification):
        return None, "verification_incomplete"
    return payload, str(path)


def publish_ready_certification(
    root: Path,
    identity: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> Path:
    if verification.get("status") != "READY":
        raise ValueError("Only READY validation may be certified")
    payload = {
        "certification_contract_version": CERTIFICATION_CONTRACT_VERSION,
        "certification_id": identity["certification_id"],
        "parents": identity["parents"],
        "validation_contract_version": VALIDATION_CONTRACT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "status": "READY",
        "base_row_count": verification["base_artifact"].get("row_count"),
        "enriched_row_count": verification["enriched_artifact"].get("row_count"),
        "resolved_symbol_count": verification["symbol_resolution"].get("resolved_symbol_count"),
        "duplicate_count": verification.get("duplicate_economic_row_count"),
        "mismatch_counts": {
            "base_only": verification["alignment"].get("base_only_count"),
            "enriched_only": verification["alignment"].get("enriched_only_count"),
            "target": verification["target_alignment"].get("target_mismatch_count"),
            "benchmark": verification["target_alignment"].get("benchmark_mismatch_count"),
            "timestamp": verification["target_alignment"].get("timestamp_mismatch_count"),
        },
        "population_checksums": {
            "base": verification["base_artifact"].get("row_population_checksum"),
            "enriched": verification["enriched_artifact"].get("row_population_checksum"),
        },
        "spine_dataset_id": verification["spine_dataset_id"],
        "price_feature_dataset_id": verification["price_feature_dataset_id"],
        "semantic_output_checksum": verification["logical_output_checksum"],
        "performance_diagnostics": verification.get("streaming_diagnostics", {}),
        "verification": dict(verification),
    }
    payload["certification_payload_checksum"] = _payload_checksum(payload)
    path = certification_path(root, str(identity["certification_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing, reason = load_ready_certification(root, identity)
        if existing is None:
            raise FileExistsError(f"Incompatible immutable certification exists: {path} ({reason})")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def certification_path(root: Path, certification_id: str) -> Path:
    return Path(root) / certification_id[:2] / f"{certification_id}.json"


def _artifact_identity(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_groups = [
        {
            "index": index,
            "num_rows": metadata.row_group(index).num_rows,
            "total_byte_size": metadata.row_group(index).total_byte_size,
        }
        for index in range(metadata.num_row_groups)
    ]
    return {
        **_content_identity(path),
        "parquet_metadata": {
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "num_row_groups": metadata.num_row_groups,
            "schema": str(parquet.schema_arrow),
            "row_groups": row_groups,
        },
    }


def _content_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path).upper(),
    }


def _archive_semantic_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = (
        "status", "dataset_logical_partition_hash", "row_count", "symbol_count",
        "date_min", "date_max", "dataset_root",
    )
    return {field: payload.get(field) for field in fields}


def _config_semantic_identity(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        "semantic_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest().upper(),
    }


def verify_registry_run_binding(
    *,
    registry_manifest: Path,
    registry_path: Path,
    aliases_path: Path,
    selector_run_id: str | None,
) -> dict[str, Any]:
    """Verify the current run's Stage 2 manifest without making it certification identity."""
    manifest_path = Path(registry_manifest)
    blockers: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
        blockers.append("registry_run_binding_manifest_unreadable")
    bound_run_id = _manifest_run_id(manifest_path, payload)
    if selector_run_id and bound_run_id != selector_run_id:
        blockers.append("registry_run_binding_run_id_mismatch")
    if payload.get("status") != "READY":
        blockers.append("registry_run_binding_not_ready")
    if payload.get("publication_status") not in (None, "complete"):
        blockers.append("registry_run_binding_publication_incomplete")
    if payload.get("validation_status") != "VERIFIED":
        blockers.append("registry_run_binding_not_verified")
    if _resolved(payload.get("registry_path")) != registry_path.resolve():
        blockers.append("registry_run_binding_registry_path_mismatch")
    if str(payload.get("registry_content_checksum", "")).lower() != file_sha256(registry_path).lower():
        blockers.append("registry_run_binding_registry_checksum_mismatch")
    if _resolved(payload.get("alias_registry_path")) != aliases_path.resolve():
        blockers.append("registry_run_binding_alias_path_mismatch")
    if str(payload.get("alias_registry_checksum", "")).lower() != file_sha256(aliases_path).lower():
        blockers.append("registry_run_binding_alias_checksum_mismatch")
    return {
        "status": "BLOCKED" if blockers else "READY",
        "blockers": blockers,
        "selector_run_id": selector_run_id,
        "manifest_run_id": bound_run_id,
        "manifest_path": str(manifest_path),
        "manifest_checksum": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "registry_content_hash": (
            payload.get("registry_content_hash")
            or payload.get("row_identity_checksum")
            or payload.get("dataset_id")
        ),
    }


def _manifest_run_id(path: Path, payload: Mapping[str, Any]) -> str | None:
    explicit = payload.get("run_id")
    if explicit:
        return str(explicit)
    for parent in path.resolve().parents:
        if parent.name.startswith("run="):
            return parent.name[4:]
    return None


def _resolved(value: Any) -> Path | None:
    return Path(str(value)).resolve() if value else None


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
