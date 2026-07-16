from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LINEAGE_ASSESSMENT_VERSION = "frozen_selector_dataset_lineage_assessment_v1"
REPAIRED_MANIFEST_VERSION = "authoritative_frozen_selector_dataset_v2"
LOGICAL_CHECKSUM_EXCLUDED_FIELDS = {
    "creation_timestamp", "generated_at", "updated_at",
    "logical_checksum", "manifest_checksum",
}


def logical_manifest_checksum(payload: Mapping[str, Any]) -> str:
    logical = {
        key: value for key, value in payload.items()
        if key not in LOGICAL_CHECKSUM_EXCLUDED_FIELDS
    }
    return hashlib.sha256(
        json.dumps(
            logical, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()


def verify_dataset_lineage_manifest(
    manifest_path: Path, *, dataset_root: Path | None = None
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed selector dataset lineage manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Malformed selector dataset lineage manifest: expected object")
    required = {
        "manifest_schema_version", "dataset_id", "symbol_registry_identity",
        "symbol_registry_checksum", "daily_stock_spine_identity",
        "daily_stock_spine_checksum", "daily_feature_store_identity",
        "daily_feature_store_checksum", "target_contract",
        "target_contract_checksum", "row_population_checksum",
        "feature_schema_checksum", "target_schema_checksum",
        "builder_run_identity", "git_commit", "logical_checksum",
        "publication_status", "validation_status",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Incomplete selector dataset lineage manifest: {missing}")
    if payload["publication_status"] != "complete":
        raise ValueError("Incomplete atomic selector dataset publication")
    expected = logical_manifest_checksum(payload)
    if payload["logical_checksum"] != expected:
        raise ValueError("Selector dataset logical checksum mismatch")
    root = dataset_root or manifest_path.parent
    checksums = payload.get("checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Incomplete selector dataset checksums")
    for name in ("rows.parquet", "feature_schema.json", "target_schema.json"):
        path = root / name
        if not path.exists():
            raise ValueError(f"Incomplete atomic selector dataset publication: missing {name}")
        if checksums.get(name) != _sha256(path):
            raise ValueError(f"Selector dataset artifact checksum mismatch: {name}")
    if payload["feature_schema_checksum"] != checksums["feature_schema.json"]:
        raise ValueError("Selector dataset feature-schema mismatch")
    if payload["target_schema_checksum"] != checksums["target_schema.json"]:
        raise ValueError("Selector dataset target-schema mismatch")
    from core.research.ml.registries import RegistryResolver, load_registry_bundle

    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", str(payload["target_contract"]), role="selector"
    )
    if payload["target_contract_checksum"] != target.entry.entry_hash:
        raise ValueError("Selector dataset target-contract mismatch")
    return {
        "status": "VERIFIED",
        "dataset_id": payload["dataset_id"],
        "logical_checksum": payload["logical_checksum"],
        "dataset_bytes_changed": False,
    }


def assess_lineage_repair(*, dataset_root: Path, daily_spine_manifest: Path, symbol_registry_manifest: Path) -> dict[str, Any]:
    manifest = _read_json(dataset_root / "manifest.json") or {}; spine = _read_json(daily_spine_manifest) or {}; registry = _read_json(symbol_registry_manifest) or {}
    rows = dataset_root / "rows.parquet"; blockers = []
    if not rows.exists() or manifest.get("checksums", {}).get("rows.parquet") != _sha256(rows): blockers.append("FROZEN_DATASET_CHECKSUM_MISMATCH")
    if spine.get("status") != "READY" or spine.get("dataset_type") != "canonical_daily_stock_spine": blockers.append("AUTHORITATIVE_DAILY_SPINE_NOT_READY")
    if not spine.get("source_artifact_path") or not spine.get("source_artifact_checksum"): blockers.append("DAILY_SPINE_SOURCE_RELATIONSHIP_UNPROVEN")
    if registry.get("status") != "READY" or registry.get("dataset_type") != "canonical_asset_registry_audit": blockers.append("AUTHORITATIVE_SYMBOL_REGISTRY_NOT_READY")
    if not registry.get("registry_path") or not registry.get("registry_content_checksum"): blockers.append("SYMBOL_REGISTRY_CONTENT_RELATIONSHIP_UNPROVEN")
    exact_parent = not blockers and Path(str(spine["source_artifact_path"])).resolve() == Path(str(manifest.get("source_path", ""))).resolve() and spine["source_artifact_checksum"] == manifest.get("source_sha256")
    if not exact_parent: blockers.append("FROZEN_DATASET_PARENT_RELATIONSHIP_UNPROVEN")
    classification = "METADATA_REPUBLICATION_SAFE" if not blockers else "AUTHORITATIVE_DATASET_REBUILD_REQUIRED"
    return {"lineage_assessment_version": LINEAGE_ASSESSMENT_VERSION, "classification": classification, "status": "READY" if classification == "METADATA_REPUBLICATION_SAFE" else "BLOCKED", "dataset_root": str(dataset_root), "dataset_checksum": manifest.get("checksums", {}).get("rows.parquet"), "daily_spine_candidate": {"path": str(daily_spine_manifest), "identity": spine.get("dataset_id"), "status": spine.get("status"), "schema_version": spine.get("schema_version")}, "symbol_registry_candidate": {"path": str(symbol_registry_manifest), "identity": registry.get("dataset_id"), "status": registry.get("status"), "version": registry.get("symbol_registry_version")}, "blocking_reasons": sorted(set(blockers)), "dataset_bytes_changed": False, "model_fitting_performed": False}


def publish_metadata_republication(*, old_manifest: Path, new_manifest: Path, spine: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    old = json.loads(old_manifest.read_text(encoding="utf-8")); rows = old_manifest.parent / "rows.parquet"
    if old.get("checksums", {}).get("rows.parquet") != _sha256(rows): raise ValueError("Unsafe metadata republication: dataset bytes changed")
    if spine.get("status") != "READY" or spine.get("source_artifact_checksum") != old.get("source_sha256") or Path(str(spine.get("source_artifact_path", ""))).resolve() != Path(str(old.get("source_path", ""))).resolve(): raise ValueError("Unsafe metadata republication: daily-spine relationship unproven")
    if registry.get("status") != "READY" or not registry.get("registry_content_checksum") or not registry.get("registry_path"): raise ValueError("Unsafe metadata republication: symbol-registry relationship unproven")
    payload = {**old, "manifest_schema_version": REPAIRED_MANIFEST_VERSION, "previous_manifest": {"path": str(old_manifest), "checksum": _sha256(old_manifest)}, "daily_stock_spine_identity": spine["dataset_id"], "daily_stock_spine_version": spine["schema_version"], "daily_stock_spine_checksum": logical_manifest_checksum(spine), "symbol_registry_identity": registry["dataset_id"], "symbol_registry_version": registry["symbol_registry_version"], "symbol_registry_checksum": logical_manifest_checksum(registry), "creation_timestamp": datetime.now(timezone.utc).isoformat(), "publication_status": "complete", "validation_status": "VERIFIED", "dataset_bytes_changed": False}
    payload["logical_checksum"] = logical_manifest_checksum(payload)
    payload["manifest_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    if new_manifest.exists():
        existing = json.loads(new_manifest.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key not in {"creation_timestamp", "manifest_checksum"}} == {key: value for key, value in payload.items() if key not in {"creation_timestamp", "manifest_checksum"}}: return {"status": "skipped_identical", "path": str(new_manifest)}
        raise FileExistsError(f"Immutable repaired manifest conflict: {new_manifest}")
    new_manifest.parent.mkdir(parents=True, exist_ok=True); temp = new_manifest.with_name(f".{new_manifest.name}.{os.getpid()}.tmp"); temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); os.replace(temp, new_manifest)
    return {"status": "published", "path": str(new_manifest), "dataset_bytes_changed": False}


def _read_json(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError): return None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()
