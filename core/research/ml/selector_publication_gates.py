from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.selector_dataset_lineage import logical_manifest_checksum


GATE_CONTRACT_VERSION = "selector_parent_publication_gate.v1"
SUPPORTED_DATASET_MANIFEST = "authoritative_frozen_selector_dataset_v2"
BLOCKER_PRIORITY = (
    "LEGACY_OR_ARBITRARY_ROOT", "MISSING_PARENT", "MALFORMED_MANIFEST",
    "PARENT_NOT_READY", "REGISTRY_MISMATCH", "SPINE_MISMATCH",
    "FEATURE_STORE_MISMATCH", "TARGET_CONTRACT_MISMATCH",
    "DATASET_MISMATCH", "POPULATION_MISMATCH",
    "DATE_COVERAGE_INCOMPLETE", "CHECKSUM_MISMATCH",
    "LOGICAL_CHECKSUM_MISMATCH",
)


def evaluate_selector_parent_publication_gate(
    *,
    registry_manifest: Path,
    spine_manifest: Path,
    feature_manifest: Path,
    dataset_manifest: Path,
    operational_dates_manifest: Path,
    required_operational_dates: Sequence[str],
    approved_root: Path,
) -> dict[str, Any]:
    configured = {
        "registry": registry_manifest,
        "spine": spine_manifest,
        "feature_store": feature_manifest,
        "selector_dataset": dataset_manifest,
        "operational_dates": operational_dates_manifest,
    }
    blockers: list[str] = []
    warnings: list[str] = []
    root = approved_root.resolve()
    documents: dict[str, dict[str, Any]] = {}
    manifest_checksums: dict[str, str | None] = {}

    for name, path in configured.items():
        if not _within(path, root):
            blockers.append("LEGACY_OR_ARBITRARY_ROOT")
            continue
        if not path.is_file():
            blockers.append("MISSING_PARENT")
            continue
        manifest_checksums[name] = _sha256(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append("MALFORMED_MANIFEST")
            continue
        if not isinstance(value, dict):
            blockers.append("MALFORMED_MANIFEST")
            continue
        documents[name] = value

    registry = documents.get("registry", {})
    spine = documents.get("spine", {})
    feature = documents.get("feature_store", {})
    dataset = documents.get("selector_dataset", {})
    dates_index = documents.get("operational_dates", {})

    _ready(registry, blockers, validation=("VERIFIED",))
    _ready(spine, blockers, validation=("READY", "VERIFIED"))
    _ready(feature, blockers, validation=("READY", "VERIFIED"))
    _ready(dataset, blockers, validation=("VERIFIED",))
    _ready(dates_index, blockers, validation=("READY", "VERIFIED"))

    registry_id = registry.get("dataset_id")
    registry_version = registry.get("symbol_registry_version")
    registry_content = registry.get("registry_content_checksum")
    if (
        registry.get("dataset_type") != "canonical_asset_registry_audit"
        or not registry_id or not registry_version or not registry_content
    ):
        blockers.append("REGISTRY_MISMATCH")

    spine_id = spine.get("dataset_id")
    spine_version = spine.get("schema_version")
    spine_population = (
        spine.get("row_population_checksum")
        or spine.get("row_identity_checksum")
    )
    spine_artifact = spine.get("spine_artifact_checksum")
    if (
        spine.get("dataset_type") != "canonical_daily_stock_spine"
        or not spine_id or not spine_version or not spine_population
        or not spine_artifact
    ):
        blockers.append("SPINE_MISMATCH")
    if (
        spine.get("canonical_symbol_registry_identity") != registry_id
        or spine.get("canonical_symbol_registry_version") != registry_version
        or spine.get("canonical_symbol_registry_manifest_checksum")
        != manifest_checksums.get("registry")
    ):
        blockers.append("REGISTRY_MISMATCH")

    feature_id = feature.get("dataset_id")
    feature_version = feature.get("schema_version")
    feature_sources = feature.get("source_dataset_ids")
    if (
        feature.get("dataset_type") != "daily_price_features"
        or not feature_id or not feature_version
        or feature_sources != [spine_id]
        or feature.get("source_spine_manifest_checksum")
        != manifest_checksums.get("spine")
    ):
        blockers.append("FEATURE_STORE_MISMATCH")

    if dataset.get("manifest_schema_version") != SUPPORTED_DATASET_MANIFEST:
        blockers.append("DATASET_MISMATCH")
    if dataset:
        expected_logical = logical_manifest_checksum(dataset)
        if dataset.get("logical_checksum") != expected_logical:
            blockers.append("LOGICAL_CHECKSUM_MISMATCH")
    if (
        dataset.get("symbol_registry_identity") != registry_id
        or dataset.get("symbol_registry_version") != registry_version
        or dataset.get("symbol_registry_checksum")
        != manifest_checksums.get("registry")
    ):
        blockers.append("REGISTRY_MISMATCH")
    if (
        dataset.get("daily_stock_spine_identity") != spine_id
        or dataset.get("daily_stock_spine_version") != spine_version
        or dataset.get("daily_stock_spine_checksum")
        != manifest_checksums.get("spine")
    ):
        blockers.append("SPINE_MISMATCH")
    if (
        dataset.get("daily_feature_store_identity") != feature_id
        or dataset.get("daily_feature_store_version") != feature_version
        or dataset.get("daily_feature_store_checksum")
        != manifest_checksums.get("feature_store")
    ):
        blockers.append("FEATURE_STORE_MISMATCH")

    target_id = dataset.get("target_contract")
    target_hash = dataset.get("target_contract_checksum")
    try:
        from core.research.ml.registries import RegistryResolver, load_registry_bundle

        target = RegistryResolver(load_registry_bundle()).resolve(
            "target_contracts", str(target_id), role="selector"
        )
        if target.canonical_id != target_id or target.entry.entry_hash != target_hash:
            blockers.append("TARGET_CONTRACT_MISMATCH")
    except (KeyError, ValueError):
        blockers.append("TARGET_CONTRACT_MISMATCH")

    dataset_population = dataset.get("row_population_checksum")
    if not dataset_population or dates_index.get("row_population_checksum") != dataset_population:
        blockers.append("POPULATION_MISMATCH")
    if dates_index.get("selector_dataset_id") != dataset.get("dataset_id"):
        blockers.append("DATASET_MISMATCH")
    if dates_index.get("selector_dataset_manifest_checksum") != manifest_checksums.get("selector_dataset"):
        blockers.append("CHECKSUM_MISMATCH")

    available_dates = sorted({str(value) for value in dates_index.get("available_operational_dates", [])})
    required_dates = sorted({str(value) for value in required_operational_dates})
    if not set(required_dates).issubset(available_dates):
        blockers.append("DATE_COVERAGE_INCOMPLETE")

    checksums = dataset.get("checksums", {})
    if (
        not dataset.get("dataset_checksum")
        or dataset.get("dataset_checksum") != checksums.get("rows.parquet")
        or dataset.get("feature_schema_checksum") != checksums.get("feature_schema.json")
        or dataset.get("target_schema_checksum") != checksums.get("target_schema.json")
    ):
        blockers.append("CHECKSUM_MISMATCH")

    blockers = sorted(set(blockers), key=lambda item: BLOCKER_PRIORITY.index(item))
    result = {
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "canonical_registry_id": registry_id,
        "canonical_registry_version": registry_version,
        "canonical_registry_manifest_checksum": manifest_checksums.get("registry"),
        "canonical_registry_content_checksum": registry_content,
        "daily_spine_id": spine_id,
        "daily_spine_version": spine_version,
        "daily_spine_manifest_checksum": manifest_checksums.get("spine"),
        "daily_spine_artifact_checksum": spine_artifact,
        "daily_spine_population_checksum": spine_population,
        "feature_store_id": feature_id,
        "feature_store_version": feature_version,
        "feature_store_manifest_checksum": manifest_checksums.get("feature_store"),
        "feature_store_source_spine_identity": (
            feature_sources[0] if isinstance(feature_sources, list) and len(feature_sources) == 1 else None
        ),
        "target_contract_id": target_id,
        "target_contract_entry_hash": target_hash,
        "selector_dataset_id": dataset.get("dataset_id"),
        "selector_dataset_version": dataset.get("frozen_dataset_version"),
        "selector_dataset_manifest_checksum": manifest_checksums.get("selector_dataset"),
        "selector_dataset_artifact_checksum": dataset.get("dataset_checksum"),
        "selector_ordered_population_checksum": dataset_population,
        "selector_feature_schema_checksum": dataset.get("feature_schema_checksum"),
        "selector_target_schema_checksum": dataset.get("target_schema_checksum"),
        "required_operational_dates": required_dates,
        "available_operational_dates": available_dates,
        "source_commit": dataset.get("git_commit"),
        "blockers": blockers,
        "warnings": sorted(set(warnings)),
        "status": "READY" if not blockers else "BLOCKED",
        "primary_blocker": blockers[0] if blockers else None,
    }
    result["logical_checksum"] = _logical_checksum(result)
    return result


def _ready(
    payload: Mapping[str, Any], blockers: list[str], *,
    validation: tuple[str, ...],
) -> None:
    if not payload:
        return
    if (
        payload.get("status", "READY") != "READY"
        or payload.get("publication_status") != "complete"
        or payload.get("validation_status") not in validation
    ):
        blockers.append("PARENT_NOT_READY")


def _logical_checksum(payload: Mapping[str, Any]) -> str:
    logical = {
        key: value for key, value in payload.items()
        if key not in {
            "logical_checksum", "generated_at", "creation_timestamp", "report_path",
            "canonical_registry_manifest_checksum", "daily_spine_manifest_checksum",
            "feature_store_manifest_checksum", "selector_dataset_manifest_checksum",
        }
    }
    return hashlib.sha256(
        json.dumps(
            logical, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()
