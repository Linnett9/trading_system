from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_BUILD_MANIFEST_VERSION = "dataset_build_manifest_v1"
DATASET_LINEAGE_CHECK_VERSION = "dataset_lineage_check_v1"

STATUS_CURRENT = "CURRENT"
STATUS_STALE = "STALE"
STATUS_UNVERIFIED = "UNVERIFIED"
STATUS_MISSING_PARENT = "MISSING_PARENT"
STATUS_CONFLICTING_PARENT = "CONFLICTING_PARENT"
STATUS_LEGACY_NO_MANIFEST = "LEGACY_NO_MANIFEST"

PERMITTED_PROMOTION = "PROMOTION_GRADE"
PERMITTED_RESEARCH = "RESEARCH_ONLY"
PERMITTED_DIAGNOSTIC = "DIAGNOSTIC_ONLY"
PERMITTED_BLOCKED = "BLOCKED"

_PERMITTED_ORDER = {
    PERMITTED_BLOCKED: 0,
    PERMITTED_DIAGNOSTIC: 1,
    PERMITTED_RESEARCH: 2,
    PERMITTED_PROMOTION: 3,
}

_HASH_FIELDS_TO_EXCLUDE = {"manifest_hash"}
_CSV_SUFFIXES = {".csv", ".tsv"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


@dataclass(frozen=True)
class DatasetLineageExpectation:
    dataset_id: str | None = None
    dataset_type: str | None = None
    schema_version: str | None = None
    producer_command: str | None = None
    producer_module: str | None = None
    canonical_price_authority_version: str | None = None
    universe_authority_version: str | None = None
    identity_authority_version: str | None = None
    corporate_action_authority_version: str | None = None
    market_calendar_authority_version: str | None = None
    market_calendar_authority: Mapping[str, Any] = field(default_factory=dict)
    target_contract_version: str | None = None
    feature_code_version: str | None = None
    label_code_version: str | None = None
    configuration_hash: str | None = None
    random_seed: int | str | None = None
    source_content_hashes: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "DatasetLineageExpectation":
        payload = dict(values or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: payload[key] for key in known if key in payload})


def dataset_manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.name}.manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_identity(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": None,
        "size_bytes": None,
    }
    if path.exists() and path.is_file():
        result["sha256"] = file_sha256(path)
        result["size_bytes"] = path.stat().st_size
    return result


def code_version_hash(paths: Iterable[Path]) -> str:
    identities = []
    for path in sorted({Path(value) for value in paths}, key=lambda item: item.as_posix()):
        identities.append(file_identity(path))
    return canonical_hash(identities)


def configuration_hash(payload: Any) -> str:
    return canonical_hash(payload)


def stable_source_content_hash(name: str, payload: Any) -> tuple[str, str]:
    return (name, canonical_hash(payload))


def manifest_hash(payload: Mapping[str, Any]) -> str:
    identity = {
        key: value for key, value in payload.items()
        if key not in _HASH_FIELDS_TO_EXCLUDE
    }
    return canonical_hash(identity)


def serialize_manifest(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = serialize_manifest(payload)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, path)


def build_dataset_build_manifest(
    *,
    dataset_id: str,
    dataset_type: str,
    schema_version: str,
    producer_command: str,
    producer_module: str,
    output_paths: Sequence[Path],
    source_paths: Sequence[Path] = (),
    source_dataset_ids: Sequence[str] = (),
    source_manifest_paths: Sequence[Path] = (),
    source_content_hashes: Mapping[str, str] | None = None,
    canonical_price_authority_version: str | None = None,
    universe_authority_version: str | None = None,
    identity_authority_version: str | None = None,
    corporate_action_authority_version: str | None = None,
    market_calendar_authority_version: str | None = None,
    market_calendar_authority: Mapping[str, Any] | None = None,
    target_contract_version: str | None = None,
    feature_code_version: str | None = None,
    label_code_version: str | None = None,
    configuration_hash_value: str | None = None,
    random_seed: int | str | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    dataset_path: Path | None = None,
    key_fields: Sequence[str] | None = None,
    partition_information: Mapping[str, Any] | None = None,
    parent_artifact_ids: Sequence[str] = (),
    rebuildability_status: str = "REBUILDABLE_FROM_MANIFEST",
    build_timestamp: str | None = None,
    source_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_control_identity = dict(source_control or _source_provenance())
    if rows is not None:
        stats = dataset_stats_from_rows(rows, key_fields=key_fields)
    elif dataset_path is not None:
        stats = dataset_stats(dataset_path, key_fields=key_fields)
    else:
        stats = _empty_dataset_stats(key_fields or ())
    outputs = [file_identity(path) for path in output_paths]
    parent_manifests = [
        {
            "path": str(path),
            "dataset_id": _manifest_dataset_id(path),
            "sha256": file_sha256(path) if path.exists() else None,
            "exists": path.exists(),
        }
        for path in sorted({Path(value) for value in source_manifest_paths}, key=lambda item: item.as_posix())
    ]
    payload: dict[str, Any] = {
        "manifest_schema_version": DATASET_BUILD_MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "dataset_type": dataset_type,
        "schema_version": schema_version,
        "build_timestamp": build_timestamp or datetime.now(timezone.utc).isoformat(),
        "producer_command": producer_command,
        "producer_module": producer_module,
        "source_paths": [
            file_identity(path)
            for path in sorted({Path(value) for value in source_paths}, key=lambda item: item.as_posix())
        ],
        "source_dataset_ids": sorted(str(value) for value in source_dataset_ids),
        "source_manifest_hashes": parent_manifests,
        "source_content_hashes": {
            str(key): str(value).upper()
            for key, value in sorted(dict(source_content_hashes or {}).items())
        },
        "canonical_price_authority_version": canonical_price_authority_version,
        "universe_authority_version": universe_authority_version,
        "identity_authority_version": identity_authority_version,
        "corporate_action_authority_version": corporate_action_authority_version,
        "market_calendar_authority_version": market_calendar_authority_version,
        "market_calendar_authority": _market_calendar_authority_payload(
            market_calendar_authority,
            market_calendar_authority_version=market_calendar_authority_version,
        ),
        "target_contract_version": target_contract_version,
        "feature_code_version": feature_code_version,
        "label_code_version": label_code_version,
        "configuration_hash": configuration_hash_value,
        "random_seed": random_seed,
        "row_count": stats["row_count"],
        "key_count": stats["key_count"],
        "duplicate_key_count": stats["duplicate_key_count"],
        "symbol_entity_count": stats["symbol_entity_count"],
        "earliest_decision_timestamp": stats["earliest_decision_timestamp"],
        "latest_decision_timestamp": stats["latest_decision_timestamp"],
        "earliest_knowledge_cutoff": stats["earliest_knowledge_cutoff"],
        "latest_knowledge_cutoff": stats["latest_knowledge_cutoff"],
        "partition_information": {
            "format": _dataset_format(output_paths[0]) if output_paths else None,
            "partitioned": False,
            "key_fields": list(key_fields or stats["key_fields"]),
            **dict(partition_information or {}),
        },
        "output_hashes": outputs,
        "parent_artifact_ids": sorted(str(value) for value in parent_artifact_ids),
        "source_control": source_control_identity,
        "dirty_tree": bool(source_control_identity.get("dirty_worktree")),
        "rebuildability_status": rebuildability_status,
    }
    payload["manifest_hash"] = manifest_hash(payload)
    return payload


def dataset_stats(path: Path, *, key_fields: Sequence[str] | None = None) -> dict[str, Any]:
    rows = _read_dataset_rows(path)
    return dataset_stats_from_rows(rows, key_fields=key_fields)


def dataset_stats_from_rows(
    rows: Sequence[Mapping[str, Any]], *, key_fields: Sequence[str] | None = None
) -> dict[str, Any]:
    fieldnames = _fieldnames(rows)
    resolved_key_fields = tuple(key_fields or _default_key_fields(fieldnames))
    keys = [
        tuple(_normalize_cell(row.get(field)) for field in resolved_key_fields)
        for row in rows
    ]
    unique_keys = set(keys)
    symbols = set()
    decision_values = []
    cutoff_values = []
    for row in rows:
        for field in ("asset_id", "canonical_symbol", "symbol"):
            value = _normalize_cell(row.get(field))
            if value:
                symbols.add(value.upper())
        selected = _normalize_cell(row.get("selected_symbols"))
        if selected:
            symbols.update(item.strip().upper() for item in selected.split(",") if item.strip())
        decision = _first_present(row, ("decision_timestamp", "decision_session_date", "feature_date", "rebalance_date"))
        cutoff = _first_present(row, ("knowledge_cutoff", "knowledge_cutoff_timestamp", "label_available_timestamp", "label_end_date", "outcome_end_date"))
        if decision:
            decision_values.append(decision)
        if cutoff:
            cutoff_values.append(cutoff)
    return {
        "row_count": len(rows),
        "key_fields": list(resolved_key_fields),
        "key_count": len(unique_keys),
        "duplicate_key_count": len(keys) - len(unique_keys),
        "symbol_entity_count": len(symbols),
        "earliest_decision_timestamp": min(decision_values) if decision_values else None,
        "latest_decision_timestamp": max(decision_values) if decision_values else None,
        "earliest_knowledge_cutoff": min(cutoff_values) if cutoff_values else None,
        "latest_knowledge_cutoff": max(cutoff_values) if cutoff_values else None,
    }


def check_dataset_lineage(
    *,
    dataset_path: Path | None = None,
    manifest_path: Path | None = None,
    expected: DatasetLineageExpectation | Mapping[str, Any] | None = None,
    intended_use: str = PERMITTED_RESEARCH,
) -> dict[str, Any]:
    expected_values = (
        expected
        if isinstance(expected, DatasetLineageExpectation)
        else DatasetLineageExpectation.from_mapping(expected)
    )
    resolved_manifest = _resolve_manifest_path(dataset_path, manifest_path)
    resolved_dataset = Path(dataset_path) if dataset_path is not None else None
    intended = normalize_intended_use(intended_use)
    if resolved_manifest is None or not resolved_manifest.exists():
        return _result(
            status=STATUS_LEGACY_NO_MANIFEST,
            reasons=("DATASET_MANIFEST_MISSING",),
            dataset_path=resolved_dataset,
            manifest_path=resolved_manifest,
            intended_use=intended,
            lineage=[],
        )
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _result(
            status=STATUS_UNVERIFIED,
            reasons=(f"MANIFEST_UNREADABLE:{type(exc).__name__}",),
            dataset_path=resolved_dataset,
            manifest_path=resolved_manifest,
            intended_use=intended,
            lineage=[],
        )
    if not isinstance(manifest, dict):
        return _result(
            status=STATUS_UNVERIFIED,
            reasons=("MANIFEST_NOT_OBJECT",),
            dataset_path=resolved_dataset,
            manifest_path=resolved_manifest,
            intended_use=intended,
            lineage=[],
        )
    dataset = resolved_dataset or _primary_output_path(manifest)
    reasons: list[str] = []
    changed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    conflicts: list[str] = []

    if manifest.get("manifest_schema_version") != DATASET_BUILD_MANIFEST_VERSION:
        reasons.append("MANIFEST_SCHEMA_VERSION_UNSUPPORTED")
    recorded_hash = manifest.get("manifest_hash")
    if recorded_hash:
        expected_hash = manifest_hash(manifest)
        if str(recorded_hash).upper() != expected_hash:
            conflicts.append("MANIFEST_HASH_MISMATCH")
            reasons.append("MANIFEST_HASH_MISMATCH")
    else:
        reasons.append("MANIFEST_HASH_MISSING")

    _check_required_contract_fields(manifest, reasons)
    _check_expected_fields(manifest, expected_values, reasons, changed)
    _check_source_content_hashes(manifest, expected_values, reasons, changed)
    _check_source_paths(manifest, reasons, changed, missing)
    _check_source_manifests(manifest, reasons, changed, missing, conflicts)
    _check_output_hashes(manifest, reasons, changed, missing)
    _check_dataset_statistics(manifest, dataset, reasons, changed, missing)

    source_control = manifest.get("source_control")
    dirty = bool(manifest.get("dirty_tree") or (
        isinstance(source_control, Mapping) and source_control.get("dirty_worktree")
    ))
    if dirty:
        reasons.append("DIRTY_TREE_BUILD")
    if manifest.get("rebuildability_status") not in {
        "REBUILDABLE_FROM_MANIFEST",
        "REBUILDABLE",
        "REPRODUCIBLE",
    }:
        reasons.append("REBUILDABILITY_UNVERIFIED")

    status = _status_from_findings(reasons, changed, missing, conflicts, dirty)
    lineage = _dataset_lineage(manifest, resolved_manifest)
    return _result(
        status=status,
        reasons=tuple(sorted(set(reasons))),
        changed_parents=changed,
        missing_parents=missing,
        dataset_path=dataset,
        manifest_path=resolved_manifest,
        manifest_version=manifest.get("manifest_schema_version"),
        intended_use=intended,
        lineage=lineage,
    )


def normalize_intended_use(value: str | None) -> str:
    text = str(value or PERMITTED_RESEARCH).strip().upper().replace("-", "_")
    aliases = {
        "PROMOTION": PERMITTED_PROMOTION,
        "PROMOTION_GRADE": PERMITTED_PROMOTION,
        "PRODUCTION": PERMITTED_PROMOTION,
        "RESEARCH": PERMITTED_RESEARCH,
        "RESEARCH_ONLY": PERMITTED_RESEARCH,
        "DIAGNOSTIC": PERMITTED_DIAGNOSTIC,
        "DIAGNOSTIC_ONLY": PERMITTED_DIAGNOSTIC,
    }
    if text not in aliases:
        raise ValueError(f"Unsupported intended use: {value}")
    return aliases[text]


def _result(
    *,
    status: str,
    reasons: Sequence[str],
    dataset_path: Path | None,
    manifest_path: Path | None,
    intended_use: str,
    changed_parents: Sequence[Mapping[str, Any]] = (),
    missing_parents: Sequence[Mapping[str, Any]] = (),
    manifest_version: Any = None,
    lineage: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    permitted = _permitted_use(status, reasons)
    authorized = (
        permitted != PERMITTED_BLOCKED
        and _PERMITTED_ORDER[permitted] >= _PERMITTED_ORDER[intended_use]
    )
    final_reasons = sorted(set(reasons))
    if not authorized:
        final_reasons = sorted(set(final_reasons) | {f"INTENDED_USE_NOT_PERMITTED:{intended_use}"})
    return {
        "lineage_check_version": DATASET_LINEAGE_CHECK_VERSION,
        "status": status,
        "reasons": final_reasons,
        "changed_parents": _stable_rows(changed_parents),
        "missing_parents": _stable_rows(missing_parents),
        "manifest_version": manifest_version,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "dataset_lineage": list(lineage),
        "intended_use": intended_use,
        "permitted_use": permitted,
        "use_authorized": authorized,
        "dataset_rebuilt": False,
        "dataset_modified": False,
        "source_modified": False,
    }


def _permitted_use(status: str, reasons: Sequence[str]) -> str:
    if status == STATUS_CURRENT:
        return PERMITTED_PROMOTION
    if status == STATUS_UNVERIFIED:
        return PERMITTED_RESEARCH
    if status == STATUS_LEGACY_NO_MANIFEST:
        return PERMITTED_DIAGNOSTIC
    return PERMITTED_BLOCKED


def _status_from_findings(
    reasons: Sequence[str],
    changed: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    conflicts: Sequence[str],
    dirty: bool,
) -> str:
    if conflicts:
        return STATUS_CONFLICTING_PARENT
    if missing:
        return STATUS_MISSING_PARENT
    if changed:
        return STATUS_STALE
    if any(
        reason in {
            "MANIFEST_SCHEMA_VERSION_UNSUPPORTED",
            "MANIFEST_HASH_MISSING",
            "REBUILDABILITY_UNVERIFIED",
        }
        or reason.endswith("_MISSING")
        or reason.endswith("_UNVERIFIED")
        for reason in reasons
    ):
        return STATUS_UNVERIFIED
    if dirty:
        return STATUS_UNVERIFIED
    return STATUS_CURRENT


def _check_expected_fields(
    manifest: Mapping[str, Any],
    expected: DatasetLineageExpectation,
    reasons: list[str],
    changed: list[dict[str, Any]],
) -> None:
    mapping = {
        "dataset_id": "DATASET_ID_CHANGED",
        "dataset_type": "DATASET_TYPE_CHANGED",
        "schema_version": "SCHEMA_VERSION_CHANGED",
        "producer_command": "PRODUCER_COMMAND_CHANGED",
        "producer_module": "PRODUCER_MODULE_CHANGED",
        "canonical_price_authority_version": "CANONICAL_PRICE_AUTHORITY_CHANGED",
        "universe_authority_version": "UNIVERSE_AUTHORITY_CHANGED",
        "identity_authority_version": "IDENTITY_AUTHORITY_CHANGED",
        "corporate_action_authority_version": "CORPORATE_ACTION_AUTHORITY_CHANGED",
        "market_calendar_authority_version": "MARKET_CALENDAR_AUTHORITY_CHANGED",
        "target_contract_version": "TARGET_CONTRACT_CHANGED",
        "feature_code_version": "FEATURE_CODE_CHANGED",
        "label_code_version": "LABEL_CODE_CHANGED",
        "configuration_hash": "CONFIGURATION_CHANGED",
        "random_seed": "RANDOM_SEED_CHANGED",
    }
    for field_name, reason in mapping.items():
        expected_value = getattr(expected, field_name)
        if expected_value is None:
            continue
        actual = manifest.get(field_name)
        if str(actual) != str(expected_value):
            reasons.append(reason)
            changed.append({
                "parent": field_name,
                "reason": reason,
                "manifest_value": actual,
                "current_value": expected_value,
            })


def _check_required_contract_fields(
    manifest: Mapping[str, Any],
    reasons: list[str],
) -> None:
    required = (
        "dataset_id",
        "dataset_type",
        "schema_version",
        "producer_command",
        "producer_module",
        "universe_authority_version",
        "identity_authority_version",
        "corporate_action_authority_version",
        "feature_code_version",
        "label_code_version",
        "configuration_hash",
    )
    for field_name in required:
        value = manifest.get(field_name)
        if value in (None, ""):
            reasons.append(f"{field_name.upper()}_MISSING")
            continue
        text = str(value).strip().upper()
        if text in {"UNKNOWN", "UNVERIFIED", "NONE"} or "UNVERIFIED" in text:
            reasons.append(f"{field_name.upper()}_UNVERIFIED")


def _check_source_content_hashes(
    manifest: Mapping[str, Any],
    expected: DatasetLineageExpectation,
    reasons: list[str],
    changed: list[dict[str, Any]],
) -> None:
    manifest_hashes = {
        str(key): str(value).upper()
        for key, value in dict(manifest.get("source_content_hashes") or {}).items()
    }
    for key, expected_hash in sorted(dict(expected.source_content_hashes or {}).items()):
        actual = manifest_hashes.get(str(key))
        normalized = str(expected_hash).upper()
        if actual != normalized:
            reason = f"SOURCE_CONTENT_HASH_CHANGED:{key}"
            reasons.append(reason)
            changed.append({
                "parent": str(key),
                "reason": reason,
                "manifest_value": actual,
                "current_value": normalized,
            })


def _check_source_paths(
    manifest: Mapping[str, Any],
    reasons: list[str],
    changed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    for row in _manifest_rows(manifest.get("source_paths")):
        path = Path(str(row.get("path", "")))
        expected = _upper_or_none(row.get("sha256"))
        if not path.exists():
            reasons.append("SOURCE_PATH_MISSING")
            missing.append({"parent": str(path), "reason": "SOURCE_PATH_MISSING"})
            continue
        if expected and file_sha256(path) != expected:
            reasons.append("SOURCE_PATH_HASH_CHANGED")
            changed.append({
                "parent": str(path),
                "reason": "SOURCE_PATH_HASH_CHANGED",
                "manifest_value": expected,
                "current_value": file_sha256(path),
            })


def _check_source_manifests(
    manifest: Mapping[str, Any],
    reasons: list[str],
    changed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    conflicts: list[str],
) -> None:
    by_dataset_id: dict[str, str] = {}
    for row in _manifest_rows(manifest.get("source_manifest_hashes")):
        dataset_id = str(row.get("dataset_id") or row.get("path") or "")
        expected = _upper_or_none(row.get("sha256") or row.get("hash") or row.get("checksum"))
        if dataset_id and expected:
            previous = by_dataset_id.setdefault(dataset_id, expected)
            if previous != expected:
                conflicts.append("CONFLICTING_PARENT")
                reasons.append("CONFLICTING_PARENT")
        path_value = row.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists():
            reasons.append("SOURCE_MANIFEST_MISSING")
            missing.append({"parent": str(path), "reason": "SOURCE_MANIFEST_MISSING"})
            continue
        current = file_sha256(path)
        if expected and current != expected:
            reasons.append("SOURCE_MANIFEST_HASH_CHANGED")
            changed.append({
                "parent": str(path),
                "reason": "SOURCE_MANIFEST_HASH_CHANGED",
                "manifest_value": expected,
                "current_value": current,
            })


def _check_output_hashes(
    manifest: Mapping[str, Any],
    reasons: list[str],
    changed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    for row in _manifest_rows(manifest.get("output_hashes")):
        path = Path(str(row.get("path", "")))
        expected = _upper_or_none(row.get("sha256"))
        if not path.exists():
            reasons.append("OUTPUT_PATH_MISSING")
            missing.append({"parent": str(path), "reason": "OUTPUT_PATH_MISSING"})
            continue
        current = file_sha256(path)
        if expected and current != expected:
            reasons.append("OUTPUT_HASH_CHANGED")
            changed.append({
                "parent": str(path),
                "reason": "OUTPUT_HASH_CHANGED",
                "manifest_value": expected,
                "current_value": current,
            })


def _check_dataset_statistics(
    manifest: Mapping[str, Any],
    dataset_path: Path | None,
    reasons: list[str],
    changed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    if dataset_path is None:
        return
    if not dataset_path.exists():
        reasons.append("DATASET_PATH_MISSING")
        missing.append({"parent": str(dataset_path), "reason": "DATASET_PATH_MISSING"})
        return
    key_fields = None
    partition = manifest.get("partition_information")
    if isinstance(partition, Mapping) and isinstance(partition.get("key_fields"), list):
        key_fields = tuple(str(value) for value in partition["key_fields"])
    try:
        stats = dataset_stats(dataset_path, key_fields=key_fields)
    except Exception as exc:
        reasons.append(f"DATASET_STATISTICS_UNREADABLE:{type(exc).__name__}")
        return
    fields = {
        "row_count": "ROW_COUNT_CHANGED",
        "key_count": "KEY_COUNT_CHANGED",
        "duplicate_key_count": "DUPLICATE_KEY_COUNT_CHANGED",
        "symbol_entity_count": "SYMBOL_ENTITY_COUNT_CHANGED",
        "earliest_decision_timestamp": "DECISION_TIMESTAMP_RANGE_CHANGED",
        "latest_decision_timestamp": "DECISION_TIMESTAMP_RANGE_CHANGED",
        "earliest_knowledge_cutoff": "KNOWLEDGE_CUTOFF_RANGE_CHANGED",
        "latest_knowledge_cutoff": "KNOWLEDGE_CUTOFF_RANGE_CHANGED",
    }
    for field_name, reason in fields.items():
        if manifest.get(field_name) != stats.get(field_name):
            reasons.append(reason)
            changed.append({
                "parent": field_name,
                "reason": reason,
                "manifest_value": manifest.get(field_name),
                "current_value": stats.get(field_name),
            })


def _dataset_lineage(manifest: Mapping[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    rows = [{
        "dataset_id": manifest.get("dataset_id"),
        "dataset_type": manifest.get("dataset_type"),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest.get("manifest_hash"),
    }]
    for item in _manifest_rows(manifest.get("source_manifest_hashes")):
        rows.append({
            "dataset_id": item.get("dataset_id"),
            "dataset_type": item.get("dataset_type"),
            "manifest_path": item.get("path"),
            "manifest_hash": item.get("sha256") or item.get("hash") or item.get("checksum"),
        })
    return rows


def _market_calendar_authority_payload(
    value: Mapping[str, Any] | None,
    *,
    market_calendar_authority_version: str | None,
) -> dict[str, Any]:
    if value:
        payload = dict(value)
        payload.setdefault("market_calendar_authority_version", market_calendar_authority_version)
        return payload
    if market_calendar_authority_version in (None, ""):
        return {}
    return {"market_calendar_authority_version": market_calendar_authority_version}


def _read_dataset_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix in _PARQUET_SUFFIXES:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported dataset format for lineage stats: {path}")


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(str(key))
    return tuple(names)


def _default_key_fields(fieldnames: Sequence[str]) -> tuple[str, ...]:
    available = set(fieldnames)
    for candidate in (
        ("row_id",),
        ("feature_id",),
        ("asset_id", "decision_timestamp"),
        ("asset_id", "decision_session_date"),
        ("symbol", "decision_timestamp"),
        ("symbol", "feature_date"),
        ("variant_id", "feature_date"),
        ("variant_id", "rebalance_date"),
    ):
        if set(candidate) <= available:
            return candidate
    return (fieldnames[0],) if fieldnames else ()


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = _normalize_cell(row.get(field))
        if value:
            return value
    return None


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _empty_dataset_stats(key_fields: Sequence[str]) -> dict[str, Any]:
    return {
        "row_count": 0,
        "key_fields": list(key_fields),
        "key_count": 0,
        "duplicate_key_count": 0,
        "symbol_entity_count": 0,
        "earliest_decision_timestamp": None,
        "latest_decision_timestamp": None,
        "earliest_knowledge_cutoff": None,
        "latest_knowledge_cutoff": None,
    }


def _dataset_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    return suffix.lstrip(".") or "unknown"


def _manifest_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _resolve_manifest_path(dataset_path: Path | None, manifest_path: Path | None) -> Path | None:
    if manifest_path is not None:
        return manifest_path
    if dataset_path is not None:
        return dataset_manifest_path(dataset_path)
    return None


def _primary_output_path(manifest: Mapping[str, Any]) -> Path | None:
    rows = _manifest_rows(manifest.get("output_hashes"))
    if not rows:
        return None
    path = rows[0].get("path")
    return Path(str(path)) if path else None


def _manifest_dataset_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(payload.get("dataset_id")) if isinstance(payload, Mapping) and payload.get("dataset_id") else None


def _upper_or_none(value: Any) -> str | None:
    return str(value).upper() if value not in (None, "") else None


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: canonical_json_bytes(row),
    )


def _source_provenance(repo_root: Path = Path("."), *, changed_path_limit: int = 100) -> dict[str, Any]:
    root = repo_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    tracked = sorted(
        set(filter(None, _git(root, "diff", "--name-only").splitlines()))
        | set(filter(None, _git(root, "diff", "--cached", "--name-only").splitlines()))
    )
    untracked = sorted(filter(None, _git(root, "ls-files", "--others", "--exclude-standard").splitlines()))
    return {
        "contract_version": "source_worktree_provenance_v1",
        "git_commit": commit or None,
        "git_branch": branch or None,
        "dirty_worktree": bool(tracked or untracked),
        "tracked_changed_path_count": len(tracked),
        "tracked_changed_paths": tracked[:changed_path_limit],
        "tracked_changed_paths_truncated": len(tracked) > changed_path_limit,
        "tracked_changed_paths_hash": canonical_hash(tracked),
        "untracked_file_count": len(untracked),
        "untracked_files_present": bool(untracked),
        "untracked_paths_hash": canonical_hash(untracked),
    }


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
