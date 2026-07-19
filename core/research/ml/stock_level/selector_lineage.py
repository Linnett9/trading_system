from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence

LINEAGE_CONTRACT_VERSION = "selector_parent_child_lineage.v1"
SPINE_CONTRACT_VERSION = "canonical_selector_daily_spine.v1"
FROZEN_PREFLIGHT_VERSION = "frozen_selector_preflight.v1"
CURRENT_ECONOMIC_TARGET_ID = "forward_return_10d"
CURRENT_TARGET_PROVENANCE_VERSION = "stock_level_target_provenance_v2"
SELECTOR_ROW_ID_CONTRACT_VERSION = "stock_selector_row_id_v1"

TARGET_COLUMNS = (
    "actual_forward_return_10d",
    "actual_benchmark_return_10d",
    "actual_market_residual_return_10d",
    "target_status",
)
TARGET_TIMESTAMP_COLUMNS = (
    "target_start_timestamp",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_timestamp",
    "benchmark_target_start_timestamp",
    "benchmark_label_start_timestamp",
    "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp",
)
DECISION_COLUMNS = (
    "decision_timestamp",
    "feature_data_cutoff_timestamp",
)
PROTECTED_ENRICHMENT_COLUMNS = frozenset(
    {
        "row_id",
        "asset_id",
        "canonical_symbol",
        "symbol",
        "rebalance_date",
        "decision_session_date",
        "economic_target_id",
        "target_provenance_contract_version",
        "target_horizon_trading_days",
        "overlapping_targets",
        "required_purge_horizon_trading_days",
        *TARGET_COLUMNS,
        *TARGET_TIMESTAMP_COLUMNS,
        *DECISION_COLUMNS,
    }
)
REQUIRED_PARENT_IDENTITIES = (
    "canonical_daily_dataset_version",
    "canonical_daily_logical_checksum",
    "asset_registry_version",
    "asset_registry_checksum",
    "daily_spine_identity",
    "daily_spine_logical_checksum",
    "calendar_version",
    "decision_timing_contract",
    "configuration_hash",
    "git_commit",
    "builder_contract_version",
    "feature_schema_version",
)


class LineageStatus(StrEnum):
    READY = "READY"
    ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
    ROW_POPULATION_MISMATCH = "ROW_POPULATION_MISMATCH"
    ROW_ID_DUPLICATE = "ROW_ID_DUPLICATE"
    ECONOMIC_TARGET_MISMATCH = "ECONOMIC_TARGET_MISMATCH"
    PROVENANCE_CONTRACT_MISMATCH = "PROVENANCE_CONTRACT_MISMATCH"
    TARGET_VALUE_MISMATCH = "TARGET_VALUE_MISMATCH"
    BENCHMARK_VALUE_MISMATCH = "BENCHMARK_VALUE_MISMATCH"
    TARGET_TIMESTAMP_MISMATCH = "TARGET_TIMESTAMP_MISMATCH"
    DECISION_TIMESTAMP_MISMATCH = "DECISION_TIMESTAMP_MISMATCH"
    PARENT_IDENTITY_MISSING = "PARENT_IDENTITY_MISSING"
    SOURCE_DATASET_MISMATCH = "SOURCE_DATASET_MISMATCH"
    MANIFEST_ROW_CONTRADICTION = "MANIFEST_ROW_CONTRADICTION"


@dataclass(frozen=True)
class LineageValidationResult:
    status: LineageStatus
    blockers: tuple[str, ...]
    row_count: int
    row_id_checksum: str
    target_checksum: str

    @property
    def ready(self) -> bool:
        return self.status is LineageStatus.READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": LINEAGE_CONTRACT_VERSION,
            "status": self.status.value,
            "blockers": list(self.blockers),
            "row_count": self.row_count,
            "row_id_checksum": self.row_id_checksum,
            "target_checksum": self.target_checksum,
        }


def selector_row_id(row: Mapping[str, Any]) -> str:
    existing = str(row.get("row_id", "")).strip()
    if existing:
        return existing
    asset = str(
        row.get("asset_id")
        or row.get("canonical_symbol")
        or row.get("symbol")
        or ""
    ).strip()
    decision = str(
        row.get("decision_timestamp")
        or row.get("decision_session_date")
        or row.get("rebalance_date")
        or ""
    ).strip()
    if not asset or not decision:
        raise ValueError("selector row identity requires asset and decision timestamp")
    identity = "|".join(
        (
            SELECTOR_ROW_ID_CONTRACT_VERSION,
            "canonical_daily_v2",
            asset,
            decision,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def canonical_timestamp(value: Any, *, allow_date: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) == 10:
        if not allow_date:
            raise ValueError(f"timezone-aware timestamp required, found date {text}")
        parsed = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"timezone-aware timestamp required: {text}")
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def timestamp_at_or_before(value: Any, cutoff: Any) -> bool:
    return canonical_timestamp(value) <= canonical_timestamp(cutoff)


def validate_selector_parent_child_lineage(
    *,
    parent_manifest: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    parent_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
) -> LineageValidationResult:
    blockers: list[LineageStatus] = []
    parent_ids, parent_duplicate = _indexed(parent_rows)
    child_ids, child_duplicate = _indexed(child_rows)
    if parent_duplicate or child_duplicate:
        blockers.append(LineageStatus.ROW_ID_DUPLICATE)
    if len(parent_rows) != len(child_rows):
        blockers.append(LineageStatus.ROW_COUNT_MISMATCH)
    if set(parent_ids) != set(child_ids):
        blockers.append(LineageStatus.ROW_POPULATION_MISMATCH)
    missing = [
        field
        for field in REQUIRED_PARENT_IDENTITIES
        if not str(parent_manifest.get(field, "")).strip()
        or not str(child_manifest.get(field, "")).strip()
    ]
    if missing:
        blockers.append(LineageStatus.PARENT_IDENTITY_MISSING)
    source_fields = (
        "canonical_daily_dataset_version",
        "canonical_daily_logical_checksum",
        "asset_registry_version",
        "asset_registry_checksum",
        "daily_spine_identity",
        "daily_spine_logical_checksum",
        "calendar_version",
        "decision_timing_contract",
        "configuration_hash",
    )
    if any(parent_manifest.get(field) != child_manifest.get(field) for field in source_fields):
        blockers.append(LineageStatus.SOURCE_DATASET_MISMATCH)
    parent_economic = str(parent_manifest.get("economic_target_id", ""))
    child_economic = str(child_manifest.get("economic_target_id", ""))
    parent_provenance = str(
        parent_manifest.get("target_provenance_contract_version", "")
    )
    child_provenance = str(
        child_manifest.get("target_provenance_contract_version", "")
    )
    for manifest in (parent_manifest, child_manifest):
        deprecated_identity = str(manifest.get("target_contract", "")).strip()
        if deprecated_identity and deprecated_identity not in {
            str(manifest.get("economic_target_id", "")),
            str(manifest.get("target_provenance_contract_version", "")),
        }:
            blockers.append(LineageStatus.MANIFEST_ROW_CONTRADICTION)
    try:
        from core.research.ml.registries.target_identity import (
            resolve_target_identity,
        )

        resolution = resolve_target_identity(
            economic_target_id=parent_economic,
            target_provenance_contract_version=parent_provenance,
        )
    except ValueError:
        resolution = None
    if (
        parent_economic != child_economic
        or parent_economic != CURRENT_ECONOMIC_TARGET_ID
    ):
        blockers.append(LineageStatus.ECONOMIC_TARGET_MISMATCH)
    if (
        parent_provenance != child_provenance
        or parent_provenance != CURRENT_TARGET_PROVENANCE_VERSION
        or resolution is None
        or not resolution.supported
    ):
        blockers.append(LineageStatus.PROVENANCE_CONTRACT_MISMATCH)
    common = sorted(set(parent_ids) & set(child_ids))
    for row_id in common:
        parent = parent_ids[row_id]
        child = child_ids[row_id]
        if _population_key(parent) != _population_key(child):
            blockers.append(LineageStatus.ROW_POPULATION_MISMATCH)
        if not _same_values(parent, child, ("actual_forward_return_10d",)):
            blockers.append(LineageStatus.TARGET_VALUE_MISMATCH)
        if not _same_values(
            parent,
            child,
            ("actual_benchmark_return_10d", "actual_market_residual_return_10d"),
        ):
            blockers.append(LineageStatus.BENCHMARK_VALUE_MISMATCH)
        if not _same_values(parent, child, ("target_status",)):
            blockers.append(LineageStatus.TARGET_VALUE_MISMATCH)
        if not _same_timestamps(parent, child, TARGET_TIMESTAMP_COLUMNS):
            blockers.append(LineageStatus.TARGET_TIMESTAMP_MISMATCH)
        if not _same_timestamps(parent, child, DECISION_COLUMNS):
            blockers.append(LineageStatus.DECISION_TIMESTAMP_MISMATCH)
        if (
            str(child.get("target_provenance_contract_version", ""))
            != child_provenance
            or str(parent.get("target_provenance_contract_version", ""))
            != parent_provenance
        ):
            blockers.append(LineageStatus.MANIFEST_ROW_CONTRADICTION)
        row_economic = str(
            child.get("economic_target_id", CURRENT_ECONOMIC_TARGET_ID)
        )
        if row_economic != child_economic:
            blockers.append(LineageStatus.MANIFEST_ROW_CONTRADICTION)
    unique = tuple(dict.fromkeys(blockers))
    status = unique[0] if unique else LineageStatus.READY
    return LineageValidationResult(
        status,
        tuple(value.value for value in unique),
        len(parent_rows),
        row_id_checksum(parent_rows),
        target_checksum(parent_rows),
    )


def merge_enrichment_preserving_base(
    base_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base_index, base_duplicate = _indexed(base_rows)
    feature_index, feature_duplicate = _indexed(feature_rows)
    if base_duplicate or feature_duplicate:
        raise ValueError("ROW_ID_DUPLICATE")
    unknown = sorted(set(feature_index) - set(base_index))
    if unknown:
        raise ValueError(f"unknown enrichment row IDs: {unknown[:5]}")
    output: list[dict[str, Any]] = []
    for base in base_rows:
        row_id = selector_row_id(base)
        merged = dict(base)
        merged["row_id"] = row_id
        feature = feature_index.get(row_id)
        if feature is not None:
            for key, value in feature.items():
                if key not in PROTECTED_ENRICHMENT_COLUMNS:
                    merged[key] = value
        merged["feature_coverage_status"] = (
            "AVAILABLE" if feature is not None else "MISSING"
        )
        output.append(merged)
    return output


def build_ready_daily_spine_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_identity: Mapping[str, Any],
    physical_sha256: str,
    configuration_hash: str,
    git_commit: str,
) -> dict[str, Any]:
    required = (
        "canonical_daily_dataset_version",
        "canonical_daily_logical_checksum",
        "asset_registry_version",
        "asset_registry_checksum",
        "calendar_version",
        "decision_timing_contract",
    )
    missing = [field for field in required if not parent_identity.get(field)]
    if missing:
        raise ValueError(f"daily spine parent identity missing: {missing}")
    timestamps = [
        canonical_timestamp(
            row.get("decision_timestamp")
            or row.get("decision_session_date"),
            allow_date=True,
        )
        for row in rows
    ]
    payload = {
        "status": "READY",
        "dataset_id": (
            "canonical_selector_daily_spine-"
            + row_id_checksum(rows)[:16].lower()
        ),
        "row_count": len(rows),
        "row_id_contract": "stock_selector_row_id_v1",
        "row_id_checksum": row_id_checksum(rows),
        "symbol_count": len({str(row.get("symbol", "")) for row in rows}),
        "decision_date_count": len(
            {str(row.get("decision_session_date") or row.get("rebalance_date")) for row in rows}
        ),
        "minimum_decision_timestamp": min(timestamps),
        "maximum_decision_timestamp": max(timestamps),
        **dict(parent_identity),
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
        "target_horizon_sessions": 10,
        "physical_sha256": physical_sha256,
        "configuration_hash": configuration_hash,
        "git_commit": git_commit,
        "builder_contract_version": SPINE_CONTRACT_VERSION,
    }
    payload["logical_checksum"] = _hash(payload)
    return payload


def preflight_frozen_selector_dataset(
    *,
    daily_spine_manifest: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    enriched_manifest: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    enriched_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    if daily_spine_manifest.get("status") != "READY":
        blockers.append("DAILY_SPINE_NOT_READY")
    for name, manifest in (
        ("BASE", base_manifest),
        ("ENRICHED", enriched_manifest),
    ):
        if manifest.get("status") != "READY":
            blockers.append(f"{name}_ARTIFACT_NOT_READY")
    lineage = validate_selector_parent_child_lineage(
        parent_manifest=base_manifest,
        child_manifest=enriched_manifest,
        parent_rows=base_rows,
        child_rows=enriched_rows,
    )
    blockers.extend(lineage.blockers)
    forbidden_features = set(TARGET_COLUMNS) | set(TARGET_TIMESTAMP_COLUMNS)
    if any(
        str(column) in forbidden_features
        or str(column).startswith(("actual_", "target_", "label_"))
        for column in feature_columns
    ):
        blockers.append("TARGET_COLUMN_IN_FEATURE_SCHEMA")
    if (
        daily_spine_manifest.get("logical_checksum")
        != base_manifest.get("daily_spine_logical_checksum")
    ):
        blockers.append("DAILY_SPINE_PARENT_MISMATCH")
    blockers = list(dict.fromkeys(blockers))
    return {
        "contract_version": FROZEN_PREFLIGHT_VERSION,
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "lineage": lineage.as_dict(),
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
    }


def row_id_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    return _hash(sorted(selector_row_id(row) for row in rows))


def target_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
                "row_id": selector_row_id(row),
                **{field: row.get(field) for field in (*TARGET_COLUMNS, *TARGET_TIMESTAMP_COLUMNS)},
        }
        for row in rows
    ]
    return _hash(sorted(payload, key=lambda row: str(row["row_id"])))


def _indexed(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], bool]:
    output: dict[str, Mapping[str, Any]] = {}
    duplicate = False
    for row in rows:
        row_id = selector_row_id(row)
        duplicate = duplicate or row_id in output
        output[row_id] = row
    return output, duplicate


def _population_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("asset_id") or row.get("canonical_symbol") or row.get("symbol")),
        str(row.get("decision_session_date") or row.get("rebalance_date")),
    )


def _same_values(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]
) -> bool:
    return all(left.get(field) == right.get(field) for field in fields)


def _same_timestamps(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]
) -> bool:
    try:
        return all(
            canonical_timestamp(left.get(field), allow_date=True)
            == canonical_timestamp(right.get(field), allow_date=True)
            for field in fields
        )
    except ValueError:
        return False


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
