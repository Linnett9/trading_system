from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


CURRENT_ECONOMIC_TARGET_ID = "forward_return_10d"
CURRENT_TARGET_PROVENANCE_VERSION = "stock_level_target_provenance_v2"
LEGACY_TARGET_PROVENANCE_VERSION = "stock_level_target_provenance_v1"
DEPRECATED_TARGET_PROVENANCE_VERSION = "stock_level_target_provenance_v4"
TARGET_REGISTRY_SCHEMA_VERSION = "selector_target_identity.v1"


class TargetIdentityStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNKNOWN_ECONOMIC_TARGET = "UNKNOWN_ECONOMIC_TARGET"
    UNKNOWN_PROVENANCE_VERSION = "UNKNOWN_PROVENANCE_VERSION"
    UNSUPPORTED_TARGET_PROVENANCE_PAIR = "UNSUPPORTED_TARGET_PROVENANCE_PAIR"
    LEGACY_INCOMPATIBLE_PROVENANCE = "LEGACY_INCOMPATIBLE_PROVENANCE"
    DEPRECATED_ERRONEOUS_IDENTIFIER = "DEPRECATED_ERRONEOUS_IDENTIFIER"


@dataclass(frozen=True)
class TargetIdentityResolution:
    economic_target_id: str
    target_provenance_contract_version: str
    status: TargetIdentityStatus

    @property
    def supported(self) -> bool:
        return self.status is TargetIdentityStatus.SUPPORTED


def resolve_target_identity(
    *,
    economic_target_id: str,
    target_provenance_contract_version: str,
) -> TargetIdentityResolution:
    economic = str(economic_target_id).strip()
    provenance = str(target_provenance_contract_version).strip()
    provenance_namespace = {
        LEGACY_TARGET_PROVENANCE_VERSION,
        CURRENT_TARGET_PROVENANCE_VERSION,
        DEPRECATED_TARGET_PROVENANCE_VERSION,
    }
    if economic in provenance_namespace:
        raise ValueError(
            "economic_target_id requires an economic-target identifier, "
            f"not provenance identifier {economic}"
        )
    if provenance == CURRENT_ECONOMIC_TARGET_ID:
        raise ValueError(
            "target_provenance_contract_version requires a provenance identifier, "
            f"not economic target {provenance}"
        )
    if economic != CURRENT_ECONOMIC_TARGET_ID:
        status = TargetIdentityStatus.UNKNOWN_ECONOMIC_TARGET
    elif provenance == LEGACY_TARGET_PROVENANCE_VERSION:
        status = TargetIdentityStatus.LEGACY_INCOMPATIBLE_PROVENANCE
    elif provenance == DEPRECATED_TARGET_PROVENANCE_VERSION:
        status = TargetIdentityStatus.DEPRECATED_ERRONEOUS_IDENTIFIER
    elif provenance != CURRENT_TARGET_PROVENANCE_VERSION:
        status = TargetIdentityStatus.UNKNOWN_PROVENANCE_VERSION
    elif (economic, provenance) != (
        CURRENT_ECONOMIC_TARGET_ID,
        CURRENT_TARGET_PROVENANCE_VERSION,
    ):
        status = TargetIdentityStatus.UNSUPPORTED_TARGET_PROVENANCE_PAIR
    else:
        status = TargetIdentityStatus.SUPPORTED
    return TargetIdentityResolution(economic, provenance, status)


def validate_target_identity_manifest(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> TargetIdentityResolution:
    if "target_contract" in manifest:
        raise ValueError(
            "ambiguous target_contract is not accepted; use economic_target_id "
            "and target_provenance_contract_version"
        )
    required = (
        "economic_target_id",
        "target_provenance_contract_version",
    )
    missing = [field for field in required if not str(manifest.get(field, "")).strip()]
    if missing:
        raise ValueError(f"target identity manifest missing required fields: {missing}")
    economic_values = {
        str(row.get("economic_target_id", "")).strip()
        for row in rows
        if str(row.get("economic_target_id", "")).strip()
    }
    provenance_values = {
        str(row.get("target_provenance_contract_version", "")).strip()
        for row in rows
        if str(row.get("target_provenance_contract_version", "")).strip()
    }
    if len(economic_values) != 1:
        raise ValueError(
            "rows require exactly one non-null economic_target_id; "
            f"found={sorted(economic_values)}"
        )
    if len(provenance_values) != 1:
        raise ValueError(
            "rows require exactly one non-null target_provenance_contract_version; "
            f"found={sorted(provenance_values)}"
        )
    row_economic = next(iter(economic_values))
    row_provenance = next(iter(provenance_values))
    if str(manifest["economic_target_id"]) != row_economic:
        raise ValueError("manifest and rows have different economic_target_id")
    if str(manifest["target_provenance_contract_version"]) != row_provenance:
        raise ValueError(
            "manifest and rows have different target_provenance_contract_version"
        )
    resolution = resolve_target_identity(
        economic_target_id=row_economic,
        target_provenance_contract_version=row_provenance,
    )
    if not resolution.supported:
        raise ValueError(f"unsupported target identity: {resolution.status.value}")
    return resolution


AUTHORITATIVE_REBUILD_PARENT = {
    "classification": "AUTHORITATIVE_REBUILD_PARENT",
    "path": (
        "reports/ml/development/ticket_7b3_daily_large_history/"
        "regeneration_canonical_v2/benchmark/"
        "stock_level_prediction_artifacts.parquet"
    ),
    "row_count": 836_074,
    "symbol_count": 379,
    "decision_date_count": 2_206,
    "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
    "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
    "sha256": "c2487d7f378121069ea5e92a1d0cf0444f42dfc1da237566d24c650ae8558d38",
    "logical_content_sha256": (
        "c564a0187ef1a32ae7f979c37ab2cc553959871c747922553ef5e7486b42b446"
    ),
}
