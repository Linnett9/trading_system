from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


IMPLEMENTATION_STATUSES = frozenset({
    "BASELINE_COMPLETE", "IMPLEMENTED_AND_BOUNDED_RUNNABLE",
    "IMPLEMENTED_AND_AUTHORITATIVE_RUNNABLE",
    "IMPLEMENTED_BUT_UNVALIDATED", "PARTIAL_IMPLEMENTATION",
    "PLACEHOLDER_OR_NAME_ONLY", "BLOCKED_BY_DATA", "SUSPENDED",
    "REDUNDANT", "UNKNOWN", "PLANNED",
})
ARTIFACT_KINDS = frozenset({
    "MODEL_EXPERIMENT", "BASELINE_EXPERIMENT", "RESEARCH_DIAGNOSTIC",
    "DATA_READINESS_AUDIT", "PORTFOLIO_REPLAY", "EXPOSURE_EXPERIMENT", "POLICY_SWEEP",
    "ARTIFACT_LINEAGE_AUDIT",
    "SELECTOR_PREDICTION_PARTITION", "EXPOSURE_DATASET", "LEGACY_ARTIFACT_EVIDENCE_IMPORT",
    "SELECTOR_EVALUATION", "MULTI_REGIME_AGGREGATE", "ORCHESTRATION_AUDIT",
})
COMMON_REQUIRED_FIELDS = (
    "registry_contract_version", "canonical_id", "aliases", "display_name",
    "category", "implementation_status", "implementation_owner",
    "config_owner", "feature_schema", "target_contract", "model_role",
    "research_only", "point_in_time_requirements", "worker_support",
    "seed_support", "checkpoint_support", "bounded_runner_support",
    "ordinary_runner_support", "dependency_requirements", "notes",
)


class RegistryValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RegistryEntry:
    registry_kind: str
    source_path: Path
    payload: Mapping[str, Any]
    entry_hash: str

    @property
    def canonical_id(self) -> str:
        return str(self.payload["canonical_id"])

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(self.payload.get("aliases", ()))


@dataclass(frozen=True)
class RegistryDocument:
    registry_kind: str
    source_path: Path
    contract_version: str
    entries: tuple[RegistryEntry, ...]
    registry_hash: str


@dataclass(frozen=True)
class RegistryResolution:
    requested_id: str
    canonical_id: str
    entry: RegistryEntry
    registry_hash: str
    registry_set_hash: str
