from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


MIB = 1024**2
STAGE_CAPACITY_PROFILE_SCHEMA = "DS24_CLEAN_V2_STAGE_CAPACITY_PROFILE_V1"


@dataclass(frozen=True)
class StageCalibrationPolicy:
    """Versioned evidence rule for conservative stage-specific calibration."""

    policy_id: str = "DS24_CLEAN_V2_STAGE_CALIBRATION_POLICY_V1"
    minimum_comparable_successful_samples: int = 3
    guarded_high_water_cushion_bytes: int = 512 * MIB
    minimum_requested_floor_bytes: int = 512 * MIB
    workload_shape_bucket_bytes: int = 512 * MIB
    upward_calibration_immediate: bool = True

    def payload(self) -> dict[str, Any]:
        return asdict(self)


STAGE_CALIBRATION_POLICY = StageCalibrationPolicy()


@dataclass(frozen=True)
class StageCapacitySample:
    family: str
    stage: str
    workload_shape_id: str
    source_hash: str
    operational_policy_hash: str
    resource_policy_hash: str
    started_at_utc: str
    completed_at_utc: str
    start_resident_bytes: int
    start_private_commit_bytes: int
    maximum_resident_bytes: int
    maximum_private_commit_bytes: int
    incremental_resident_growth_bytes: int
    incremental_commit_growth_bytes: int
    wall_seconds: float
    pressure_occurred: bool
    paging_or_resource_exhaustion_occurred: bool
    publication_completed: bool | None
    successful: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageCalibrationDecision:
    requested_physical_bytes: int
    requested_commit_bytes: int
    fallback_physical_bytes: int
    fallback_commit_bytes: int
    guarded_physical_bytes: int
    guarded_commit_bytes: int
    sample_count: int
    calibrated: bool
    classification: str
    workload_shape_id: str
    observed_upper_resident_growth_bytes: int
    observed_upper_commit_growth_bytes: int

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def stage_workload_shape_id(
    *,
    family: str,
    stage: str,
    requested_physical_bytes: int,
    requested_commit_bytes: int,
    policy: StageCalibrationPolicy = STAGE_CALIBRATION_POLICY,
) -> str:
    """Return a stable coarse shape identity without merging larger workloads."""

    def upper_bucket(value: int) -> int:
        size = policy.workload_shape_bucket_bytes
        return max(size, int(math.ceil(max(0, int(value)) / size) * size))

    payload = {
        "family": str(family),
        "stage": canonical_stage_name(family=family, stage=stage),
        "physical_upper_bucket_bytes": upper_bucket(requested_physical_bytes),
        "commit_upper_bucket_bytes": upper_bucket(requested_commit_bytes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_stage_name(*, family: str, stage: str) -> str:
    prefix = f"{family}_"
    return stage[len(prefix) :] if stage.startswith(prefix) else stage


def profile_key(
    *,
    family: str,
    stage: str,
    workload_shape_id: str,
    source_hash: str,
    operational_policy_hash: str,
    resource_policy_hash: str,
) -> str:
    payload = {
        "family": family,
        "stage": canonical_stage_name(family=family, stage=stage),
        "workload_shape_id": workload_shape_id,
        "source_hash": source_hash,
        "operational_policy_hash": operational_policy_hash,
        "resource_policy_hash": resource_policy_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def empty_profile_document(
    policy: StageCalibrationPolicy = STAGE_CALIBRATION_POLICY,
) -> dict[str, Any]:
    return {
        "schema": STAGE_CAPACITY_PROFILE_SCHEMA,
        "calibration_policy": policy.payload(),
        "profiles": {},
        "updated_at_utc": None,
    }


def validate_sample(sample: StageCapacitySample) -> None:
    required_text = {
        "family": sample.family,
        "stage": sample.stage,
        "workload_shape_id": sample.workload_shape_id,
        "source_hash": sample.source_hash,
        "operational_policy_hash": sample.operational_policy_hash,
        "resource_policy_hash": sample.resource_policy_hash,
        "started_at_utc": sample.started_at_utc,
        "completed_at_utc": sample.completed_at_utc,
    }
    missing = sorted(name for name, value in required_text.items() if not value)
    if missing:
        raise ValueError(f"Stage-capacity sample fields are empty: {missing}")
    byte_values = (
        sample.start_resident_bytes,
        sample.start_private_commit_bytes,
        sample.maximum_resident_bytes,
        sample.maximum_private_commit_bytes,
        sample.incremental_resident_growth_bytes,
        sample.incremental_commit_growth_bytes,
    )
    if any(value < 0 for value in byte_values):
        raise ValueError("Stage-capacity sample byte values cannot be negative")
    if sample.maximum_resident_bytes < sample.start_resident_bytes:
        raise ValueError("Stage resident maximum precedes its start value")
    if sample.maximum_private_commit_bytes < sample.start_private_commit_bytes:
        raise ValueError("Stage commit maximum precedes its start value")
    if sample.wall_seconds < 0:
        raise ValueError("Stage wall time cannot be negative")


def merge_stage_sample(
    document: Mapping[str, Any],
    sample: StageCapacitySample,
    *,
    policy: StageCalibrationPolicy = STAGE_CALIBRATION_POLICY,
) -> dict[str, Any]:
    """Return a compact profile document with one comparable sample merged."""

    validate_sample(sample)
    result = empty_profile_document(policy)
    existing_profiles = document.get("profiles")
    if isinstance(existing_profiles, Mapping):
        result["profiles"] = {
            str(key): dict(value)
            for key, value in existing_profiles.items()
            if isinstance(value, Mapping)
        }
    key = profile_key(
        family=sample.family,
        stage=sample.stage,
        workload_shape_id=sample.workload_shape_id,
        source_hash=sample.source_hash,
        operational_policy_hash=sample.operational_policy_hash,
        resource_policy_hash=sample.resource_policy_hash,
    )
    current = dict(result["profiles"].get(key) or {})
    successful_comparable = bool(
        sample.successful
        and not sample.pressure_occurred
        and not sample.paging_or_resource_exhaustion_occurred
    )
    successful_count = int(current.get("successful_sample_count", 0))
    total_count = int(current.get("total_sample_count", 0)) + 1
    if successful_comparable:
        successful_count += 1
    upper_resident = int(current.get("maximum_incremental_resident_growth_bytes", 0))
    upper_commit = int(current.get("maximum_incremental_commit_growth_bytes", 0))
    if successful_comparable:
        upper_resident = max(upper_resident, sample.incremental_resident_growth_bytes)
        upper_commit = max(upper_commit, sample.incremental_commit_growth_bytes)
    result["profiles"][key] = {
        "profile_key": key,
        "family": sample.family,
        "stage": canonical_stage_name(family=sample.family, stage=sample.stage),
        "workload_shape_id": sample.workload_shape_id,
        "source_hash": sample.source_hash,
        "operational_policy_hash": sample.operational_policy_hash,
        "resource_policy_hash": sample.resource_policy_hash,
        "successful_sample_count": successful_count,
        "total_sample_count": total_count,
        "maximum_incremental_resident_growth_bytes": upper_resident,
        "maximum_incremental_commit_growth_bytes": upper_commit,
        "maximum_stage_resident_bytes": max(
            int(current.get("maximum_stage_resident_bytes", 0)),
            sample.maximum_resident_bytes,
        ),
        "maximum_stage_private_commit_bytes": max(
            int(current.get("maximum_stage_private_commit_bytes", 0)),
            sample.maximum_private_commit_bytes,
        ),
        "maximum_wall_seconds": max(
            float(current.get("maximum_wall_seconds", 0.0)), sample.wall_seconds
        ),
        "pressure_sample_count": int(current.get("pressure_sample_count", 0))
        + int(sample.pressure_occurred),
        "paging_or_resource_exhaustion_sample_count": int(
            current.get("paging_or_resource_exhaustion_sample_count", 0)
        )
        + int(sample.paging_or_resource_exhaustion_occurred),
        "publication_success_count": int(
            current.get("publication_success_count", 0)
        )
        + int(sample.publication_completed is True),
        "first_sample_at_utc": current.get("first_sample_at_utc")
        or sample.completed_at_utc,
        "last_sample_at_utc": sample.completed_at_utc,
        "latest_sample": sample.payload(),
    }
    result["updated_at_utc"] = sample.completed_at_utc
    return result


def calibration_decision(
    profile: Mapping[str, Any] | None,
    *,
    workload_shape_id: str,
    fallback_physical_bytes: int,
    fallback_commit_bytes: int,
    allocation_safety_factor: float,
    policy: StageCalibrationPolicy = STAGE_CALIBRATION_POLICY,
) -> StageCalibrationDecision:
    """Resolve a safe requested estimate from comparable successful evidence."""

    if allocation_safety_factor <= 0:
        raise ValueError("Allocation safety factor must be positive")
    fallback_physical = max(0, int(fallback_physical_bytes))
    fallback_commit = max(0, int(fallback_commit_bytes))
    if not profile or profile.get("workload_shape_id") != workload_shape_id:
        return StageCalibrationDecision(
            fallback_physical,
            fallback_commit,
            fallback_physical,
            fallback_commit,
            int(math.ceil(fallback_physical * allocation_safety_factor)),
            int(math.ceil(fallback_commit * allocation_safety_factor)),
            0,
            False,
            "CONSERVATIVE_FALLBACK_UNKNOWN_OR_INCOMPATIBLE_SHAPE",
            workload_shape_id,
            0,
            0,
        )
    sample_count = int(profile.get("successful_sample_count", 0))
    observed_physical = int(
        profile.get("maximum_incremental_resident_growth_bytes", 0)
    )
    observed_commit = int(
        profile.get("maximum_incremental_commit_growth_bytes", 0)
    )
    guarded_physical = observed_physical + policy.guarded_high_water_cushion_bytes
    guarded_commit = observed_commit + policy.guarded_high_water_cushion_bytes
    fallback_guarded_physical = int(
        math.ceil(fallback_physical * allocation_safety_factor)
    )
    fallback_guarded_commit = int(
        math.ceil(fallback_commit * allocation_safety_factor)
    )
    upward = bool(
        guarded_physical > fallback_guarded_physical
        or guarded_commit > fallback_guarded_commit
    )
    enough_samples = sample_count >= policy.minimum_comparable_successful_samples
    if not upward and not enough_samples:
        return StageCalibrationDecision(
            fallback_physical,
            fallback_commit,
            fallback_physical,
            fallback_commit,
            fallback_guarded_physical,
            fallback_guarded_commit,
            sample_count,
            False,
            "CALIBRATION_PENDING_MINIMUM_COMPARABLE_SAMPLES",
            workload_shape_id,
            observed_physical,
            observed_commit,
        )
    requested_physical = max(
        policy.minimum_requested_floor_bytes,
        int(math.ceil(guarded_physical / allocation_safety_factor)),
    )
    requested_commit = max(
        policy.minimum_requested_floor_bytes,
        int(math.ceil(guarded_commit / allocation_safety_factor)),
    )
    return StageCalibrationDecision(
        requested_physical,
        requested_commit,
        fallback_physical,
        fallback_commit,
        int(math.ceil(requested_physical * allocation_safety_factor)),
        int(math.ceil(requested_commit * allocation_safety_factor)),
        sample_count,
        True,
        (
            "IMMEDIATE_UPWARD_CALIBRATION"
            if upward
            else "CALIBRATED_AFTER_MINIMUM_COMPARABLE_SAMPLES"
        ),
        workload_shape_id,
        observed_physical,
        observed_commit,
    )
