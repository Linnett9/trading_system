from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.research.ml.ds24.clean_v2_resources import (
    AllocationEstimate,
    read_json_object_with_retry,
    write_json_object_atomic,
)
from core.research.ml.ds24.stage_capacity_calibration import (
    STAGE_CALIBRATION_POLICY,
    StageCalibrationDecision,
    StageCapacitySample,
    calibration_decision,
    empty_profile_document,
    merge_stage_sample,
    profile_key,
    stage_workload_shape_id,
)


class StageCapacityProfileStore:
    """Atomic persistence adapter for compact operational stage profiles."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return empty_profile_document()
        payload = read_json_object_with_retry(self.path)
        if not isinstance(payload, dict):
            raise ValueError("Stage-capacity profile document must be an object")
        return payload

    def record(self, sample: StageCapacitySample) -> dict[str, Any]:
        updated = merge_stage_sample(self.load(), sample)
        write_json_object_atomic(self.path, updated)
        return updated

    def resolve(
        self,
        *,
        family: str,
        stage: str,
        fallback: AllocationEstimate,
        source_hash: str,
        operational_policy_hash: str,
        resource_policy_hash: str,
        allocation_safety_factor: float,
    ) -> StageCalibrationDecision:
        shape_id = stage_workload_shape_id(
            family=family,
            stage=stage,
            requested_physical_bytes=fallback.physical_bytes,
            requested_commit_bytes=fallback.commit_bytes,
        )
        key = profile_key(
            family=family,
            stage=stage,
            workload_shape_id=shape_id,
            source_hash=source_hash,
            operational_policy_hash=operational_policy_hash,
            resource_policy_hash=resource_policy_hash,
        )
        profiles = self.load().get("profiles") or {}
        profile: Mapping[str, Any] | None = (
            profiles.get(key) if isinstance(profiles, Mapping) else None
        )
        return calibration_decision(
            profile,
            workload_shape_id=shape_id,
            fallback_physical_bytes=fallback.physical_bytes,
            fallback_commit_bytes=fallback.commit_bytes,
            allocation_safety_factor=allocation_safety_factor,
            policy=STAGE_CALIBRATION_POLICY,
        )
