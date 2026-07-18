from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

PROFILE_CONTRACT = "compute_machine_profile.v1"
GIB = 1024**3


def _checksum(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MachineProfile:
    profile_id: str
    os_family: str
    cpu_model: str
    physical_cores: int
    logical_processors: int
    total_ram_bytes: int
    schedulable_ram_bytes: int
    normal_reserve_bytes: int
    critical_reserve_bytes: int
    cpu_capacity: int
    max_lightweight_jobs: int
    default_inner_threads: int
    deep_model_limit: int
    gpu_inventory: tuple[str, ...]
    source: str
    source_git_commit: str
    generated_at: str
    contract_version: str = PROFILE_CONTRACT
    logical_checksum: str = ""

    def __post_init__(self) -> None:
        required = (
            self.profile_id, self.os_family, self.cpu_model, self.source,
            self.source_git_commit,
        )
        if not all(value.strip() for value in required):
            raise ValueError("Machine profile identity fields are required")
        if self.contract_version != PROFILE_CONTRACT:
            raise ValueError("Unsupported machine profile contract")
        if self.source not in {
            "explicit_configuration", "detected_runtime", "test_fixture"
        }:
            raise ValueError("Unsupported machine profile source")
        if self.total_ram_bytes <= 0 or self.schedulable_ram_bytes <= 0:
            raise ValueError("RAM capacities must be positive")
        if self.schedulable_ram_bytes > self.total_ram_bytes:
            raise ValueError("Schedulable RAM exceeds total RAM")
        if min(self.normal_reserve_bytes, self.critical_reserve_bytes) < 0:
            raise ValueError("Reserve thresholds cannot be negative")
        if self.critical_reserve_bytes > self.normal_reserve_bytes:
            raise ValueError("Critical reserve exceeds normal reserve")
        capacities = (
            self.physical_cores, self.logical_processors, self.cpu_capacity,
            self.max_lightweight_jobs, self.default_inner_threads,
            self.deep_model_limit,
        )
        if min(capacities) <= 0:
            raise ValueError("CPU and concurrency policies must be positive")
        expected = _checksum(self.logical_payload())
        if self.logical_checksum and self.logical_checksum != expected:
            raise ValueError("Machine profile logical checksum mismatch")
        if not self.logical_checksum:
            object.__setattr__(self, "logical_checksum", expected)

    def logical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        for transient in ("generated_at", "logical_checksum"):
            payload.pop(transient, None)
        payload["gpu_inventory"] = list(self.gpu_inventory)
        return payload


@dataclass(frozen=True)
class RuntimeResources:
    os_family: str
    cpu_model: str
    physical_cores: int | None
    logical_processors: int
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    gpu_inventory: tuple[str, ...] = ()


class DetectionComparison(str, Enum):
    MATCH = "MATCH"
    WARNING_DIFFERENCE = "WARNING_DIFFERENCE"
    INCOMPATIBLE = "INCOMPATIBLE"


def dell_i5_10500_profile(
    *, source_git_commit: str, generated_at: str | None = None
) -> MachineProfile:
    return MachineProfile(
        profile_id="dell-i5-10500-32gb-v1",
        os_family="Windows",
        cpu_model="Intel Core i5-10500",
        physical_cores=6,
        logical_processors=12,
        total_ram_bytes=32 * GIB,
        schedulable_ram_bytes=24 * GIB,
        normal_reserve_bytes=8 * GIB,
        critical_reserve_bytes=6 * GIB,
        cpu_capacity=3,
        max_lightweight_jobs=4,
        default_inner_threads=1,
        deep_model_limit=1,
        gpu_inventory=("NVIDIA Quadro P620",),
        source="explicit_configuration",
        source_git_commit=source_git_commit,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
    )


def detect_runtime_resources() -> RuntimeResources:
    total = available = physical = None
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        total, available = int(memory.total), int(memory.available)
        physical = psutil.cpu_count(logical=False)
    except (ImportError, OSError, RuntimeError):
        pass
    return RuntimeResources(
        os_family=platform.system() or "UNKNOWN",
        cpu_model=platform.processor() or "UNKNOWN",
        physical_cores=physical,
        logical_processors=os.cpu_count() or 1,
        total_ram_bytes=total,
        available_ram_bytes=available,
        gpu_inventory=_detect_nvidia_gpus(),
    )


def _detect_nvidia_gpus() -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def compare_runtime(
    profile: MachineProfile,
    detected: RuntimeResources,
    *,
    require_gpu: bool = False,
) -> tuple[DetectionComparison, tuple[str, ...]]:
    incompatible: list[str] = []
    warnings: list[str] = []
    if detected.total_ram_bytes is None:
        warnings.append("TOTAL_RAM_UNAVAILABLE")
    elif detected.total_ram_bytes < profile.total_ram_bytes:
        incompatible.append("DETECTED_RAM_BELOW_PROFILE")
    elif detected.total_ram_bytes > profile.total_ram_bytes:
        warnings.append("DETECTED_RAM_ABOVE_PROFILE")
    if detected.logical_processors < profile.logical_processors:
        incompatible.append("DETECTED_LOGICAL_PROCESSORS_BELOW_PROFILE")
    elif detected.logical_processors > profile.logical_processors:
        warnings.append("DETECTED_LOGICAL_PROCESSORS_ABOVE_PROFILE")
    missing_gpu = bool(profile.gpu_inventory) and not detected.gpu_inventory
    if missing_gpu:
        (incompatible if require_gpu else warnings).append("PROFILE_GPU_NOT_DETECTED")
    reasons = tuple(incompatible + warnings)
    if incompatible:
        return DetectionComparison.INCOMPATIBLE, reasons
    if warnings:
        return DetectionComparison.WARNING_DIFFERENCE, reasons
    return DetectionComparison.MATCH, reasons
