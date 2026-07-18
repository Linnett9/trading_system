from __future__ import annotations

from dataclasses import replace

import pytest

from core.research.compute.machine_profile import GIB, dell_i5_10500_profile
from core.research.compute.resource_governor import (
    LeaseStatus,
    ResourceGovernor,
    ResourceRequest,
)


def request(job: str, *, ram: int = 2 * GIB, cpu: int = 1, group: str = "EVALUATION", **kwargs: object) -> ResourceRequest:
    return ResourceRequest(
        pipeline="test", stage="stage", job_id=job, run_id="run",
        resource_class="SMALL", estimated_peak_ram_bytes=ram,
        cpu_weight=cpu, inner_threads=1, gpu_required=False,
        concurrency_group=group, estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity="fixture", **kwargs,
    )


def governor(available: list[int], **kwargs: object) -> ResourceGovernor:
    return ResourceGovernor(
        dell_i5_10500_profile(source_git_commit="abc", generated_at="now"),
        available_memory=lambda: available[0],
        **kwargs,
    )


def test_request_validation_and_ram_cpu_admission_release() -> None:
    with pytest.raises(ValueError, match="RAM estimate"):
        request("invalid", ram=0)
    available = [30 * GIB]
    service = governor(available)
    first = service.request_lease(request("one", ram=12 * GIB, cpu=2))
    assert first.status == LeaseStatus.GRANTED
    service.mark_process_started(first, 101)
    assert first.status == LeaseStatus.ACTIVE
    second = service.request_lease(request("two", ram=8 * GIB, cpu=2))
    assert second.status == LeaseStatus.WAITING
    assert second.blocked_reasons == ("CPU_WEIGHT_CAPACITY",)
    service.release_lease(first, "PROCESS_FAILURE")
    assert first.status == LeaseStatus.RELEASED
    assert second.status == LeaseStatus.GRANTED
    assert service.status()["active_reserved_ram_bytes"] == 8 * GIB


def test_low_memory_deep_groups_duplicates_and_fifo() -> None:
    available = [7 * GIB]
    service = governor(available)
    low = service.request_lease(request("low", ram=1 * GIB))
    assert low.blocked_reasons == ("NORMAL_MEMORY_RESERVE",)
    available[0] = 5 * GIB
    critical = service.request_lease(request("critical"))
    # Frozen FIFO means later jobs cannot bypass the first blocked request.
    assert critical.status == LeaseStatus.WAITING
    with pytest.raises(ValueError, match="Duplicate"):
        service.request_lease(request("critical"))

    available[0] = 30 * GIB
    deep_service = governor(available)
    first = deep_service.request_lease(request("deep1", group="SELECTOR_DEEP"))
    second = deep_service.request_lease(request("deep2", group="EXPOSURE_DEEP"))
    assert first.status == LeaseStatus.GRANTED
    assert second.blocked_reasons == ("DEEP_MODEL_CONCURRENCY_LIMIT",)
    deep_service.release_lease(first, "STARTUP_FAILURE")
    assert first.status == LeaseStatus.FAILED_TO_START
    assert second.status == LeaseStatus.GRANTED
    deep_service.release_lease(second, "CANCELLED")
    assert second.status == LeaseStatus.CANCELLED


def test_group_gpu_and_reservation_before_start() -> None:
    available = [30 * GIB]
    service = governor(
        available,
        incompatible_groups=(("DATA_FINALISATION", "SELECTOR_TREE"),),
        available_gpus=lambda: (),
    )
    first = service.request_lease(request("tree", group="SELECTOR_TREE"))
    assert service.status()["active_reserved_ram_bytes"] == 2 * GIB
    blocked = service.request_lease(request("data", group="DATA_FINALISATION"))
    assert blocked.blocked_reasons == ("CONCURRENCY_GROUP_INCOMPATIBLE",)
    service.release_lease(first)
    assert blocked.status == LeaseStatus.GRANTED
    service.release_lease(blocked)
    gpu = service.request_lease(replace(request("gpu"), gpu_required=True, logical_checksum=""))
    assert gpu.blocked_reasons == ("REQUIRED_GPU_UNAVAILABLE",)
