from __future__ import annotations

import pytest

from core.research.compute.machine_profile import (
    DetectionComparison,
    GIB,
    MachineProfile,
    RuntimeResources,
    compare_runtime,
    dell_i5_10500_profile,
)


def test_dell_profile_and_validation() -> None:
    profile = dell_i5_10500_profile(source_git_commit="abc", generated_at="now")
    assert profile.total_ram_bytes == 32 * GIB
    assert profile.schedulable_ram_bytes == 24 * GIB
    assert profile.logical_checksum
    values = profile.__dict__ | {"schedulable_ram_bytes": 33 * GIB, "logical_checksum": ""}
    with pytest.raises(ValueError, match="exceeds total"):
        MachineProfile(**values)
    values = profile.__dict__ | {
        "critical_reserve_bytes": 9 * GIB, "logical_checksum": ""
    }
    with pytest.raises(ValueError, match="Critical reserve"):
        MachineProfile(**values)


def test_runtime_comparison_is_conservative() -> None:
    profile = dell_i5_10500_profile(source_git_commit="abc", generated_at="now")
    exact = RuntimeResources(
        "Windows", "Intel Core i5-10500", 6, 12, 32 * GIB, 20 * GIB,
        ("NVIDIA Quadro P620",),
    )
    assert compare_runtime(profile, exact)[0] == DetectionComparison.MATCH
    less_ram = RuntimeResources(
        "Windows", "Intel Core i5-10500", 6, 12, 31 * GIB, 20 * GIB, ()
    )
    status, reasons = compare_runtime(profile, less_ram)
    assert status == DetectionComparison.INCOMPATIBLE
    assert "DETECTED_RAM_BELOW_PROFILE" in reasons
    no_gpu = exact.__class__(
        exact.os_family, exact.cpu_model, exact.physical_cores,
        exact.logical_processors, exact.total_ram_bytes,
        exact.available_ram_bytes, (),
    )
    assert compare_runtime(profile, no_gpu)[0] == DetectionComparison.WARNING_DIFFERENCE
    assert compare_runtime(profile, no_gpu, require_gpu=True)[0] == DetectionComparison.INCOMPATIBLE
