from __future__ import annotations

import json

from core.research.compute.process_telemetry import ProcessTelemetry, TelemetrySample


def sample(elapsed: float, rss: int | None, child: int | None, available: int | None) -> TelemetrySample:
    return TelemetrySample(
        timestamp=f"t{elapsed}", run_id="run", job_id="job", process_id=1,
        execution_phase="fit", elapsed_seconds=elapsed,
        process_rss_bytes=rss, child_rss_bytes=child,
        system_available_bytes=available, process_cpu_percent=25.0,
    )


def test_bounded_samples_and_resource_summary(tmp_path) -> None:
    telemetry = ProcessTelemetry(minimum_interval_seconds=10)
    assert telemetry.record(sample(0, 100, 20, 1_000))
    assert not telemetry.record(sample(5, 999, 999, 1))
    assert telemetry.record(sample(15, 150, 50, 700))
    summary = telemetry.summarize(
        estimated_peak_ram_bytes=180, normal_reserve_bytes=800
    )
    assert summary["telemetry_sample_count"] == 2
    assert summary["measured_peak_process_ram_bytes"] == 150
    assert summary["measured_peak_process_plus_children_ram_bytes"] == 200
    assert summary["estimate_exceeded"] is True
    assert summary["reserve_threshold_approached"] is True

    telemetry.write_csv(tmp_path / "telemetry.csv")
    telemetry.write_summary(
        tmp_path / "resource_summary.json",
        estimated_peak_ram_bytes=180,
    )
    assert (tmp_path / "telemetry.csv").read_text().count("\n") == 3
    assert json.loads((tmp_path / "resource_summary.json").read_text())[
        "contract_version"
    ] == "compute_resource_summary.v1"


def test_missing_values_are_unknown_and_failures_are_incomplete() -> None:
    telemetry = ProcessTelemetry()
    telemetry.record(sample(0, None, None, None))
    telemetry.failures.append("metrics unavailable")
    summary = telemetry.summarize(estimated_peak_ram_bytes=100)
    assert summary["measured_peak_process_ram_bytes"] is None
    assert summary["estimate_exceeded"] is None
    assert summary["telemetry_status"] == "TELEMETRY_INCOMPLETE"
