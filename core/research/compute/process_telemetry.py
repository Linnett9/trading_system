from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

TELEMETRY_CONTRACT = "compute_process_telemetry.v1"
SUMMARY_CONTRACT = "compute_resource_summary.v1"


@dataclass(frozen=True)
class TelemetrySample:
    timestamp: str
    run_id: str
    job_id: str
    process_id: int
    execution_phase: str
    elapsed_seconds: float
    process_rss_bytes: int | None = None
    process_vms_bytes: int | None = None
    child_rss_bytes: int | None = None
    system_total_bytes: int | None = None
    system_available_bytes: int | None = None
    scheduler_reserved_bytes: int | None = None
    process_cpu_percent: float | None = None
    system_cpu_percent: float | None = None
    checkpoint_identity: str | None = None
    completed_items: int | None = None
    telemetry_status: str = "OK"
    contract_version: str = TELEMETRY_CONTRACT


class ProcessTelemetry:
    def __init__(self, *, minimum_interval_seconds: float = 1.0) -> None:
        if not 1 <= minimum_interval_seconds <= 3600:
            raise ValueError("Telemetry interval must be between 1 and 3600 seconds")
        self.minimum_interval_seconds = minimum_interval_seconds
        self.samples: list[TelemetrySample] = []
        self.failures: list[str] = []

    def record(self, sample: TelemetrySample) -> bool:
        if self.samples and (
            sample.elapsed_seconds - self.samples[-1].elapsed_seconds
            < self.minimum_interval_seconds
        ):
            return False
        self.samples.append(sample)
        return True

    def monitor_process(
        self,
        *,
        process: object,
        sample_provider: Callable[[object, float], TelemetrySample],
        interval_seconds: float = 15.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        interval = max(interval_seconds, self.minimum_interval_seconds)
        started = time.monotonic()
        while getattr(process, "poll")() is None:
            try:
                self.record(sample_provider(process, time.monotonic() - started))
            except Exception as exc:  # telemetry cannot mask workload outcome
                self.failures.append(f"{type(exc).__name__}: {exc}")
            sleep(interval)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(TelemetrySample.__dataclass_fields__)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.samples)
        os.replace(temporary, path)

    def summarize(
        self,
        *,
        estimated_peak_ram_bytes: int,
        resource_wait_seconds: float | None = None,
        lease_acquired_at: str | None = None,
        process_started_at: str | None = None,
        process_ended_at: str | None = None,
        normal_reserve_bytes: int | None = None,
    ) -> dict[str, object]:
        rss = [row.process_rss_bytes for row in self.samples if row.process_rss_bytes is not None]
        combined = [
            row.process_rss_bytes + row.child_rss_bytes
            for row in self.samples
            if row.process_rss_bytes is not None and row.child_rss_bytes is not None
        ]
        available = [row.system_available_bytes for row in self.samples if row.system_available_bytes is not None]
        cpu = [row.process_cpu_percent for row in self.samples if row.process_cpu_percent is not None]
        measured = max(combined or rss, default=None)
        status = "TELEMETRY_INCOMPLETE" if self.failures or not self.samples else "COMPLETE"
        summary: dict[str, object] = {
            "contract_version": SUMMARY_CONTRACT,
            "telemetry_status": status,
            "estimated_peak_ram_bytes": estimated_peak_ram_bytes,
            "measured_peak_process_ram_bytes": max(rss, default=None),
            "measured_peak_process_plus_children_ram_bytes": max(combined, default=None),
            "average_process_ram_bytes": (sum(rss) / len(rss)) if rss else None,
            "minimum_system_available_ram_bytes": min(available, default=None),
            "peak_process_cpu_percent": max(cpu, default=None),
            "wall_time_seconds": (
                max((row.elapsed_seconds for row in self.samples), default=None)
            ),
            "resource_wait_seconds": resource_wait_seconds,
            "lease_acquired_timestamp": lease_acquired_at,
            "process_start_timestamp": process_started_at,
            "process_end_timestamp": process_ended_at,
            "telemetry_sample_count": len(self.samples),
            "telemetry_failures": list(self.failures),
            "estimate_to_measured_ratio": (
                estimated_peak_ram_bytes / measured if measured else None
            ),
            "estimate_exceeded": (
                measured > estimated_peak_ram_bytes if measured is not None else None
            ),
            "reserve_threshold_approached": (
                min(available) <= normal_reserve_bytes
                if available and normal_reserve_bytes is not None else None
            ),
        }
        return summary

    def write_summary(self, path: Path, **kwargs: object) -> dict[str, object]:
        summary = self.summarize(**kwargs)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return summary
