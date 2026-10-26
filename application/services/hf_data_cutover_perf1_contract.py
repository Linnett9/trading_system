"""Immutable operational configuration for the bounded PERF1 experiment."""

from __future__ import annotations

from dataclasses import dataclass


PERF1_SCHEMA = "hf_data_cutover_r1.perf1.v1"
PERF1_SAMPLE_SIZE = 24
PERF1_BENCHMARK_WORKERS = 8
PERF1_PACING_REQUESTS_PER_MINUTE = 60
PERF1_HARD_REQUEST_CEILING = 150
PERF1_SYMBOLS_PER_MONTH = 5
PERF1_PROVIDER_LIMIT = 10_000
PERF1_ADOPTION_RATIO = 1.20
PERF1_SCRATCH_MAX_BYTES = 10 * 1024**3


@dataclass(frozen=True, slots=True)
class Perf1TransportConfiguration:
    variant: str
    shard_geometry: str
    session_mode: str
    workers: int
    pacing_requests_per_minute: int = PERF1_PACING_REQUESTS_PER_MINUTE
    hard_request_ceiling: int = PERF1_HARD_REQUEST_CEILING
    provider_limit: int = PERF1_PROVIDER_LIMIT


__all__ = [
    "PERF1_ADOPTION_RATIO",
    "PERF1_BENCHMARK_WORKERS",
    "PERF1_HARD_REQUEST_CEILING",
    "PERF1_PACING_REQUESTS_PER_MINUTE",
    "PERF1_PROVIDER_LIMIT",
    "PERF1_SAMPLE_SIZE",
    "PERF1_SCHEMA",
    "PERF1_SCRATCH_MAX_BYTES",
    "PERF1_SYMBOLS_PER_MONTH",
    "Perf1TransportConfiguration",
]
