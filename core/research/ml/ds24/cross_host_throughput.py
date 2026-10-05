"""Host-neutral DS24 CLEAN V2 package profiling contracts.

The module is deliberately free of filesystem and process inspection.  Workers on
either host emit immutable :class:`PackageProfile` documents; reporting code then
aggregates those documents without consulting process liveness or inferring
unobserved timings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


PROFILE_SCHEMA_ID = "DS24_CLEAN_V2_V11X_PACKAGE_PROFILE_V1"
SUPPORTED_HOSTS = frozenset({"dell", "mac"})
HOST_FAMILIES: Mapping[str, frozenset[str]] = {
    "dell": frozenset({"random_forest", "huber", "elastic_net_C5"}),
    "mac": frozenset(
        {
            "lightgbm_rank_xendcg",
            "momentum_transformer",
            "market_context_encoder",
            "temporal_fusion_transformer",
        }
    ),
}
ALL_HOST_FAMILY_LANES = tuple(
    (host, family)
    for host in sorted(HOST_FAMILIES)
    for family in sorted(HOST_FAMILIES[host])
)


class ProfileContractError(ValueError):
    """Raised when profiling evidence is incomplete or internally inconsistent."""


class StageStatus(str, Enum):
    MEASURED = "MEASURED"
    UNINSTRUMENTED = "UNINSTRUMENTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ThroughputStage(str, Enum):
    SCHEDULER_DELAY = "scheduler_delay"
    RESOURCE_CAPACITY_WAIT = "resource_capacity_wait"
    CACHE_LOOKUP = "cache_lookup"
    DATA_PARQUET_READ = "data_parquet_read"
    PANEL_ASSEMBLY = "panel_assembly"
    POPULATION_SLICING = "population_slicing"
    SEQUENCE_QUERY_CONSTRUCTION = "sequence_query_construction"
    FEATURE_SELECTION = "feature_selection"
    PREPROCESSING = "preprocessing"
    IMPUTATION_SCALING = "imputation_scaling"
    TENSOR_ARRAY_CONVERSION = "tensor_array_conversion"
    MODEL_FIT = "model_fit"
    MODEL_PERSISTENCE = "model_persistence"
    SCORING_INPUT_PREPARATION = "scoring_input_preparation"
    PREDICTION_INFERENCE = "prediction_inference"
    SCORE_MATRIX_CREATION = "score_matrix_creation"
    GROUP_EVALUATION = "group_evaluation"
    PREDICTION_LEDGER_PREPARATION = "prediction_ledger_preparation"
    ATOMIC_LEDGER_PUBLICATION = "atomic_ledger_publication"
    METRICS_CALCULATION = "metrics_calculation"
    CHECKPOINT_PUBLICATION = "checkpoint_publication"
    CACHE_WRITE_EVICTION = "cache_write_eviction"
    PACKAGE_COMPLETION = "package_completion"


WAIT_STAGES = frozenset(
    {ThroughputStage.SCHEDULER_DELAY, ThroughputStage.RESOURCE_CAPACITY_WAIT}
)
IO_STAGES = frozenset(
    {
        ThroughputStage.CACHE_LOOKUP,
        ThroughputStage.DATA_PARQUET_READ,
        ThroughputStage.CACHE_WRITE_EVICTION,
    }
)
FIT_STAGES = frozenset(
    {
        ThroughputStage.FEATURE_SELECTION,
        ThroughputStage.PREPROCESSING,
        ThroughputStage.IMPUTATION_SCALING,
        ThroughputStage.TENSOR_ARRAY_CONVERSION,
        ThroughputStage.MODEL_FIT,
        ThroughputStage.MODEL_PERSISTENCE,
    }
)
SCORE_EVALUATION_STAGES = frozenset(
    {
        ThroughputStage.SCORING_INPUT_PREPARATION,
        ThroughputStage.PREDICTION_INFERENCE,
        ThroughputStage.SCORE_MATRIX_CREATION,
        ThroughputStage.GROUP_EVALUATION,
        ThroughputStage.METRICS_CALCULATION,
    }
)
PUBLISH_STAGES = frozenset(
    {
        ThroughputStage.PREDICTION_LEDGER_PREPARATION,
        ThroughputStage.ATOMIC_LEDGER_PUBLICATION,
        ThroughputStage.CHECKPOINT_PUBLICATION,
        ThroughputStage.PACKAGE_COMPLETION,
    }
)


@dataclass(frozen=True)
class StageResourceObservation:
    """Optional counters captured over one stage, never reconstructed later."""

    cpu_seconds: float | None = None
    resident_memory_growth_bytes: int | None = None
    private_commit_growth_bytes: int | None = None
    rss_growth_bytes: int | None = None
    swap_growth_bytes: int | None = None
    job_commit_growth_bytes: int | None = None
    io_read_bytes: int | None = None
    io_write_bytes: int | None = None

    def __post_init__(self) -> None:
        monotonic_counters = (
            "cpu_seconds",
            "io_read_bytes",
            "io_write_bytes",
        )
        for name in monotonic_counters:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ProfileContractError(f"{name} must be non-negative")

    def payload(self) -> dict[str, float | int | None]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StageResourceObservation":
        return cls(**{name: payload.get(name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class WorkloadCounters:
    """Typed package-shape, cache, and compute counters captured when available."""

    rows_read: int | None = None
    columns_read: int | None = None
    bytes_read: int | None = None
    partitions_opened: int | None = None
    cache_hit: bool | None = None
    cache_miss_reason: str | None = None
    bytes_reused: int | None = None
    bytes_regenerated: int | None = None
    cache_entry_size_bytes: int | None = None
    eviction_count: int | None = None
    panel_rows: int | None = None
    training_rows: int | None = None
    scoring_rows: int | None = None
    asset_count: int | None = None
    session_count: int | None = None
    query_count: int | None = None
    sequence_endpoint_count: int | None = None
    sequence_length: int | None = None
    batch_count: int | None = None
    time_waiting_seconds: float | None = None
    time_computing_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name in {"cache_hit", "cache_miss_reason"} or value is None:
                continue
            if value < 0:
                raise ProfileContractError(f"{name} must be non-negative")
        if self.cache_hit is True and self.cache_miss_reason is not None:
            raise ProfileContractError(
                "cache_miss_reason must be absent when cache_hit is true"
            )

    def payload(self) -> dict[str, int | float | str | bool | None]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WorkloadCounters":
        return cls(**{name: payload.get(name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class StageObservation:
    stage: ThroughputStage
    status: StageStatus
    wall_seconds: float | None = None
    resource: StageResourceObservation = field(default_factory=StageResourceObservation)

    def __post_init__(self) -> None:
        if self.status is StageStatus.MEASURED:
            if self.wall_seconds is None or not math.isfinite(self.wall_seconds):
                raise ProfileContractError(
                    f"Measured stage {self.stage.value} requires finite wall_seconds"
                )
            if self.wall_seconds < 0:
                raise ProfileContractError("wall_seconds must be non-negative")
        elif self.wall_seconds is not None:
            raise ProfileContractError(
                f"{self.status.value} stage {self.stage.value} cannot have wall_seconds"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "wall_seconds": self.wall_seconds,
            "resource": self.resource.payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StageObservation":
        return cls(
            stage=ThroughputStage(str(payload["stage"])),
            status=StageStatus(str(payload["status"])),
            wall_seconds=(
                None
                if payload.get("wall_seconds") is None
                else float(payload["wall_seconds"])
            ),
            resource=StageResourceObservation.from_payload(payload.get("resource") or {}),
        )


def uninstrumented_stage_observations() -> tuple[StageObservation, ...]:
    """Return an explicit, complete blank profile; no missing stage is implicit."""

    return tuple(
        StageObservation(stage=stage, status=StageStatus.UNINSTRUMENTED)
        for stage in ThroughputStage
    )


class PackageProfileBuilder:
    """Build one complete package profile from directly observed stage timings.

    The builder performs no timing itself.  Callers supply durations and resource
    deltas measured at the worker boundary, which keeps clocks/process APIs out of
    the scientific contract and makes host adapters independently testable.
    """

    def __init__(
        self,
        *,
        host: str,
        family: str,
        run_id: str,
        package_id: str,
        refit_timestamp: str,
        source_hash: str,
        scientific_authority_hash: str,
    ) -> None:
        self._identity = {
            "host": host,
            "family": family,
            "run_id": run_id,
            "package_id": package_id,
            "refit_timestamp": refit_timestamp,
            "source_hash": source_hash,
            "scientific_authority_hash": scientific_authority_hash,
        }
        self._observations: dict[ThroughputStage, StageObservation] = {}

    def record(
        self,
        stage: ThroughputStage,
        *,
        wall_seconds: float,
        resource: StageResourceObservation | None = None,
    ) -> None:
        if stage in self._observations:
            raise ProfileContractError(f"Stage already recorded: {stage.value}")
        self._observations[stage] = StageObservation(
            stage=stage,
            status=StageStatus.MEASURED,
            wall_seconds=wall_seconds,
            resource=resource or StageResourceObservation(),
        )

    def mark_not_applicable(self, *stages: ThroughputStage) -> None:
        for stage in stages:
            if stage in self._observations:
                raise ProfileContractError(f"Stage already recorded: {stage.value}")
            self._observations[stage] = StageObservation(
                stage=stage, status=StageStatus.NOT_APPLICABLE
            )

    def build(
        self,
        *,
        wall_seconds: float,
        workload: WorkloadCounters | None = None,
    ) -> "PackageProfile":
        stages = tuple(
            self._observations.get(
                stage,
                StageObservation(stage=stage, status=StageStatus.UNINSTRUMENTED),
            )
            for stage in ThroughputStage
        )
        return PackageProfile(
            **self._identity,
            wall_seconds=wall_seconds,
            stages=stages,
            workload=workload or WorkloadCounters(),
        )


@dataclass(frozen=True)
class PackageProfile:
    host: str
    family: str
    run_id: str
    package_id: str
    refit_timestamp: str
    source_hash: str
    scientific_authority_hash: str
    wall_seconds: float
    stages: tuple[StageObservation, ...]
    workload: WorkloadCounters = field(default_factory=WorkloadCounters)
    schema_id: str = PROFILE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != PROFILE_SCHEMA_ID:
            raise ProfileContractError(f"Unsupported profile schema: {self.schema_id}")
        if self.host not in SUPPORTED_HOSTS:
            raise ProfileContractError(f"Unsupported host: {self.host}")
        if self.family not in HOST_FAMILIES[self.host]:
            raise ProfileContractError(
                f"Family {self.family!r} is not owned by host {self.host!r}"
            )
        if not self.run_id or not self.package_id:
            raise ProfileContractError("run_id and package_id are required")
        if not self.source_hash or not self.scientific_authority_hash:
            raise ProfileContractError("source and scientific authority hashes are required")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ProfileContractError("wall_seconds must be finite and non-negative")
        observed = [row.stage for row in self.stages]
        expected = list(ThroughputStage)
        if len(observed) != len(set(observed)):
            raise ProfileContractError("Each throughput stage must occur exactly once")
        if set(observed) != set(expected):
            missing = sorted(stage.value for stage in set(expected) - set(observed))
            extra = sorted(stage.value for stage in set(observed) - set(expected))
            raise ProfileContractError(f"Stage coverage mismatch missing={missing} extra={extra}")
        measured = sum(
            row.wall_seconds or 0.0
            for row in self.stages
            if row.status is StageStatus.MEASURED
        )
        tolerance = max(1e-6, self.wall_seconds * 1e-6)
        if measured > self.wall_seconds + tolerance:
            raise ProfileContractError(
                "Measured stage time exceeds package wall time; overlapping stages "
                "must not be summed into this schema"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "host": self.host,
            "family": self.family,
            "run_id": self.run_id,
            "package_id": self.package_id,
            "refit_timestamp": self.refit_timestamp,
            "source_hash": self.source_hash,
            "scientific_authority_hash": self.scientific_authority_hash,
            "wall_seconds": self.wall_seconds,
            "stages": [row.payload() for row in self.stages],
            "workload": self.workload.payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PackageProfile":
        return cls(
            schema_id=str(payload["schema_id"]),
            host=str(payload["host"]),
            family=str(payload["family"]),
            run_id=str(payload["run_id"]),
            package_id=str(payload["package_id"]),
            refit_timestamp=str(payload["refit_timestamp"]),
            source_hash=str(payload["source_hash"]),
            scientific_authority_hash=str(payload["scientific_authority_hash"]),
            wall_seconds=float(payload["wall_seconds"]),
            stages=tuple(
                StageObservation.from_payload(row) for row in payload["stages"]
            ),
            workload=WorkloadCounters.from_payload(payload.get("workload") or {}),
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ProfileContractError("Cannot calculate a percentile without samples")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stage_percentage(
    profiles: Sequence[PackageProfile], stages: frozenset[ThroughputStage]
) -> float:
    total_wall = sum(profile.wall_seconds for profile in profiles)
    if total_wall <= 0:
        return 0.0
    elapsed = sum(
        row.wall_seconds or 0.0
        for profile in profiles
        for row in profile.stages
        if row.stage in stages and row.status is StageStatus.MEASURED
    )
    return 100.0 * elapsed / total_wall


def summarize_family_profiles(
    profiles: Iterable[PackageProfile], *, remaining_packages: int | None = None
) -> dict[str, Any]:
    """Summarize one host/family without blending incompatible authorities."""

    rows = tuple(profiles)
    if not rows:
        raise ProfileContractError("At least one package profile is required")
    host_families = {(row.host, row.family) for row in rows}
    if len(host_families) != 1:
        raise ProfileContractError("A family summary cannot combine host/family lanes")
    identities = {
        (row.run_id, row.source_hash, row.scientific_authority_hash) for row in rows
    }
    if len(identities) != 1:
        raise ProfileContractError("A family summary cannot combine authority identities")
    if remaining_packages is not None and remaining_packages < 0:
        raise ProfileContractError("remaining_packages must be non-negative")
    package_ids = [row.package_id for row in rows]
    if len(package_ids) != len(set(package_ids)):
        raise ProfileContractError("Duplicate package profiles cannot be aggregated")

    wall = [row.wall_seconds for row in rows]
    stage_seconds = {
        stage: sum(
            observation.wall_seconds or 0.0
            for profile in rows
            for observation in profile.stages
            if observation.stage is stage
            and observation.status is StageStatus.MEASURED
        )
        for stage in ThroughputStage
    }
    primary_stage = max(stage_seconds, key=stage_seconds.get)
    uninstrumented = sorted(
        {
            observation.stage.value
            for profile in rows
            for observation in profile.stages
            if observation.status is StageStatus.UNINSTRUMENTED
        }
    )
    mean_wall = sum(wall) / len(wall)
    remaining_hours = (
        None if remaining_packages is None else mean_wall * remaining_packages / 3600.0
    )
    return {
        "host": rows[0].host,
        "family": rows[0].family,
        "sample_count": len(rows),
        "sec_per_refit_p50": median(wall),
        "sec_per_refit_p90": _percentile(wall, 0.90),
        "sec_per_refit_p99": _percentile(wall, 0.99),
        "wait_percent": _stage_percentage(rows, WAIT_STAGES),
        "io_percent": _stage_percentage(rows, IO_STAGES),
        "fit_percent": _stage_percentage(rows, FIT_STAGES),
        "score_evaluation_percent": _stage_percentage(
            rows, SCORE_EVALUATION_STAGES
        ),
        "publish_percent": _stage_percentage(rows, PUBLISH_STAGES),
        "primary_bottleneck": (
            primary_stage.value if stage_seconds[primary_stage] > 0 else "UNINSTRUMENTED"
        ),
        "remaining_packages": remaining_packages,
        "estimated_remaining_hours": remaining_hours,
        "uninstrumented_stages": uninstrumented,
    }


def distributed_critical_path(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the longest evidenced host/family lane, never filling missing ETAs."""

    rows = [dict(row) for row in summaries]
    evidenced = [
        row for row in rows if row.get("estimated_remaining_hours") is not None
    ]
    if not evidenced:
        return {
            "classification": "UNAVAILABLE_MISSING_REMAINING_PACKAGE_COUNTS",
            "expected_final_straggler": None,
            "estimated_distributed_remaining_hours": None,
        }
    slowest = max(evidenced, key=lambda row: float(row["estimated_remaining_hours"]))
    complete = len(evidenced) == len(rows)
    return {
        "classification": (
            "COMPLETE_EVIDENCE" if complete else "PARTIAL_EVIDENCE_DO_NOT_DEPLOY"
        ),
        "expected_final_straggler": {
            "host": slowest["host"],
            "family": slowest["family"],
        },
        "estimated_distributed_remaining_hours": slowest[
            "estimated_remaining_hours"
        ],
    }
