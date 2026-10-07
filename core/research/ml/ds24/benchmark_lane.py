"""Contracts for supervisor-owned, read-only CLEAN V2 benchmark work.

Benchmark control and outputs deliberately live outside the production run root.
The supervisor remains the only component allowed to admit and launch a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.ds24.clean_v2_contracts import stable_hash


BENCHMARK_REQUEST_SCHEMA = "DS24_CLEAN_V2_GOVERNED_BENCHMARK_REQUEST_V1"
BENCHMARK_RESULT_SCHEMA = "DS24_CLEAN_V2_GOVERNED_BENCHMARK_RESULT_V1"
INCREMENTAL_PANEL_REQUEST_SCHEMA = (
    "DS24_CLEAN_V2_GOVERNED_INCREMENTAL_PANEL_REQUEST_V1"
)
INCREMENTAL_PANEL_RESULT_SCHEMA = (
    "DS24_CLEAN_V2_GOVERNED_INCREMENTAL_PANEL_RESULT_V1"
)
QUALIFICATION_WINDOW_SCHEMA = "DS24_CLEAN_V2_QUALIFICATION_WINDOW_V1"
SUPPORTED_BENCHMARK_FAMILIES = frozenset({"random_forest"})
SUPPORTED_RF_N_JOBS = (1, 2, 3)
SUPPORTED_INCREMENTAL_PANEL_FAMILIES = (
    "random_forest",
    "huber",
    "elastic_net_C5",
)
QUALIFICATION_HOLD_FAMILIES = frozenset(
    {"random_forest", "huber", "elastic_net_C5"}
)


class BenchmarkLaneError(ValueError):
    """Raised when a benchmark request could touch production state."""


def _resolved_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved != resolved_root and resolved_root in resolved.parents


@dataclass(frozen=True)
class GovernedBenchmarkRequest:
    request_id: str
    family: str
    package_ordinal: int
    data_root: Path
    output_root: Path
    production_run_root: Path
    source_hash: str
    feature_authority_hash: str
    target_authority_hash: str
    requested_rf_n_jobs: tuple[int, ...] = SUPPORTED_RF_N_JOBS
    schema: str = BENCHMARK_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BENCHMARK_REQUEST_SCHEMA:
            raise BenchmarkLaneError("Unsupported governed benchmark request schema")
        if not self.request_id or Path(self.request_id).name != self.request_id:
            raise BenchmarkLaneError("Benchmark request_id must be one path-safe name")
        if self.family not in SUPPORTED_BENCHMARK_FAMILIES:
            raise BenchmarkLaneError(f"Unsupported benchmark family: {self.family}")
        if self.package_ordinal < 0:
            raise BenchmarkLaneError("Benchmark package ordinal must be non-negative")
        if tuple(self.requested_rf_n_jobs) != SUPPORTED_RF_N_JOBS:
            raise BenchmarkLaneError("RF qualification must compare n_jobs 1, 2, and 3")
        if not self.data_root.is_dir():
            raise BenchmarkLaneError("Benchmark data root is unavailable")
        if _resolved_within(self.output_root, self.production_run_root):
            raise BenchmarkLaneError("Benchmark output cannot be inside production state")
        if self.output_root.resolve() == self.production_run_root.resolve():
            raise BenchmarkLaneError("Benchmark output cannot equal production state")
        if not self.source_hash:
            raise BenchmarkLaneError("Benchmark source hash is required")
        if not self.feature_authority_hash or not self.target_authority_hash:
            raise BenchmarkLaneError("Benchmark data authority hashes are required")

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.payload(include_identity=False))

    def payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "request_id": self.request_id,
            "family": self.family,
            "package_ordinal": self.package_ordinal,
            "data_root": str(self.data_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "production_run_root": str(self.production_run_root.resolve()),
            "source_hash": self.source_hash,
            "feature_authority_hash": self.feature_authority_hash,
            "target_authority_hash": self.target_authority_hash,
            "requested_rf_n_jobs": list(self.requested_rf_n_jobs),
        }
        if include_identity:
            result["request_identity_hash"] = self.identity_hash
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GovernedBenchmarkRequest":
        request = cls(
            schema=str(payload.get("schema") or ""),
            request_id=str(payload.get("request_id") or ""),
            family=str(payload.get("family") or ""),
            package_ordinal=int(payload.get("package_ordinal", -1)),
            data_root=Path(str(payload.get("data_root") or "")),
            output_root=Path(str(payload.get("output_root") or "")),
            production_run_root=Path(
                str(payload.get("production_run_root") or "")
            ),
            source_hash=str(payload.get("source_hash") or ""),
            feature_authority_hash=str(
                payload.get("feature_authority_hash") or ""
            ),
            target_authority_hash=str(
                payload.get("target_authority_hash") or ""
            ),
            requested_rf_n_jobs=tuple(
                int(value) for value in payload.get("requested_rf_n_jobs", ())
            ),
        )
        supplied_hash = str(payload.get("request_identity_hash") or "")
        if supplied_hash != request.identity_hash:
            raise BenchmarkLaneError("Benchmark request identity hash mismatch")
        return request


@dataclass(frozen=True)
class GovernedIncrementalPanelRequest:
    """Read-only full-universe panel-reuse qualification request."""

    request_id: str
    package_ordinal: int
    data_root: Path
    output_root: Path
    production_run_root: Path
    source_hash: str
    feature_authority_hash: str
    target_authority_hash: str
    families: tuple[str, ...] = SUPPORTED_INCREMENTAL_PANEL_FAMILIES
    schema: str = INCREMENTAL_PANEL_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INCREMENTAL_PANEL_REQUEST_SCHEMA:
            raise BenchmarkLaneError(
                "Unsupported incremental-panel benchmark request schema"
            )
        if not self.request_id or Path(self.request_id).name != self.request_id:
            raise BenchmarkLaneError("Benchmark request_id must be one path-safe name")
        if self.package_ordinal <= 0:
            raise BenchmarkLaneError(
                "Incremental-panel package ordinal must have a predecessor"
            )
        if tuple(self.families) != SUPPORTED_INCREMENTAL_PANEL_FAMILIES:
            raise BenchmarkLaneError(
                "Incremental-panel qualification must cover RF, Huber, and Elastic C5"
            )
        if not self.data_root.is_dir():
            raise BenchmarkLaneError("Benchmark data root is unavailable")
        if _resolved_within(self.output_root, self.production_run_root):
            raise BenchmarkLaneError("Benchmark output cannot be inside production state")
        if self.output_root.resolve() == self.production_run_root.resolve():
            raise BenchmarkLaneError("Benchmark output cannot equal production state")
        if not self.source_hash:
            raise BenchmarkLaneError("Benchmark source hash is required")
        if not self.feature_authority_hash or not self.target_authority_hash:
            raise BenchmarkLaneError("Benchmark data authority hashes are required")

    @property
    def family(self) -> str:
        """Reservation family used for the single isolated benchmark process."""

        return "random_forest"

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.payload(include_identity=False))

    def payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "request_id": self.request_id,
            "package_ordinal": self.package_ordinal,
            "data_root": str(self.data_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "production_run_root": str(self.production_run_root.resolve()),
            "source_hash": self.source_hash,
            "feature_authority_hash": self.feature_authority_hash,
            "target_authority_hash": self.target_authority_hash,
            "families": list(self.families),
        }
        if include_identity:
            result["request_identity_hash"] = self.identity_hash
        return result

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "GovernedIncrementalPanelRequest":
        request = cls(
            schema=str(payload.get("schema") or ""),
            request_id=str(payload.get("request_id") or ""),
            package_ordinal=int(payload.get("package_ordinal", -1)),
            data_root=Path(str(payload.get("data_root") or "")),
            output_root=Path(str(payload.get("output_root") or "")),
            production_run_root=Path(
                str(payload.get("production_run_root") or "")
            ),
            source_hash=str(payload.get("source_hash") or ""),
            feature_authority_hash=str(
                payload.get("feature_authority_hash") or ""
            ),
            target_authority_hash=str(
                payload.get("target_authority_hash") or ""
            ),
            families=tuple(str(value) for value in payload.get("families", ())),
        )
        if str(payload.get("request_identity_hash") or "") != request.identity_hash:
            raise BenchmarkLaneError("Benchmark request identity hash mismatch")
        return request


GovernedRequest = GovernedBenchmarkRequest | GovernedIncrementalPanelRequest


def governed_request_from_payload(payload: Mapping[str, Any]) -> GovernedRequest:
    schema = str(payload.get("schema") or "")
    if schema == BENCHMARK_REQUEST_SCHEMA:
        return GovernedBenchmarkRequest.from_payload(payload)
    if schema == INCREMENTAL_PANEL_REQUEST_SCHEMA:
        return GovernedIncrementalPanelRequest.from_payload(payload)
    raise BenchmarkLaneError("Unsupported governed benchmark request schema")


@dataclass(frozen=True)
class DedicatedQualificationWindow:
    """Source- and generation-bound authority for temporary family holds."""

    host: str
    source_hash: str
    attempt_generation: int
    request_id: str
    held_families: tuple[str, ...]
    opened_at_utc: str
    schema: str = QUALIFICATION_WINDOW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUALIFICATION_WINDOW_SCHEMA:
            raise BenchmarkLaneError("Unsupported qualification-window schema")
        if self.host != "dell":
            raise BenchmarkLaneError("Dedicated qualification is Dell-only")
        if not self.source_hash:
            raise BenchmarkLaneError("Qualification source hash is required")
        if self.attempt_generation <= 0:
            raise BenchmarkLaneError("Qualification generation must be positive")
        if not self.request_id or Path(self.request_id).name != self.request_id:
            raise BenchmarkLaneError("Qualification request_id is invalid")
        normalized = tuple(sorted(set(self.held_families)))
        if normalized != self.held_families:
            raise BenchmarkLaneError("Qualification holds must be sorted and unique")
        if not normalized or not set(normalized) <= QUALIFICATION_HOLD_FAMILIES:
            raise BenchmarkLaneError("Qualification holds are not authorized")
        if "huber" not in normalized:
            raise BenchmarkLaneError("Qualification must hold Huber before Elastic")
        if "random_forest" in normalized and set(normalized) != set(
            QUALIFICATION_HOLD_FAMILIES
        ):
            raise BenchmarkLaneError(
                "Qualification may hold RF only after Huber and Elastic"
            )
        if not self.opened_at_utc:
            raise BenchmarkLaneError("Qualification opening timestamp is required")

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.payload(include_identity=False))

    def payload(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "host": self.host,
            "source_hash": self.source_hash,
            "attempt_generation": self.attempt_generation,
            "request_id": self.request_id,
            "held_families": list(self.held_families),
            "opened_at_utc": self.opened_at_utc,
        }
        if include_identity:
            result["identity_hash"] = self.identity_hash
        return result

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "DedicatedQualificationWindow":
        window = cls(
            schema=str(payload.get("schema") or ""),
            host=str(payload.get("host") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            attempt_generation=int(payload.get("attempt_generation", 0) or 0),
            request_id=str(payload.get("request_id") or ""),
            held_families=tuple(str(value) for value in payload.get("held_families", ())),
            opened_at_utc=str(payload.get("opened_at_utc") or ""),
        )
        if str(payload.get("identity_hash") or "") != window.identity_hash:
            raise BenchmarkLaneError("Qualification-window identity mismatch")
        return window


def benchmark_control_root(repository_root: Path) -> Path:
    return (
        Path(repository_root)
        / "research_runs"
        / "ds24_clean_v2_benchmarks"
        / "governed_lane"
    )


def qualification_window_path(repository_root: Path) -> Path:
    return benchmark_control_root(repository_root) / "qualification_window.json"


def validate_benchmark_namespace(
    *, repository_root: Path, request: GovernedRequest
) -> None:
    root = benchmark_control_root(repository_root)
    if not _resolved_within(request.output_root, root):
        raise BenchmarkLaneError(
            "Benchmark output must be inside the governed benchmark root"
        )
