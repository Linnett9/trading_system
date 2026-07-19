from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import multiprocessing
import os
import pickle
import shutil
import subprocess
import traceback
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import date as calendar_date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow.lib import ArrowTypeError

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.stock_level_alpha_features_audit import _audit
from core.research.ml.stock_level.stock_level_alpha_features_builder import (
    _add_cross_sectional_features,
    _build_symbol_rows,
    _prepare_history,
    _time_series_features,
)
from core.research.ml.stock_level.selector_lineage import (
    CURRENT_ECONOMIC_TARGET_ID,
    merge_enrichment_preserving_base,
)
from core.research.ml.stock_level.stock_level_alpha_features_io import (
    _load_price_histories,
    _markdown,
    _write_audit_csv,
)
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    ENGINEERED_FEATURE_COLUMNS,
    ENRICHMENT_METADATA_COLUMNS,
    FEATURE_DEFINITIONS,
    StockLevelAlphaFeaturePaths,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    canonical_artifact_path,
    read_stock_level_artifact,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    CONTEXT_COLUMNS,
    DECISION_CONTEXT_COLUMNS,
    PREDICTION_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
)


DEFAULT_REPORT_ROOT = Path("reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2")
EXPECTED_CANONICAL_HASH = "c2ab57992c9363c118d854f01da18ea34122b9c0775af3d0676afe5ff80bad56"
EXPECTED_BASE_HASH = "739a2b984cdd0a160d65ea546d9523b75637be3921c14734dd5483a093357e89"
ALPHA_ENRICHMENT_CONTRACT_VERSION = "canonical_v2_alpha_enrichment_v2"
ALPHA_BASE_CONTRACT_VERSION = "canonical_v2_alpha_base_v1"
ALPHA_BASE_NAMESPACE_CONTRACT_VERSION = "canonical_v2_alpha_base_namespace_v1"
ALPHA_BASE_NAMESPACE_KEY_HEX_LENGTH = 20
ALPHA_PARTITION_NAMESPACE_CONTRACT_VERSION = (
    "canonical_v2_alpha_partition_namespace_v1"
)
ALPHA_PARTITION_NAMESPACE_KEY_HEX_LENGTH = 20
TARGET_PROVENANCE_V2 = "stock_level_target_provenance_v2"
REQUIRED_BASE_COLUMNS = {
    "rebalance_date",
    "symbol",
    "decision_timestamp",
    "target_provenance_contract_version",
}
BOOL_COLUMNS = {
    "selector_eligible",
    "provider_transition_flag",
    "true_stock_level_row",
    "overlapping_targets",
}
STRICT_BOOL_COLUMNS = {"true_stock_level_row", "overlapping_targets"}
NON_NULLABLE_COLUMNS = {"true_stock_level_row"}
TEMPORAL_DATE_COLUMNS = {
    "rebalance_date",
    "decision_session_date",
    "first_actionable_session",
    "market_context_source_date",
}
TEMPORAL_TIMESTAMP_COLUMNS = {
    "feature_timestamp",
    "feature_data_cutoff_timestamp",
    "decision_timestamp",
    "target_start_timestamp",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_timestamp",
    "benchmark_target_start_timestamp",
    "benchmark_label_start_timestamp",
    "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp",
    "context_source_timestamp",
    "market_context_availability_timestamp",
    "fundamentals_available_timestamp",
    "fundamentals_latest_filing_timestamp",
}
TEMPORAL_COLUMNS = TEMPORAL_DATE_COLUMNS | TEMPORAL_TIMESTAMP_COLUMNS
INT_COLUMNS = {
    "target_horizon_trading_days",
    "required_purge_horizon_trading_days",
    "target_observation_count",
    "context_age_calendar_days",
    "breadth_eligible_symbol_count",
    "breadth_observed_symbol_count",
    "industry_peer_count",
    "fundamentals_data_age_days",
    "fundamental_coverage_count",
}
INTERMEDIATE_NUMERIC_COLUMNS = {
    "_stock_above_200d_average",
    "_stock_momentum_20d",
    "_stock_momentum_60d",
}
NUMERIC_COLUMNS = {
    *ENGINEERED_FEATURE_COLUMNS,
    *FEATURE_DEFINITIONS,
    *PREDICTION_COLUMNS,
    *ACTUAL_COLUMNS,
    *INTERMEDIATE_NUMERIC_COLUMNS,
    "industry_mapping_available",
    "model_close",
    "average_dollar_volume_21d",
    "average_dollar_volume_63d",
    *CONTEXT_COLUMNS,
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "actual_future_volatility",
    "actual_future_drawdown",
    "actual_benchmark_return_10d",
    "actual_market_residual_return_10d",
    "actual_vol_adjusted_forward_return_10d",
    "actual_drawdown_adjusted_forward_return_10d",
    "actual_rank_normalized_forward_return_10d",
    "actual_top_decile_label_10d",
}
BASE_ARTIFACT_FIXED_COLUMNS = {
    "rebalance_date",
    "symbol",
    "benchmark_symbol",
    "sector",
    "average_dollar_volume_21d",
    "average_dollar_volume_63d",
    "source",
    "source_feature_id",
    "source_model_type",
    "source_split",
    "source_dataset_hash",
    "true_stock_level_row",
}
LEGACY_SPINE_BASE_COLUMNS = {
    "asset_id",
    "canonical_symbol",
    "source_provider",
    "compatibility_tier",
    "eligibility_reason",
    "selector_eligible",
    "provider_transition_flag",
    "provider_transition_id",
}
ALPHA_EXPECTED_OUTPUT_COLUMNS = frozenset(
    {
        *BASE_ARTIFACT_FIXED_COLUMNS,
        *LEGACY_SPINE_BASE_COLUMNS,
        *PREDICTION_COLUMNS,
        *ACTUAL_COLUMNS,
        *TARGET_PROVENANCE_COLUMNS,
        *CONTEXT_COLUMNS,
        *DECISION_CONTEXT_COLUMNS,
        *ENGINEERED_FEATURE_COLUMNS,
        *ENRICHMENT_METADATA_COLUMNS,
        *FEATURE_DEFINITIONS,
        *INTERMEDIATE_NUMERIC_COLUMNS,
        "model_close",
    }
)


def _build_alpha_output_schema_map(
    *,
    expected_columns: Sequence[str] = tuple(sorted(ALPHA_EXPECTED_OUTPUT_COLUMNS)),
    bool_columns: Sequence[str] = tuple(sorted(BOOL_COLUMNS)),
    int_columns: Sequence[str] = tuple(sorted(INT_COLUMNS)),
    numeric_columns: Sequence[str] = tuple(
        sorted(NUMERIC_COLUMNS - INT_COLUMNS - BOOL_COLUMNS)
    ),
    temporal_columns: Sequence[str] = tuple(sorted(TEMPORAL_COLUMNS)),
) -> dict[str, tuple[str, bool]]:
    kinds = {
        "bool": set(bool_columns),
        "int": set(int_columns),
        "float": set(numeric_columns),
        "temporal": set(temporal_columns),
    }
    conflicts: dict[str, list[str]] = {}
    for column in set().union(*kinds.values()):
        assigned = [kind for kind, columns in kinds.items() if column in columns]
        if len(assigned) > 1:
            conflicts[column] = assigned
    if conflicts:
        raise ValueError(
            "conflicting alpha output schema classifications: "
            + json.dumps(conflicts, sort_keys=True)
        )
    expected = set(expected_columns)
    classified = set().union(*kinds.values())
    unexpected = sorted(classified - expected)
    if unexpected:
        raise ValueError(
            f"alpha output schema classifies unexpected columns: {unexpected}"
        )
    schema = {
        column: (
            next(
                (
                    kind
                    for kind, columns in kinds.items()
                    if column in columns
                ),
                "string",
            ),
            column not in NON_NULLABLE_COLUMNS,
        )
        for column in sorted(expected)
    }
    return schema


ALPHA_OUTPUT_SCHEMA = _build_alpha_output_schema_map()


def _canonical_utc_timestamp(value: Any, *, column: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"temporal column {column} received naive datetime"
            )
        parsed = value.astimezone(timezone.utc)
    elif isinstance(value, calendar_date):
        if column not in TEMPORAL_DATE_COLUMNS:
            raise ValueError(
                f"timestamp column {column} received date without timezone"
            )
        return value.isoformat()
    elif isinstance(value, str):
        text = value.strip()
        if column in TEMPORAL_DATE_COLUMNS:
            if len(text) != 10:
                raise ValueError(
                    f"date column {column} requires YYYY-MM-DD"
                )
            try:
                return calendar_date.fromisoformat(text).isoformat()
            except ValueError as exc:
                raise ValueError(
                    f"date column {column} received invalid ISO date"
                ) from exc
        if len(text) == 10:
            try:
                day = calendar_date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(
                    f"timestamp column {column} received invalid ISO date"
                ) from exc
            parsed = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"timestamp column {column} received invalid ISO-8601 text"
                ) from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(
                    f"timestamp column {column} requires an explicit timezone"
                )
            parsed = parsed.astimezone(timezone.utc)
    else:
        raise ValueError(
            f"temporal column {column} received {type(value).__name__}"
        )
    timespec = "microseconds" if parsed.microsecond else "seconds"
    return parsed.isoformat(timespec=timespec).replace("+00:00", "Z")


class PartitionBuildError(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        super().__init__(str(payload.get("exception_message", "")))


_ALPHA_WORKER_CONFIG: dict[str, Any] | None = None
_ALPHA_WORKER_SPY: list[dict[str, float | str]] | None = None
_ALPHA_WORKER_INPUT_RESOLUTION: dict[str, Any] | None = None


def validate_alpha_base_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an immutable alpha parent without materialising the base table."""
    settings = StockLevelResearchConfig.from_mapping(config)
    path = settings.base_artifact_path
    if path.name.startswith(".") or ".tmp" in path.name.lower() or path.suffix.lower() != ".parquet":
        raise ValueError(f"alpha base must be a published Parquet path, not a temporary path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"alpha base artifact is missing: {path}")
    try:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        schema = parquet.schema_arrow
    except Exception as exc:
        raise ValueError(f"alpha base Parquet metadata is unreadable: {path}: {exc}") from exc
    if metadata.num_rows <= 0:
        raise ValueError("alpha base artifact has zero rows")
    missing = sorted(REQUIRED_BASE_COLUMNS - set(schema.names))
    if missing:
        raise ValueError(f"alpha base artifact is missing required columns: {missing}")

    configured_manifest = str(
        dict(config.get("ml", {}) or {}).get(
            "canonical_v2_alpha_base_manifest_path", ""
        )
    ).strip()
    sidecar_path = (
        Path(configured_manifest)
        if configured_manifest
        else path.with_name("stock_level_prediction_artifacts.json")
    )
    sidecar = _read_json(sidecar_path)
    identity = dict(sidecar.get("canonical_artifact", {}) or {})
    if not sidecar or not identity:
        raise ValueError(f"alpha base publication identity is missing: {sidecar_path}")
    if identity.get("completion_status") != "complete":
        raise ValueError("alpha base publication is not complete")
    if int(identity.get("row_count", -1)) != metadata.num_rows:
        raise ValueError("alpha base publication row count does not match Parquet metadata")
    recorded_path = Path(str(identity.get("resolved_artifact_path", "")))
    if (
        not configured_manifest
        and recorded_path
        and recorded_path.resolve() != path.resolve()
    ):
        raise ValueError("alpha base publication identity points to a different artifact")
    recorded_sha256 = str(identity.get("sha256", "")).lower()
    if len(recorded_sha256) != 64:
        raise ValueError("alpha base publication checksum is missing or malformed")
    if int(identity.get("file_size_bytes", -1)) != path.stat().st_size:
        raise ValueError("alpha base publication size does not match the artifact")
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != recorded_sha256:
        raise ValueError("alpha base publication checksum does not match the artifact")

    versions: set[str] = set()
    key_hasher = hashlib.sha256()
    observed_rows = 0
    for batch in parquet.iter_batches(
        batch_size=int(dict(config.get("ml", {}) or {}).get("canonical_v2_alpha_validation_batch_rows", 65_536)),
        columns=["rebalance_date", "symbol", "target_provenance_contract_version"],
    ):
        dates = batch.column(0).to_pylist()
        symbols = batch.column(1).to_pylist()
        provenance = batch.column(2).to_pylist()
        observed_rows += batch.num_rows
        for date, symbol, version in zip(dates, symbols, provenance):
            versions.add(str(version or ""))
            key_hasher.update(f"{str(date)[:10]}\x1f{str(symbol).upper()}\n".encode("utf-8"))
    if observed_rows != metadata.num_rows:
        raise ValueError("alpha base projected scan row count mismatch")
    if versions != {TARGET_PROVENANCE_V2}:
        raise ValueError(
            "alpha base target provenance must contain only "
            f"{TARGET_PROVENANCE_V2}; observed={sorted(versions)}"
        )
    return {
        "status": "VALID",
        "contract_version": ALPHA_BASE_CONTRACT_VERSION,
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "row_count": metadata.num_rows,
        "column_count": len(schema.names),
        "column_names": list(schema.names),
        "required_columns": sorted(REQUIRED_BASE_COLUMNS),
        "schema_fingerprint": _schema_fingerprint(schema),
        "sha256": observed_sha256,
        "logical_content_sha256": identity.get("logical_content_sha256"),
        "economic_key_sha256": key_hasher.hexdigest(),
        "target_provenance_contract_versions": sorted(versions),
        "publication_identity_path": str(sidecar_path),
        "publication_complete": True,
        "full_table_materialized": False,
        "validation_projection": [
            "rebalance_date",
            "symbol",
            "target_provenance_contract_version",
        ],
    }


def write_partitioned_canonical_v2_alpha_features(
    config: dict[str, Any],
) -> StockLevelAlphaFeaturePaths:
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(
        str(
            ml.get(
                "canonical_v2_alpha_report_root",
                DEFAULT_REPORT_ROOT / "alpha_enrichment",
            )
        )
    )
    report_root.mkdir(parents=True, exist_ok=True)
    run_id = f"alpha-only-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    manifest_path = report_root / "alpha_only_run_manifest.json"
    base = validate_alpha_base_artifact(config)
    state = {
        "contract_version": "canonical_v2_alpha_only_run.v1",
        "run_id": run_id,
        "mode": "ml-stock-level-alpha-features",
        "status": "RUNNING",
        "source_commit": _source_commit(config),
        "immutable_parent": base,
        "stages": {
            "stock_artifact": {
                "status": "completed_existing",
                "path": base["path"],
                "sha256": base["sha256"],
            },
            "alpha_features": {"status": "running"},
        },
        "stock_artifact_generation_invoked": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, state)
    try:
        result = _write_partitioned_canonical_v2_alpha_features(config)
    except BaseException as exc:
        lifecycle = _read_json(report_root / "partition_dataset_status.json")
        state["status"] = "FAILED"
        state["partition_generation_status"] = (
            "COMPLETE"
            if lifecycle.get("partition_processing_status") == "complete"
            else "FAILED"
        )
        state["consolidation_status"] = lifecycle.get(
            "consolidation_status", "FAILED"
        ).upper()
        state["publication_status"] = "NOT_PUBLISHED"
        state["overall_status"] = "FAILED"
        state["stages"]["alpha_features"] = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
        state["ended_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, state)
        raise
    state["status"] = "COMPLETE"
    lifecycle = _read_json(report_root / "partition_dataset_status.json")
    state["partition_generation_status"] = (
        "COMPLETE_REUSED"
        if int(lifecycle.get("partitions_recomputed", 0) or 0) == 0
        else "COMPLETE"
    )
    state["consolidation_status"] = "COMPLETE"
    state["publication_status"] = "COMPLETE"
    state["overall_status"] = "COMPLETE"
    state["reused_namespace_identity"] = lifecycle.get(
        "partition_namespace_identity"
    )
    state["reused_partition_count"] = int(
        lifecycle.get("partitions_reused", 0) or 0
    )
    state["workers_submitted"] = int(
        lifecycle.get("workers_submitted", 0) or 0
    )
    state["final_artifact"] = lifecycle.get("final_artifact")
    state["stages"]["alpha_features"] = {
        "status": "completed",
        "output_paths": _path_payload_for_alpha(result),
    }
    state["ended_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, state)
    return result


def _path_payload_for_alpha(paths: StockLevelAlphaFeaturePaths) -> dict[str, str]:
    return {
        name: str(value)
        for name, value in vars(paths).items()
        if isinstance(value, Path)
    }


def _write_partitioned_canonical_v2_alpha_features(config: dict[str, Any]) -> StockLevelAlphaFeaturePaths:
    config = {**config, "ml": dict(config.get("ml", {}) or {})}
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_stock_alpha_worker_caps(config)
    ml = config["ml"]
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    report_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _write_json(
        report_root / "preflight_status.json",
        {
            "status": "RUNNING",
            "source_commit": _source_commit(config),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_json(
        report_root / "failed_partitions.json",
        {"failed_partition_count": 0, "failed_partitions": []},
    )
    _progress(report_root, 0, 0, 0, 0, started)
    base_validation = validate_alpha_base_artifact(config)
    schema_coverage = _validate_alpha_output_schema_coverage(
        base_validation["column_names"]
    )
    _write_json(
        report_root / "preflight_status.json",
        {
            "status": "COMPLETE",
            "source_commit": _source_commit(config),
            "schema_coverage": schema_coverage,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    config["ml"]["canonical_v2_alpha_validated_base_sha256"] = base_validation["sha256"]
    config["ml"]["canonical_v2_alpha_validated_base_key_sha256"] = base_validation["economic_key_sha256"]
    input_resolution = resolve_inputs(config)
    input_resolution["validated_alpha_base"] = base_validation
    _write_json(report_root / "input_resolution.json", input_resolution)
    if not input_resolution["gates_passed"]:
        raise ValueError(f"canonical-v2 alpha input gates failed: {input_resolution['blocking_issues']}")
    base_partition_manifest = prepare_alpha_base_partitions(
        config,
        base_validation=base_validation,
    )
    config["ml"]["canonical_v2_alpha_base_partition_root"] = base_partition_manifest["path"]

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_namespace, namespace_identity = _resolve_alpha_partition_namespace(
        report_root,
        config,
        base_validation=base_validation,
    )
    partition_root = partition_namespace / "partitions"
    manifest_root = partition_namespace / "partition_manifests"
    base_partition_root = Path(str(base_partition_manifest["path"]))
    partition_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    base_path = settings.base_artifact_path
    available_symbols = {
        str(row["symbol"]).upper()
        for row in base_partition_manifest["partitions"]
    }

    configured_symbols = ml.get("canonical_v2_alpha_symbols")
    if configured_symbols:
        requested = {str(symbol).upper() for symbol in configured_symbols}
        symbols = [symbol for symbol in sorted(requested) if symbol in available_symbols]
        missing = sorted(requested - set(symbols))
        if missing:
            raise ValueError(f"missing certified base partition for requested symbols: {missing[:10]}")
    else:
        symbols = sorted(available_symbols)
    completed_before = _completed_compatible_symbols(manifest_root, config)
    pending = [symbol for symbol in symbols if symbol not in completed_before]
    workers = settings.alpha_feature_n_jobs
    plan = {
        "partition_unit": "symbol",
        "planned_partitions": len(symbols),
        "completed_before": len(completed_before),
        "pending_partitions": len(pending),
        "requested_workers": workers,
        "effective_workers": (
            min(max(1, workers), len(pending)) if pending else 0
        ),
        "workers_submitted": len(pending),
        "partition_root": str(partition_root),
        "manifest_root": str(manifest_root),
        "base_artifact_path": str(base_path),
        "base_artifact_sha256": base_validation["sha256"],
        "base_economic_key_sha256": base_validation["economic_key_sha256"],
        "base_partition_root": str(base_partition_root),
        "partition_namespace": namespace_identity,
        "output_schema_coverage": schema_coverage,
        "source_mode": "certified_published_base_partition",
        "canonical_price_root": str(settings.parquet_dir),
        "monolithic_base_read": False,
        "resume_enabled": True,
        "retry_only_failed_command": "python .\\main.py --mode ml-stock-level-alpha-features --config .\\config\\config.ticket_6d_alpha_only_resume.yaml",
    }
    _write_json(report_root / "partition_plan.json", plan)
    _write_json(
        report_root / "failed_partitions.json",
        {"failed_partition_count": 0, "failed_partitions": []},
    )
    _progress(
        report_root, len(symbols), len(completed_before), 0, 0, started
    )
    failed: list[dict[str, Any]] = []
    rows_processed = 0
    aborted_early = False
    abort_reason = ""
    dominant_signature = ""
    tasks_cancelled = 0
    abort_threshold_failure_count = 0
    worker_failure_record_count = 0
    in_flight_at_abort = 0
    fail_fast = _fail_fast_settings(ml)
    if pending:
        effective_workers = min(max(1, workers), len(pending))
        if effective_workers == 1:
            prepared_spy = _prepare_history(_load_price_histories(settings.parquet_dir, [settings.spy_symbol]).get(settings.spy_symbol, []))
            for symbol in pending:
                try:
                    result = _build_partition(
                        symbol,
                        config,
                        prepared_spy,
                        partition_root,
                        manifest_root,
                        input_resolution=input_resolution,
                    )
                    rows_processed += int(result["row_count"])
                except Exception as exc:
                    failed.append(_failure_record(symbol, exc))
                    _write_json(report_root / "failed_partitions.json", {"failed_partition_count": len(failed), "failed_partitions": failed})
                    _progress(report_root, len(symbols), len(_completed_compatible_symbols(manifest_root, config)), len(failed), rows_processed, started)
                    raise
                _progress(report_root, len(symbols), len(_completed_compatible_symbols(manifest_root, config)), len(failed), rows_processed, started)
        else:
            execution = _execute_alpha_process_pool(
                pending,
                config=config,
                input_resolution=input_resolution,
                partition_root=partition_root,
                manifest_root=manifest_root,
                report_root=report_root,
                workers=effective_workers,
                fail_fast=fail_fast,
                planned_partitions=len(symbols),
                started=started,
            )
            failed.extend(execution["failures"])
            rows_processed += int(execution["rows_processed"])
            aborted_early = bool(execution["aborted_early"])
            abort_reason = str(execution["abort_reason"])
            dominant_signature = str(execution["dominant_signature"])
            tasks_cancelled += int(execution["tasks_cancelled"])
            abort_threshold_failure_count = int(execution["abort_threshold_failure_count"])
            worker_failure_record_count = int(execution["worker_failure_record_count"])
            in_flight_at_abort = int(execution["in_flight_at_abort"])
    _write_json(
        report_root / "failed_partitions.json",
        {
            "failed_partition_count": len(failed),
            "failed_partitions": failed,
            "abort_threshold_failure_count": abort_threshold_failure_count,
            "worker_failure_record_count": worker_failure_record_count,
            "in_flight_at_abort": in_flight_at_abort,
            "failure_count_explanation": (
                "failed_partition_count contains failures observed by the parent "
                "before fail-fast; worker_failure_record_count also includes "
                "independently persisted failures from tasks already in flight"
            ),
        },
    )
    if aborted_early:
        raise RuntimeError(f"canonical-v2 alpha aborted early: {abort_reason}; dominant_signature={dominant_signature}")
    if failed:
        raise RuntimeError(f"canonical-v2 alpha partitions failed: {failed[:5]}")

    partition_paths = _completed_partition_paths(manifest_root, expected_symbols=symbols, config=config)
    partition_validation = _validate_partition_dataset(partition_paths, report_root=report_root)
    reused_partition_count = len(completed_before)
    workers_submitted = len(pending)
    _write_json(
        report_root / "partition_dataset_status.json",
        {
            "partition_processing_status": "complete",
            "partition_validation_status": "complete",
            "consolidation_status": "pending",
            "publication_status": "not_published",
            "overall_status": "running",
            "partitions_reused": reused_partition_count,
            "partitions_recomputed": len(pending),
            "workers_submitted": workers_submitted,
            "partition_namespace_identity": namespace_identity,
            "consolidation_retried": True,
            **partition_validation,
        },
    )
    paths = StockLevelAlphaFeaturePaths(
        enriched_parquet_path=canonical_artifact_path(output_dir, "stock_level_prediction_artifacts_enriched", config),
        audit_csv_path=output_dir / "stock_level_alpha_feature_audit.csv",
        audit_json_path=output_dir / "stock_level_alpha_feature_audit.json",
        audit_markdown_path=output_dir / "stock_level_alpha_feature_audit.md",
        enriched_sample_csv_path=output_dir / "stock_level_prediction_artifacts_enriched_sample.csv",
    )
    try:
        identity = _consolidate_partition_parquets(
            partition_paths,
            paths.enriched_parquet_path,
            config=config,
            sample_path=paths.enriched_sample_csv_path,
            expected_row_count=int(partition_validation["row_count"]),
            report_root=report_root,
            preflight_validation=partition_validation,
        )
    except Exception as exc:
        _write_json(
            report_root / "partition_dataset_status.json",
            {
                "partition_processing_status": "complete",
                "partition_validation_status": "complete",
                "consolidation_status": "failed",
                "publication_status": "not_published",
                "overall_status": "failed",
                "consolidation_error": f"{type(exc).__name__}: {exc}",
                "partitions_reused": reused_partition_count,
                "partitions_recomputed": len(pending),
                "workers_submitted": workers_submitted,
                "partition_namespace_identity": namespace_identity,
                "consolidation_retried": True,
                **partition_validation,
            },
        )
        raise
    audit = {
        "mode": "stock_level_alpha_features_research_only",
        "source_path": str(base_path),
        "source_base_sha256": base_validation["sha256"],
        "source_base_economic_key_sha256": base_validation["economic_key_sha256"],
        "row_count": identity["row_count"],
        "features": identity.pop("feature_coverage"),
        "parallelism": {
            "requested_workers": workers,
            "effective_workers": plan["effective_workers"],
            "partition": "symbol",
            "symbol_count": len(symbols),
            "partitioned_resume": True,
            "worker_local_initialization": True,
            "bounded_in_flight_tasks": True,
        },
        "bounded_memory": {
            "full_base_materialization": False,
            "full_enriched_materialization": False,
            "maximum_cross_section_rows": len(symbols),
        },
    }
    audit.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=base_path))
    audit["canonical_v2_input_resolution"] = input_resolution
    audit["canonical_artifact"] = identity
    audit["artifact_format"] = identity["artifact_format"]
    audit["artifact_path"] = identity["resolved_artifact_path"]
    audit["artifact_sha256"] = identity["sha256"]
    audit["logical_content_sha256"] = identity["logical_content_sha256"]
    audit["schema_fingerprint"] = identity["schema_fingerprint"]
    audit["target_contract_version"] = identity.get("target_contract_version")
    audit["benchmark_contract_version"] = "stock_level_benchmark_return_10d_v1"
    _write_audit_csv(paths.audit_csv_path, audit["features"])
    writer = ResearchArtifactWriter()
    writer.write_json(paths.audit_json_path, audit)
    writer.write_markdown(paths.audit_markdown_path, _markdown(audit))
    feature_coverage = audit["features"]
    _write_csv(report_root / "feature_coverage.csv", feature_coverage, ["feature", "definition", "populated_count", "missing_count", "availability_rate"])
    validation = validate_enriched_artifact(paths.enriched_parquet_path, input_resolution=input_resolution)
    validation.update(
        {
            "completed_partitions": len(_completed_compatible_symbols(manifest_root, config)),
            "failed_partitions": len(failed),
            "planned_partitions": len(symbols),
            "worker_count": workers,
            "artifact_hash": identity["sha256"],
            "feature_count": len(ENGINEERED_FEATURE_COLUMNS),
            "populated_feature_count": sum(1 for row in feature_coverage if int(row.get("populated_count", 0) or 0) > 0),
        }
    )
    _write_json(
        report_root / "partition_dataset_status.json",
        {
            "partition_processing_status": "complete",
            "partition_validation_status": "complete",
            "consolidation_status": "complete",
            "publication_status": "complete",
            "overall_status": "complete",
            "partitions_reused": reused_partition_count,
            "partitions_recomputed": len(pending),
            "workers_submitted": workers_submitted,
            "partition_namespace_identity": namespace_identity,
            "consolidation_retried": True,
            "final_artifact": identity,
            **partition_validation,
        },
    )
    _write_json(report_root / "final_validation.json", validation)
    _progress(report_root, len(symbols), len(_completed_compatible_symbols(manifest_root, config)), len(failed), validation["row_count"], started)
    return paths


def resolve_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    canonical_manifest_path = Path(str(ml.get("canonical_daily_v2_manifest_path", "reports/data_lineage/canonical_daily_v2/build_manifest.json")))
    canonical_root = Path(str(ml.get("canonical_daily_v2_root", "data/processed/market_data/canonical_daily_v2/full")))
    labeled_manifest_path = Path(str(ml.get("canonical_v2_labeled_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/labeled_spine_manifest.json")))
    inference_manifest_path = Path(str(ml.get("canonical_v2_inference_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/inference_spine_manifest.json")))
    recovered = Path("reports/ml/development/ticket_7b3_daily_large_history/regeneration/benchmark/stock_level_prediction_artifacts.parquet")
    settings = StockLevelResearchConfig.from_mapping(config)
    blocking: list[str] = []
    canonical_manifest = _read_json(canonical_manifest_path)
    labeled_manifest = _read_json(labeled_manifest_path)
    inference_manifest = _read_json(inference_manifest_path)
    canonical_hash = canonical_manifest.get("dataset_logical_partition_hash")
    if canonical_manifest.get("status") != "COMPLETE":
        blocking.append("canonical_manifest_not_complete")
    if int(canonical_manifest.get("completed_partitions", 0) or 0) != 514:
        blocking.append("canonical_completed_partitions_not_514")
    if canonical_hash != EXPECTED_CANONICAL_HASH:
        blocking.append("canonical_hash_mismatch")
    validation_path = canonical_manifest_path.with_name("validation.json")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True:
        blocking.append("canonical_validation_not_valid")
    if labeled_manifest.get("status") != "BUILT":
        blocking.append("labeled_spine_not_built")
    if inference_manifest.get("status") != "BUILT":
        blocking.append("inference_spine_not_built")
    recovered_hash = _file_sha256(recovered) if recovered.exists() else None
    if recovered_hash != EXPECTED_BASE_HASH:
        blocking.append("recovered_artifact_hash_mismatch")
    if str(ml.get("stock_selector_market_data_source", "")).lower() != "canonical_daily_v2":
        blocking.append("selector_source_not_canonical_v2")
    if Path(str(ml.get("stooq_parquet_dir", ""))) != canonical_root:
        blocking.append("alpha_price_root_not_canonical_v2")
    if "expanded_rebalance_dataset" in str(settings.base_artifact_path).lower():
        blocking.append("base_artifact_points_to_expanded_rebalance_cache")
    return {
        "gates_passed": not blocking,
        "blocking_issues": blocking,
        "canonical_dataset": {
            "path": str(canonical_root),
            "manifest_path": str(canonical_manifest_path),
            "hash": canonical_hash,
            "row_count": canonical_manifest.get("row_count"),
            "symbol_count": canonical_manifest.get("symbol_count"),
            "date_min": canonical_manifest.get("date_min"),
            "date_max": canonical_manifest.get("date_max"),
            "completed_partitions": canonical_manifest.get("completed_partitions"),
            "validation_path": str(validation_path),
            "validation_valid": validation.get("valid"),
        },
        "labeled_spine": _manifest_summary(labeled_manifest, labeled_manifest_path),
        "inference_spine": _manifest_summary(inference_manifest, inference_manifest_path),
        "base_artifact": {
            "path": str(settings.base_artifact_path),
            "exists": settings.base_artifact_path.exists(),
            "recovered_reference_path": str(recovered),
            "recovered_reference_hash": recovered_hash,
        },
        "worker_configuration": {
            "stock_alpha_feature_n_jobs": settings.alpha_feature_n_jobs,
            "stock_level_dataset_workers": settings.dataset_workers,
        },
        "output_paths": {
            "output_dir": str(settings.output_dir),
            "enriched_artifact": str(canonical_artifact_path(settings.output_dir, "stock_level_prediction_artifacts_enriched", config)),
        },
        "resume_manifest_paths": {
            "partition_plan": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "partition_plan.json"),
            "progress_manifest": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "progress_manifest.json"),
            "failed_partitions": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "failed_partitions.json"),
        },
        "stooq_fallback_used": False,
    }


def prepare_alpha_base_partitions(
    config: Mapping[str, Any],
    *,
    base_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream the immutable published base into a checksum-owned symbol dataset."""
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    namespace_identity = _alpha_base_namespace_identity(base_validation)
    checksum = namespace_identity["source_base_sha256"]
    configured_root = ml.get("canonical_v2_alpha_base_partition_root")
    cache_parent = report_root / "alpha_base_partitions_v2"
    if configured_root:
        target_root = Path(str(configured_root))
        return _validate_alpha_base_partition_cache(
            target_root, base_validation=base_validation
        )
    legacy_root = cache_parent / checksum
    if legacy_root.is_dir():
        try:
            return _validate_alpha_base_partition_cache(
                legacy_root,
                base_validation=base_validation,
                allow_legacy_namespace=True,
            )
        except ValueError:
            pass
    elif legacy_root.exists():
        raise ValueError(
            f"legacy alpha base cache path is not a directory: {legacy_root}"
        )
    target_root = cache_parent / f"id-{namespace_identity['namespace_key']}"
    manifest_path = target_root / "base_partition_manifest.json"
    if target_root.exists():
        return _validate_alpha_base_partition_cache(
            target_root, base_validation=base_validation
        )

    attempt_root = _alpha_base_attempt_root(target_root, checksum)
    attempt_root.mkdir(parents=True, exist_ok=False)
    try:
        parquet = pq.ParquetFile(settings.base_artifact_path)
        compression = str(ml.get("stock_level_parquet_compression", "zstd")).lower()
        writers: dict[str, pq.ParquetWriter] = {}
        row_counts: dict[str, int] = {}
        paths: dict[str, Path] = {}
        try:
            for batch in parquet.iter_batches(
                batch_size=int(ml.get("canonical_v2_alpha_base_partition_batch_rows", 32_768))
            ):
                table = pa.Table.from_batches([batch])
                for scalar in pc.unique(table["symbol"]):
                    symbol = str(scalar.as_py() or "").upper()
                    if not symbol:
                        raise ValueError("alpha base contains a blank symbol")
                    selected = table.filter(pc.equal(table["symbol"], pa.scalar(scalar.as_py())))
                    path = attempt_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
                    if symbol not in writers:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writers[symbol] = pq.ParquetWriter(
                            path, parquet.schema_arrow, compression=compression
                        )
                        paths[symbol] = path
                        row_counts[symbol] = 0
                    writers[symbol].write_table(selected)
                    row_counts[symbol] += selected.num_rows
        finally:
            for writer in writers.values():
                writer.close()
        observed_rows = sum(row_counts.values())
        if observed_rows != int(base_validation["row_count"]):
            raise ValueError(
                f"alpha base partition row count {observed_rows} does not match "
                f"validated base row count {base_validation['row_count']}"
            )
        partitions: list[dict[str, Any]] = []
        for symbol in sorted(paths):
            path = paths[symbol]
            published_path = target_root / path.relative_to(attempt_root)
            identity = {
                "contract_version": ALPHA_BASE_CONTRACT_VERSION,
                "status": "COMPLETE",
                "symbol": symbol,
                "path": str(published_path),
                "row_count": row_counts[symbol],
                "sha256": _file_sha256(path),
                "source_base_sha256": checksum,
                "source_base_logical_content_sha256": namespace_identity[
                    "source_base_logical_content_sha256"
                ],
                "source_base_schema_fingerprint": base_validation["schema_fingerprint"],
                "source_base_economic_key_sha256": base_validation["economic_key_sha256"],
                "base_namespace_identity": namespace_identity,
            }
            _write_json(_base_partition_identity_path(path), identity)
            partitions.append(identity)
        payload = {
            "contract_version": ALPHA_BASE_CONTRACT_VERSION,
            "status": "COMPLETE",
            "path": str(target_root),
            "source_base_path": str(settings.base_artifact_path),
            "source_base_sha256": checksum,
            "source_base_logical_content_sha256": namespace_identity[
                "source_base_logical_content_sha256"
            ],
            "source_base_schema_fingerprint": base_validation["schema_fingerprint"],
            "source_base_economic_key_sha256": base_validation["economic_key_sha256"],
            "source_base_column_count": int(base_validation["column_count"]),
            "base_namespace_identity": namespace_identity,
            "row_count": observed_rows,
            "symbol_count": len(partitions),
            "partition_count": len(partitions),
            "partitions": partitions,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "streaming_source_read": True,
            "full_table_materialized": False,
        }
        _write_json(attempt_root / manifest_path.name, payload)
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists():
            raise FileExistsError(
                f"incompatible alpha base partition root already exists: {target_root}"
            )
        os.replace(attempt_root, target_root)
        return _validate_alpha_base_partition_cache(
            target_root, base_validation=base_validation
        )
    except BaseException:
        if attempt_root.exists():
            shutil.rmtree(attempt_root)
        raise


def _alpha_base_attempt_root(target_root: Path, checksum: str) -> Path:
    """Return a bounded sibling name without duplicating the full checksum."""
    return target_root.with_name(
        f".attempt-{checksum[:8]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )


def _alpha_base_namespace_identity(
    base_validation: Mapping[str, Any],
) -> dict[str, Any]:
    authoritative = {
        "source_base_sha256": str(base_validation["sha256"]),
        "source_base_logical_content_sha256": str(
            base_validation.get("logical_content_sha256") or ""
        ),
        "source_base_schema_fingerprint": str(
            base_validation["schema_fingerprint"]
        ),
        "source_base_economic_key_sha256": str(
            base_validation["economic_key_sha256"]
        ),
        "source_base_row_count": int(base_validation["row_count"]),
        "source_base_column_count": int(base_validation["column_count"]),
        "target_provenance_contract_versions": list(
            base_validation.get("target_provenance_contract_versions", [])
        ),
        "partition_contract_version": ALPHA_BASE_CONTRACT_VERSION,
    }
    key = hashlib.sha256(
        json.dumps(
            authoritative, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:ALPHA_BASE_NAMESPACE_KEY_HEX_LENGTH]
    return {
        "contract_version": ALPHA_BASE_NAMESPACE_CONTRACT_VERSION,
        "status": "COMPLETE",
        "namespace_key": key,
        **authoritative,
    }


def _alpha_base_identity_mismatches(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "field": field,
            "expected": expected.get(field),
            "observed": observed.get(field),
        }
        for field in expected
        if observed.get(field) != expected.get(field)
    ]


def _validate_alpha_base_partition_cache(
    root: Path,
    *,
    base_validation: Mapping[str, Any],
    allow_legacy_namespace: bool = False,
) -> dict[str, Any]:
    manifest_path = root / "base_partition_manifest.json"
    manifest = _read_json(manifest_path)
    expected_namespace = _alpha_base_namespace_identity(base_validation)
    observed_namespace = dict(manifest.get("base_namespace_identity") or {})
    if allow_legacy_namespace and not observed_namespace:
        legacy_expected = {
            "status": "COMPLETE",
            "source_base_sha256": expected_namespace["source_base_sha256"],
            "source_base_schema_fingerprint": expected_namespace[
                "source_base_schema_fingerprint"
            ],
            "source_base_economic_key_sha256": expected_namespace[
                "source_base_economic_key_sha256"
            ],
            "row_count": expected_namespace["source_base_row_count"],
        }
        mismatches = _alpha_base_identity_mismatches(manifest, legacy_expected)
    else:
        mismatches = _alpha_base_identity_mismatches(
            observed_namespace, expected_namespace
        )
    if mismatches:
        raise ValueError(
            "alpha base cache namespace identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    partitions = list(manifest.get("partitions") or [])
    if (
        manifest.get("status") != "COMPLETE"
        or int(manifest.get("partition_count", -1)) != len(partitions)
        or not partitions
    ):
        raise ValueError(
            f"alpha base cache manifest is missing or incomplete: {manifest_path}"
        )
    for row in partitions:
        symbol = str(row.get("symbol") or "")
        path = Path(str(row.get("path") or ""))
        identity = _read_json(_base_partition_identity_path(path))
        expected_partition = {
            "contract_version": ALPHA_BASE_CONTRACT_VERSION,
            "status": "COMPLETE",
            "symbol": symbol,
            "path": str(path),
            "row_count": int(row.get("row_count", -1)),
            "sha256": str(row.get("sha256") or ""),
            "source_base_sha256": expected_namespace["source_base_sha256"],
            "source_base_schema_fingerprint": expected_namespace[
                "source_base_schema_fingerprint"
            ],
            "source_base_economic_key_sha256": expected_namespace[
                "source_base_economic_key_sha256"
            ],
        }
        if not allow_legacy_namespace:
            expected_partition.update(
                {
                    "source_base_logical_content_sha256": expected_namespace[
                        "source_base_logical_content_sha256"
                    ],
                    "base_namespace_identity": expected_namespace,
                }
            )
        partition_mismatches = _alpha_base_identity_mismatches(
            identity, expected_partition
        )
        if not path.is_file():
            partition_mismatches.append(
                {"field": "path_exists", "expected": True, "observed": False}
            )
        elif _file_sha256(path) != expected_partition["sha256"]:
            partition_mismatches.append(
                {
                    "field": "sha256_observed",
                    "expected": expected_partition["sha256"],
                    "observed": _file_sha256(path),
                }
            )
        if partition_mismatches:
            raise ValueError(
                f"alpha base cache partition identity mismatch for {symbol}: "
                + json.dumps(partition_mismatches, sort_keys=True)
            )
    return manifest


def _alpha_partition_namespace_identity(
    config: Mapping[str, Any],
    *,
    base_validation: Mapping[str, Any],
) -> dict[str, Any]:
    base_sha256 = str(base_validation["sha256"])
    feature_schema_sha256 = _feature_schema_identity()
    configuration_sha256 = _alpha_configuration_identity(config)
    authoritative = {
        "base_artifact_sha256": base_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "configuration_sha256": configuration_sha256,
    }
    namespace_key = hashlib.sha256(
        json.dumps(
            authoritative, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:ALPHA_PARTITION_NAMESPACE_KEY_HEX_LENGTH]
    return {
        "contract_version": ALPHA_PARTITION_NAMESPACE_CONTRACT_VERSION,
        "status": "COMPLETE",
        "namespace_key": namespace_key,
        **authoritative,
    }


def _resolve_alpha_partition_namespace(
    report_root: Path,
    config: Mapping[str, Any],
    *,
    base_validation: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    identity = _alpha_partition_namespace_identity(
        config, base_validation=base_validation
    )
    namespace_parent = report_root / "alpha_partitions_v2"
    legacy_root = (
        namespace_parent
        / identity["base_artifact_sha256"]
        / identity["feature_schema_sha256"]
    )
    if legacy_root.is_dir():
        return legacy_root, {
            **identity,
            "layout": "legacy_full_hash_v2",
            "path": str(legacy_root),
        }
    if legacy_root.exists():
        raise ValueError(
            f"legacy alpha partition namespace is not a directory: {legacy_root}"
        )

    namespace_root = namespace_parent / f"id-{identity['namespace_key']}"
    manifest_path = namespace_root / "namespace_manifest.json"
    if namespace_root.exists():
        observed = _read_json(manifest_path)
        expected = {**identity, "layout": "bounded_v1", "path": str(namespace_root)}
        if observed != expected:
            raise ValueError(
                "alpha partition namespace identity mismatch for bounded key "
                f"{identity['namespace_key']}: {manifest_path}"
            )
        return namespace_root, observed

    attempt_root = namespace_parent / (
        f".attempt-{identity['namespace_key']}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    expected = {**identity, "layout": "bounded_v1", "path": str(namespace_root)}
    try:
        attempt_root.mkdir(parents=True, exist_ok=False)
        _write_json(attempt_root / manifest_path.name, expected)
        os.replace(attempt_root, namespace_root)
    except BaseException:
        if attempt_root.exists():
            shutil.rmtree(attempt_root)
        if namespace_root.is_dir():
            observed = _read_json(manifest_path)
            if observed == expected:
                return namespace_root, observed
        raise
    return namespace_root, expected


def _small_alpha_worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    allowed = {
        "output_dir",
        "stooq_parquet_dir",
        "stock_ranker_spy_symbol",
        "stock_ranker_market_symbol",
        "stock_level_base_prediction_artifacts_path",
        "stock_level_artifact_format",
        "stock_level_parquet_compression",
        "stock_alpha_feature_n_jobs",
        "canonical_v2_alpha_report_root",
        "canonical_v2_alpha_base_partition_root",
        "canonical_v2_alpha_validated_base_sha256",
        "canonical_v2_alpha_validated_base_key_sha256",
        "canonical_daily_v2_root",
        "canonical_daily_v2_manifest_path",
        "canonical_v2_labeled_spine_root",
        "canonical_v2_labeled_spine_manifest_path",
        "canonical_v2_inference_spine_manifest_path",
        "sector_reference_path",
        "sector_by_symbol",
        "feature_lookback_days",
    }
    payload = {"ml": {key: value for key, value in ml.items() if key in allowed}}
    encoded = pickle.dumps(payload)
    if len(encoded) > 131_072:
        raise ValueError(f"alpha worker configuration payload is too large: {len(encoded)} bytes")
    return payload


def _alpha_worker_initialize(
    worker_config: dict[str, Any],
    input_resolution: dict[str, Any],
    startup_root: str,
) -> None:
    global _ALPHA_WORKER_CONFIG, _ALPHA_WORKER_SPY, _ALPHA_WORKER_INPUT_RESOLUTION
    pid = os.getpid()
    path = Path(startup_root) / f"{pid}.json"
    try:
        settings = StockLevelResearchConfig.from_mapping(worker_config)
        _ALPHA_WORKER_CONFIG = worker_config
        _ALPHA_WORKER_INPUT_RESOLUTION = input_resolution
        _ALPHA_WORKER_SPY = _prepare_history(
            _load_price_histories(settings.parquet_dir, [settings.spy_symbol]).get(
                settings.spy_symbol, []
            )
        )
        _write_json(
            path,
            {
                "status": "STARTED",
                "worker_pid": pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "prepared_spy_rows": len(_ALPHA_WORKER_SPY),
            },
        )
    except BaseException as exc:
        _write_json(
            path,
            {
                "status": "FAILED",
                "worker_pid": pid,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


def _alpha_worker_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _ALPHA_WORKER_CONFIG is None or _ALPHA_WORKER_SPY is None or _ALPHA_WORKER_INPUT_RESOLUTION is None:
        raise RuntimeError("alpha worker was not initialized")
    symbol = str(task["symbol"])
    event_root = Path(str(task["event_root"]))
    _write_json(
        event_root / f"{_safe_symbol(symbol)}.json",
        {
            "status": "STARTED",
            "symbol": symbol,
            "worker_pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        result = _build_partition(
            symbol,
            _ALPHA_WORKER_CONFIG,
            _ALPHA_WORKER_SPY,
            Path(str(task["partition_root"])),
            Path(str(task["manifest_root"])),
            input_resolution=_ALPHA_WORKER_INPUT_RESOLUTION,
        )
    except Exception as exc:
        failure = _failure_record(symbol, exc)
        failure["worker_pid"] = os.getpid()
        _write_json(
            event_root / f"{_safe_symbol(symbol)}.json",
            {
                "status": "FAILED",
                "symbol": symbol,
                "worker_pid": os.getpid(),
                "failure": failure,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {"task_status": "FAILED", "failure": failure}
    result["worker_pid"] = os.getpid()
    result["task_status"] = "COMPLETE"
    _write_json(
        event_root / f"{_safe_symbol(symbol)}.json",
        {
            "status": "COMPLETED",
            "symbol": symbol,
            "worker_pid": os.getpid(),
            "row_count": result["row_count"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


def _execute_alpha_process_pool(
    symbols: Sequence[str],
    *,
    config: Mapping[str, Any],
    input_resolution: Mapping[str, Any],
    partition_root: Path,
    manifest_root: Path,
    report_root: Path,
    workers: int,
    fail_fast: Mapping[str, Any],
    planned_partitions: int,
    started: float,
    executor_cls: type = ProcessPoolExecutor,
) -> dict[str, Any]:
    startup_root = report_root / "worker_startup"
    event_root = report_root / "worker_tasks"
    startup_root.mkdir(parents=True, exist_ok=True)
    event_root.mkdir(parents=True, exist_ok=True)
    worker_config = _small_alpha_worker_config(config)
    source_commit = _source_commit(config)
    context = multiprocessing.get_context("spawn")
    maximum_in_flight = max(workers, workers * 2)
    queue = iter(symbols)
    futures: dict[Any, str] = {}
    failures: list[dict[str, Any]] = []
    rows_processed = 0
    submitted = 0
    started_count = 0
    completed = 0
    cancelled = 0
    aborted = False
    abort_reason = ""
    dominant = ""
    abort_threshold_failure_count = 0
    in_flight_at_abort = 0
    worker_failure_record_count = 0
    worker_exit_information: list[dict[str, Any]] = []
    executor_phase = "creation"
    executor = executor_cls(
        max_workers=workers,
        mp_context=context,
        initializer=_alpha_worker_initialize,
        initargs=(worker_config, dict(input_resolution), str(startup_root)),
    )
    try:
        executor_phase = "submission"

        def submit_one(symbol: str) -> None:
            nonlocal submitted
            task = {
                "symbol": symbol,
                "partition_root": str(partition_root),
                "manifest_root": str(manifest_root),
                "event_root": str(event_root),
            }
            if len(pickle.dumps(task)) > 16_384:
                raise ValueError(f"alpha task payload is too large for {symbol}")
            futures[executor.submit(_alpha_worker_task, task)] = symbol
            submitted += 1

        for _ in range(min(maximum_in_flight, len(symbols))):
            submit_one(next(queue))
        executor_phase = "collection"
        while futures and not aborted:
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                symbol = futures.pop(future)
                event = _read_json(event_root / f"{_safe_symbol(symbol)}.json")
                if event.get("status") in {"STARTED", "COMPLETED"}:
                    started_count += 1
                if future.cancelled():
                    cancelled += 1
                    continue
                try:
                    result = future.result()
                    if result.get("task_status") == "FAILED":
                        failure = dict(result["failure"])
                        failure.update(
                            {
                                "task_identity": f"alpha-symbol:{symbol}",
                                "partition_identity": symbol,
                                "worker_pid": result["failure"].get("worker_pid"),
                                "executor_phase": executor_phase,
                                "worker_count": workers,
                                "multiprocessing_start_method": context.get_start_method(),
                                "source_commit": source_commit,
                            }
                        )
                        failures.append(failure)
                        should_abort, abort_reason, dominant = _should_abort_fail_fast(
                            failures,
                            completed=completed,
                            settings=fail_fast,
                        )
                        aborted = should_abort
                    else:
                        completed += 1
                        rows_processed += int(result["row_count"])
                except Exception as exc:
                    failure = _failure_record(symbol, exc)
                    failure.update(
                        {
                            "task_identity": f"alpha-symbol:{symbol}",
                            "partition_identity": symbol,
                            "worker_pid": event.get("worker_pid"),
                            "executor_phase": executor_phase,
                            "worker_count": workers,
                            "multiprocessing_start_method": context.get_start_method(),
                            "source_commit": source_commit,
                            "parent_memory": _parent_memory_snapshot(),
                            "worker_startup_diagnostics": (
                                _worker_startup_diagnostics(startup_root)
                            ),
                        }
                    )
                    failures.append(failure)
                    should_abort, abort_reason, dominant = _should_abort_fail_fast(
                        failures,
                        completed=completed,
                        settings=fail_fast,
                    )
                    if isinstance(exc, BrokenProcessPool):
                        should_abort = len(failures) >= min(3, len(symbols))
                        abort_reason = (
                            f"{completed} completed partitions and {len(failures)} "
                            "broken-pool failures"
                        )
                        dominant = failure["failure_signature"]
                    aborted = should_abort
                if aborted and not abort_threshold_failure_count:
                    abort_threshold_failure_count = len(failures)
                    in_flight_at_abort = len(futures)
                _write_json(
                    report_root / "failed_partitions.json",
                    {
                        "failed_partition_count": len(failures),
                        "failed_partitions": failures,
                    },
                )
                _progress(
                    report_root,
                    planned_partitions,
                    completed,
                    len(failures),
                    rows_processed,
                    started,
                    aborted_early=aborted,
                    abort_reason=abort_reason,
                    dominant_failure_signature=dominant,
                    tasks_cancelled=cancelled,
                )
                if aborted:
                    break
                if not aborted:
                    try:
                        submit_one(next(queue))
                    except StopIteration:
                        pass
            _write_json(
                report_root / "executor_lifecycle.json",
                {
                    "status": "ABORTING" if aborted else "RUNNING",
                    "worker_count": workers,
                    "multiprocessing_start_method": context.get_start_method(),
                    "maximum_in_flight": maximum_in_flight,
                    "submitted": submitted,
                    "started": started_count,
                    "completed": completed,
                    "failed": len(failures),
                    "cancelled": cancelled,
                    "executor_phase": executor_phase,
                    "source_commit": source_commit,
                    "worker_startup_diagnostics": _worker_startup_diagnostics(startup_root),
                    "worker_exit_information": _executor_process_snapshot(executor),
                },
            )
        if aborted:
            for future in futures:
                future.cancel()
                cancelled += 1
    finally:
        executor_phase = "shutdown"
        worker_exit_information = _executor_process_snapshot(executor)
        executor.shutdown(wait=True, cancel_futures=True)
        worker_failure_record_count = sum(
            1
            for path in event_root.glob("*.json")
            if _read_json(path).get("status") == "FAILED"
        )
        _progress(
            report_root,
            planned_partitions,
            completed,
            len(failures),
            rows_processed,
            started,
            aborted_early=aborted,
            abort_reason=abort_reason,
            dominant_failure_signature=dominant,
            tasks_cancelled=cancelled,
            abort_threshold_failure_count=abort_threshold_failure_count,
            worker_failure_record_count=worker_failure_record_count,
            in_flight_at_abort=in_flight_at_abort,
        )
        _write_json(
            report_root / "executor_lifecycle.json",
            {
                "status": "FAILED" if failures else "COMPLETE",
                "worker_count": workers,
                "multiprocessing_start_method": context.get_start_method(),
                "maximum_in_flight": maximum_in_flight,
                "submitted": submitted,
                "started": started_count,
                "completed": completed,
                "failed": len(failures),
                "abort_threshold_failure_count": abort_threshold_failure_count,
                "worker_failure_record_count": worker_failure_record_count,
                "in_flight_at_abort": in_flight_at_abort,
                "failure_count_explanation": (
                    "failed is the parent-observed fail-fast count; "
                    "worker_failure_record_count includes persisted results from "
                    "tasks that were already in flight during shutdown"
                ),
                "cancelled": cancelled,
                "executor_phase": "shutdown_complete",
                "source_commit": source_commit,
                "worker_startup_diagnostics": _worker_startup_diagnostics(startup_root),
                "worker_exit_information": worker_exit_information,
            },
        )
    return {
        "failures": failures,
        "rows_processed": rows_processed,
        "aborted_early": aborted,
        "abort_reason": abort_reason,
        "dominant_signature": dominant,
        "tasks_cancelled": cancelled,
        "abort_threshold_failure_count": abort_threshold_failure_count,
        "worker_failure_record_count": worker_failure_record_count,
        "in_flight_at_abort": in_flight_at_abort,
    }


def _worker_startup_diagnostics(root: Path) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(root.glob("*.json"))][-64:]


def _executor_process_snapshot(executor: Any) -> list[dict[str, Any]]:
    processes = getattr(executor, "_processes", None) or {}
    return [
        {"worker_pid": process.pid, "exitcode": process.exitcode}
        for process in processes.values()
    ]


def _parent_memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        return {
            "parent_pid": os.getpid(),
            "rss_bytes": memory.rss,
            "vms_bytes": memory.vms,
            "available_bytes": psutil.virtual_memory().available,
        }
    except Exception:
        return {"parent_pid": os.getpid(), "available": False}


def _source_commit(config: Mapping[str, Any]) -> str:
    configured = dict(config.get("ml", {}) or {}).get("source_commit")
    if configured:
        return str(configured)
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_base_artifact_from_labeled_spines(config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    rows: list[dict[str, Any]] = []
    spy_returns = _spine_return_by_date(root / "symbol=SPY" / "spine.parquet")
    for path in sorted(root.glob("symbol=*/spine.parquet")):
        for row in _read_parquet_file(path):
            if not row.get("selector_eligible") or not row.get("is_labeled"):
                continue
            rows.append(_base_row(row, spy_returns, input_resolution=input_resolution))
    rows.sort(key=lambda row: (row["rebalance_date"], row["symbol"]))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["rebalance_date", "symbol"]
    identity = _write_large_parquet_artifact(
        settings.base_artifact_path,
        rows,
        fieldnames=fieldnames,
        config=config,
        inspection_sample_path=settings.output_dir / "stock_level_prediction_artifacts_sample.csv",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(
        settings.output_dir / "stock_level_prediction_artifacts.json",
        {
            "mode": "canonical_v2_labeled_spine_base_artifact",
            "row_count": len(rows),
            "symbol_count": len({row["symbol"] for row in rows}),
            "date_min": min((row["rebalance_date"] for row in rows), default=None),
            "date_max": max((row["rebalance_date"] for row in rows), default=None),
            "canonical_artifact": identity,
            "source_spine_root": str(root),
            "research_only": True,
            "trading_impact": "none",
            "production_validated": False,
        },
    )
    return identity


def ensure_base_symbol_partitions(config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
    labeled_root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    configured_symbols = ml.get("canonical_v2_alpha_symbols")
    requested = {str(symbol).upper() for symbol in configured_symbols} if configured_symbols else None
    manifest_path = report_root / "base_partition_manifest.json"
    base_partition_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        path
        for path in base_partition_root.glob("symbol=*/rows.parquet")
        if _base_partition_compatible(path, config, input_resolution=input_resolution)
    )
    if existing:
        return _read_json(manifest_path) or {"status": "BUILT", "partition_count": len(existing), "path": str(base_partition_root)}
    spy_returns = _spine_return_by_date(labeled_root / "symbol=SPY" / "spine.parquet")
    partitions = []
    started = time.perf_counter()
    for spine_path in sorted(labeled_root.glob("symbol=*/spine.parquet")):
        symbol = spine_path.parent.name.split("=", 1)[1]
        if requested is not None and symbol.upper() not in requested:
            continue
        spine_rows = _read_parquet_file(spine_path)
        rows = _base_rows_from_spine(spine_rows, spy_returns, input_resolution=input_resolution)
        if not rows:
            continue
        target = base_partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
        _write_partition_parquet(target, rows, list(rows[0]))
        _write_json(_base_partition_identity_path(target), _base_partition_identity(target, config, input_resolution=input_resolution))
        partitions.append({"symbol": symbol, "row_count": len(rows), "path": str(target), "sha256": _file_sha256(target)})
    payload = {
        "status": "BUILT",
        "path": str(base_partition_root),
        "partition_count": len(partitions),
        "row_count": sum(int(row["row_count"]) for row in partitions),
        "symbol_count": len(partitions),
        "elapsed_seconds": time.perf_counter() - started,
        "base_artifact_path": str(settings.base_artifact_path),
        "source_labeled_spine_root": str(labeled_root),
        "partitions": partitions,
    }
    _write_json(manifest_path, payload)
    return payload


def validate_enriched_artifact(path: Path, *, input_resolution: Mapping[str, Any]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    required = {
        "rebalance_date",
        "symbol",
        "asset_id",
        "actual_forward_return_10d",
        "target_provenance_contract_version",
        *ENGINEERED_FEATURE_COLUMNS,
    }
    missing_columns = sorted(required - set(parquet.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"enriched artifact is missing required columns: {missing_columns}")
    feature_missingness = {feature: 0 for feature in ENGINEERED_FEATURE_COLUMNS}
    previous_asset_key: tuple[Any, str] | None = None
    previous_symbol_key: tuple[str, str] | None = None
    duplicate_asset = 0
    duplicate_symbol = 0
    symbols: set[str] = set()
    dates: set[str] = set()
    versions: set[str] = set()
    tier_d_rows = 0
    quarantined_rows = 0
    label_violations = 0
    row_count = 0
    key_hasher = hashlib.sha256()
    columns = [
        name
        for name in (
            "rebalance_date",
            "symbol",
            "asset_id",
            "target_provenance_contract_version",
            "compatibility_tier",
            "eligibility_reason",
            "label_available_timestamp",
            "decision_timestamp",
            *ENGINEERED_FEATURE_COLUMNS,
        )
        if name in parquet.schema_arrow.names
    ]
    for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            row_count += 1
            symbol = str(row.get("symbol", "")).upper()
            date = str(row.get("rebalance_date", ""))[:10]
            asset_key = (row.get("asset_id"), date)
            symbol_key = (symbol, date)
            if asset_key == previous_asset_key:
                duplicate_asset += 1
            if symbol_key == previous_symbol_key:
                duplicate_symbol += 1
            previous_asset_key = asset_key
            previous_symbol_key = symbol_key
            symbols.add(symbol)
            dates.add(date)
            versions.add(str(row.get("target_provenance_contract_version") or ""))
            key_hasher.update(f"{date}\x1f{symbol}\n".encode("utf-8"))
            for feature in ENGINEERED_FEATURE_COLUMNS:
                if row.get(feature) in (None, "", "nan"):
                    feature_missingness[feature] += 1
            tier_d_rows += row.get("compatibility_tier") == "TIER_D_SYMBOL_QUARANTINE"
            quarantined_rows += str(row.get("eligibility_reason", "")).startswith("quarantined:")
            label_available = str(row.get("label_available_timestamp", ""))[:10]
            decision = str(row.get("decision_timestamp", ""))[:10]
            label_violations += bool(
                label_available and decision and label_available < decision
            )
    base = dict(input_resolution.get("validated_alpha_base", {}) or {})
    economic_alignment = (
        not base.get("economic_key_sha256")
        or key_hasher.hexdigest() == base["economic_key_sha256"]
    )
    return {
        "valid": (
            duplicate_asset == 0
            and duplicate_symbol == 0
            and tier_d_rows == 0
            and quarantined_rows == 0
            and label_violations == 0
            and versions == {TARGET_PROVENANCE_V2}
            and economic_alignment
            and row_count == parquet.metadata.num_rows
        ),
        "path": str(path),
        "row_count": row_count,
        "symbol_count": len(symbols),
        "date_min": min(dates, default=None),
        "date_max": max(dates, default=None),
        "duplicate_asset_session_rows": duplicate_asset,
        "duplicate_symbol_session_rows": duplicate_symbol,
        "tier_d_rows": tier_d_rows,
        "quarantined_rows": quarantined_rows,
        "label_availability_violations": label_violations,
        "feature_missingness": feature_missingness,
        "target_provenance_contract_versions": sorted(versions),
        "economic_key_sha256": key_hasher.hexdigest(),
        "economic_key_alignment_valid": economic_alignment,
        "full_table_materialized": False,
        "canonical_source_identity": dict(input_resolution.get("canonical_dataset", {})),
    }


def _build_partition(
    symbol: str,
    config: Mapping[str, Any],
    prepared_spy: list[dict[str, float | str]],
    partition_root: Path,
    manifest_root: Path,
    *,
    input_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    started = time.perf_counter()
    timings: dict[str, float] = {}
    certified_base_sha = str(ml.get("canonical_v2_alpha_validated_base_sha256", ""))
    if certified_base_sha:
        source_base_partition_path = str(
            Path(str(ml["canonical_v2_alpha_base_partition_root"]))
            / f"symbol={_safe_symbol(symbol)}"
            / "rows.parquet"
        )
        source = {
            "base_partition_reused": True,
            "base_partition_path": source_base_partition_path,
            "spine_path": "",
        }
        source_mode = "certified_published_base_partition"
    else:
        source = _resolve_symbol_source(config, symbol)
        source_mode = "labeled_spine_partition"
    monolithic_base_read = False
    base_partition_reused = source["base_partition_reused"]
    source_base_partition_path = source["base_partition_path"]
    source_spine_path = source["spine_path"]
    phase = "spine_read"
    spine_rows_read = 0
    price_history_rows_read = 0
    try:
        phase_started = time.perf_counter()
        if certified_base_sha:
            base_identity = _read_json(
                _base_partition_identity_path(Path(source_base_partition_path))
            )
            if (
                base_identity.get("status") != "COMPLETE"
                or base_identity.get("source_base_sha256") != certified_base_sha
            ):
                raise ValueError(
                    f"certified alpha base partition identity mismatch for {symbol}"
                )
            rows = _read_parquet_file(Path(source_base_partition_path))
            if {str(row.get("symbol", "")).upper() for row in rows} != {symbol.upper()}:
                raise ValueError(f"certified alpha base partition symbol mismatch for {symbol}")
            source_meta = {
                "spine_read_seconds": 0.0,
                "base_derivation_seconds": 0.0,
                "base_partition_reused": True,
                "source_rows_read": len(rows),
            }
        else:
            rows, source_meta = _read_symbol_source_rows_from_spine(
                symbol,
                Path(source_spine_path),
                Path(source_base_partition_path),
                config=config,
                input_resolution=input_resolution or resolve_inputs(config),
            )
        timings["spine_read_seconds"] = float(source_meta.get("spine_read_seconds", time.perf_counter() - phase_started) or 0.0)
        timings["base_derivation_seconds"] = float(source_meta.get("base_derivation_seconds", 0.0) or 0.0)
        base_partition_reused = bool(source_meta["base_partition_reused"])
        spine_rows_read = int(source_meta.get("source_rows_read", len(rows)) or len(rows))

        phase = "price_history_read"
        phase_started = time.perf_counter()
        history = _load_price_histories(settings.parquet_dir, [symbol]).get(symbol, [])
        price_history_rows_read = len(history)
        timings["price_history_read_seconds"] = time.perf_counter() - phase_started

        phase = "feature_compute"
        phase_started = time.perf_counter()
        prepared_history = _prepare_history(history)
        enriched = _build_symbol_rows((rows, prepared_history, prepared_spy))
        enriched = merge_enrichment_preserving_base(rows, enriched)
        timings["feature_compute_seconds"] = time.perf_counter() - phase_started
    except Exception as exc:
        failure = _partition_failure_payload(
            symbol,
            exc,
            phase=phase,
            source_spine_path=source_spine_path,
            source_base_partition_path=source_base_partition_path,
            monolithic_base_read=monolithic_base_read,
            base_partition_reused=base_partition_reused,
            source_rows_read=spine_rows_read,
            price_history_rows_read=price_history_rows_read,
            timings=timings,
        )
        _write_json(manifest_root.parent / "partition_failures" / f"{_safe_symbol(symbol)}.json", failure)
        raise PartitionBuildError(failure) from exc
    path = partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    diagnosis_path = manifest_root.parent / "schema_diagnostics" / f"{_safe_symbol(symbol)}.json"
    inventory = _column_type_inventory(enriched)
    _write_json(diagnosis_path, {"symbol": symbol, "phase": "pre_normalization", "column_type_inventory": inventory, "timestamp": datetime.now(timezone.utc).isoformat()})
    try:
        phase = "normalisation"
        phase_started = time.perf_counter()
        normalized, schema_report = _normalize_partition_rows(enriched)
        _validate_normalized_rows(enriched, normalized, schema_report)
        table = pa.Table.from_pylist(normalized, schema=_schema_for_fieldnames(list(normalized[0]) if normalized else []))
        timings["normalisation_seconds"] = time.perf_counter() - phase_started
    except (ArrowTypeError, TypeError, ValueError) as exc:
        failure = _schema_failure_payload(
            symbol,
            exc,
            inventory,
            phase=phase,
            source_spine_path=source_spine_path,
            source_base_partition_path=source_base_partition_path,
            monolithic_base_read=monolithic_base_read,
            base_partition_reused=base_partition_reused,
            source_rows_read=spine_rows_read,
            price_history_rows_read=price_history_rows_read,
            timings=timings,
        )
        _write_json(manifest_root.parent / "partition_failures" / f"{_safe_symbol(symbol)}.json", failure)
        raise PartitionBuildError(failure) from exc
    _write_json(manifest_root.parent / "schema_validation" / f"{_safe_symbol(symbol)}.json", schema_report)
    phase_started = time.perf_counter()
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    timings["parquet_write_seconds"] = time.perf_counter() - phase_started
    timings["total_seconds"] = time.perf_counter() - started
    manifest = {
        "symbol": symbol,
        "status": "COMPLETE",
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": (
            _target_provenance_contract_version()
        ),
        "row_count": len(enriched),
        "path": str(path),
        "sha256": _file_sha256(path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "compatibility_identity": _partition_compatibility_identity(
            symbol,
            config,
            source_base_partition_path=source_base_partition_path,
        ),
        "source_rows_read": spine_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": timings,
    }
    _write_json(manifest_root / f"{_safe_symbol(symbol)}.json", manifest)
    return manifest


def _base_rows_from_spine(
    spine_rows: Sequence[Mapping[str, Any]],
    spy_returns: Mapping[str, float],
    *,
    input_resolution: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in spine_rows
        if row.get("selector_eligible") and row.get("is_labeled")
    ]
    eligible.sort(key=lambda row: str(row.get("session_date", ""))[:10])
    sessions = [str(row.get("session_date", ""))[:10] for row in eligible]
    return [
        _base_row(row, spy_returns, next_session=sessions[index + 1] if index + 1 < len(sessions) else "", input_resolution=input_resolution)
        for index, row in enumerate(eligible)
    ]


def _target_provenance_contract_version() -> str:
    from core.research.ml.stock_level.prediction_artifacts.types import TARGET_PROVENANCE_CONTRACT_VERSION

    return TARGET_PROVENANCE_CONTRACT_VERSION


def _label_observation_timestamp(session: str) -> str:
    from core.research.ml.stock_level.prediction_artifacts.targets import label_observation_timestamp

    return label_observation_timestamp(session)


def _base_row(
    row: Mapping[str, Any],
    spy_returns: Mapping[str, float],
    *,
    next_session: str = "",
    input_resolution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    date = str(row["session_date"])[:10]
    end = str(row.get("target_end_session_date") or date)[:10]
    actual = _float(row.get("actual_forward_return_10d"))
    benchmark = spy_returns.get(date)
    residual = actual - benchmark if actual is not None and benchmark is not None else ""
    label_start = _label_observation_timestamp(next_session) if next_session else ""
    label_end = _label_observation_timestamp(end) if end else ""
    label_available = label_end
    benchmark_label_start = label_start if benchmark is not None else ""
    benchmark_label_end = label_end if benchmark is not None else ""
    benchmark_label_available = label_available if benchmark is not None else ""
    return {
        "rebalance_date": date,
        "symbol": str(row.get("symbol") or row.get("canonical_symbol", "")).upper(),
        "asset_id": row.get("asset_id"),
        "canonical_symbol": row.get("canonical_symbol"),
        "source_provider": row.get("source_provider"),
        "compatibility_tier": row.get("compatibility_tier"),
        "eligibility_reason": row.get("eligibility_reason"),
        "selector_eligible": bool(row.get("selector_eligible")),
        "provider_transition_flag": bool(row.get("provider_transition_flag")),
        "provider_transition_id": row.get("provider_transition_id"),
        "target_provenance_contract_version": _target_provenance_contract_version(),
        "feature_timestamp": date,
        "feature_data_cutoff_timestamp": f"{date} 20:00:00+00:00",
        "decision_timestamp": f"{date} 20:05:00+00:00",
        "decision_session_date": date,
        "first_actionable_session": "",
        "decision_grid_version": "canonical_daily_v2_labeled_spine_v1",
        "decision_grid_identity": input_resolution.get("canonical_dataset", {}).get("hash") if input_resolution else "",
        "exchange_calendar_identity": "canonical_daily_v2_sessions",
        "decision_frequency": "daily",
        "target_horizon_trading_days": int(row.get("target_horizon_trading_days") or 10),
        "overlapping_targets": None,
        "required_purge_horizon_trading_days": 10,
        "target_horizon": "10_trading_observations",
        "target_observation_count": 10,
        "target_start_timestamp": date,
        "label_start_timestamp": label_start,
        "label_end_timestamp": label_end,
        "label_available_timestamp": label_available,
        "target_price_convention": "canonical_daily_v2_model_close_to_close",
        "benchmark_target_start_timestamp": date if benchmark is not None else "",
        "benchmark_label_start_timestamp": benchmark_label_start,
        "benchmark_label_end_timestamp": benchmark_label_end,
        "benchmark_label_available_timestamp": benchmark_label_available,
        "target_status": "realized",
        "actual_forward_return_10d": actual,
        "actual_forward_return_5d": "",
        "actual_future_volatility": "",
        "actual_future_drawdown": "",
        "actual_benchmark_return_10d": benchmark if benchmark is not None else "",
        "actual_market_residual_return_10d": residual,
        "actual_vol_adjusted_forward_return_10d": actual,
        "actual_drawdown_adjusted_forward_return_10d": actual,
        "actual_rank_normalized_forward_return_10d": "",
        "actual_top_decile_label_10d": "",
        "source_dataset_hash": input_resolution.get("canonical_dataset", {}).get("hash") if input_resolution else "",
    }


def _spine_return_by_date(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {
        str(row["session_date"])[:10]: float(row["actual_forward_return_10d"])
        for row in _read_parquet_file(path, columns=["session_date", "actual_forward_return_10d"])
        if row.get("actual_forward_return_10d") is not None
    }


def _artifact_symbols(path: Path, *, base_partition_root: Path | None = None) -> list[str]:
    if base_partition_root is not None and base_partition_root.exists():
        symbols = [
            p.parent.name.split("=", 1)[1].upper()
            for p in sorted(base_partition_root.glob("symbol=*/rows.parquet"))
        ]
        if symbols:
            return symbols
    table = pq.read_table(path, columns=["symbol"])
    return sorted({str(value).upper() for value in table.column("symbol").to_pylist() if value})


def _symbol_spine_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ml = dict(config.get("ml", {}) or {})
    manifest_path = Path(str(ml.get("canonical_v2_labeled_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/labeled_spine_manifest.json")))
    manifest = _read_json(manifest_path)
    entries = manifest.get("partition_manifests") or []
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("status") != "BUILT":
            continue
        symbol = str(entry.get("canonical_symbol") or "").upper()
        path = Path(str(entry.get("path") or ""))
        if symbol and path.exists():
            index[symbol] = {**entry, "path": str(path)}
    if index:
        return index
    root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    for path in sorted(root.glob("symbol=*/spine.parquet")):
        symbol = path.parent.name.split("=", 1)[1].upper()
        index[symbol] = {"canonical_symbol": symbol, "path": str(path), "status": "BUILT"}
    return index


def _resolve_symbol_source(config: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
    spine_index = _symbol_spine_index(config)
    canonical_symbol = symbol.upper()
    if canonical_symbol not in spine_index:
        raise FileNotFoundError(f"missing labeled spine partition for symbol {canonical_symbol}")
    base_partition_path = base_partition_root / f"symbol={_safe_symbol(canonical_symbol)}" / "rows.parquet"
    return {
        "symbol": canonical_symbol,
        "source_mode": "labeled_spine_partition",
        "spine_path": spine_index[canonical_symbol]["path"],
        "base_partition_path": str(base_partition_path),
        "base_partition_reused": base_partition_path.exists(),
        "monolithic_base_read": False,
    }


def _read_source_rows(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    if columns:
        return _read_parquet_file(path, columns=columns)
    return _read_parquet_file(path)


def _read_symbol_source_rows_from_spine(
    symbol: str,
    spine_path: Path,
    base_partition_path: Path,
    *,
    config: Mapping[str, Any],
    input_resolution: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spine_path.exists():
        raise FileNotFoundError(f"missing labeled spine partition for symbol {symbol}: {spine_path}")
    if base_partition_path.exists() and _base_partition_compatible(base_partition_path, config, input_resolution=input_resolution):
        rows = _read_parquet_file(base_partition_path)
        return rows, {
            "base_partition_reused": True,
            "source_rows_read": len(rows),
            "spine_read_seconds": 0.0,
            "base_derivation_seconds": 0.0,
        }
    spine_started = time.perf_counter()
    spine_rows = _read_parquet_file(spine_path)
    spine_seconds = time.perf_counter() - spine_started
    derive_started = time.perf_counter()
    ml = dict(config.get("ml", {}) or {})
    labeled_root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    spy_returns = _spine_return_by_date(labeled_root / "symbol=SPY" / "spine.parquet")
    rows = _base_rows_from_spine(spine_rows, spy_returns, input_resolution=input_resolution)
    if not rows:
        raise ValueError(f"labeled spine partition for symbol {symbol} produced no eligible labeled rows")
    _write_partition_parquet(base_partition_path, rows, list(rows[0]))
    _write_json(_base_partition_identity_path(base_partition_path), _base_partition_identity(base_partition_path, config, input_resolution=input_resolution))
    return rows, {
        "base_partition_reused": False,
        "source_rows_read": len(spine_rows),
        "spine_read_seconds": spine_seconds,
        "base_derivation_seconds": time.perf_counter() - derive_started,
    }


def _read_symbol_source_rows(path: Path, symbol: str, *, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if config is not None:
        ml = dict(config.get("ml", {}) or {})
        report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
        base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
        partition = base_partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
        if partition.exists() and _base_partition_compatible(partition, config, input_resolution=resolve_inputs(config)):
            return _read_parquet_file(partition)
    try:
        table = pq.read_table(path, filters=[("symbol", "=", symbol)])
        rows = table.to_pylist()
        if rows:
            return rows
    except Exception:
        pass
    return [
        row
        for row in _read_parquet_file(path)
        if str(row.get("symbol", "")).upper() == symbol
    ]


def _read_partition_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("symbol=*/rows.parquet")):
        rows.extend(_read_parquet_file(path))
    return rows


def _completed_partition_paths(manifest_root: Path, *, expected_symbols: Sequence[str], config: Mapping[str, Any] | None = None) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for symbol in expected_symbols:
        manifest = _read_json(manifest_root / f"{_safe_symbol(symbol)}.json")
        path = Path(str(manifest.get("path") or ""))
        compatible = True
        if config is not None:
            compatible = manifest.get("compatibility_identity") == _partition_compatibility_identity(
                str(symbol),
                config,
                source_base_partition_path=str(manifest.get("source_base_partition_path") or ""),
            )
        if manifest.get("status") != "COMPLETE" or not path.exists() or not compatible:
            missing.append(symbol)
            continue
        paths.append(path)
    if missing:
        raise FileNotFoundError(f"missing completed alpha partitions for symbols: {missing[:10]}")
    return paths


def _read_parquet_file(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    return pq.ParquetFile(path).read(columns=list(columns) if columns else None).to_pylist()


def _fieldnames(source_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    source_columns = list(source_rows[0]) if source_rows else []
    return list(dict.fromkeys([*source_columns, *ENGINEERED_FEATURE_COLUMNS, *ENRICHMENT_METADATA_COLUMNS]))


def _write_large_parquet_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    config: Mapping[str, Any],
    sample_path: Path | None,
) -> dict[str, Any]:
    compression = str(dict(config.get("ml", {}) or {}).get("stock_level_parquet_compression", "zstd")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    schema = _schema_for_rows(rows, fieldnames)
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(rows), 100_000):
            chunk = [
                {name: row.get(name) for name in fieldnames}
                for row in rows[start : start + 100_000]
            ]
            table = pa.Table.from_pylist(chunk, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=compression)
            writer.write_table(table)
        if writer is None:
            writer = pq.ParquetWriter(tmp, schema, compression=compression)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(path)
    if sample_path is not None:
        _write_csv(sample_path, [{name: row.get(name) for name in fieldnames} for row in rows[:100]], fieldnames)
    parquet = pq.ParquetFile(path)
    column_order = list(parquet.schema_arrow.names)
    decision_dates = _column_values(path, "rebalance_date")
    symbols = _column_values(path, "symbol")
    target_versions = sorted(set(_column_values(path, "target_provenance_contract_version")))
    dataset_hashes = sorted(set(_column_values(path, "source_dataset_hash")))
    file_hash = _file_sha256(path)
    return {
        "artifact_format": "parquet",
        "compression": compression,
        "resolved_artifact_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_hash,
        "logical_content_sha256": file_hash,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
        "stable_column_order": column_order,
        "row_count": parquet.metadata.num_rows,
        "column_count": len(column_order),
        "symbol_count": len({str(value).upper() for value in symbols if value}),
        "decision_date_count": len({str(value)[:10] for value in decision_dates if value}),
        "minimum_decision_timestamp": min((str(value) for value in decision_dates if value), default=None),
        "maximum_decision_timestamp": max((str(value) for value in decision_dates if value), default=None),
        "target_contract_version": target_versions[0] if len(target_versions) == 1 else None,
        "target_contract_versions": target_versions,
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "source_dataset_hash_count": len(dataset_hashes),
        "source_dataset_hashes": dataset_hashes[:10],
        "completion_status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_partition_dataset(partition_paths: Sequence[Path], *, report_root: Path | None = None) -> dict[str, Any]:
    if not partition_paths:
        raise ValueError("no completed alpha partitions available for consolidation")
    ordered_paths = sorted(partition_paths, key=lambda path: str(path).lower())
    canonical_schema = _schema_for_fieldnames(
        pq.ParquetFile(ordered_paths[0]).schema_arrow.names
    )
    partition_reports: list[dict[str, Any]] = []
    row_count = 0
    duplicate_keys = 0
    seen_keys: set[tuple[str, str]] = set()
    for path in ordered_paths:
        parquet = pq.ParquetFile(path)
        partition_row_count = 0
        partition_mismatches: dict[str, dict[str, str]] = {}
        for batch in parquet.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch])
            report, casted = _validate_and_cast_partition_table(
                path,
                table,
                canonical_schema,
                row_offset=partition_row_count,
            )
            for mismatch in report["type_mismatches"]:
                partition_mismatches[mismatch["column"]] = mismatch
            symbols = casted["symbol"] if "symbol" in casted.column_names else None
            dates = casted["rebalance_date"] if "rebalance_date" in casted.column_names else None
            if symbols is not None and dates is not None:
                for symbol, date in zip(_iter_chunked_values(symbols), _iter_chunked_values(dates)):
                    key = (str(symbol).upper(), str(date)[:10])
                    if key in seen_keys:
                        duplicate_keys += 1
                    seen_keys.add(key)
            partition_row_count += table.num_rows
        partition_reports.append(
            {
                "partition_path": str(path),
                "row_count": partition_row_count,
                "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
                "missing_columns": [],
                "unexpected_columns": [],
                "type_mismatches": list(partition_mismatches.values()),
                "cast_operations": [
                    {
                        "column": mismatch["column"],
                        "from": mismatch["actual_type"],
                        "to": mismatch["expected_type"],
                    }
                    for mismatch in partition_mismatches.values()
                ],
            }
        )
        row_count += partition_row_count
    if duplicate_keys:
        raise ValueError(f"duplicate symbol/date keys in partition dataset: {duplicate_keys}")
    payload = {
        "partition_count": len(partition_paths),
        "row_count": row_count,
        "canonical_schema_fingerprint": _schema_fingerprint(canonical_schema),
        "duplicate_symbol_date_keys": duplicate_keys,
        "partitions": partition_reports,
    }
    if report_root is not None:
        _write_json(report_root / "partition_schema_validation.json", payload)
    return payload


def _consolidate_partition_parquets(
    partition_paths: Sequence[Path],
    output_path: Path,
    *,
    config: Mapping[str, Any],
    sample_path: Path | None,
    expected_row_count: int,
    report_root: Path | None = None,
    preflight_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not partition_paths:
        raise ValueError("no completed alpha partitions available for consolidation")
    ml = dict(config.get("ml", {}) or {})
    compression = str(ml.get("stock_level_parquet_compression", "zstd")).lower()
    preflight = dict(
        preflight_validation
        or _validate_partition_dataset(partition_paths, report_root=report_root)
    )
    if int(preflight["row_count"]) != expected_row_count:
        raise ValueError(
            "pre-consolidation partition row count "
            f"{preflight['row_count']} does not match expected {expected_row_count}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    canonical_schema = _schema_for_fieldnames(
        pq.ParquetFile(partition_paths[0]).schema_arrow.names
    )
    row_count = 0
    duplicate_keys = 0
    previous_key: tuple[str, str] | None = None
    key_hasher = hashlib.sha256()
    symbols_seen: set[str] = set()
    dates_seen: set[str] = set()
    target_versions: set[str] = set()
    dataset_hashes: set[str] = set()
    feature_non_null = {feature: 0 for feature in ENGINEERED_FEATURE_COLUMNS}
    writer: pq.ParquetWriter | None = None
    promoted: pq.ParquetFile | None = None
    validated_temporary_sha256: str | None = None
    try:
        writer = pq.ParquetWriter(tmp, canonical_schema, compression=compression)
        date_rows: list[dict[str, Any]] = []
        active_date: str | None = None
        for row in _iter_partition_rows_date_major(
            partition_paths,
            batch_rows=int(
                ml.get("canonical_v2_alpha_consolidation_batch_rows", 16_384)
            ),
        ):
            date = str(row.get("rebalance_date", ""))[:10]
            if active_date is not None and date != active_date:
                _add_cross_sectional_features(date_rows)
                writer.write_table(_rows_to_table(date_rows, canonical_schema))
                date_rows = []
            active_date = date
            date_rows.append(row)
        if date_rows:
            _add_cross_sectional_features(date_rows)
            writer.write_table(_rows_to_table(date_rows, canonical_schema))
        writer.close()
        writer = None

        promoted = pq.ParquetFile(tmp)
        promoted_metadata = promoted.metadata
        promoted_schema = promoted.schema_arrow
        for batch in promoted.iter_batches(
            batch_size=int(
                ml.get("canonical_v2_alpha_validation_batch_rows", 65_536)
            )
        ):
            for row in _batch_rows(batch):
                symbol = str(row.get("symbol", "")).upper()
                date = str(row.get("rebalance_date", ""))[:10]
                key = (date, symbol)
                if previous_key == key:
                    duplicate_keys += 1
                if previous_key is not None and key < previous_key:
                    raise ValueError(
                        "enriched artifact economic keys are not date-major sorted"
                    )
                previous_key = key
                key_hasher.update(
                    f"{date}\x1f{symbol}\n".encode("utf-8")
                )
                symbols_seen.add(symbol)
                dates_seen.add(date)
                target_versions.add(
                    str(row.get("target_provenance_contract_version") or "")
                )
                dataset_hash = str(row.get("source_dataset_hash") or "")
                if dataset_hash:
                    dataset_hashes.add(dataset_hash)
                for feature in ENGINEERED_FEATURE_COLUMNS:
                    if row.get(feature) not in (None, "", "nan"):
                        feature_non_null[feature] += 1
                row_count += 1
        promoted.close()
        promoted = None
        if duplicate_keys:
            raise ValueError(
                f"duplicate symbol/date keys during consolidation: {duplicate_keys}"
            )
        if row_count != expected_row_count:
            raise ValueError(f"consolidated row count {row_count} does not match expected {expected_row_count}")
        if promoted_metadata.num_rows != expected_row_count:
            raise ValueError(f"temporary artifact row count {promoted_metadata.num_rows} does not match expected {expected_row_count}")
        if _schema_fingerprint(promoted_schema) != _schema_fingerprint(canonical_schema):
            raise ValueError("temporary artifact schema fingerprint mismatch")
        expected_keys = str(
            ml.get("canonical_v2_alpha_validated_base_key_sha256", "")
        )
        if expected_keys and key_hasher.hexdigest() != expected_keys:
            raise ValueError(
                "enriched artifact economic keys do not align with the certified base"
            )
        if (
            ml.get("canonical_v2_alpha_validated_base_sha256")
            and target_versions != {TARGET_PROVENANCE_V2}
        ):
            raise ValueError(
                "enriched artifact target provenance mismatch: "
                f"{sorted(target_versions)}"
            )
        validated_temporary_sha256 = _file_sha256(tmp)
        tmp.replace(output_path)
    except Exception:
        if promoted is not None:
            promoted.close()
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise
    if sample_path is not None:
        _write_parquet_sample_csv(output_path, sample_path, limit=100)
    parquet = pq.ParquetFile(output_path)
    column_order = list(parquet.schema_arrow.names)
    file_hash = _file_sha256(output_path)
    if file_hash != validated_temporary_sha256:
        raise ValueError(
            "published artifact checksum differs from validated temporary artifact"
        )
    feature_coverage = [
        {
            "feature": feature,
            "definition": "",
            "populated_count": feature_non_null[feature],
            "missing_count": row_count - feature_non_null[feature],
            "availability_rate": (
                feature_non_null[feature] / row_count if row_count else 0.0
            ),
        }
        for feature in ENGINEERED_FEATURE_COLUMNS
    ]
    identity = {
        "artifact_format": "parquet",
        "compression": compression,
        "resolved_artifact_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "sha256": file_hash,
        "logical_content_sha256": file_hash,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
        "stable_column_order": column_order,
        "row_count": parquet.metadata.num_rows,
        "column_count": len(column_order),
        "symbol_count": len(symbols_seen),
        "decision_date_count": len(dates_seen),
        "minimum_decision_timestamp": min(dates_seen, default=None),
        "maximum_decision_timestamp": max(dates_seen, default=None),
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": (
            next(iter(target_versions)) if len(target_versions) == 1 else None
        ),
        "target_contract_version": next(iter(target_versions)) if len(target_versions) == 1 else None,
        "target_contract_versions": sorted(target_versions),
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "source_dataset_hash_count": len(dataset_hashes),
        "source_dataset_hashes": sorted(dataset_hashes)[:10],
        "source_base_sha256": ml.get("canonical_v2_alpha_validated_base_sha256"),
        "source_base_economic_key_sha256": ml.get(
            "canonical_v2_alpha_validated_base_key_sha256"
        ),
        "economic_key_sha256": key_hasher.hexdigest(),
        "economic_key_alignment_valid": True,
        "feature_coverage": feature_coverage,
        "completion_status": "complete",
        "atomic_publication": {
            "status": "COMPLETE",
            "temporary_artifact_validated": True,
            "validated_temporary_sha256": validated_temporary_sha256,
            "published_sha256": file_hash,
            "replacement_method": "same_filesystem_replace",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": "validated_symbol_partitions",
        "source_partition_count": len(partition_paths),
    }
    if report_root is not None:
        _write_json(
            report_root / "consolidation_manifest.json",
            {
                "status": "COMPLETE",
                "row_count": row_count,
                "partition_count": len(partition_paths),
                "duplicate_symbol_date_keys": duplicate_keys,
                "schema_fingerprint": identity["schema_fingerprint"],
                "source_base_sha256": identity["source_base_sha256"],
                "economic_key_alignment_valid": True,
                "artifact": identity,
            },
        )
    return identity


def _iter_partition_rows_date_major(
    partition_paths: Sequence[Path],
    *,
    batch_rows: int,
):
    iterators = [
        iter(_iter_parquet_rows(path, batch_rows=batch_rows))
        for path in partition_paths
    ]
    heap: list[tuple[str, str, int, dict[str, Any]]] = []
    for index, iterator in enumerate(iterators):
        try:
            row = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                str(row.get("rebalance_date", ""))[:10],
                str(row.get("symbol", "")).upper(),
                index,
                row,
            ),
        )
    while heap:
        _date, _symbol, index, row = heapq.heappop(heap)
        yield row
        try:
            next_row = next(iterators[index])
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (
                str(next_row.get("rebalance_date", ""))[:10],
                str(next_row.get("symbol", "")).upper(),
                index,
                next_row,
            ),
        )


def _iter_parquet_rows(path: Path, *, batch_rows: int):
    previous: tuple[str, str] | None = None
    parquet = pq.ParquetFile(path)
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=batch_rows):
        for batch_index, row in enumerate(_batch_rows(batch)):
            row["__source_partition_path"] = str(path)
            row["__source_row_index"] = row_offset + batch_index
            key = (
                str(row.get("rebalance_date", ""))[:10],
                str(row.get("symbol", "")).upper(),
            )
            if previous is not None and key < previous:
                raise ValueError(
                    f"alpha partition is not deterministically sorted: {path}"
                )
            previous = key
            yield row
        row_offset += batch.num_rows


def _batch_rows(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    return pa.Table.from_batches([batch]).to_pylist()


def _rows_to_table(
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    *,
    source_partition_path: Path | str | None = None,
    row_offset: int = 0,
    allow_whitespace_numeric_missing: bool = False,
) -> pa.Table:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        unknown = sorted(
            set(row)
            - set(ALPHA_OUTPUT_SCHEMA)
            - {"__source_partition_path", "__source_row_index"}
        )
        if unknown:
            raise ValueError(
                "canonical alpha row contains unknown columns: "
                f"{unknown}; symbol={row.get('symbol')!r} "
                f"rebalance_date={row.get('rebalance_date')!r} "
                f"source_partition_path={row.get('__source_partition_path') or source_partition_path!s} "
                f"row_index={row.get('__source_row_index', row_offset + index)}"
            )
        output: dict[str, Any] = {}
        for field in schema:
            value = row.get(field.name)
            try:
                output[field.name], _coerced = _normalize_value(
                    field.name,
                    value,
                    _column_kind(field.name),
                    allow_whitespace_numeric_missing=allow_whitespace_numeric_missing,
                )
            except ValueError as exc:
                raise ValueError(
                    "canonical alpha value is incompatible: "
                    f"column={field.name} expected_kind={_column_kind(field.name)} "
                    f"expected_type={field.type} value={value!r} "
                    f"symbol={row.get('symbol')!r} "
                    f"rebalance_date={row.get('rebalance_date')!r} "
                    f"source_partition_path={row.get('__source_partition_path') or source_partition_path!s} "
                    f"row_index={row.get('__source_row_index', row_offset + index)}; "
                    f"{exc}"
                ) from exc
        normalized.append(output)
    return pa.Table.from_pylist(normalized, schema=schema)


def _validate_and_cast_partition_table(
    path: Path,
    table: pa.Table,
    canonical_schema: pa.Schema,
    *,
    row_offset: int = 0,
) -> tuple[dict[str, Any], pa.Table]:
    missing = [name for name in canonical_schema.names if name not in table.column_names]
    unexpected = [name for name in table.column_names if name not in canonical_schema.names]
    type_mismatches: list[dict[str, Any]] = []
    casts: list[dict[str, str]] = []
    if missing or unexpected:
        raise ValueError(f"partition schema columns mismatch for {path}: missing={missing[:10]} unexpected={unexpected[:10]}")
    for field in canonical_schema:
        actual_type = table[field.name].type
        if not actual_type.equals(field.type):
            type_mismatches.append({"column": field.name, "actual_type": str(actual_type), "expected_type": str(field.type)})
            casts.append({"column": field.name, "from": str(actual_type), "to": str(field.type)})
    rows = table.to_pylist()
    for row_index, row in enumerate(rows):
        row["__source_partition_path"] = str(path)
        row["__source_row_index"] = row_offset + row_index
    casted_table = _rows_to_table(
        rows,
        canonical_schema,
        source_partition_path=path,
        allow_whitespace_numeric_missing=False,
    )
    report = {
        "partition_path": str(path),
        "row_count": table.num_rows,
        "schema_fingerprint": _schema_fingerprint(table.schema),
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "type_mismatches": type_mismatches,
        "cast_operations": casts,
    }
    return report, casted_table


def _safe_cast_partition_column(path: Path, name: str, column: pa.ChunkedArray, expected_type: pa.DataType) -> pa.ChunkedArray:
    actual_type = column.type
    if pa.types.is_null(actual_type):
        return column.cast(expected_type)
    if pa.types.is_string(expected_type):
        if pa.types.is_dictionary(actual_type) or pa.types.is_large_string(actual_type):
            return column.cast(expected_type)
    if pa.types.is_floating(expected_type):
        if pa.types.is_integer(actual_type) or pa.types.is_floating(actual_type):
            return column.cast(expected_type)
        if pa.types.is_string(actual_type) or pa.types.is_large_string(actual_type):
            representative = _first_non_null_string(column)
            raise ValueError(
                "numeric column contains text during consolidation: "
                f"partition={path} column={name} row_index={representative['row_index']} "
                f"value={representative['value']!r}"
            )
    if pa.types.is_int64(expected_type) and pa.types.is_integer(actual_type):
        return column.cast(expected_type)
    raise ValueError(f"unsafe partition schema cast for {path}: column={name} actual={actual_type} expected={expected_type}")


def _first_non_null_string(column: pa.ChunkedArray) -> dict[str, Any]:
    offset = 0
    for chunk in column.chunks:
        for index, scalar in enumerate(chunk):
            value = scalar.as_py()
            if value is not None:
                return {"row_index": offset + index, "value": value}
        offset += len(chunk)
    return {"row_index": None, "value": None}


def _iter_chunked_values(column: pa.ChunkedArray):
    for chunk in column.chunks:
        for scalar in chunk:
            yield scalar.as_py()


def _write_parquet_sample_csv(path: Path, sample_path: Path, *, limit: int) -> None:
    parquet = pq.ParquetFile(path)
    table = (
        parquet.read_row_group(0).slice(0, limit)
        if parquet.metadata.num_row_groups
        else pa.Table.from_batches([], schema=parquet.schema_arrow)
    )
    fieldnames = table.column_names
    rows: list[dict[str, Any]] = []
    columns = {name: table[name].to_pylist() for name in fieldnames}
    for index in range(table.num_rows):
        rows.append({name: columns[name][index] for name in fieldnames})
    _write_csv(sample_path, rows, fieldnames)


def _write_partition_parquet(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    schema = _schema_for_rows(rows, fieldnames)
    table = pa.Table.from_pylist([{name: row.get(name) for name in fieldnames} for row in rows], schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _column_type_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = list(dict.fromkeys(name for row in rows for name in row))
    inventory: dict[str, Any] = {}
    for column in columns:
        counts: dict[str, int] = {}
        examples: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            value = row.get(column)
            type_name = "empty_string" if value == "" else ("null" if value is None else type(value).__name__)
            counts[type_name] = counts.get(type_name, 0) + 1
            if type_name not in examples:
                examples[type_name] = {
                    "value": repr(value),
                    "row_index": index,
                    "symbol": row.get("symbol"),
                    "rebalance_date": row.get("rebalance_date"),
                }
        inventory[column] = {
            "python_type_counts": dict(sorted(counts.items())),
            "representative_values": examples,
        }
    return inventory


def _normalize_partition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_whitespace_numeric_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    normalized: list[dict[str, Any]] = []
    coerced_nulls: dict[str, int] = {name: 0 for name in fieldnames}
    invalid_values: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        output: dict[str, Any] = {}
        for column in fieldnames:
            value = row.get(column)
            kind = _column_kind(column)
            try:
                normalized_value, coerced = _normalize_value(
                    column,
                    value,
                    kind,
                    allow_whitespace_numeric_missing=allow_whitespace_numeric_missing,
                )
            except ValueError as exc:
                invalid_values.setdefault(column, []).append(
                    {
                        "row_index": row_index,
                        "symbol": row.get("symbol"),
                        "rebalance_date": row.get("rebalance_date"),
                        "value": repr(value),
                        "error": str(exc),
                    }
                )
                normalized_value, coerced = None, False
            if coerced:
                coerced_nulls[column] += 1
            output[column] = normalized_value
        normalized.append(output)
    duplicate_count = _duplicate_symbol_date_count(normalized)
    report = {
        "input_row_count": len(rows),
        "output_row_count": len(normalized),
        "column_count": len(fieldnames),
        "columns": [
            {"name": name, "kind": _column_kind(name), "arrow_type": str(_arrow_type_for_column(name))}
            for name in fieldnames
        ],
        "values_coerced_to_null_by_column": {k: v for k, v in coerced_nulls.items() if v},
        "invalid_values_rejected_by_column": invalid_values,
        "duplicate_symbol_date_keys": duplicate_count,
        "valid": len(rows) == len(normalized) and duplicate_count == 0 and not invalid_values,
    }
    if invalid_values:
        first_column = next(iter(invalid_values))
        first = invalid_values[first_column][0]
        raise ValueError(f"Invalid value for column {first_column}: {first['value']} ({first['error']})")
    return normalized, report


def _normalize_value(
    column: str,
    value: Any,
    kind: str,
    *,
    allow_whitespace_numeric_missing: bool = False,
) -> tuple[Any, bool]:
    if value is None:
        if column in NON_NULLABLE_COLUMNS:
            raise ValueError(f"non-nullable column {column} received null")
        return None, False
    if value == "":
        if kind in {"float", "int"}:
            if not ALPHA_OUTPUT_SCHEMA[column][1]:
                raise ValueError(
                    f"non-nullable {kind} column {column} received empty string"
                )
            return None, True
        if kind == "temporal":
            return None, True
        if kind == "string":
            return "", False
        raise ValueError(f"{kind} column {column} received empty string")
    if (
        isinstance(value, str)
        and not value.strip()
        and kind in {"float", "int"}
    ):
        if not allow_whitespace_numeric_missing:
            raise ValueError(
                f"{kind} column {column} received whitespace-only text"
            )
        if not ALPHA_OUTPUT_SCHEMA[column][1]:
            raise ValueError(
                f"non-nullable {kind} column {column} received whitespace-only text"
            )
        return None, True
    if kind == "string":
        if isinstance(value, str):
            return value, False
        raise ValueError(f"text column {column} received {type(value).__name__}")
    if kind == "bool":
        if isinstance(value, bool):
            return value, False
        if column in STRICT_BOOL_COLUMNS:
            raise ValueError(
                f"bool column {column} received {type(value).__name__}"
            )
        if isinstance(value, int) and value in {0, 1}:
            return bool(value), False
        if isinstance(value, float) and value in {0.0, 1.0}:
            return bool(value), False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ValueError(f"bool column {column} received non-binary numeric value {value!r}")
        raise ValueError(f"bool column {column} received {type(value).__name__}")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"int column {column} received bool")
        if isinstance(value, int):
            return value, False
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value), False
        if isinstance(value, str):
            raise ValueError(f"int column {column} received text")
        raise ValueError(f"int column {column} received non-integer value {value!r}")
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"float column {column} received bool")
        if isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            raise ValueError(f"float column {column} received text")
        else:
            raise ValueError(f"float column {column} received {type(value).__name__}")
        return (None, True) if math.isnan(parsed) else (parsed, False)
    if kind == "temporal":
        return _canonical_utc_timestamp(value, column=column), False
    raise ValueError(f"unknown column kind {kind} for {column}")


def _validate_normalized_rows(
    source_rows: Sequence[Mapping[str, Any]],
    normalized_rows: Sequence[Mapping[str, Any]],
    schema_report: Mapping[str, Any],
) -> None:
    if len(source_rows) != len(normalized_rows):
        raise ValueError("normalised row count differs from enriched row count")
    before_keys = [(row.get("symbol"), row.get("rebalance_date")) for row in source_rows]
    after_keys = [(row.get("symbol"), row.get("rebalance_date")) for row in normalized_rows]
    if before_keys != after_keys:
        raise ValueError("symbol/date keys changed during normalisation")
    if int(schema_report.get("duplicate_symbol_date_keys", 0) or 0):
        raise ValueError("duplicate symbol/date keys introduced during normalisation")
    before_columns = set(name for row in source_rows for name in row)
    after_columns = set(name for row in normalized_rows for name in row)
    if before_columns - after_columns:
        raise ValueError(f"fields dropped during normalisation: {sorted(before_columns - after_columns)}")
    for before, after in zip(source_rows, normalized_rows):
        old = before.get("actual_forward_return_10d")
        new = after.get("actual_forward_return_10d")
        if old in (None, "") and new is None:
            continue
        if old not in (None, "") and abs(float(old) - float(new)) > 1e-12:
            raise ValueError("actual_forward_return_10d changed during normalisation")


def _schema_failure_payload(
    symbol: str,
    exc: BaseException,
    inventory: Mapping[str, Any],
    *,
    phase: str,
    source_spine_path: str | None = None,
    source_base_partition_path: str | None = None,
    monolithic_base_read: bool | None = None,
    base_partition_reused: bool | None = None,
    source_rows_read: int | None = None,
    price_history_rows_read: int | None = None,
    timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
        "traceback": traceback.format_exc(),
        "column_type_inventory": inventory,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "source_rows_read": source_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": dict(timings or {}),
    }


def _partition_failure_payload(
    symbol: str,
    exc: BaseException,
    *,
    phase: str,
    source_spine_path: str | None,
    source_base_partition_path: str | None,
    monolithic_base_read: bool,
    base_partition_reused: bool,
    source_rows_read: int,
    price_history_rows_read: int,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "source_rows_read": source_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": dict(timings),
    }


def _duplicate_symbol_date_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [(row.get("symbol"), row.get("rebalance_date")) for row in rows]
    return len(keys) - len(set(keys))


def _column_kind(column: str) -> str:
    field = ALPHA_OUTPUT_SCHEMA.get(column)
    if field is None:
        raise ValueError(f"unknown canonical-v2 alpha output column: {column}")
    return field[0]


def _validate_alpha_output_schema_coverage(
    base_columns: Sequence[str],
    *,
    producer_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unknown_base_columns = sorted(set(base_columns) - set(ALPHA_OUTPUT_SCHEMA))
    producer_columns = set(_time_series_features([], []))
    unknown_producer_columns = sorted(
        producer_columns - set(ALPHA_OUTPUT_SCHEMA)
    )
    missing_expected_columns = sorted(
        ALPHA_EXPECTED_OUTPUT_COLUMNS - set(ALPHA_OUTPUT_SCHEMA)
    )
    if (
        unknown_base_columns
        or unknown_producer_columns
        or missing_expected_columns
    ):
        raise ValueError(
            "canonical-v2 alpha output schema coverage is incomplete: "
            f"unknown_base_columns={unknown_base_columns} "
            f"unknown_producer_columns={unknown_producer_columns} "
            f"missing_expected_columns={missing_expected_columns}"
        )
    value_contract = _validate_alpha_output_value_contract(
        producer_values=producer_values
    )
    return {
        "status": "COMPLETE",
        "base_column_count": len(set(base_columns)),
        "producer_column_count": len(producer_columns),
        "expected_output_column_count": len(ALPHA_EXPECTED_OUTPUT_COLUMNS),
        "schema_map_column_count": len(ALPHA_OUTPUT_SCHEMA),
        "schema_identity": _feature_schema_identity(),
        "value_contract": value_contract,
    }


def _validate_alpha_output_value_contract(
    *,
    producer_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    representative: dict[str, Any] = {}
    for column, (kind, nullable) in ALPHA_OUTPUT_SCHEMA.items():
        value: Any
        if kind == "bool":
            value = True
        elif kind == "int":
            value = 1
        elif kind == "float":
            value = 1.25
        elif kind == "temporal":
            value = (
                calendar_date(2026, 1, 2)
                if column in TEMPORAL_DATE_COLUMNS
                else datetime(2026, 1, 2, 20, 5, tzinfo=timezone.utc)
            )
        else:
            value = "contract-value"
        normalized, _ = _normalize_value(column, value, kind)
        representative[column] = normalized
        if nullable:
            null_value, _ = _normalize_value(column, None, kind)
            if null_value is not None:
                raise ValueError(
                    f"nullable alpha output column did not preserve null: {column}"
                )

    if producer_values is None:
        history = [
            {
                "date": (
                    calendar_date(2025, 1, 1) + timedelta(days=index)
                ).isoformat(),
                "close": float(100 + index),
                "high": float(101 + index),
                "low": float(99 + index),
            }
            for index in range(260)
        ]
        observed_producer = _time_series_features(history, history)
    else:
        observed_producer = dict(producer_values)
    unknown = sorted(set(observed_producer) - set(ALPHA_OUTPUT_SCHEMA))
    if unknown:
        raise ValueError(
            f"alpha producer value contract has unknown columns: {unknown}"
        )
    for column, value in observed_producer.items():
        if value in (None, ""):
            continue
        try:
            _normalize_value(column, value, _column_kind(column))
        except ValueError as exc:
            raise ValueError(
                f"alpha producer value contract mismatch for {column}: {exc}"
            ) from exc

    identity = hashlib.sha256(
        json.dumps(
            representative, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "COMPLETE",
        "representative_column_count": len(representative),
        "producer_non_null_column_count": sum(
            value not in (None, "") for value in observed_producer.values()
        ),
        "identity": identity,
    }


def _arrow_type_for_column(column: str) -> pa.DataType:
    kind = _column_kind(column)
    if kind == "bool":
        return pa.bool_()
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "temporal":
        return pa.string()
    return pa.string()


def _schema_for_fieldnames(fieldnames: Sequence[str]) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                name,
                _arrow_type_for_column(name),
                nullable=ALPHA_OUTPUT_SCHEMA[name][1],
            )
            for name in fieldnames
        ]
    )


def _schema_for_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> pa.Schema:
    fields = []
    sample = rows[: min(len(rows), 100_000)]
    for name in fieldnames:
        value_type = pa.string()
        for row in sample:
            value = row.get(name)
            if value in (None, ""):
                continue
            if isinstance(value, bool):
                value_type = pa.bool_()
            elif isinstance(value, int) and not isinstance(value, bool):
                value_type = pa.int64()
            elif isinstance(value, float):
                value_type = pa.float64()
            else:
                value_type = pa.string()
            break
        fields.append(pa.field(name, value_type))
    return pa.schema(fields)


def _column_values(path: Path, column: str) -> list[Any]:
    try:
        return pq.read_table(path, columns=[column]).column(column).to_pylist()
    except Exception:
        return []


def _schema_fingerprint(schema: pa.Schema) -> str:
    payload = "|".join(f"{field.name}:{field.type}" for field in schema)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _completed_symbols(manifest_root: Path) -> set[str]:
    completed = set()
    for path in manifest_root.glob("*.json"):
        payload = _read_json(path)
        if payload.get("status") == "COMPLETE" and payload.get("symbol"):
            completed.add(str(payload["symbol"]).upper())
    return completed


def _completed_compatible_symbols(manifest_root: Path, config: Mapping[str, Any]) -> set[str]:
    completed = set()
    for path in manifest_root.glob("*.json"):
        payload = _read_json(path)
        symbol = str(payload.get("symbol") or "").upper()
        if payload.get("status") != "COMPLETE" or not symbol:
            continue
        partition_path = Path(str(payload.get("path") or ""))
        if not partition_path.exists():
            continue
        source_base_partition_path = payload.get("source_base_partition_path")
        expected = _partition_compatibility_identity(
            symbol,
            config,
            source_base_partition_path=str(source_base_partition_path or ""),
        )
        if payload.get("compatibility_identity") == expected:
            completed.add(symbol)
    return completed


def _partition_compatibility_identity(
    symbol: str,
    config: Mapping[str, Any],
    *,
    source_base_partition_path: str,
) -> dict[str, Any]:
    return {
        "alpha_enrichment_contract_version": ALPHA_ENRICHMENT_CONTRACT_VERSION,
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": _target_provenance_contract_version(),
        "source_base_artifact_sha256": str(
            dict(config.get("ml", {}) or {}).get(
                "canonical_v2_alpha_validated_base_sha256", ""
            )
        ),
        "source_base_economic_key_sha256": str(
            dict(config.get("ml", {}) or {}).get(
                "canonical_v2_alpha_validated_base_key_sha256", ""
            )
        ),
        "feature_schema_identity": _feature_schema_identity(),
        "configuration_identity": _alpha_configuration_identity(config),
        "implementation_identity": "canonical_v2_alpha_enrichment.partitioned.v2",
        "source_base_partition_path": source_base_partition_path,
        "source_base_partition_sha256": _file_sha256(Path(source_base_partition_path)) if source_base_partition_path and Path(source_base_partition_path).exists() else "",
        "symbol": symbol.upper(),
    }


def _feature_schema_identity() -> str:
    payload = {
        "engineered": list(ENGINEERED_FEATURE_COLUMNS),
        "enrichment_metadata": list(ENRICHMENT_METADATA_COLUMNS),
        "output_schema": {
            column: {"kind": kind, "nullable": nullable}
            for column, (kind, nullable) in sorted(ALPHA_OUTPUT_SCHEMA.items())
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _alpha_configuration_identity(config: Mapping[str, Any]) -> str:
    ml = dict(config.get("ml", {}) or {})
    payload = {
        "stock_alpha_feature_n_jobs": ml.get("stock_alpha_feature_n_jobs"),
        "stock_level_artifact_format": ml.get("stock_level_artifact_format", "parquet"),
        "stock_level_parquet_compression": ml.get("stock_level_parquet_compression", "zstd"),
        "canonical_v2_labeled_spine_manifest_path": ml.get("canonical_v2_labeled_spine_manifest_path"),
        "canonical_v2_labeled_spine_root": ml.get("canonical_v2_labeled_spine_root"),
        "canonical_v2_alpha_validated_base_sha256": ml.get(
            "canonical_v2_alpha_validated_base_sha256"
        ),
        "canonical_v2_alpha_validated_base_key_sha256": ml.get(
            "canonical_v2_alpha_validated_base_key_sha256"
        ),
        "stooq_parquet_dir": ml.get("stooq_parquet_dir"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _base_partition_identity_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".identity.json")


def _base_partition_identity(path: Path, config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "alpha_enrichment_contract_version": ALPHA_ENRICHMENT_CONTRACT_VERSION,
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": _target_provenance_contract_version(),
        "configuration_identity": _alpha_configuration_identity(config),
        "implementation_identity": "canonical_v2_alpha_enrichment.base_partition.v2",
        "canonical_dataset_hash": (input_resolution or {}).get("canonical_dataset", {}).get("hash", ""),
        "path": str(path),
        "sha256": _file_sha256(path) if path.exists() else "",
    }


def _base_partition_compatible(path: Path, config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None) -> bool:
    if not path.exists():
        return False
    identity_path = _base_partition_identity_path(path)
    if not identity_path.exists():
        return False
    payload = _read_json(identity_path)
    return payload == _base_partition_identity(path, config, input_resolution=input_resolution)


def _failure_record(symbol: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, PartitionBuildError):
        return dict(exc.payload)
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    signature = f"{type(exc).__name__}: {first_line}"
    return {
        "symbol": symbol,
        "phase": "unknown",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": signature,
        "traceback": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fail_fast_settings(ml: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(ml.get("canonical_v2_alpha_fail_fast", {}) or {})
    return {
        "minimum_failures_before_abort": int(raw.get("minimum_failures_before_abort", 3)),
        "abort_when_completed_is_zero": bool(raw.get("abort_when_completed_is_zero", True)),
        "same_failure_signature_threshold": int(raw.get("same_failure_signature_threshold", 3)),
        "maximum_failure_ratio": float(raw.get("maximum_failure_ratio", 0.25)),
    }


def _should_abort_fail_fast(
    failed: Sequence[Mapping[str, Any]],
    *,
    completed: int,
    settings: Mapping[str, Any],
) -> tuple[bool, str, str]:
    if len(failed) < int(settings["minimum_failures_before_abort"]):
        return False, "", ""
    counts: dict[str, int] = {}
    for row in failed:
        signature = str(row.get("failure_signature") or row.get("signature", ""))
        counts[signature] = counts.get(signature, 0) + 1
    dominant, count = max(counts.items(), key=lambda item: item[1])
    if (
        bool(settings["abort_when_completed_is_zero"])
        and completed == 0
        and count >= int(settings["same_failure_signature_threshold"])
    ):
        return True, f"0 completed partitions and {count} identical failures", dominant
    return False, "", dominant


def _progress(
    report_root: Path,
    planned: int,
    completed: int,
    failed: int,
    rows: int,
    started: float,
    *,
    aborted_early: bool = False,
    abort_reason: str = "",
    dominant_failure_signature: str = "",
    tasks_cancelled: int = 0,
    abort_threshold_failure_count: int = 0,
    worker_failure_record_count: int = 0,
    in_flight_at_abort: int = 0,
) -> None:
    elapsed = max(0.001, time.perf_counter() - started)
    payload = {
        "planned_partitions": planned,
        "completed_partitions": completed,
        "pending_partitions": max(0, planned - completed - failed),
        "failed_partitions": failed,
        "elapsed_seconds": elapsed,
        "rows_processed": rows,
        "rows_per_second": rows / elapsed,
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "dominant_failure_signature": dominant_failure_signature,
        "tasks_cancelled": tasks_cancelled,
        "abort_threshold_failure_count": abort_threshold_failure_count,
        "worker_failure_record_count": worker_failure_record_count,
        "in_flight_at_abort": in_flight_at_abort,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(report_root / "progress_manifest.json", payload)
    print(
        "[canonical-v2-alpha] "
        f"completed={completed}/{planned} pending={payload['pending_partitions']} "
        f"failed={failed} rows={rows} elapsed={elapsed:.1f}s rps={payload['rows_per_second']:.1f}",
        flush=True,
    )


def _manifest_summary(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "manifest_path": str(path),
        "status": payload.get("status"),
        "path": payload.get("path"),
        "row_count": payload.get("row_count"),
        "symbol_count": payload.get("symbol_count"),
        "partition_count": payload.get("partition_count"),
        "date_min": payload.get("date_min"),
        "date_max": payload.get("date_max"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan
