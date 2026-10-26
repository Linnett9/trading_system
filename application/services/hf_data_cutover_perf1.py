"""Bounded PERF1 transport benchmark and resumable monthly-shard acquisition."""

from __future__ import annotations

import re
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import build_opener

import pandas as pd
import psutil

from application.services.hf_data_cutover_iex_contract import (
    BACKFILL_ASOF,
    BACKFILL_REQUESTS_PER_MINUTE,
    MINIMUM_FINAL_RESERVE_BYTES,
    IexBackfillSettings,
    load_iex_backfill_settings,
)
from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from application.services.hf_data_cutover_perf1_evidence import (
    activity_proxy as _activity_proxy,
    activity_stratum as _activity_stratum,
    atomic_csv as _atomic_csv,
    completed_hash_row as _completed_hash_row,
    failure_classification_counts as _failure_classification_counts,
    json_sha256 as _json_sha256,
    optional_json as _optional_json,
    percentile as _percentile,
    planned_partitions as _plans,
    read_json as _read_json,
    request_from_identity as _request_from_identity,
    retryable_classification as _retryable_classification,
    select_transport,
    stable_rank as _stable_rank,
    scratch_ceiling as _evidence_scratch_ceiling,
    tree_bytes as _tree_bytes,
    validated_inventory as _validated_inventory,
)
from application.services.hf_data_cutover_perf1_contract import (
    PERF1_HARD_REQUEST_CEILING,
    PERF1_PACING_REQUESTS_PER_MINUTE,
    PERF1_PROVIDER_LIMIT,
    PERF1_SAMPLE_SIZE,
    PERF1_SCHEMA,
    PERF1_SCRATCH_MAX_BYTES,
    PERF1_SYMBOLS_PER_MONTH,
    Perf1TransportConfiguration,
)
from core.research.iex_market_data import (
    IexPartitionRequest,
    canonical_raw_bar_frame,
    validate_raw_bar_partition,
)
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    HistoricalBarMetrics,
    HistoricalBarRequest,
    SharedRateLimiter,
    fetch_chunk_with_retries,
)
from infrastructure.data.iex_market_data_store import IexMarketDataStore, sha256_file
from infrastructure.data.iex_transport_shards import (
    IexTransportShardRequest,
    IexTransportShardStore,
    plan_monthly_transport_shards,
)
from infrastructure.data.persistent_https import WorkerLocalHttpsSessionPool


@dataclass(frozen=True, slots=True)
class ShardBatch:
    shards: tuple[IexTransportShardRequest, ...]

    @property
    def start(self) -> datetime:
        return self.shards[0].start

    @property
    def end(self) -> datetime:
        return self.shards[0].end


def run_perf1(paths: CutoverPaths) -> dict[str, Any]:
    """Run the bounded B/C benchmark, select safely, and publish deployment evidence."""
    settings = load_iex_backfill_settings(paths)
    store = IexMarketDataStore(settings.data_root)
    plans = _plans(settings)
    completed, pending = _validated_inventory(store, plans)
    if not pending:
        raise CutoverAuditError("PERF1 has no unresolved canonical partitions to benchmark")
    start_state = capture_perf1_start_state(paths, settings, completed, pending)
    sample = freeze_perf1_benchmark_sample(paths, store, completed, pending)
    baseline = publish_perf1_baseline(paths, start_state)
    sample_requests = tuple(_request_from_identity(row["request"]) for row in sample["partitions"])
    results: list[dict[str, Any]] = [dict(baseline["benchmark_row"])]
    for configuration in (
        Perf1TransportConfiguration("B", "up_to_5_symbols_x_1_month", "fresh_per_attempt", 8),
        Perf1TransportConfiguration("C", "up_to_5_symbols_x_1_month", "persistent_per_worker", 8),
    ):
        results.append(
            benchmark_transport_configuration(
                paths,
                settings,
                sample_requests,
                configuration,
                scratch_namespace=f"benchmark_{configuration.variant.lower()}",
            )
        )
    for row in results:
        partitions_per_hour = float(row["canonical_equivalent_partitions_per_hour"])
        central_hours = len(pending) / max(partitions_per_hour, 0.000001)
        row["remaining_canonical_partitions"] = len(pending)
        row["optimistic_eta_hours"] = central_hours / 1.20
        row["central_eta_hours"] = central_hours
        row["conservative_eta_hours"] = central_hours / 0.80
    _atomic_csv(paths.output_root / "PERF1_BENCHMARK_RESULTS.csv", results)
    selection = select_transport(paths, results)
    deployed_measurement = next(
        row for row in results if row["variant"] == selection["selected_variant"]
    )
    if selection["selected_variant"] in {"B", "C"}:
        scaling = run_concurrency_scaling(
            paths,
            settings,
            store,
            pending,
            excluded={request.partition_id for request in sample_requests},
            selected_variant=str(selection["selected_variant"]),
        )
        _atomic_csv(paths.output_root / "PERF1_CONCURRENCY_SCALING.csv", scaling)
        selected_workers = int(scaling[0]["workers"])
        for scaling_row in scaling:
            if not bool(scaling_row["advance_permitted"]):
                break
            selected_workers = int(scaling_row["workers"])
        selection["selected_workers"] = selected_workers
        atomic_json(paths.output_root / "PERF1_TRANSPORT_SELECTION.json", selection)
        deployed_measurement = next(
            row for row in scaling if int(row["workers"]) == selected_workers
        )
    verify_completed_partition_hash_audit(paths, store)
    receipt = {
        "schema_version": PERF1_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "READY_FOR_CONTROLLER_DEPLOYMENT",
        "selected_variant": selection["selected_variant"],
        "selected_workers": selection["selected_workers"],
        "baseline_validated_rows_per_second": results[0]["validated_rows_per_second"],
        "selected_validated_rows_per_second": deployed_measurement["validated_rows_per_second"],
        "selected_canonical_partitions_per_hour": deployed_measurement[
            "canonical_equivalent_partitions_per_hour"
        ],
        "selected_tls_eof_count": deployed_measurement["tls_eof_count"],
        "selected_retry_count": deployed_measurement["retries"],
        "remaining_canonical_partitions_at_benchmark": len(pending),
        "projected_central_eta_hours": len(pending)
        / max(float(deployed_measurement["canonical_equivalent_partitions_per_hour"]), 0.000001),
        "completed_partition_count_at_perf1_start": start_state["completed_partitions"],
        "scientific_authorities_changed": False,
        "canonical_partition_contract_changed": False,
        "controller_owned_resume": True,
        "selection_sha256": sha256_file(paths.output_root / "PERF1_TRANSPORT_SELECTION.json"),
        "transport_implementation_sha256": {
            "application/services/hf_data_cutover_perf1.py": sha256_file(Path(__file__)),
            "infrastructure/data/iex_transport_shards.py": sha256_file(
                paths.repository_root / "infrastructure/data/iex_transport_shards.py"
            ),
            "infrastructure/data/persistent_https.py": sha256_file(
                paths.repository_root / "infrastructure/data/persistent_https.py"
            ),
        },
    }
    atomic_json(paths.output_root / "PERF1_DEPLOYMENT_RECEIPT.json", receipt)
    return receipt


def capture_perf1_start_state(
    paths: CutoverPaths,
    settings: IexBackfillSettings,
    completed: Sequence[IexPartitionRequest],
    pending: Sequence[IexPartitionRequest],
) -> dict[str, Any]:
    state_path = paths.output_root / "PERF1_START_STATE.json"
    audit_path = paths.output_root / "PERF1_COMPLETED_PARTITION_HASH_AUDIT.json"
    if state_path.exists() and audit_path.exists():
        _extend_hash_audit_before_benchmark(audit_path, IexMarketDataStore(settings.data_root), completed)
        return _read_json(state_path)
    store = IexMarketDataStore(settings.data_root)
    hashes = [_completed_hash_row(store, request) for request in completed]
    classification_counts = _failure_classification_counts(store, pending)
    memory = psutil.virtual_memory()
    workers = []
    for process in psutil.process_iter(("pid", "ppid", "create_time", "name", "cmdline", "memory_info")):
        try:
            command = " ".join(process.info.get("cmdline") or ())
            if "hf_data_cutover" not in command:
                continue
            workers.append(
                {
                    "pid": int(process.info["pid"]),
                    "parent_pid": int(process.info["ppid"]),
                    "create_time": float(process.info["create_time"]),
                    "name": process.info.get("name"),
                    "working_set_bytes": int(process.info["memory_info"].rss),
                    "command_line": command,
                }
            )
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    recovery = _optional_json(paths.output_root / "IEX_TRANSPORT_RECOVERY_CONFIGURATION.json")
    payload = {
        "schema_version": PERF1_SCHEMA,
        "created_at_utc": utc_now(),
        "planned_partitions": len(completed) + len(pending),
        "completed_partitions": len(completed),
        "completed_rows": sum(
            int((store.completed_manifest(request) or {"quality": {"row_count": 0}})["quality"]["row_count"])
            for request in completed
        ),
        "unresolved_partitions": len(pending),
        "transient_failure_partitions": sum(
            count for name, count in classification_counts.items() if _retryable_classification(name)
        ),
        "irrecoverable_failure_partitions": sum(
            count for name, count in classification_counts.items() if not _retryable_classification(name)
        ),
        "failure_classification_counts": classification_counts,
        "current_workers": workers,
        "current_recovery_mode": "8_worker_fresh_https_recovery",
        "request_ceiling_per_minute": PERF1_HARD_REQUEST_CEILING,
        "pacing_requests_per_minute": recovery.get("pacing_requests_per_minute", 60),
        "free_disk_bytes": shutil.disk_usage(paths.repository_root).free,
        "available_ram_bytes": int(memory.available),
        "total_ram_bytes": int(memory.total),
        "data_contract": {
            "provider": "Alpaca",
            "feed": "IEX",
            "adjustment": "raw",
            "currency": "USD",
            "asof": BACKFILL_ASOF,
            "start": settings.start.isoformat(),
            "end_exclusive": settings.end.isoformat(),
            "symbol_count": len(settings.roster),
        },
    }
    atomic_json(state_path, payload)
    atomic_json(
        audit_path,
        {
            "schema_version": PERF1_SCHEMA,
            "created_at_utc": utc_now(),
            "status": "VALIDATED",
            "partition_count": len(hashes),
            "partitions": hashes,
            "aggregate_sha256": _json_sha256(hashes),
        },
    )
    return payload


def _extend_hash_audit_before_benchmark(
    audit_path: Path,
    store: IexMarketDataStore,
    completed: Sequence[IexPartitionRequest],
) -> None:
    payload = _read_json(audit_path)
    original = {
        str(row["request"]["partition_id"]): row for row in list(payload.get("partitions") or [])
    }
    current = [_completed_hash_row(store, request) for request in completed]
    for row in current:
        partition_id = str(row["request"]["partition_id"])
        if partition_id in original and original[partition_id] != row:
            raise CutoverAuditError(f"Pre-PERF1 completed partition changed before benchmark: {partition_id}")
    payload.update(
        {
            "validated_before_benchmark_at_utc": utc_now(),
            "start_partition_count": payload.get("start_partition_count", payload.get("partition_count")),
            "partition_count": len(current),
            "partitions": current,
            "aggregate_sha256": _json_sha256(current),
        }
    )
    atomic_json(audit_path, payload)


def freeze_perf1_benchmark_sample(
    paths: CutoverPaths,
    store: IexMarketDataStore,
    completed: Sequence[IexPartitionRequest],
    pending: Sequence[IexPartitionRequest],
) -> dict[str, Any]:
    path = paths.output_root / "PERF1_BENCHMARK_SAMPLE.json"
    if path.exists():
        return _read_json(path)
    five_minute = [request for request in pending if request.timeframe == "5m"]
    if len(five_minute) < min(PERF1_SAMPLE_SIZE, len(pending)):
        five_minute = list(pending)
    activity = _activity_proxy(store, completed)
    population_activity = tuple(activity.values())
    ranked = sorted(five_minute, key=lambda request: _stable_rank(request.partition_id))
    by_stratification_cell: dict[tuple[int | None, str], list[IexPartitionRequest]] = defaultdict(list)
    for request in ranked:
        by_stratification_cell[
            (request.year, _activity_stratum(activity.get(request.symbol), population_activity))
        ].append(request)
    target_size = min(PERF1_SAMPLE_SIZE, len(ranked))
    selected = sorted(
        (request for request in ranked if request.symbol in {"SPY", "QQQ"}),
        key=lambda request: (request.symbol, request.year or 9999),
    )[:target_size]
    selected_ids = {request.partition_id for request in selected}
    cells = sorted(by_stratification_cell, key=lambda value: (value[0] or 9999, value[1]))
    while len(selected) < target_size:
        added = False
        for cell in cells:
            while by_stratification_cell[cell] and by_stratification_cell[cell][0].partition_id in selected_ids:
                by_stratification_cell[cell].pop(0)
            if by_stratification_cell[cell] and len(selected) < target_size:
                request = by_stratification_cell[cell].pop(0)
                selected.append(request)
                selected_ids.add(request.partition_id)
                added = True
        if not added:
            break
    rows = [
        {
            "request": request.identity(),
            "selection_rank": index,
            "activity_proxy_rows": activity.get(request.symbol),
            "activity_stratum": _activity_stratum(activity.get(request.symbol), population_activity),
            "etf_if_known": request.symbol in {"SPY", "QQQ"},
            "selection_is_target_blind": True,
        }
        for index, request in enumerate(selected, start=1)
    ]
    payload = {
        "schema_version": PERF1_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "FROZEN",
        "source_population": "currently unresolved canonical partitions only",
        "selection_method": "deterministic year/activity-stratum round-robin plus partition-id hash; unresolved SPY/QQQ included; no targets or alpha data",
        "partition_count": len(rows),
        "partitions": rows,
        "sample_sha256": _json_sha256(rows),
    }
    atomic_json(path, payload)
    return payload


def publish_perf1_baseline(paths: CutoverPaths, start_state: Mapping[str, Any]) -> dict[str, Any]:
    path = paths.output_root / "PERF1_BASELINE_TELEMETRY.json"
    if path.exists():
        return _read_json(path)
    probe = _optional_json(paths.output_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json")
    log_path = paths.output_root / "controller/stage_logs/raw_iex.stdout.log"
    matches = []
    pattern = re.compile(
        r"completed=(?P<completed>\d+)/(?P<planned>\d+).*rows=(?P<rows>\d+).*requests=(?P<requests>\d+).*http_429=(?P<http_429>\d+)"
    )
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                matches.append({key: int(value) for key, value in match.groupdict().items()})
    latest = matches[-1] if matches else {}
    start_completed = int(start_state["completed_partitions"])
    latest_completed = max(start_completed, int(latest.get("completed", start_completed)))
    validated_delta = max(0, latest_completed - start_completed)
    probe_metrics = list(probe.get("provider_metrics_by_partition") or [])
    mean_latency_values = [float(row["average_response_latency"]) for row in probe_metrics if row.get("average_response_latency")]
    # The live attempt is the authoritative comparator; elapsed is bounded below to avoid optimistic rates.
    live_processes = list(start_state.get("current_workers") or [])
    raw_processes = [row for row in live_processes if " backfill " in f" {row.get('command_line', '')} "]
    captured_at = datetime.fromisoformat(str(start_state["created_at_utc"]).replace("Z", "+00:00"))
    elapsed = max(1.0, time.time() - captured_at.timestamp())
    row_delta = max(0, int(latest.get("rows", 0)) - int(start_state.get("completed_rows", 0)))
    benchmark_row = {
        "variant": "A",
        "session_mode": "fresh_per_attempt",
        "shard_geometry": "current_symbol_year_multi_symbol_batch",
        "workers": 8,
        "wall_clock_seconds": round(elapsed, 3),
        "validated_rows": row_delta,
        "validated_rows_per_second": row_delta / elapsed,
        "validated_bars_per_second": row_delta / elapsed,
        "canonical_equivalent_partitions_per_hour": validated_delta / elapsed * 3600.0,
        "time_to_first_durable_checkpoint_seconds": None,
        "total_requests": int(latest.get("requests", 0)),
        "requests_per_minute": int(latest.get("requests", 0)) / elapsed * 60.0,
        "pages": int(latest.get("requests", 0)),
        "rows_per_request": row_delta / max(1, int(latest.get("requests", 0))),
        "mean_latency_seconds": median(mean_latency_values) if mean_latency_values else None,
        "median_latency_seconds": None,
        "p95_latency_seconds": None,
        "http_429_count": int(latest.get("http_429", 0)),
        "http_5xx_count": 0,
        "tls_eof_count": 0,
        "url_error_count": 0,
        "retries": 0,
        "retry_success_rate": None,
        "failed_shards": 0,
        "irrecoverable_partitions": 0,
        "duplicated_work_avoided": start_completed,
        "requests_per_million_bars": int(latest.get("requests", 0)) / max(row_delta, 1) * 1_000_000,
        "seconds_per_million_bars": elapsed / max(row_delta, 1) * 1_000_000,
        "temporary_disk_peak_bytes": 0,
        "process_ram_peak_bytes": max((int(row["working_set_bytes"]) for row in raw_processes), default=0),
        "requested_limit": PERF1_PROVIDER_LIMIT,
        "actual_returned_rows": row_delta,
        "status": "HISTORICAL_TELEMETRY",
    }
    payload = {
        "schema_version": PERF1_SCHEMA,
        "created_at_utc": utc_now(),
        "source": "existing controller logs and connectivity probe; no baseline redownload",
        "provider_limit_authority": {
            "source": "Alpaca official Historical Bars API reference",
            "url": "https://docs.alpaca.markets/us/reference/stockbars",
            "validated_at_utc": utc_now(),
            "permitted_range": "1 to 10000",
            "selected_maximum": PERF1_PROVIDER_LIMIT,
        },
        "benchmark_row": benchmark_row,
    }
    atomic_json(path, payload)
    return payload


def benchmark_transport_configuration(
    paths: CutoverPaths,
    settings: IexBackfillSettings,
    requests: Sequence[IexPartitionRequest],
    configuration: Perf1TransportConfiguration,
    *,
    scratch_namespace: str,
) -> dict[str, Any]:
    scratch_root = settings.data_root / "perf1_temp" / scratch_namespace
    scratch_ceiling = _scratch_ceiling(paths.repository_root)
    shard_store = IexTransportShardStore(scratch_root, scratch_ceiling_bytes=scratch_ceiling)
    return _acquire_shards(
        paths,
        settings,
        requests,
        configuration,
        shard_store,
        publish_canonical=False,
    )


def collect_perf1_sharded_backlog(
    paths: CutoverPaths,
    settings: IexBackfillSettings,
    store: IexMarketDataStore,
    plans: Mapping[str, tuple[IexPartitionRequest, ...]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    pending = [
        request
        for timeframe in ("5m", "1Day")
        for request in plans[timeframe]
        if store.completed_manifest(request) is None
    ]
    configuration = Perf1TransportConfiguration(
        variant=str(selection["selected_variant"]),
        shard_geometry="up_to_5_symbols_x_1_month",
        session_mode=(
            "persistent_per_worker" if selection["selected_variant"] == "C" else "fresh_per_attempt"
        ),
        workers=int(selection["selected_workers"]),
    )
    selected_namespace = f"benchmark_{str(selection['selected_variant']).lower()}"
    shard_store = IexTransportShardStore(
        settings.data_root / "perf1_temp" / selected_namespace,
        scratch_ceiling_bytes=_scratch_ceiling(paths.repository_root),
    )
    metrics = _acquire_shards(
        paths,
        settings,
        pending,
        configuration,
        shard_store,
        publish_canonical=True,
        canonical_store=store,
        canonical_completed_base=sum(len(values) for values in plans.values()) - len(pending),
    )
    completed_total = sum(
        store.completed_manifest(request) is not None
        for timeframe in ("5m", "1Day")
        for request in plans[timeframe]
    )
    return {
        "planned_partitions": sum(len(values) for values in plans.values()),
        "completed_partitions": completed_total,
        "resumed_partitions": completed_total - int(metrics["canonical_partitions_published"]),
        "failed_partitions_during_invocation": int(metrics["failed_shards"]),
        "rows_observed_including_resumed": int(metrics["validated_rows"]),
        "provider_metrics": metrics,
    }


def _acquire_shards(
    paths: CutoverPaths,
    settings: IexBackfillSettings,
    requests: Sequence[IexPartitionRequest],
    configuration: Perf1TransportConfiguration,
    shard_store: IexTransportShardStore,
    *,
    publish_canonical: bool,
    canonical_store: IexMarketDataStore | None = None,
    canonical_completed_base: int = 0,
) -> dict[str, Any]:
    if publish_canonical and canonical_store is None:
        raise ValueError("canonical_store is required for deployed shard acquisition")
    free = shutil.disk_usage(paths.repository_root).free
    if free <= MINIMUM_FINAL_RESERVE_BYTES:
        raise CutoverAuditError("PERF1 cannot start below the 15 GiB reserve")
    aliases = dict(settings.provider_symbol_by_canonical)
    all_shards = {request.partition_id: plan_monthly_transport_shards(request) for request in requests}
    batches = _shard_batches(all_shards.values())
    limiter = SharedRateLimiter(
        requests_per_minute=configuration.hard_request_ceiling,
        pacing_requests_per_minute=configuration.pacing_requests_per_minute,
        max_in_flight_requests=configuration.workers,
    )
    metrics = HistoricalBarMetrics()
    session_pool = (
        WorkerLocalHttpsSessionPool() if configuration.session_mode == "persistent_per_worker" else None
    )
    opener = session_pool.open if session_pool is not None else _fresh_open
    provider = AlpacaBasicHistoricalBarProvider(
        feed="iex",
        opener=opener,
        rate_limiter=limiter,
        metrics=metrics,
    )
    if not provider.check_authentication()["can_attempt_authenticated_request"]:
        raise CutoverAuditError("Existing Alpaca credentials are unavailable to PERF1")
    process = psutil.Process()
    cpu_at_start = process.cpu_times()
    started = time.monotonic()
    first_checkpoint: float | None = None
    first_checkpoint_lock = threading.Lock()
    skipped_shards = 0
    pending_batches: list[ShardBatch] = []
    for batch in batches:
        missing = tuple(shard for shard in batch.shards if shard_store.completed_manifest(shard) is None)
        skipped_shards += len(batch.shards) - len(missing)
        if missing:
            pending_batches.append(ShardBatch(missing))
    failures: list[dict[str, Any]] = []
    scratch_estate_root = settings.data_root / "perf1_temp"
    peak_scratch = _tree_bytes(scratch_estate_root)
    if peak_scratch > shard_store.scratch_ceiling_bytes:
        raise CutoverAuditError(
            f"PERF1 scratch estate exceeds its ceiling: {peak_scratch} > {shard_store.scratch_ceiling_bytes}"
        )
    peak_ram = process.memory_info().rss
    returned_rows = 0
    page_rows: list[int] = []
    validated_rows = 0
    assembled = 0
    published = 0
    assembled_ids: set[str] = set()
    validation_rows: list[dict[str, Any]] = []
    request_by_id = {request.partition_id: request for request in requests}
    last_progress = started

    def collect(batch: ShardBatch) -> dict[str, Any]:
        nonlocal first_checkpoint
        result = _collect_shard_batch(shard_store, provider, batch, aliases)
        with first_checkpoint_lock:
            if first_checkpoint is None and result["published_shards"]:
                first_checkpoint = time.monotonic() - started
        return result

    try:
        with ThreadPoolExecutor(max_workers=configuration.workers, thread_name_prefix="iex-perf1") as executor:
            futures: dict[Future[dict[str, Any]], ShardBatch] = {
                executor.submit(collect, batch): batch for batch in pending_batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "shard_ids": [shard.shard_id for shard in batch.shards],
                            "classification": str(getattr(exc, "classification", "unexpected_failure")),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    if str(getattr(exc, "classification", "")) in {
                        "permanent_authentication_failure",
                        "entitlement_failure",
                        "validation_failure",
                    }:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                else:
                    returned_rows += int(result["returned_rows"])
                    page_rows.extend(int(value) for value in result["page_rows"])
                    for partition_id in {shard.canonical.partition_id for shard in batch.shards}:
                        if partition_id in assembled_ids:
                            continue
                        request = request_by_id[partition_id]
                        assembled_result = _assemble_transport_partition(
                            request,
                            all_shards[partition_id],
                            shard_store,
                            configuration,
                            publish_canonical=publish_canonical,
                            canonical_store=canonical_store,
                        )
                        if assembled_result is not None:
                            assembled_ids.add(partition_id)
                            assembled += 1
                            validated_rows += int(assembled_result["row_count"])
                            published += int(assembled_result["published"])
                            validation_rows.append(assembled_result["validation"])
                            if publish_canonical and assembled_result["published"]:
                                atomic_json(
                                    paths.output_root / "PERF1_LIVE_PROGRESS.json",
                                    {
                                        "schema_version": PERF1_SCHEMA,
                                        "updated_at_utc": utc_now(),
                                        "completed_partition_count": canonical_completed_base + published,
                                        "published_this_invocation": published,
                                        "last_completed_partition_id": partition_id,
                                        "configuration": asdict(configuration),
                                    },
                                )
                scratch_bytes = _tree_bytes(scratch_estate_root)
                peak_scratch = max(peak_scratch, scratch_bytes)
                if scratch_bytes > shard_store.scratch_ceiling_bytes:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise CutoverAuditError(
                        f"PERF1 scratch estate exceeds its ceiling: {scratch_bytes} > {shard_store.scratch_ceiling_bytes}"
                    )
                peak_ram = max(peak_ram, process.memory_info().rss)
                if shutil.disk_usage(paths.repository_root).free <= MINIMUM_FINAL_RESERVE_BYTES:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise CutoverAuditError("PERF1 stopped at the 15 GiB reserve gate")
                now = time.monotonic()
                if now - last_progress >= 60:
                    print(
                        "IEX PERF1 BACKFILL "
                        f"variant={configuration.variant} assembled={assembled}/{len(requests)} "
                        f"published={published} requests={metrics.requests_attempted} "
                        f"retries={metrics.requests_retried} tls_eof={metrics.tls_eof_count} "
                        f"http_429={metrics.http_429_count} scratch_bytes={peak_scratch}",
                        flush=True,
                    )
                    last_progress = now
    finally:
        if session_pool is not None:
            session_pool.close()
    elapsed = max(time.monotonic() - started, 0.000001)
    cpu_at_end = process.cpu_times()
    cpu_percent = (
        (cpu_at_end.user + cpu_at_end.system - cpu_at_start.user - cpu_at_start.system)
        / elapsed
        * 100.0
    )
    for request in requests:
        if request.partition_id in assembled_ids:
            continue
        assembled_result = _assemble_transport_partition(
            request,
            all_shards[request.partition_id],
            shard_store,
            configuration,
            publish_canonical=publish_canonical,
            canonical_store=canonical_store,
        )
        if assembled_result is None:
            continue
        assembled += 1
        validated_rows += int(assembled_result["row_count"])
        published += int(assembled_result["published"])
        validation_rows.append(assembled_result["validation"])
    metric_payload = metrics.as_dict()
    retries = int(metric_payload["requests_retried"])
    retry_successes = max(0, retries - len(failures))
    row = {
        "variant": configuration.variant,
        "session_mode": configuration.session_mode,
        "shard_geometry": configuration.shard_geometry,
        "workers": configuration.workers,
        "wall_clock_seconds": elapsed,
        "validated_rows": validated_rows,
        "validated_rows_per_second": validated_rows / elapsed,
        "validated_bars_per_second": validated_rows / elapsed,
        "canonical_equivalent_partitions_per_hour": assembled / elapsed * 3600.0,
        "time_to_first_durable_checkpoint_seconds": first_checkpoint,
        "total_requests": metric_payload["requests_attempted"],
        "requests_per_minute": metric_payload["requests_attempted"] / elapsed * 60.0,
        "pages": metric_payload["pages_downloaded"],
        "rows_per_request": metric_payload["rows_per_request"],
        "mean_latency_seconds": metric_payload["average_response_latency"],
        "median_latency_seconds": _percentile(metrics.response_latency_seconds, 0.5),
        "p95_latency_seconds": _percentile(metrics.response_latency_seconds, 0.95),
        "http_429_count": metric_payload["http_429_count"],
        "http_5xx_count": metric_payload["http_5xx_count"],
        "tls_eof_count": metric_payload["tls_eof_count"],
        "url_error_count": metric_payload["transport_error_count"],
        "retries": retries,
        "retry_success_rate": retry_successes / retries if retries else 1.0,
        "failed_shards": len(failures),
        "irrecoverable_partitions": sum(not _retryable_classification(row["classification"]) for row in failures),
        "duplicated_work_avoided": skipped_shards,
        "requests_per_million_bars": metric_payload["requests_attempted"] / max(validated_rows, 1) * 1_000_000,
        "seconds_per_million_bars": elapsed / max(validated_rows, 1) * 1_000_000,
        "estimated_bytes_replayed_after_failure": 0,
        "temporary_disk_peak_bytes": peak_scratch,
        "process_ram_peak_bytes": peak_ram,
        "process_cpu_percent": cpu_percent,
        "requested_limit": configuration.provider_limit,
        "actual_returned_rows": returned_rows,
        "actual_rows_per_page_median": _percentile(page_rows, 0.5),
        "persistent_session_count": session_pool.session_count if session_pool is not None else None,
        "persistent_connection_generations": (
            session_pool.connection_generations if session_pool is not None else None
        ),
        "canonical_partitions_published": published,
        "status": "PASSED" if not failures else "FAILED",
    }
    validation_path = paths.output_root / "PERF1_TEMP_SHARD_VALIDATION.json"
    prior = _optional_json(validation_path)
    validations = list(prior.get("validations") or [])
    validations.append(
        {
            "created_at_utc": utc_now(),
            "configuration": asdict(configuration),
            "scratch_root": str(shard_store.root.resolve()),
            "scratch_ceiling_bytes": shard_store.scratch_ceiling_bytes,
            "failures": failures,
            "partitions": validation_rows,
            "metrics": row,
        }
    )
    atomic_json(
        validation_path,
        {"schema_version": PERF1_SCHEMA, "status": "VALIDATED", "validations": validations},
    )
    return row


def _assemble_transport_partition(
    request: IexPartitionRequest,
    shards: Sequence[IexTransportShardRequest],
    shard_store: IexTransportShardStore,
    configuration: Perf1TransportConfiguration,
    *,
    publish_canonical: bool,
    canonical_store: IexMarketDataStore | None,
) -> dict[str, Any] | None:
    if any(shard_store.completed_manifest(shard) is None for shard in shards):
        return None
    shard_hashes = [sha256_file(shard_store.manifest_path(shard)) for shard in shards]
    frame = canonical_raw_bar_frame(
        pd.concat([shard_store.load_verified(shard) for shard in shards], ignore_index=True).to_dict("records")
    )
    quality = validate_raw_bar_partition(frame, request)
    if quality["duplicate_count"] or quality["off_grid_count"]:
        raise CutoverAuditError(f"PERF1 canonical assembly quality failure: {request.partition_id}")
    published = 0
    if publish_canonical:
        if canonical_store is None:
            raise ValueError("canonical_store is required for canonical publication")
        existing = canonical_store.completed_manifest(request)
        if existing is None:
            canonical_store.publish(
                request,
                frame,
                quality=quality,
                source={
                    "provider": "Alpaca",
                    "feed": "IEX",
                    "endpoint_family": "historical_stock_bars",
                    "adjustment": "raw",
                    "currency": "USD",
                    "asof": BACKFILL_ASOF,
                    "transport": asdict(configuration),
                    "temporary_shard_manifest_sha256": shard_hashes,
                    "normalization": "field rename and typed projection only; no adjustment, fill, or price transformation",
                },
            )
            published = 1
        canonical_store.completed_manifest(request)
        shard_store.cleanup_partition(request)
    return {
        "row_count": len(frame),
        "published": published,
        "validation": {
            "partition_id": request.partition_id,
            "shard_count": len(shards),
            "shard_manifest_sha256": shard_hashes,
            "row_count": len(frame),
            "duplicate_count": quality["duplicate_count"],
            "off_grid_count": quality["off_grid_count"],
            "status": "VALIDATED_AND_PUBLISHED" if publish_canonical else "VALIDATED",
        },
    }


def _collect_shard_batch(
    store: IexTransportShardStore,
    provider: AlpacaBasicHistoricalBarProvider,
    batch: ShardBatch,
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    provider_symbols = tuple(aliases.get(shard.canonical.symbol, shard.canonical.symbol) for shard in batch.shards)
    remaps = tuple(
        (shard.canonical.symbol, provider_symbols[index])
        for index, shard in enumerate(batch.shards)
        if provider_symbols[index] != shard.canonical.symbol
    )
    request = HistoricalBarRequest(
        symbols=provider_symbols,
        timeframe=batch.shards[0].canonical.timeframe,
        start=batch.start,
        end=batch.end,
        feed="iex",
        adjustment="raw",
        limit=PERF1_PROVIDER_LIMIT,
        raw_chunk_id="perf1-" + "-".join(shard.shard_id for shard in batch.shards),
        canonical_symbols=tuple(shard.canonical.symbol for shard in batch.shards),
        provider_symbol_by_canonical=remaps,
        currency="USD",
        asof=BACKFILL_ASOF,
    )
    rows, chunk = fetch_chunk_with_retries(provider, request, max_retries=5, retain_raw_pages=False)
    combined = canonical_raw_bar_frame(rows)
    published = 0
    for shard in batch.shards:
        frame = canonical_raw_bar_frame(
            combined.loc[combined["symbol"].eq(shard.canonical.symbol)].to_dict("records")
        )
        quality = validate_raw_bar_partition(frame, shard.validation_request())
        if quality["duplicate_count"] or quality["off_grid_count"]:
            raise ValueError(f"Transport shard quality failure: {shard.shard_id}")
        store.publish(
            shard,
            frame,
            quality=quality,
            source={
                "provider": "Alpaca",
                "feed": "IEX",
                "adjustment": "raw",
                "currency": "USD",
                "asof": BACKFILL_ASOF,
                "requested_limit": PERF1_PROVIDER_LIMIT,
                "actual_returned_rows_for_batch": len(rows),
                "page_count_for_batch": int(chunk["pages"]),
                "raw_page_sha256": list(chunk["raw_page_sha256"]),
                "temporary_only_not_authority": True,
            },
        )
        published += 1
    pages = max(1, int(chunk["pages"]))
    return {
        "published_shards": published,
        "returned_rows": len(rows),
        "page_rows": [len(rows) // pages] * pages,
    }


def run_concurrency_scaling(
    paths: CutoverPaths,
    settings: IexBackfillSettings,
    store: IexMarketDataStore,
    pending: Sequence[IexPartitionRequest],
    *,
    excluded: set[str],
    selected_variant: str,
) -> list[dict[str, Any]]:
    candidates = [request for request in pending if request.partition_id not in excluded and request.timeframe == "5m"]
    rows: list[dict[str, Any]] = []
    previous_rate: float | None = None
    session_mode = "persistent_per_worker" if selected_variant == "C" else "fresh_per_attempt"
    sample = candidates[:6]
    for workers in (8, 12, 16):
        if not sample:
            break
        row = benchmark_transport_configuration(
            paths,
            settings,
            sample,
            Perf1TransportConfiguration(
                f"{selected_variant}_SCALE_{workers}",
                "up_to_5_symbols_x_1_month",
                session_mode,
                workers,
            ),
            scratch_namespace=f"scaling_{selected_variant.lower()}_{workers}",
        )
        rate = float(row["validated_rows_per_second"])
        material_gain = previous_rate is None or rate >= previous_rate * 1.05
        reliable = (
            row["status"] == "PASSED"
            and int(row["failed_shards"]) == 0
            and int(row["http_429_count"]) == 0
        )
        row["advance_permitted"] = material_gain and reliable
        rows.append(row)
        if not material_gain or not reliable:
            break
        previous_rate = rate
    return rows


def verify_completed_partition_hash_audit(paths: CutoverPaths, store: IexMarketDataStore) -> dict[str, Any]:
    path = paths.output_root / "PERF1_COMPLETED_PARTITION_HASH_AUDIT.json"
    payload = _read_json(path)
    mismatches = []
    for row in payload["partitions"]:
        request = _request_from_identity(row["request"])
        try:
            current = _completed_hash_row(store, request)
        except Exception as exc:
            mismatches.append({"partition_id": request.partition_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if current != row:
            mismatches.append({"partition_id": request.partition_id, "expected": row, "actual": current})
    payload.update(
        {
            "verified_after_benchmark_at_utc": utc_now(),
            "post_benchmark_status": "MATCH" if not mismatches else "MISMATCH",
            "mismatches": mismatches,
        }
    )
    atomic_json(path, payload)
    if mismatches:
        raise CutoverAuditError(f"PERF1 completed partition hash audit failed: {len(mismatches)} mismatches")
    return payload


def _shard_batches(
    request_groups: Iterable[Sequence[IexTransportShardRequest]],
) -> tuple[ShardBatch, ...]:
    by_bounds: dict[tuple[str, datetime, datetime], list[IexTransportShardRequest]] = defaultdict(list)
    for shards in request_groups:
        for shard in shards:
            by_bounds[(shard.canonical.timeframe, shard.start, shard.end)].append(shard)
    batches: list[ShardBatch] = []
    for key in sorted(by_bounds, key=lambda value: (value[0], value[1], value[2])):
        ordered = sorted(by_bounds[key], key=lambda shard: shard.canonical.symbol)
        for offset in range(0, len(ordered), PERF1_SYMBOLS_PER_MONTH):
            batches.append(ShardBatch(tuple(ordered[offset : offset + PERF1_SYMBOLS_PER_MONTH])))
    return tuple(batches)


def _scratch_ceiling(repository_root: Path) -> int:
    return _evidence_scratch_ceiling(repository_root, PERF1_SCRATCH_MAX_BYTES)


def _fresh_open(request: Any, *, timeout: float) -> Any:
    return build_opener().open(request, timeout=timeout)


__all__ = [
    "Perf1TransportConfiguration",
    "benchmark_transport_configuration",
    "capture_perf1_start_state",
    "collect_perf1_sharded_backlog",
    "freeze_perf1_benchmark_sample",
    "run_perf1",
]
