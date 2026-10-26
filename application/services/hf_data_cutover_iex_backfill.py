"""Application service for the resumable canonical Alpaca IEX raw backfill."""

from __future__ import annotations

import json
import hashlib
import shutil
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import build_opener

from application.services.hf_data_cutover_service import CutoverAuditError, CutoverPaths, atomic_json, utc_now
from application.services.hf_data_cutover_iex_contract import (
    BACKFILL_ASOF,
    BACKFILL_PACING_REQUESTS_PER_MINUTE,
    BACKFILL_REQUESTS_PER_MINUTE,
    BACKFILL_WORKERS,
    INITIAL_VALIDATED_BACKFILL_WORKERS,
    IexBackfillSettings,
    MINIMUM_FINAL_RESERVE_BYTES,
    TRANSPORT_PROBE_PARTITIONS,
    TRANSPORT_PROBE_WORKERS,
    alpaca_symbol_aliases,
    load_iex_backfill_settings,
)
from core.research.iex_market_data import (
    IexPartitionRequest,
    canonical_raw_bar_frame,
    plan_iex_partitions,
    validate_raw_bar_partition,
)
from infrastructure.data.historical_bar_providers import (
    AlpacaBasicHistoricalBarProvider,
    AlpacaHistoricalBarError,
    HistoricalBarRequest,
    SharedRateLimiter,
    fetch_chunk_with_retries,
)
from infrastructure.data.iex_market_data_store import IexMarketDataStore, sha256_file
from infrastructure.data.market_data_cutover_audit import resource_state


FIVE_MINUTE_RETRIEVAL_BATCH_SIZE = 10
DAILY_RETRIEVAL_BATCH_SIZE = 25


class RawBackfillDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    RESUMABLE_INCOMPLETE = "RESUMABLE_INCOMPLETE"
    IRRECOVERABLE_FAILURE = "IRRECOVERABLE_FAILURE"


_RETRYABLE_FAILURE_CLASSIFICATIONS = frozenset(
    {
        "missing",
        "missing_unattempted",
        "retryable_failure",
        "transient_network_failure",
        "transient_provider_failure",
        "rate_limit_failure",
        "timeout",
        "connection_failure",
        "network_failure",
    }
)
_RETRYABLE_ERROR_TYPES = frozenset({"urlerror", "timeouterror", "connectionerror"})
_RETRYABLE_ERROR_MARKERS = (
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "connection reset",
    "remote end closed",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
    "retry-after",
)


def run_iex_raw_backfill(paths: CutoverPaths) -> dict[str, Any]:
    """Acquire the frozen IEX raw 5m and daily authorities, resuming safely."""

    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    allowed_states = {
        "IEX_COVERAGE_PROBE_COMPLETE_FULL_BUILD_PENDING",
        "IEX_RAW_BACKFILL_IN_PROGRESS",
        "IEX_RAW_DATA_INCOMPLETE",
    }
    if run_manifest.get("state") not in allowed_states:
        raise CutoverAuditError(
            f"IEX raw backfill requires a completed coverage probe, found {run_manifest.get('state')}"
        )
    settings = load_iex_backfill_settings(paths)
    _preflight(paths, settings)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    store = IexMarketDataStore(settings.data_root)
    provider = AlpacaBasicHistoricalBarProvider(
        feed="iex",
        opener=_fresh_https_open,
        rate_limiter=SharedRateLimiter(
            requests_per_minute=BACKFILL_REQUESTS_PER_MINUTE,
            pacing_requests_per_minute=BACKFILL_PACING_REQUESTS_PER_MINUTE,
            max_in_flight_requests=BACKFILL_WORKERS,
        ),
    )
    auth = provider.check_authentication()
    if not auth["can_attempt_authenticated_request"]:
        raise CutoverAuditError("Existing Alpaca credentials are unavailable to the IEX backfill")
    plans = {
        "5m": plan_iex_partitions(
            settings.roster,
            start=settings.start,
            end=settings.end,
            timeframe="5m",
        ),
        "1Day": plan_iex_partitions(
            settings.roster,
            start=settings.start,
            end=settings.end,
            timeframe="1Day",
        ),
    }
    plan_identity = _plan_identity(plans)
    previous_plan = (run_manifest.get("iex_raw_backfill") or {}).get("plan_identity")
    if previous_plan and previous_plan != plan_identity:
        raise CutoverAuditError("IEX backfill plan changed after execution started")
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_RAW_BACKFILL_IN_PROGRESS",
            "iex_raw_backfill": {
                **dict(run_manifest.get("iex_raw_backfill") or {}),
                "plan_identity": plan_identity,
                "start": settings.start.isoformat(),
                "end_exclusive": settings.end.isoformat(),
                "symbol_count": len(settings.roster),
                "planned_5m_partitions": len(plans["5m"]),
                "planned_daily_partitions": len(plans["1Day"]),
                "data_root": str(settings.data_root.resolve()),
                "requests_per_minute": BACKFILL_REQUESTS_PER_MINUTE,
                "pacing_requests_per_minute": BACKFILL_PACING_REQUESTS_PER_MINUTE,
                "max_in_flight_requests": BACKFILL_WORKERS,
                "initial_validated_max_in_flight_requests": INITIAL_VALIDATED_BACKFILL_WORKERS,
                "transport_recovery_max_in_flight_requests": BACKFILL_WORKERS,
                "minimum_final_reserve_bytes": MINIMUM_FINAL_RESERVE_BYTES,
            },
            "next_gate": "Complete and validate all raw IEX partitions.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)

    selection_path = paths.output_root / "PERF1_TRANSPORT_SELECTION.json"
    deployment_path = paths.output_root / "PERF1_DEPLOYMENT_RECEIPT.json"
    selection = _read_json(selection_path) if selection_path.exists() else {}
    deployment = _read_json(deployment_path) if deployment_path.exists() else {}
    perf1_selected = (
        selection.get("selected_variant") in {"B", "C"}
        and deployment.get("status") == "READY_FOR_CONTROLLER_DEPLOYMENT"
    )
    if perf1_selected:
        from application.services.hf_data_cutover_perf1 import collect_perf1_sharded_backlog

        summary = collect_perf1_sharded_backlog(paths, settings, store, plans, selection)
    else:
        summary = _collect_all(
            store=store,
            provider=provider,
            plans=plans,
            provider_symbol_map=dict(settings.provider_symbol_by_canonical),
        )
    run_manifest, assessment = _publish_raw_invocation(
        paths=paths,
        store=store,
        plans=plans,
        provider_metrics=dict(summary["provider_metrics"]),
        summary=summary,
        invocation_kind="PERF1_SHARDED_BACKLOG" if perf1_selected else "BACKLOG",
    )
    complete = assessment["disposition"] == RawBackfillDisposition.COMPLETE.value
    if not complete:
        unresolved = assessment["unresolved_partition_count"]
        raise CutoverAuditError(
            "IEX raw backfill "
            f"{assessment['disposition'].lower()}: {unresolved} unresolved partitions "
            f"({assessment['resumable_incomplete_partition_count']} resumable, "
            f"{assessment['irrecoverable_failure_partition_count']} irrecoverable); "
            "rerun preserves completed work"
        )
    return run_manifest


def run_iex_transport_connectivity_probe(paths: CutoverPaths) -> dict[str, Any]:
    """Probe three incomplete partitions sequentially before recovery backlog work."""
    settings = load_iex_backfill_settings(paths)
    _preflight(paths, settings)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    store = IexMarketDataStore(settings.data_root)
    plans = {
        "5m": plan_iex_partitions(settings.roster, start=settings.start, end=settings.end, timeframe="5m"),
        "1Day": plan_iex_partitions(settings.roster, start=settings.start, end=settings.end, timeframe="1Day"),
    }
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    previous_plan = (run_manifest.get("iex_raw_backfill") or {}).get("plan_identity")
    plan_identity = _plan_identity(plans)
    if previous_plan and previous_plan != plan_identity:
        raise CutoverAuditError("IEX transport probe plan differs from the frozen raw acquisition plan")

    completed, pending = _validated_partition_inventory(store, plans)
    baseline_hash = _completed_manifest_set_sha256(store, completed)
    selected = tuple(sorted(pending, key=lambda request: request.partition_id)[:TRANSPORT_PROBE_PARTITIONS])
    if not selected:
        report = {
            "schema_version": "hf_data_cutover_r1.iex_transport_connectivity_probe.v1",
            "created_at_utc": utc_now(),
            "status": "NOT_REQUIRED_COMPLETE",
            "probe_workers": TRANSPORT_PROBE_WORKERS,
            "selected_partition_count": 0,
            "completed_partition_count_before": len(completed),
            "completed_partition_count_after": len(completed),
            "completed_manifest_set_sha256_before": baseline_hash,
            "completed_manifest_set_sha256_after": baseline_hash,
            "results": [],
        }
        atomic_json(paths.output_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json", report)
        return report
    if len(selected) < TRANSPORT_PROBE_PARTITIONS:
        raise CutoverAuditError(
            f"Transport probe requires {TRANSPORT_PROBE_PARTITIONS} incomplete partitions; found {len(selected)}"
        )

    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_TRANSPORT_PROBE_IN_PROGRESS",
            "next_gate": "Establish sequential IEX TLS connectivity before backlog recovery.",
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    aliases = dict(settings.provider_symbol_by_canonical)
    results: list[dict[str, Any]] = []
    provider_metrics: list[dict[str, Any]] = []
    successful = 0
    irrecoverable = 0
    for request in selected:
        if store.completed_manifest(request) is not None:
            raise CutoverAuditError(f"Transport probe refused already-complete partition: {request.partition_id}")
        provider = AlpacaBasicHistoricalBarProvider(
            feed="iex",
            opener=_fresh_https_open,
            rate_limiter=SharedRateLimiter(
                requests_per_minute=BACKFILL_REQUESTS_PER_MINUTE,
                pacing_requests_per_minute=BACKFILL_PACING_REQUESTS_PER_MINUTE,
                max_in_flight_requests=TRANSPORT_PROBE_WORKERS,
            ),
        )
        auth = provider.check_authentication()
        if not auth["can_attempt_authenticated_request"]:
            raise CutoverAuditError("Existing Alpaca credentials are unavailable to the IEX transport probe")
        try:
            outcome = _collect_batch(
                store=store,
                provider=provider,
                requests=(request,),
                provider_symbol_map=aliases,
            )[0][1]
            if isinstance(outcome, Exception):
                raise outcome
        except Exception as exc:
            classification = _exception_classification(exc)
            if classification not in _RETRYABLE_FAILURE_CLASSIFICATIONS:
                irrecoverable += 1
            failure = {
                "failed_at_utc": utc_now(),
                "classification": classification,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            store.write_failure(request, failure)
            results.append({"partition_id": request.partition_id, "status": "FAILED", **failure})
        else:
            successful += 1
            results.append(
                {
                    "partition_id": request.partition_id,
                    "status": "COMPLETE",
                    "parquet_sha256": outcome["parquet_sha256"],
                    "row_count": outcome["quality"]["row_count"],
                }
            )
        provider_metrics.append(provider.metrics.as_dict())

    completed_after, _pending_after = _validated_partition_inventory(store, plans)
    after_hash = _completed_manifest_set_sha256(store, completed_after)
    status = (
        "FAILED_IRRECOVERABLE"
        if irrecoverable
        else "PASSED"
        if successful
        else "IEX_TRANSPORT_CONNECTIVITY_BLOCKED"
    )
    report = {
        "schema_version": "hf_data_cutover_r1.iex_transport_connectivity_probe.v1",
        "created_at_utc": utc_now(),
        "status": status,
        "contract": {"feed": "IEX", "adjustment": "raw", "asof": BACKFILL_ASOF},
        "probe_workers": TRANSPORT_PROBE_WORKERS,
        "request_ceiling_per_minute": BACKFILL_REQUESTS_PER_MINUTE,
        "pacing_target_per_minute": BACKFILL_PACING_REQUESTS_PER_MINUTE,
        "fresh_https_opener_per_attempt": True,
        "bounded_retry": {"maximum_retries_per_request": 5, "backoff": "exponential_with_jitter"},
        "selected_partition_count": len(selected),
        "completed_partition_count_before": len(completed),
        "completed_partition_count_after": len(completed_after),
        "completed_manifest_set_sha256_before": baseline_hash,
        "completed_manifest_set_sha256_after": after_hash,
        "successful_partition_count": successful,
        "irrecoverable_failure_count": irrecoverable,
        "results": results,
        "provider_metrics_by_partition": provider_metrics,
    }
    report_path = paths.output_root / "IEX_TRANSPORT_CONNECTIVITY_PROBE.json"
    atomic_json(report_path, report)
    summary = {
        "planned_partitions": sum(len(requests) for requests in plans.values()),
        "completed_partitions": len(completed_after),
        "resumed_partitions": len(completed),
        "probe_partitions": len(selected),
        "probe_successful_partitions": successful,
        "rows_observed_including_resumed": 0,
        "provider_metrics": {"probe_metrics_by_partition": provider_metrics},
    }
    _publish_raw_invocation(
        paths=paths,
        store=store,
        plans=plans,
        provider_metrics={"probe_metrics_by_partition": provider_metrics},
        summary=summary,
        invocation_kind="TRANSPORT_PROBE",
    )
    run_manifest = _read_json(run_manifest_path)
    run_manifest["iex_transport_probe"] = {
        "status": status,
        "report_sha256": sha256_file(report_path),
        "completed_partition_count_before": len(completed),
        "completed_partition_count_after": len(completed_after),
    }
    if status == "IEX_TRANSPORT_CONNECTIVITY_BLOCKED":
        run_manifest["state"] = "IEX_TRANSPORT_CONNECTIVITY_BLOCKED"
        run_manifest["next_gate"] = "Transport connectivity must recover before another raw backlog attempt."
    elif status == "FAILED_IRRECOVERABLE":
        run_manifest["state"] = "IEX_RAW_DATA_INCOMPLETE"
        run_manifest["next_gate"] = "Repair the irrecoverable transport-probe failure."
    atomic_json(run_manifest_path, run_manifest)
    if status != "PASSED":
        raise CutoverAuditError(f"IEX transport connectivity probe stopped with status {status}")
    return report


def _publish_raw_invocation(
    *,
    paths: CutoverPaths,
    store: IexMarketDataStore,
    plans: Mapping[str, tuple[IexPartitionRequest, ...]],
    provider_metrics: Mapping[str, Any],
    summary: Mapping[str, Any],
    invocation_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    five_manifest = _dataset_manifest(
        store=store,
        requests=plans["5m"],
        dataset="raw_5m",
        provider_metrics=provider_metrics,
    )
    daily_manifest = _dataset_manifest(
        store=store,
        requests=plans["1Day"],
        dataset="raw_daily",
        provider_metrics=provider_metrics,
    )
    five_path = paths.output_root / "IEX_RAW_5M_MANIFEST.json"
    daily_path = paths.output_root / "IEX_DAILY_MANIFEST.json"
    atomic_json(five_path, five_manifest)
    atomic_json(daily_path, daily_manifest)
    assessment = assess_raw_backfill_manifests(five_manifest, daily_manifest)
    complete = assessment["disposition"] == RawBackfillDisposition.COMPLETE.value
    ledger_path = paths.output_root / "IEX_RAW_ACQUISITION_LEDGER.json"
    atomic_json(
        ledger_path,
        {
            "schema_version": "hf_data_cutover_r1.iex_raw_acquisition_ledger.v1",
            "created_at_utc": utc_now(),
            "invocation_kind": invocation_kind,
            **assessment,
            "raw_5m_manifest_sha256": sha256_file(five_path),
            "raw_daily_manifest_sha256": sha256_file(daily_path),
            "partition_failure_evidence": (
                "IEX_RAW_5M_MANIFEST.json and IEX_DAILY_MANIFEST.json failures arrays; "
                "per-partition failure.json files"
            ),
        },
    )
    run_manifest_path = paths.output_root / "RUN_MANIFEST.json"
    run_manifest = _read_json(run_manifest_path)
    run_manifest.update(
        {
            "updated_at_utc": utc_now(),
            "state": "IEX_RAW_DATA_COMPLETE_FEATURE_BUILD_PENDING" if complete else "IEX_RAW_DATA_INCOMPLETE",
            "iex_raw_backfill": {
                **dict(run_manifest.get("iex_raw_backfill") or {}),
                "collection_summary": dict(summary),
                "raw_5m_manifest_sha256": sha256_file(five_path),
                "daily_manifest_sha256": sha256_file(daily_path),
                "raw_acquisition_ledger_sha256": sha256_file(ledger_path),
                "completion_disposition": assessment["disposition"],
                "resumable_incomplete_partition_count": assessment["resumable_incomplete_partition_count"],
                "irrecoverable_failure_partition_count": assessment["irrecoverable_failure_partition_count"],
                "max_in_flight_requests": BACKFILL_WORKERS,
                "initial_validated_max_in_flight_requests": INITIAL_VALIDATED_BACKFILL_WORKERS,
                "transport_recovery_max_in_flight_requests": BACKFILL_WORKERS,
                "requests_per_minute": BACKFILL_REQUESTS_PER_MINUTE,
                "pacing_requests_per_minute": BACKFILL_PACING_REQUESTS_PER_MINUTE,
                "last_invocation_kind": invocation_kind,
                "final_free_bytes": shutil.disk_usage(paths.repository_root).free,
                "completed_at_utc": utc_now(),
            },
            "next_gate": (
                "Derive canonical completed 1h bars."
                if complete
                else "Resume incomplete IEX raw partitions."
                if assessment["disposition"] == RawBackfillDisposition.RESUMABLE_INCOMPLETE.value
                else "Repair explicitly classified irrecoverable IEX raw failures."
            ),
        }
    )
    atomic_json(run_manifest_path, run_manifest)
    return run_manifest, assessment


def _validated_partition_inventory(
    store: IexMarketDataStore,
    plans: Mapping[str, tuple[IexPartitionRequest, ...]],
) -> tuple[list[IexPartitionRequest], list[IexPartitionRequest]]:
    completed: list[IexPartitionRequest] = []
    pending: list[IexPartitionRequest] = []
    for timeframe in ("5m", "1Day"):
        for request in plans[timeframe]:
            if store.completed_manifest(request) is None:
                pending.append(request)
            else:
                completed.append(request)
    return completed, pending


def _completed_manifest_set_sha256(
    store: IexMarketDataStore,
    requests: Iterable[IexPartitionRequest],
) -> str:
    rows = [
        {"partition_id": request.partition_id, "manifest_sha256": sha256_file(store.manifest_path(request))}
        for request in sorted(requests, key=lambda value: value.partition_id)
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exception_classification(exc: Exception) -> str:
    classification = str(getattr(exc, "classification", "")).casefold()
    if classification:
        return classification
    if type(exc).__name__.casefold() in _RETRYABLE_ERROR_TYPES:
        return "transient_network_failure"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "integrity_failure"
    return "unexpected_failure"


def _fresh_https_open(request: Any, *, timeout: float) -> Any:
    """Use a new urllib opener for every HTTP attempt; no session state is reused."""
    return build_opener().open(request, timeout=timeout)


def _preflight(paths: CutoverPaths, settings: IexBackfillSettings) -> None:
    free_bytes = shutil.disk_usage(paths.repository_root).free
    if free_bytes <= MINIMUM_FINAL_RESERVE_BYTES:
        raise CutoverAuditError(
            f"IEX backfill reserve gate failed: free={free_bytes}, reserve={MINIMUM_FINAL_RESERVE_BYTES}"
        )
    state = resource_state(repository_root=paths.repository_root)
    atomic_json(
        paths.output_root / "IEX_RAW_BACKFILL_PREFLIGHT.json",
        {
            "schema_version": "hf_data_cutover_r1.iex_raw_backfill_preflight.v1",
            "recorded_at_utc": utc_now(),
            "start": settings.start.isoformat(),
            "end_exclusive": settings.end.isoformat(),
            "symbol_count": len(settings.roster),
            "minimum_final_reserve_bytes": MINIMUM_FINAL_RESERVE_BYTES,
            "resource_state": state,
            "protected_process_policy": "OBSERVE_ONLY_DO_NOT_STOP_OR_MUTATE",
        },
    )


def _collect_all(
    *,
    store: IexMarketDataStore,
    provider: AlpacaBasicHistoricalBarProvider,
    plans: Mapping[str, tuple[IexPartitionRequest, ...]],
    provider_symbol_map: Mapping[str, str],
) -> dict[str, Any]:
    total = sum(len(requests) for requests in plans.values())
    completed = skipped = failed = rows = 0
    started = time.monotonic()
    last_report = started
    pending: list[IexPartitionRequest] = []
    for timeframe in ("5m", "1Day"):
        for request in plans[timeframe]:
            existing = store.completed_manifest(request)
            if existing is None:
                pending.append(request)
                continue
            skipped += 1
            completed += 1
            rows += int(existing["quality"]["row_count"])
    if shutil.disk_usage(store.root).free <= MINIMUM_FINAL_RESERVE_BYTES:
        raise CutoverAuditError("IEX backfill stopped at the reserve gate before pending work")
    retrieval_batches = _retrieval_batches(pending)
    with ThreadPoolExecutor(max_workers=BACKFILL_WORKERS, thread_name_prefix="iex-backfill") as executor:
        futures: dict[Future[list[tuple[IexPartitionRequest, dict[str, Any] | Exception]]], tuple[IexPartitionRequest, ...]] = {
            executor.submit(
                _collect_batch,
                store=store,
                provider=provider,
                requests=batch,
                provider_symbol_map=provider_symbol_map,
            ): batch
            for batch in retrieval_batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                classification = getattr(exc, "classification", "unexpected_failure")
                results = [(request, exc) for request in batch]
                if classification in {"permanent_authentication_failure", "entitlement_failure"}:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
            for request, result in results:
                if isinstance(result, Exception):
                    failed += 1
                    store.write_failure(
                        request,
                        {
                            "failed_at_utc": utc_now(),
                            "classification": getattr(result, "classification", "unexpected_failure"),
                            "error_type": type(result).__name__,
                            "error_message": str(result),
                        },
                    )
                    continue
                completed += 1
                rows += int(result["quality"]["row_count"])
            now = time.monotonic()
            if now - last_report >= 60:
                free_bytes = shutil.disk_usage(store.root).free
                if free_bytes <= MINIMUM_FINAL_RESERVE_BYTES:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise CutoverAuditError(
                        f"IEX backfill stopped at reserve gate: free={free_bytes}"
                    )
                elapsed = max(now - started, 1.0)
                print(
                    "IEX RAW BACKFILL "
                    f"completed={completed}/{total} skipped={skipped} failed={failed} "
                    f"rows={rows} partitions_per_hour={completed / elapsed * 3600.0:.1f} "
                    f"requests={provider.metrics.requests_attempted} retries={provider.metrics.requests_retried} "
                    f"tls_eof={provider.metrics.tls_eof_count} http_429={provider.metrics.http_429_count} "
                    f"free_bytes={free_bytes}",
                    flush=True,
                )
                last_report = now
    return {
        "planned_partitions": total,
        "completed_partitions": completed,
        "resumed_partitions": skipped,
        "failed_partitions_during_invocation": failed,
        "rows_observed_including_resumed": rows,
        "provider_metrics": provider.metrics.as_dict(),
    }


def _retrieval_batches(
    requests: Iterable[IexPartitionRequest],
) -> tuple[tuple[IexPartitionRequest, ...], ...]:
    grouped: dict[tuple[str, datetime, datetime, int | None], list[IexPartitionRequest]] = defaultdict(list)
    for request in requests:
        grouped[(request.timeframe, request.start, request.end, request.year)].append(request)
    batches: list[tuple[IexPartitionRequest, ...]] = []
    for key in sorted(grouped, key=lambda value: (value[0], value[1], value[3] or 0)):
        ordered = sorted(grouped[key], key=lambda request: request.symbol)
        size = DAILY_RETRIEVAL_BATCH_SIZE if key[0] == "1Day" else FIVE_MINUTE_RETRIEVAL_BATCH_SIZE
        batches.extend(tuple(ordered[index : index + size]) for index in range(0, len(ordered), size))
    return tuple(batches)


def _collect_batch(
    *,
    store: IexMarketDataStore,
    provider: AlpacaBasicHistoricalBarProvider,
    requests: tuple[IexPartitionRequest, ...],
    provider_symbol_map: Mapping[str, str],
) -> list[tuple[IexPartitionRequest, dict[str, Any] | Exception]]:
    if not requests:
        return []
    first = requests[0]
    if any(
        request.timeframe != first.timeframe
        or request.start != first.start
        or request.end != first.end
        for request in requests
    ):
        raise ValueError("IEX retrieval batch contains incompatible partition bounds")
    batch_id = f"iex-{first.timeframe}-{first.year or 'all'}-{'-'.join(request.symbol for request in requests)}"
    provider_symbols = tuple(provider_symbol_map.get(request.symbol, request.symbol) for request in requests)
    remaps = tuple(
        (request.symbol, provider_symbols[index])
        for index, request in enumerate(requests)
        if provider_symbols[index] != request.symbol
    )
    provider_request = HistoricalBarRequest(
        symbols=provider_symbols,
        timeframe=first.timeframe,
        start=first.start,
        end=first.end,
        feed="iex",
        adjustment="raw",
        limit=10_000,
        raw_chunk_id=batch_id,
        currency="USD",
        asof=BACKFILL_ASOF,
        canonical_symbols=tuple(request.symbol for request in requests),
        provider_symbol_by_canonical=remaps,
    )
    try:
        rows, chunk = fetch_chunk_with_retries(
            provider,
            provider_request,
            max_retries=5,
            retain_raw_pages=False,
        )
    except AlpacaHistoricalBarError as exc:
        if len(requests) == 1 or exc.classification in {
            "permanent_authentication_failure",
            "entitlement_failure",
            "retryable_failure",
            "transient_network_failure",
        }:
            raise
        results: list[tuple[IexPartitionRequest, dict[str, Any] | Exception]] = []
        for request in requests:
            try:
                results.extend(
                    _collect_batch(
                        store=store,
                        provider=provider,
                        requests=(request,),
                        provider_symbol_map=provider_symbol_map,
                    )
                )
            except Exception as individual_exc:
                results.append((request, individual_exc))
        return results
    combined = canonical_raw_bar_frame(rows)
    results = []
    for request in requests:
        try:
            frame = canonical_raw_bar_frame(
                combined.loc[combined["symbol"].eq(request.symbol)].to_dict("records")
            )
            quality = validate_raw_bar_partition(frame, request)
            if quality["off_grid_count"]:
                raise ValueError(f"Partition {request.partition_id} contains off-grid rows")
            source = {
                "provider": "Alpaca",
                "feed": "IEX",
                "endpoint_family": "historical_stock_bars",
                "adjustment": "raw",
                "currency": "USD",
                "asof": BACKFILL_ASOF,
                "retrieval_batch_id": batch_id,
                "retrieval_symbol_batch": list(provider_request.symbols),
                "provider_symbol_map": dict(remaps),
                "page_count_for_shared_batch": int(chunk["pages"]),
                "raw_page_sha256_for_shared_batch": list(chunk["raw_page_sha256"]),
                "partition_retry_count": "not_attributable_under_shared_concurrent_provider_metrics",
                "raw_provider_fields": ["t", "o", "h", "l", "c", "v", "n", "vw"],
                "normalization": "field rename and typed projection only; no adjustment, fill, or price transformation",
            }
            manifest = store.publish(request, frame, quality=quality, source=source)
        except Exception as exc:
            results.append((request, exc))
        else:
            results.append((request, manifest))
    return results


def _dataset_manifest(
    *,
    store: IexMarketDataStore,
    requests: Iterable[IexPartitionRequest],
    dataset: str,
    provider_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for request in requests:
        try:
            manifest = store.completed_manifest(request)
        except Exception as exc:
            unresolved.append(
                {
                    "partition_id": request.partition_id,
                    "classification": "integrity_failure",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        if manifest is None:
            failure_path = store.partition_dir(request) / "failure.json"
            failure = (
                _read_json(failure_path)
                if failure_path.exists()
                else {"classification": "missing_unattempted", "error_type": None, "error_message": None}
            )
            unresolved.append({"partition_id": request.partition_id, **failure})
            continue
        completed.append(manifest)
    timestamps_first = [row["quality"]["first_timestamp"] for row in completed if row["quality"]["first_timestamp"]]
    timestamps_last = [row["quality"]["last_timestamp"] for row in completed if row["quality"]["last_timestamp"]]
    return {
        "schema_version": "hf_data_cutover_r1.iex_raw_dataset_manifest.v1",
        "dataset": dataset,
        "created_at_utc": utc_now(),
        "provider": "Alpaca",
        "feed": "IEX",
        "adjustment": "raw",
        "currency": "USD",
        "asof": BACKFILL_ASOF,
        "partitioning": "symbol/year" if dataset == "raw_5m" else "symbol",
        "planned_partition_count": len(completed) + len(unresolved),
        "completed_partition_count": len(completed),
        "failed_partition_count": sum(not _is_retryable_partition_failure(row) for row in unresolved),
        "incomplete_partition_count": sum(_is_retryable_partition_failure(row) for row in unresolved),
        "retryable_failure_partition_count": sum(
            _is_retryable_partition_failure(row)
            and str(row.get("classification", "")).casefold() not in {"missing", "missing_unattempted"}
            for row in unresolved
        ),
        "missing_partition_count": sum(
            str(row.get("classification", "")).casefold() in {"missing", "missing_unattempted"}
            for row in unresolved
        ),
        "unresolved_partition_count": len(unresolved),
        "row_count": sum(int(row["quality"]["row_count"]) for row in completed),
        "parquet_bytes": sum(int(row["parquet_bytes"]) for row in completed),
        "first_observation": min(timestamps_first) if timestamps_first else None,
        "last_observation": max(timestamps_last) if timestamps_last else None,
        "duplicate_count": sum(int(row["quality"]["duplicate_count"]) for row in completed),
        "off_grid_count": sum(int(row["quality"]["off_grid_count"]) for row in completed),
        "vwap_null_count": sum(int(row["quality"]["vwap_null_count"]) for row in completed),
        "trade_count_null_count": sum(int(row["quality"]["trade_count_null_count"]) for row in completed),
        "partition_manifest_sha256": sorted(
            sha256_file(store.manifest_path(request))
            for request in requests
            if store.manifest_path(request).exists()
        ),
        "failures": unresolved,
        "provider_metrics_last_invocation": dict(provider_metrics),
        "historical_eligibility_inferred": False,
        "research_use_only": True,
    }


def assess_raw_backfill_manifests(*manifests: Mapping[str, Any]) -> dict[str, Any]:
    """Classify incomplete raw partitions without conflating retryable and fatal failures."""
    planned = sum(int(manifest.get("planned_partition_count", 0)) for manifest in manifests)
    completed = sum(int(manifest.get("completed_partition_count", 0)) for manifest in manifests)
    failures = [row for manifest in manifests for row in list(manifest.get("failures") or [])]
    retryable = [row for row in failures if _is_retryable_partition_failure(row)]
    irrecoverable = [row for row in failures if not _is_retryable_partition_failure(row)]
    unresolved = planned - completed
    if unresolved < 0 or unresolved != len(failures):
        return {
            "disposition": RawBackfillDisposition.IRRECOVERABLE_FAILURE.value,
            "planned_partition_count": planned,
            "completed_partition_count": completed,
            "unresolved_partition_count": max(unresolved, 0),
            "resumable_incomplete_partition_count": len(retryable),
            "irrecoverable_failure_partition_count": len(irrecoverable) + 1,
            "classification_counts": {"manifest_count_or_failure_ledger_mismatch": 1},
        }
    classification_counts: dict[str, int] = {}
    for row in failures:
        classification = _normalized_failure_classification(row)
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    disposition = (
        RawBackfillDisposition.COMPLETE
        if unresolved == 0
        else RawBackfillDisposition.IRRECOVERABLE_FAILURE
        if irrecoverable
        else RawBackfillDisposition.RESUMABLE_INCOMPLETE
    )
    return {
        "disposition": disposition.value,
        "planned_partition_count": planned,
        "completed_partition_count": completed,
        "unresolved_partition_count": unresolved,
        "resumable_incomplete_partition_count": len(retryable),
        "irrecoverable_failure_partition_count": len(irrecoverable),
        "classification_counts": dict(sorted(classification_counts.items())),
    }


def _is_retryable_partition_failure(failure: Mapping[str, Any]) -> bool:
    classification = str(failure.get("classification", "")).casefold()
    if classification in _RETRYABLE_FAILURE_CLASSIFICATIONS:
        return True
    error_type = str(failure.get("error_type", "")).casefold()
    message = str(failure.get("error_message") or failure.get("error") or "").casefold()
    return error_type in _RETRYABLE_ERROR_TYPES and any(marker in message for marker in _RETRYABLE_ERROR_MARKERS)


def _normalized_failure_classification(failure: Mapping[str, Any]) -> str:
    classification = str(failure.get("classification", "unknown")).casefold()
    if _is_retryable_partition_failure(failure) and classification not in _RETRYABLE_FAILURE_CLASSIFICATIONS:
        return "legacy_transient_network_failure"
    return classification


def _plan_identity(plans: Mapping[str, tuple[IexPartitionRequest, ...]]) -> str:
    import hashlib

    payload = [request.identity() for key in sorted(plans) for request in plans[key]]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BACKFILL_PACING_REQUESTS_PER_MINUTE",
    "BACKFILL_REQUESTS_PER_MINUTE",
    "BACKFILL_WORKERS",
    "INITIAL_VALIDATED_BACKFILL_WORKERS",
    "IexBackfillSettings",
    "RawBackfillDisposition",
    "TRANSPORT_PROBE_PARTITIONS",
    "TRANSPORT_PROBE_WORKERS",
    "alpaca_symbol_aliases",
    "assess_raw_backfill_manifests",
    "load_iex_backfill_settings",
    "run_iex_raw_backfill",
    "run_iex_transport_connectivity_probe",
]
