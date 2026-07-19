"""Shared-compute orchestration for bounded Alpaca raw chunk conversion."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.research.compute.artifact_contracts import (
    ArtifactRole,
    ArtifactType,
    build_stage_artifact_manifest,
    canonical_checksum,
)
from core.research.compute.artifact_storage import publish_artifact_package
from core.research.compute.machine_profile import GIB, dell_i5_10500_profile, detect_runtime_resources
from core.research.compute.lease_storage import atomic_write_json
from core.research.compute.process_telemetry import ProcessTelemetry, TelemetrySample
from core.research.compute.resource_governor import LeaseStatus, ResourceRequest
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.compute.run_contracts import (
    build_item_status,
    build_result_record,
    build_run_manifest,
    checksum,
)
from core.research.compute.run_storage import (
    initialise_run,
    publish_artifact_inventory,
    publish_item_status,
    publish_results_snapshot,
    publish_summary,
    update_global_registry_snapshot,
    update_run_status,
)

PIPELINE = "alpaca_market_data"
STAGE = "raw_chunk_to_parquet"
CONVERSION_CONTRACT = "alpaca_raw_chunk_to_parquet.v1"


@dataclass(frozen=True)
class ConversionComputeOptions:
    row_group_size: int
    requested_workers: int
    progress: bool = False
    runs_root: Path = Path("reports/runs")
    resource_ledger_path: Path = Path("reports/compute/resource_leases/resource_lease_ledger.json")
    artifact_root: Path = Path("reports/compute/artifacts/data_conversion")
    registry_path: Path = Path("reports/runs/run_registry.json")
    invocation: Mapping[str, Any] = field(default_factory=dict)
    executor_factory: Callable[..., Any] = ProcessPoolExecutor
    telemetry_factory: Callable[..., ProcessTelemetry] = ProcessTelemetry
    ledger_factory: Callable[..., ResourceLeaseLedger] = ResourceLeaseLedger


def execute_conversion_run(
    *,
    candidates: Sequence[Mapping[str, Any]],
    compatible_skips: Sequence[Mapping[str, Any]],
    convert_one: Callable[..., dict[str, Any]],
    options: ConversionComputeOptions,
) -> dict[str, Any]:
    """Execute one deterministic bounded invocation with one item per chunk."""
    work = [dict(row) for row in candidates]
    skips = [dict(row) for row in compatible_skips]
    items = sorted(work + skips, key=lambda row: item_id(row))
    if not items:
        return {"results": [], "run_id": None, "run_root": None}
    commit = _git_commit()
    effective_workers = min(max(1, options.requested_workers), 2, max(1, len(work)))
    invocation = {
        **_portable_invocation(options.invocation),
        "conversion_contract": CONVERSION_CONTRACT,
        "worker_cap": 2,
        "effective_workers": effective_workers,
    }
    logical_run_id = f"conversion-{canonical_checksum(invocation | {'items': [item_id(row) for row in items]})[:20]}"
    run_id = _claim_attempt_run_id(options.runs_root, logical_run_id)
    attempt_id = checksum({"run_id": run_id, "purpose": "bounded-conversion"})[:24]
    profile = dell_i5_10500_profile(source_git_commit=commit)
    inventory = [
        {"item_id": item_id(row), "ordered_position": index}
        for index, row in enumerate(items, 1)
    ]
    manifest = build_run_manifest(
        run_id=run_id,
        pipeline=PIPELINE,
        stage=STAGE,
        run_purpose="Bounded Alpaca raw chunk to Parquet conversion",
        source_git_commit=commit,
        configuration_identity=canonical_checksum(invocation),
        configuration_checksum=canonical_checksum(invocation),
        machine_profile_identity=profile.logical_checksum,
        requested_resource_profile_identity="CONSERVATIVE_DEFAULT",
        parent_input_artifacts=[{"identity": item_id(row), "path": _portable(row["source_path"])} for row in items],
        expected_inventory=inventory,
        campaign_identity=logical_run_id,
    )
    run_root = initialise_run(manifest, runs_root=options.runs_root)
    positions = {row["item_id"]: row["ordered_position"] for row in inventory}
    results: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []
    for row in skips:
        result = dict(row)
        results.append(result)
        publish_item_status(run_root, _item_status(
            manifest, row, positions, attempt_id, "SKIPPED_COMPATIBLE",
            compatible_skip_evidence={
                "parquet_path": _portable(row["parquet_path"]),
                "evidence_path": _portable(row["evidence_path"]),
            },
        ))
    initial_status = update_run_status(run_root, expected_revision=None, inputs_valid=True)
    status_revision = int(initial_status["state_revision"])
    if not work:
        status = update_run_status(run_root, expected_revision=status_revision, inputs_valid=True)
        payload = _finish_publication(
            run_root, manifest, results, artifact_entries, status, options, effective_workers, None
        )
        return payload

    request = ResourceRequest(
        pipeline=PIPELINE, stage=STAGE, job_id=run_id, run_id=run_id,
        resource_class="LARGE", estimated_peak_ram_bytes=8 * GIB, cpu_weight=2,
        inner_threads=1, gpu_required=False, concurrency_group="DATA_CONVERSION",
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity=CONVERSION_CONTRACT,
        attempt_identity=attempt_id,
    )
    runtime = detect_runtime_resources()
    ledger = None
    lease = None
    telemetry = None
    pool = None
    terminal_reason = "FAILURE"
    execution_error: BaseException | None = None
    interrupted = False
    cleanup_error: BaseException | None = None
    try:
        ledger = options.ledger_factory(
            profile=profile, path=options.resource_ledger_path,
            available_memory=lambda: runtime.available_ram_bytes or profile.total_ram_bytes,
        )
        lease, revision = ledger.request_persisted_lease(request)
        if lease.status != LeaseStatus.GRANTED:
            raise RuntimeError(f"resource lease not granted: {','.join(lease.blocked_reasons)}")
        process_start = _process_start_timestamp()
        ledger.activate_persisted_lease(
            lease.logical_identity, attempt_identity=attempt_id, process_id=os.getpid(),
            process_start_timestamp=process_start, expected_revision=revision,
            command_identity=run_id,
        )
        telemetry = options.telemetry_factory()
        _record_telemetry(telemetry, run_id, lease.logical_identity, 0, effective_workers)
        for row in work:
            publish_item_status(run_root, _item_status(
                manifest, row, positions, attempt_id, "RUNNING",
                lease_identity=lease.logical_identity,
                resource_request_identity=request.logical_checksum,
            ))
        running_status = update_run_status(
            run_root, expected_revision=status_revision, inputs_valid=True,
            resource_evidence={"active_lease_identities": [lease.logical_identity],
                               "reserved_ram_bytes": 8 * GIB},
        )
        status_revision = int(running_status["state_revision"])
        if effective_workers == 1:
            converted = [convert_one(row, options.row_group_size, delete_json_after_validate=False) for row in work]
        else:
            pool = options.executor_factory(max_workers=effective_workers)
            futures = [
                pool.submit(convert_one, row, options.row_group_size, delete_json_after_validate=False)
                for row in work
            ]
            converted = [future.result() for future in as_completed(futures)]
            pool.shutdown(wait=True, cancel_futures=False)
            pool = None
        for result in converted:
            source = next(row for row in work if row["source_path"] == result["source_path"])
            result = {**result, "item_id": item_id(source), "chunk_id": source.get("chunk_id")}
            results.append(result)
            if result.get("status") != "converted":
                publish_item_status(run_root, _item_status(
                    manifest, source, positions, attempt_id, "FAILED",
                    failure_code="CONVERSION_FAILED",
                    failure_reason="; ".join(result.get("errors", [])),
                ))
                continue
            package_root, artifact_manifest = _publish_artifact(
                source, result, manifest, attempt_id, request, lease.logical_identity,
                options, effective_workers,
            )
            artifact_entries.append({
                "package_root": str(package_root),
                "owning_run_identity": manifest["run_identity"],
                "owning_item_identity": item_id(source),
            })
            publish_item_status(run_root, _item_status(
                manifest, source, positions, attempt_id, "COMPLETE",
                lease_identity=lease.logical_identity,
                resource_request_identity=request.logical_checksum,
                stage_artifact_identity=artifact_manifest["artifact_id"],
                stage_artifact_package_path=str(package_root),
                artifact_validation={"stage_artifact_valid": True},
                required_artifact_kind="STAGE",
            ))
        terminal_reason = "SUCCESS" if all(row.get("status") != "failed" for row in results) else "FAILURE"
    except KeyboardInterrupt:
        terminal_reason = "CANCELLED"
        interrupted = True
        _mark_unfinished(run_root, manifest, work, positions, attempt_id, "CANCELLED", "KeyboardInterrupt")
    except BaseException as exc:
        execution_error = exc
        _mark_unfinished(run_root, manifest, work, positions, attempt_id, "FAILED", f"{type(exc).__name__}: {exc}")
        results.append({
            "status": "failed", "source_path": str(work[0]["source_path"]),
            "parquet_path": str(work[0]["parquet_path"]),
            "errors": [f"{type(exc).__name__}: {exc}"],
            "item_id": item_id(work[0]), "chunk_id": work[0].get("chunk_id"),
        })
    finally:
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        resource_summary = None
        if telemetry is not None:
            try:
                _record_telemetry(telemetry, run_id, lease.logical_identity if lease else "", 1, effective_workers)
                telemetry.write_csv(run_root / "telemetry.csv")
                resource_summary = telemetry.write_summary(
                    run_root / "resource_summary.json",
                    estimated_peak_ram_bytes=8 * GIB,
                    lease_acquired_at=lease.granted_at if lease else None,
                    process_started_at=lease.process_started_at if lease else None,
                    process_ended_at=_now(),
                )
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        if lease is not None:
            assert ledger is not None
            try:
                if terminal_reason == "SUCCESS" and cleanup_error is None:
                    ledger.release_persisted_lease(lease.logical_identity, attempt_identity=attempt_id, reason="SUCCESS")
                elif terminal_reason == "CANCELLED":
                    ledger.cancel_persisted_lease(lease.logical_identity, attempt_identity=attempt_id, reason="INTERRUPTED")
                else:
                    ledger.fail_persisted_lease(lease.logical_identity, attempt_identity=attempt_id, reason="CONVERSION_FAILURE")
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

    if cleanup_error is not None and execution_error is None and not interrupted:
        execution_error = cleanup_error
        terminal_reason = "FAILURE"
        _mark_unfinished(
            run_root, manifest, work, positions, attempt_id, "FAILED",
            f"{type(cleanup_error).__name__}: {cleanup_error}",
        )

    status = update_run_status(
        run_root, expected_revision=status_revision, inputs_valid=True,
        cancelled=interrupted,
        resource_evidence={
            "reserved_ram_bytes": 8 * GIB,
            "measured_peak_ram_bytes": (resource_summary or {}).get("measured_peak_process_plus_children_ram_bytes"),
            "telemetry_identity": checksum(resource_summary or {}),
        },
    )
    if interrupted:
        _finish_publication(
            run_root, manifest, results, artifact_entries, status, options, effective_workers, resource_summary
        )
        raise KeyboardInterrupt
    try:
        publication = _finish_publication(
            run_root, manifest, results, artifact_entries, status, options, effective_workers, resource_summary
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        atomic_write_json(
            run_root / "publication_failure.json",
            {"error": failure, "timestamp": _now()},
        )
        _mark_unfinished(run_root, manifest, work, positions, attempt_id, "FAILED", failure)
        failed_status = update_run_status(
            run_root, expected_revision=int(status["state_revision"]), inputs_valid=True
        )
        return {
            "results": results, "run_id": manifest["run_id"], "run_root": str(run_root),
            "summary": {"terminal_status": failed_status["current_status"]},
            "error": failure,
        }
    if execution_error is not None:
        publication["error"] = f"{type(execution_error).__name__}: {execution_error}"
    return publication


def _publish_artifact(source, result, manifest, attempt_id, request, lease_id, options, workers):
    parquet = Path(result["parquet_path"])
    evidence = Path(source["source_path"]) / "parquet_conversion.json"
    metadata = _file_metadata(source, parquet, evidence, result)
    metadata.update({
        "run_id": manifest["run_id"],
        "item_id": item_id(source),
        "worker_policy": {"requested": options.requested_workers, "effective": workers, "cap": 2},
        "bounded_invocation": _portable_invocation(options.invocation),
    })
    artifact_id = f"alpaca-parquet-{canonical_checksum({'item': item_id(source), 'run': manifest['run_id']})[:20]}"
    template = build_stage_artifact_manifest(
        artifact_id=artifact_id, artifact_type=ArtifactType.DATA_STAGE_ARTIFACT.value,
        artifact_subtype="ALPACA_RAW_CHUNK_PARQUET",
        artifact_role=ArtifactRole.REFERENCE_DATA.value,
        pipeline=PIPELINE, stage=STAGE, run_id=manifest["run_id"], attempt_id=attempt_id,
        dataset_input_ancestry=[{"identity": item_id(source), "path": _portable(source["source_path"])}],
        source_artifacts=[], configuration_identity=manifest["configuration_identity"],
        configuration_checksum=manifest["configuration_checksum"],
        source_git_commit=manifest["source_git_commit"], stage_owner=STAGE,
        output_counts={"rows": int(result["parquet_row_count"]), "files": 2},
        schema_identity="alpaca_bars_parquet_schema.v1",
        coverage_evidence={"chunk_id": source.get("chunk_id"), "row_count": int(result["parquet_row_count"])},
        resumability_evidence={"evidence": _portable(evidence), "contract": CONVERSION_CONTRACT},
        conversion_metadata=metadata,
        worker_policy={"requested": options.requested_workers, "effective": workers, "cap": 2},
        bounded_invocation=dict(options.invocation),
        resource_evidence={
            "applicable": True, "machine_profile_identity": manifest["machine_profile_identity"],
            "resource_request_identity": request.logical_checksum,
            "resource_lease_identity": lease_id, "telemetry_artifact_identity": None,
            "resource_summary_identity": None,
        },
    )
    root = options.artifact_root / artifact_id
    _, published = publish_artifact_package(
        root, template,
        {"evidence/parquet_conversion.json": evidence.read_bytes()},
    )
    return root, published


def _file_metadata(source, parquet: Path, evidence: Path, result) -> dict[str, Any]:
    chunk_root = Path(source["source_path"])
    return {
        "source_identity": _portable(source["source_path"]),
        "source_size": int(source.get("source_bytes", 0)),
        "source_hash": canonical_checksum({
            name: _sha256(chunk_root / name)
            for name in ("normalized_rows.json", "provider_pages.json")
            if (chunk_root / name).exists()
        }),
        "destination_identity": _portable(parquet),
        "parquet_output_size": parquet.stat().st_size,
        "parquet_output_hash": _sha256(parquet),
        "evidence_identity": _portable(evidence),
        "evidence_size": evidence.stat().st_size,
        "evidence_hash": _sha256(evidence),
        "row_count": int(result["parquet_row_count"]),
        "conversion_contract": CONVERSION_CONTRACT,
        "compatibility_identity": canonical_checksum({"chunk": item_id(source), "contract": CONVERSION_CONTRACT}),
        "item_id": item_id(source),
    }


def _finish_publication(run_root, manifest, results, entries, status, options, workers, resources):
    records = [
        build_result_record(
            result_identity=checksum({"run": manifest["run_identity"], "item": row.get("item_id") or item_id(row)}),
            run_identity=manifest["run_identity"], item_identity=row.get("item_id") or item_id(row),
            result_kind="DATA_STAGE", pipeline=PIPELINE, stage=STAGE,
            status=("COMPLETE" if row.get("status") == "converted" else "SKIPPED_COMPATIBLE"),
            artifact_identities=[], metrics={}, dimensions={"source_path": _portable(row["source_path"])},
        )
        for row in results if row.get("status") != "failed"
    ]
    publish_results_snapshot(run_root, records)
    inventory, _ = publish_artifact_inventory(run_root, entries)
    publish_summary(run_root, artifact_inventory=inventory)
    registry = update_global_registry_snapshot(run_root, registry_path=options.registry_path)
    if registry["health"] != "HEALTHY":
        raise RuntimeError(f"registry publication failed: {registry['error']}")
    summary = {
        "candidate_count": len(results), "conversion_required_count": sum(r.get("status") != "skipped_compatible" for r in results),
        "converted_count": sum(r.get("status") == "converted" for r in results),
        "compatible_skip_count": sum(r.get("status") == "skipped_compatible" for r in results),
        "failed_count": sum(r.get("status") == "failed" for r in results),
        "total_rows": sum(int(r.get("parquet_row_count", 0) or 0) for r in results),
        "total_output_bytes": sum(int(r.get("parquet_bytes", 0) or 0) for r in results),
        "effective_workers": workers, "artifact_count": len(inventory),
        "terminal_status": status["current_status"], "telemetry": resources,
    }
    atomic_write_json(run_root / "conversion_summary.json", summary)
    return {"results": results, "run_id": manifest["run_id"], "run_root": str(run_root), "summary": summary}


def _item_status(manifest, row, positions, attempt, status, **evidence):
    return build_item_status(
        run_identity=manifest["run_identity"], item_id=item_id(row),
        ordered_position=positions[item_id(row)], pipeline=PIPELINE, stage=STAGE,
        attempt_identity=attempt, status=status,
        source_identity=_portable(row["source_path"]),
        destination_identity=_portable(row["parquet_path"]), **evidence,
    )


def _mark_unfinished(run_root, manifest, work, positions, attempt, status, reason):
    for row in work:
        try:
            publish_item_status(run_root, _item_status(
                manifest, row, positions, attempt, status,
                failure_code="EXECUTION_INTERRUPTED", failure_reason=reason,
            ))
        except Exception:
            pass


def item_id(row: Mapping[str, Any]) -> str:
    return f"chunk-{canonical_checksum({'chunk_id': row.get('chunk_id'), 'source': _portable(row['source_path']), 'output': _portable(row['parquet_path'])})[:24]}"


def _portable(value: Any) -> str:
    path = Path(str(value))
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_invocation(invocation: Mapping[str, Any]) -> dict[str, Any]:
    path_fields = {"raw_root", "parquet_root", "collection_manifest"}
    return {
        key: (_portable(value) if key in path_fields and value is not None else value)
        for key, value in invocation.items()
    }


def _claim_attempt_run_id(runs_root: Path, logical_run_id: str) -> str:
    """Atomically reserve a never-reused physical attempt root."""
    stage_root = runs_root / PIPELINE / STAGE
    stage_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        run_id = f"{logical_run_id}-attempt-{attempt:04d}"
        try:
            # mkdir is the ownership decision. It is atomic on supported
            # Windows and POSIX filesystems; no preceding existence check is
            # relied upon for correctness.
            (stage_root / run_id).mkdir()
            return run_id
        except FileExistsError:
            attempt += 1


def _record_telemetry(telemetry, run_id, lease_id, elapsed, workers):
    runtime = detect_runtime_resources()
    rss = vms = child_rss = cpu = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        rss, vms = int(memory.rss), int(memory.vms)
        child_rss = sum(int(child.memory_info().rss) for child in process.children(recursive=True))
        cpu = float(process.cpu_percent(interval=None))
    except Exception:
        pass
    telemetry.record(TelemetrySample(
        timestamp=_now(), run_id=run_id, job_id=run_id, process_id=os.getpid(),
        execution_phase="CONVERSION", elapsed_seconds=float(elapsed),
        system_total_bytes=runtime.total_ram_bytes,
        system_available_bytes=runtime.available_ram_bytes,
        process_rss_bytes=rss, process_vms_bytes=vms, child_rss_bytes=child_rss,
        process_cpu_percent=cpu,
        scheduler_reserved_bytes=8 * GIB, completed_items=None,
        checkpoint_identity=lease_id,
    ))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _process_start_timestamp() -> str:
    try:
        import psutil
        return datetime.fromtimestamp(psutil.Process(os.getpid()).create_time(), timezone.utc).isoformat()
    except Exception:
        return _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
