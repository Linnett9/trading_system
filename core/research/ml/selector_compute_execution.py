from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.research.compute import (
    LeaseStatus,
    ResourceLeaseLedger,
    ResourceRequest,
    build_item_status,
    build_result_record,
    build_run_manifest,
    dell_i5_10500_profile,
    initialise_run,
    publish_item_status,
    publish_artifact_inventory,
    publish_results_snapshot,
    publish_summary,
    update_global_registry_snapshot,
    update_run_status,
)
from core.research.compute.process_telemetry import ProcessTelemetry, TelemetrySample


SELECTOR_RESOURCE_PROFILE_CONTRACT = "selector_compute_resource_profile.v1"
SELECTOR_RESOURCE_PROFILE_ID = "selector-stage10-conservative-defaults-v1"
GIB = 1024**3


@dataclass(frozen=True)
class SelectorResourceProfile:
    resource_class: str
    estimated_peak_ram_bytes: int
    cpu_weight: int
    concurrency_group: str


RESOURCE_PROFILES = {
    "ridge": SelectorResourceProfile("SMALL", 2 * GIB, 1, "selector_tabular"),
    "elastic_net": SelectorResourceProfile(
        "SMALL", 2 * GIB, 1, "selector_tabular"
    ),
    "ordered_logit_ranker": SelectorResourceProfile(
        "MEDIUM", 4 * GIB, 1, "selector_tabular"
    ),
    "huber": SelectorResourceProfile("MEDIUM", 4 * GIB, 1, "selector_tabular"),
    "contextual_elastic_net": SelectorResourceProfile(
        "MEDIUM", 6 * GIB, 1, "selector_tabular"
    ),
    "multi_horizon_ridge": SelectorResourceProfile(
        "MEDIUM", 6 * GIB, 1, "selector_tabular"
    ),
    "multi_horizon_elastic_net": SelectorResourceProfile(
        "MEDIUM", 6 * GIB, 1, "selector_tabular"
    ),
    "lightgbm_rank_xendcg": SelectorResourceProfile(
        "LARGE", 8 * GIB, 2, "selector_heavy"
    ),
    "lightgbm_lambdarank": SelectorResourceProfile(
        "LARGE", 8 * GIB, 2, "selector_heavy"
    ),
}
COORDINATOR_PROFILE = SelectorResourceProfile(
    "SMALL", 1 * GIB, 1, "lightweight_coordinator"
)


class ModelPackageHook(Protocol):
    def __call__(
        self,
        *,
        job: Mapping[str, Any],
        component_result: Mapping[str, Any],
        run_identity: str,
    ) -> Mapping[str, Any]: ...


class CompatibleSkipHook(Protocol):
    def __call__(
        self, *, job: Mapping[str, Any], run_identity: str
    ) -> Mapping[str, Any] | None: ...


def selector_resource_request(
    job: Mapping[str, Any],
    *,
    run_id: str,
    attempt_identity: str,
) -> ResourceRequest:
    model_id = str(job.get("model_id") or "")
    try:
        profile = RESOURCE_PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(f"No selector compute resource profile: {model_id}") from exc
    return _request(
        profile,
        run_id=run_id,
        job_id=str(job["job_id"]),
        attempt_identity=attempt_identity,
        stage="stage10_component",
    )


def coordinator_resource_request(
    *, run_id: str, attempt_identity: str
) -> ResourceRequest:
    return _request(
        COORDINATOR_PROFILE,
        run_id=run_id,
        job_id="stage10_batch_coordinator",
        attempt_identity=attempt_identity,
        stage="stage10_coordinator",
        lightweight=True,
    )


def _request(
    profile: SelectorResourceProfile,
    *,
    run_id: str,
    job_id: str,
    attempt_identity: str,
    stage: str,
    lightweight: bool = False,
) -> ResourceRequest:
    return ResourceRequest(
        pipeline="selector",
        stage=stage,
        job_id=job_id,
        run_id=run_id,
        resource_class=profile.resource_class,
        estimated_peak_ram_bytes=profile.estimated_peak_ram_bytes,
        cpu_weight=profile.cpu_weight,
        inner_threads=1,
        gpu_required=False,
        concurrency_group=profile.concurrency_group,
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity=(
            f"{SELECTOR_RESOURCE_PROFILE_CONTRACT}:"
            f"{SELECTOR_RESOURCE_PROFILE_ID}"
        ),
        lightweight=lightweight,
        attempt_identity=attempt_identity,
    )


def build_selector_compute_manifest(
    *,
    jobs: Sequence[Mapping[str, Any]],
    campaign_manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    run_id: str,
    source_git_commit: str,
) -> dict[str, Any]:
    from core.research.ml.selector_component_scheduler import (
        validate_component_plan,
    )

    jobs = validate_component_plan(
        jobs, campaign_manifest=campaign_manifest
    )
    campaign_identity = str(campaign_manifest.get("campaign_identity") or "")
    campaign_checksum = str(campaign_manifest.get("logical_checksum") or "")
    plan_checksum = str(readiness.get("logical_checksum") or "")
    if not all((campaign_identity, campaign_checksum, plan_checksum)):
        raise ValueError("Selector compute run requires frozen campaign and plan identities")
    expected = [
        {
            "item_id": str(job["job_id"]),
            "ordered_position": index,
            "model_id": str(job["model_id"]),
            "date_identity": str(job["prediction_date"]),
            "horizon_identity": job.get("horizon_id"),
            "plan_job_checksum": str(job.get("logical_checksum") or ""),
        }
        for index, job in enumerate(jobs)
    ]
    configuration = {
        "campaign_identity": campaign_identity,
        "campaign_manifest_checksum": campaign_checksum,
        "stage10_plan_checksum": plan_checksum,
        "component_order": [row["item_id"] for row in expected],
        "component_identity": [
            {
                key: row.get(key)
                for key in (
                    "item_id", "model_id", "date_identity",
                    "horizon_identity", "plan_job_checksum",
                )
            }
            for row in expected
        ],
        "resource_profile_identity": SELECTOR_RESOURCE_PROFILE_ID,
    }
    configuration_checksum = _checksum(configuration)
    profile = dell_i5_10500_profile(source_git_commit=source_git_commit)
    return build_run_manifest(
        run_id=run_id,
        pipeline="selector",
        stage=str(campaign_manifest.get("campaign_id") or campaign_identity),
        run_purpose="authoritative Stage-10 selector component execution",
        source_git_commit=source_git_commit,
        configuration_identity=_checksum(
            {
                "campaign_identity": campaign_identity,
                "stage10_plan_checksum": plan_checksum,
            }
        ),
        configuration_checksum=configuration_checksum,
        machine_profile_identity=profile.logical_checksum,
        requested_resource_profile_identity=SELECTOR_RESOURCE_PROFILE_ID,
        parent_input_artifacts=[
            {
                "artifact_kind": "SELECTOR_CAMPAIGN_MANIFEST",
                "identity": campaign_identity,
                "checksum": campaign_checksum,
            },
            {
                "artifact_kind": "SELECTOR_STAGE10_PLAN",
                "identity": plan_checksum,
                "checksum": plan_checksum,
            },
        ],
        expected_inventory=expected,
        campaign_identity=campaign_identity,
    )


class SelectorComputeExecution:
    def __init__(
        self,
        *,
        jobs: Sequence[Mapping[str, Any]],
        campaign_manifest: Mapping[str, Any],
        readiness: Mapping[str, Any],
        run_id: str,
        source_git_commit: str,
        runs_root: Path,
        lease_ledger_path: Path,
        registry_path: Path,
        available_memory: Callable[[], int | None],
        model_package_hook: ModelPackageHook | None = None,
        compatible_skip_hook: CompatibleSkipHook | None = None,
        telemetry_interval_seconds: float = 15.0,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.jobs = list(jobs)
        self.run_id = run_id
        self.clock = clock
        self.registry_path = registry_path
        if model_package_hook is None or compatible_skip_hook is None:
            from core.research.ml.stock_level.selector_sklearn_model_artifacts import (
                resolve_selector_model_package,
            )

            model_package_hook = model_package_hook or resolve_selector_model_package
            compatible_skip_hook = (
                compatible_skip_hook or resolve_selector_model_package
            )
        self.model_package_hook = model_package_hook
        self.compatible_skip_hook = compatible_skip_hook
        self.telemetry_interval_seconds = telemetry_interval_seconds
        self.profile = dell_i5_10500_profile(source_git_commit=source_git_commit)
        self.manifest = build_selector_compute_manifest(
            jobs=self.jobs,
            campaign_manifest=campaign_manifest,
            readiness=readiness,
            run_id=run_id,
            source_git_commit=source_git_commit,
        )
        self.run_root = initialise_run(self.manifest, runs_root=runs_root)
        self.ledger = ResourceLeaseLedger(
            profile=self.profile,
            path=lease_ledger_path,
            available_memory=available_memory,
        )
        self.ledger.initialise_ledger()
        self._positions = {
            str(job["job_id"]): index for index, job in enumerate(self.jobs)
        }
        self._waiting_leases: list[tuple[str, str]] = []
        self._publication_lock = threading.Lock()
        self._artifact_entries: list[dict[str, Any]] = []
        self._artifact_inventory: list[dict[str, Any]] = []
        self._result_records: list[dict[str, Any]] = []
        coordinator_attempt = _checksum(
            {"run_id": run_id, "owner": "stage10_batch_coordinator"}
        )
        coordinator_request = coordinator_resource_request(
            run_id=run_id, attempt_identity=coordinator_attempt
        )
        coordinator, _ = self.ledger.request_persisted_lease(coordinator_request)
        if coordinator.status != LeaseStatus.GRANTED:
            raise RuntimeError("Selector coordinator is waiting for resources")
        coordinator_started = _process_start_timestamp(
            os.getpid(), self.clock().isoformat()
        )
        self.ledger.activate_persisted_lease(
            coordinator.logical_identity,
            attempt_identity=coordinator_attempt,
            process_id=os.getpid(),
            process_start_timestamp=coordinator_started,
            command_identity="selector_stage10_batch_coordinator",
        )
        self._coordinator = (coordinator.logical_identity, coordinator_attempt)

    def close(self, *, reason: str = "SUCCESS") -> None:
        for identity, attempt in self._waiting_leases:
            try:
                self.ledger.cancel_persisted_lease(
                    identity,
                    attempt_identity=attempt,
                    reason="BATCH_EXIT_WHILE_WAITING",
                )
            except ValueError:
                pass
        self._waiting_leases.clear()
        if self._coordinator is None:
            return
        identity, attempt = self._coordinator
        self._coordinator = None
        if reason == "SUCCESS":
            self.ledger.release_persisted_lease(
                identity, attempt_identity=attempt, reason=reason
            )
        else:
            self.ledger.fail_persisted_lease(
                identity, attempt_identity=attempt, reason=reason
            )

    def execute_component(
        self,
        *,
        job: Mapping[str, Any],
        command: Sequence[str],
        environment: Mapping[str, str],
        report_path: Path,
        transcript_path: Path,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> Mapping[str, Any]:
        job_id = str(job["job_id"])
        if job_id not in self._positions:
            raise ValueError(f"Unknown selector compute item: {job_id}")
        attempt = _checksum(
            {
                "run_id": self.run_id,
                "job_id": job_id,
                "plan_job_checksum": job.get("logical_checksum"),
                "attempt_started_at": self.clock().isoformat(),
            }
        )
        if self.compatible_skip_hook is not None:
            compatible = self.compatible_skip_hook(
                job=job, run_identity=str(self.manifest["run_identity"])
            )
            if compatible is not None:
                _validate_model_package_evidence(compatible)
                artifact_identity, results_identity = (
                    self._record_model_artifacts(job, compatible)
                )
                self._publish_item(
                    job,
                    attempt,
                    "SKIPPED_COMPATIBLE",
                    compatible_skip_evidence=dict(compatible),
                    fitted_model_artifact_identity=compatible.get(
                        "artifact_identity"
                    ),
                    prediction_artifact_identity=compatible.get(
                        "prediction_binding_identity"
                    ),
                    required_artifact_kind="NONE",
                )
                registry = self._refresh_run(
                    artifact_inventory_identity=artifact_identity,
                    results_identity=results_identity,
                )
                return {
                    "status": "SKIPPED_COMPATIBLE",
                    "compute_registry_health": registry["health"],
                    "model_package": dict(compatible),
                }
        request = selector_resource_request(
            job, run_id=self.run_id, attempt_identity=attempt
        )
        lease, _ = self.ledger.request_persisted_lease(request)
        if lease.status == LeaseStatus.WAITING:
            self._waiting_leases.append((lease.logical_identity, attempt))
            self._publish_item(
                job, attempt, "WAITING_FOR_RESOURCES",
                lease_identity=lease.logical_identity,
                resource_request_identity=request.logical_checksum,
                blocker_reason=";".join(lease.blocked_reasons),
            )
            self._refresh_run()
            return {
                "status": "WAITING_FOR_RESOURCES",
                "lease_identity": lease.logical_identity,
                "blocked_reasons": list(lease.blocked_reasons),
            }

        process = None
        telemetry = ProcessTelemetry(
            minimum_interval_seconds=self.telemetry_interval_seconds
        )
        telemetry_path = self.run_root / "telemetry" / f"{_safe(job_id)}.csv"
        summary_path = self.run_root / "resources" / f"{_safe(job_id)}.json"
        started_at = self.clock().isoformat()
        terminal_lease = False
        monitor: threading.Thread | None = None
        try:
            child_environment = dict(environment)
            child_environment.update({
                "SELECTOR_COMPUTE_RUN_ID": self.run_id,
                "SELECTOR_COMPUTE_RUN_IDENTITY": str(
                    self.manifest["run_identity"]
                ),
                "SELECTOR_COMPUTE_ATTEMPT_ID": attempt,
            })
            process = popen(
                list(command),
                cwd=Path(__file__).resolve().parents[3],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_environment,
            )
            process_start = _process_start_timestamp(process.pid, started_at)
            self.ledger.activate_persisted_lease(
                lease.logical_identity,
                attempt_identity=attempt,
                process_id=int(process.pid),
                process_start_timestamp=process_start,
                command_identity=_checksum(list(command)),
            )
            self._publish_item(
                job, attempt, "RUNNING",
                lease_identity=lease.logical_identity,
                resource_request_identity=request.logical_checksum,
                started_timestamp=started_at,
            )
            monitor = threading.Thread(
                target=telemetry.monitor_process,
                kwargs={
                    "process": process,
                    "sample_provider": lambda child, elapsed: (
                        self._sample_and_heartbeat(
                            child=child,
                            elapsed=elapsed,
                            job_id=job_id,
                            lease_identity=lease.logical_identity,
                            attempt_identity=attempt,
                            process_start_timestamp=process_start,
                        )
                    ),
                    "interval_seconds": self.telemetry_interval_seconds,
                },
                daemon=True,
            )
            monitor.start()
            stdout, stderr = process.communicate()
            monitor.join(timeout=self.telemetry_interval_seconds + 1)
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(
                (stdout or "") + (stderr or ""), encoding="utf-8"
            )
            telemetry.write_csv(telemetry_path)
            summary = telemetry.write_summary(
                summary_path,
                estimated_peak_ram_bytes=request.estimated_peak_ram_bytes,
                lease_acquired_at=lease.granted_at,
                process_started_at=started_at,
                process_ended_at=self.clock().isoformat(),
                normal_reserve_bytes=self.profile.normal_reserve_bytes,
            )
            if process.returncode:
                self.ledger.fail_persisted_lease(
                    lease.logical_identity,
                    attempt_identity=attempt,
                    reason=f"SUBPROCESS_EXIT_{process.returncode}",
                )
                terminal_lease = True
                self._publish_item(
                    job, attempt, "FAILED",
                    lease_identity=lease.logical_identity,
                    resource_request_identity=request.logical_checksum,
                    telemetry_identity=str(telemetry_path),
                    resource_summary_identity=str(summary_path),
                    failure_code="SELECTOR_COMPONENT_SUBPROCESS_FAILED",
                    failure_reason=f"exit code {process.returncode}",
                    retryable=True,
                )
                self._refresh_run(summary)
                raise RuntimeError(
                    f"Component command failed ({process.returncode}): {job_id}"
                )
            component_result = json.loads(report_path.read_text(encoding="utf-8"))
            try:
                package = self._model_package(job, component_result)
                if package.get("completion_status") == "COMPLETE":
                    _validate_model_package_evidence(package)
            except Exception as exc:
                package = {
                    "completion_status": "BLOCKED",
                    "blocker_reason": (
                        "Model-package hook failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "hook_failure": True,
                }
            if package.get("completion_status") != "COMPLETE":
                status = "INCOMPLETE"
                blocker_code = "SELECTOR_MODEL_PACKAGE_NOT_ADOPTED"
                blocker_reason = str(
                    package.get("blocker_reason")
                    or "Required fitted-model package is unavailable"
                )
            else:
                status = (
                    "SKIPPED_COMPATIBLE"
                    if component_result.get("status") == "SKIPPED_COMPATIBLE"
                    else "COMPLETE"
                )
                blocker_code = blocker_reason = None
            artifact_inventory_identity = results_identity = None
            if status in {"COMPLETE", "SKIPPED_COMPATIBLE"}:
                (
                    artifact_inventory_identity,
                    results_identity,
                ) = self._record_model_artifacts(job, package)
            self._publish_item(
                job, attempt, status,
                lease_identity=lease.logical_identity,
                resource_request_identity=request.logical_checksum,
                telemetry_identity=str(telemetry_path),
                resource_summary_identity=str(summary_path),
                fitted_model_artifact_identity=package.get("artifact_identity"),
                prediction_artifact_identity=package.get(
                    "prediction_binding_identity"
                ),
                required_artifact_kind="NONE",
                blocker_code=blocker_code,
                blocker_reason=blocker_reason,
                artifact_validation=dict(package),
                completed_timestamp=self.clock().isoformat(),
            )
            self.ledger.release_persisted_lease(
                lease.logical_identity,
                attempt_identity=attempt,
                reason="SUCCESS",
            )
            terminal_lease = True
            registry = self._refresh_run(
                summary,
                artifact_inventory_identity=artifact_inventory_identity,
                results_identity=results_identity,
            )
            return {
                **dict(component_result),
                "status": status,
                "compute_registry_health": registry["health"],
                "model_package": dict(package),
            }
        except BaseException as exc:
            if not terminal_lease:
                cancelled = isinstance(exc, KeyboardInterrupt)
                if process is not None and getattr(process, "poll")() is None:
                    terminate = getattr(process, "terminate", None)
                    if callable(terminate):
                        terminate()
                    if monitor is not None:
                        monitor.join(timeout=self.telemetry_interval_seconds + 1)
                try:
                    if cancelled:
                        self.ledger.cancel_persisted_lease(
                            lease.logical_identity,
                            attempt_identity=attempt,
                            reason="PARENT_INTERRUPTION",
                        )
                    else:
                        self.ledger.fail_persisted_lease(
                            lease.logical_identity,
                            attempt_identity=attempt,
                            reason=f"EXECUTION_FAILURE:{type(exc).__name__}",
                            startup_failure=process is None,
                        )
                except ValueError:
                    pass
                self._publish_item(
                    job, attempt, "CANCELLED" if cancelled else "FAILED",
                    lease_identity=lease.logical_identity,
                    resource_request_identity=request.logical_checksum,
                    failure_code=(
                        None if cancelled else "SELECTOR_COMPONENT_EXECUTION_FAILED"
                    ),
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                )
                self._refresh_run()
            raise

    def _model_package(
        self, job: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if self.model_package_hook is None:
            return {
                "completion_status": "BLOCKED",
                "blocker_reason": "COMPUTE-ADOPT-SELECTOR-1B/1C not installed",
            }
        return self.model_package_hook(
            job=job,
            component_result=result,
            run_identity=str(self.manifest["run_identity"]),
        )

    def _sample_and_heartbeat(
        self,
        *,
        child: Any,
        elapsed: float,
        job_id: str,
        lease_identity: str,
        attempt_identity: str,
        process_start_timestamp: str,
    ) -> TelemetrySample:
        sample = _sample(child, elapsed, self.run_id, job_id, self.profile)
        if elapsed >= self.telemetry_interval_seconds:
            self.ledger.heartbeat_persisted_lease(
                lease_identity,
                attempt_identity=attempt_identity,
                process_id=int(child.pid),
                process_start_timestamp=process_start_timestamp,
                phase="component_subprocess",
                telemetry_identity=f"{self.run_id}:{job_id}",
                measured_memory_summary={
                    "process_rss_bytes": sample.process_rss_bytes,
                    "child_rss_bytes": sample.child_rss_bytes,
                },
            )
        return sample

    def _publish_item(
        self, job: Mapping[str, Any], attempt: str, status: str, **evidence: Any
    ) -> None:
        publish_item_status(
            self.run_root,
            build_item_status(
                run_identity=str(self.manifest["run_identity"]),
                item_id=str(job["job_id"]),
                ordered_position=self._positions[str(job["job_id"])],
                pipeline="selector",
                stage=str(self.manifest["stage"]),
                attempt_identity=attempt,
                status=status,
                model_id=str(job["model_id"]),
                date_identity=str(job["prediction_date"]),
                horizon_identity=job.get("horizon_id"),
                **evidence,
            ),
        )

    def _record_model_artifacts(
        self, job: Mapping[str, Any], package: Mapping[str, Any]
    ) -> tuple[str, str]:
        if not package.get("model_package_path") or not package.get(
            "prediction_package_path"
        ):
            synthetic_identity = _checksum(dict(package))
            return synthetic_identity, synthetic_identity
        roots = (
            Path(str(package["model_package_path"])),
            Path(str(package["prediction_package_path"])),
        )
        with self._publication_lock:
            known = {
                (row["package_root"], row["owning_item_identity"])
                for row in self._artifact_entries
            }
            for root in roots:
                key = (str(root), str(job["job_id"]))
                if key not in known:
                    self._artifact_entries.append({
                        "package_root": str(root),
                        "owning_run_identity": self.manifest["run_identity"],
                        "owning_item_identity": str(job["job_id"]),
                    })
                    known.add(key)
            self._artifact_inventory, artifact_identity = (
                publish_artifact_inventory(
                    self.run_root, self._artifact_entries
                )
            )
            artifact_ids = [
                row["artifact_identity"]
                for row in self._artifact_inventory
                if row["owning_item_identity"] == str(job["job_id"])
            ]
            result = build_result_record(
                result_identity=_checksum({
                    "run_identity": self.manifest["run_identity"],
                    "item_identity": str(job["job_id"]),
                    "artifact_identities": artifact_ids,
                }),
                run_identity=str(self.manifest["run_identity"]),
                item_identity=str(job["job_id"]),
                result_kind="MODEL_COMPONENT",
                pipeline="selector",
                stage=str(self.manifest["stage"]),
                status="COMPLETE",
                artifact_identities=artifact_ids,
                metrics={},
                model_id=str(job["model_id"]),
                horizon_identity=job.get("horizon_id"),
                date_identity=str(job["prediction_date"]),
                eligibility_state="NOT_EVALUATED",
                promotion_state="NOT_PROMOTED",
            )
            by_item = {
                row["item_identity"]: row for row in self._result_records
            }
            by_item[str(job["job_id"])] = result
            self._result_records = [
                by_item[key] for key in sorted(by_item)
            ]
            results = publish_results_snapshot(
                self.run_root, self._result_records
            )
            return artifact_identity, str(results["logical_checksum"])

    def _refresh_run(
        self,
        resource_summary: Mapping[str, Any] | None = None,
        *,
        artifact_inventory_identity: str | None = None,
        results_identity: str | None = None,
    ) -> Mapping[str, Any]:
        status = update_run_status(
            self.run_root,
            expected_revision=None,
            inputs_valid=True,
            resource_evidence={
                "measured_peak_ram_bytes": (
                    resource_summary or {}
                ).get("measured_peak_process_plus_children_ram_bytes"),
                "resource_wait_seconds": (
                    resource_summary or {}
                ).get("resource_wait_seconds"),
                "estimate_exceeded": (
                    resource_summary or {}
                ).get("estimate_exceeded"),
                "artifact_inventory_identity": artifact_inventory_identity,
                "results_identity": results_identity,
            },
        )
        publish_summary(
            self.run_root, artifact_inventory=self._artifact_inventory
        )
        return update_global_registry_snapshot(
            self.run_root, registry_path=self.registry_path
        )


def _sample(
    process: Any,
    elapsed: float,
    run_id: str,
    job_id: str,
    profile: Any,
) -> TelemetrySample:
    rss = vms = child_rss = available = cpu = system_cpu = None
    telemetry_status = "OK"
    try:
        import psutil  # type: ignore[import-not-found]

        owner = psutil.Process(int(process.pid))
        memory = owner.memory_info()
        rss, vms = int(memory.rss), int(memory.vms)
        child_rss = sum(
            int(child.memory_info().rss)
            for child in owner.children(recursive=True)
        )
        available = int(psutil.virtual_memory().available)
        cpu = float(owner.cpu_percent())
        system_cpu = float(psutil.cpu_percent())
    except (ImportError, OSError, RuntimeError):
        telemetry_status = "METRICS_UNAVAILABLE"
    return TelemetrySample(
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        job_id=job_id,
        process_id=int(process.pid),
        execution_phase="component_subprocess",
        elapsed_seconds=elapsed,
        process_rss_bytes=rss,
        process_vms_bytes=vms,
        child_rss_bytes=child_rss,
        system_total_bytes=profile.total_ram_bytes,
        system_available_bytes=available,
        scheduler_reserved_bytes=None,
        process_cpu_percent=cpu,
        system_cpu_percent=system_cpu,
        telemetry_status=telemetry_status,
    )


def _process_start_timestamp(process_id: int, fallback: str) -> str:
    try:
        import psutil  # type: ignore[import-not-found]

        return datetime.fromtimestamp(
            psutil.Process(process_id).create_time(), timezone.utc
        ).isoformat()
    except (ImportError, OSError, RuntimeError):
        return fallback


def _validate_model_package_evidence(package: Mapping[str, Any]) -> None:
    required = (
        "artifact_identity",
        "package_checksum",
        "preprocessing_identity",
        "feature_order_identity",
        "prediction_binding_identity",
        "completion_status",
    )
    if package.get("completion_status") != "COMPLETE" or any(
        package.get(field) in (None, "") for field in required
    ):
        raise ValueError(
            "Compatible selector completion requires complete model-package evidence"
        )


def _checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _safe(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")
