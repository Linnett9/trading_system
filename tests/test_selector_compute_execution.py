from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from core.research.ml.selector_compute_execution import (
    GIB,
    SelectorComputeExecution,
    build_selector_compute_manifest,
    coordinator_resource_request,
    selector_resource_request,
)
from core.research.compute import (
    ResourceLeaseLedger,
    ResourceRequest,
    dell_i5_10500_profile,
)


def _campaign(jobs, identity="CAMPAIGN"):
    payload = {
        "campaign_contract": "selector_research_campaign.v1",
        "campaign_version": "v2",
        "campaign_id": "fixture",
        "campaign_identity": identity,
        "expected_component_count": len(jobs),
        "fitted_component_matrix": [
            {
                "job_id": row["job_id"],
                "model_id": row["model_id"],
                "prediction_date": row["prediction_date"],
                "horizon_id": row.get("horizon_id"),
                "component_runner": "fixture:runner",
            }
            for row in jobs
        ],
    }
    payload["logical_checksum"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest().upper()
    for job in jobs:
        job["campaign_identity"] = payload["campaign_identity"]
    return payload


def _job(model="ridge", index=0):
    return {
        "job_id": f"selector:2025-01-01:{model}:{index}",
        "model_id": model,
        "prediction_date": "2025-01-01",
        "horizon_id": str(index),
        "logical_checksum": f"JOB-{model}-{index}",
        "selector_dataset_root": "dataset",
        "authoritative_output_root": "output",
        "feature_schema": "features",
        "target_contract": "target",
        "expected_parent_gate_checksum": "gate",
        "expected_dataset_checksum": "dataset",
        "dependency_state": "ready",
        "overwrite_policy": "never",
        "resume_policy": "compatible",
    }


def test_resource_profiles_are_explicit_conservative_defaults():
    expected = {
        "ridge": (2, 1, "SMALL"),
        "elastic_net": (2, 1, "SMALL"),
        "ordered_logit_ranker": (4, 1, "MEDIUM"),
        "huber": (4, 1, "MEDIUM"),
        "contextual_elastic_net": (6, 1, "MEDIUM"),
        "multi_horizon_ridge": (6, 1, "MEDIUM"),
        "multi_horizon_elastic_net": (6, 1, "MEDIUM"),
        "lightgbm_rank_xendcg": (8, 2, "LARGE"),
        "lightgbm_lambdarank": (8, 2, "LARGE"),
    }
    for index, (model, (ram, cpu, resource_class)) in enumerate(expected.items()):
        request = selector_resource_request(
            _job(model, index), run_id="run", attempt_identity=f"attempt-{index}"
        )
        assert request.estimated_peak_ram_bytes == ram * GIB
        assert request.cpu_weight == cpu
        assert request.resource_class == resource_class
        assert request.inner_threads == 1
        assert request.estimate_source == "CONSERVATIVE_DEFAULT"
    coordinator = coordinator_resource_request(
        run_id="run", attempt_identity="coordinator"
    )
    assert coordinator.lightweight is True
    assert coordinator.estimated_peak_ram_bytes == GIB
    assert coordinator.cpu_weight == 1


def test_manifest_inventory_and_compatibility_bind_exact_campaign_and_plan():
    jobs = [_job("ridge", 0), _job("huber", 1)]
    campaign = _campaign(jobs)
    readiness = {"logical_checksum": "PLAN-A"}
    first = build_selector_compute_manifest(
        jobs=jobs,
        campaign_manifest=campaign,
        readiness=readiness,
        run_id="run",
        source_git_commit="commit",
    )
    second = build_selector_compute_manifest(
        jobs=jobs,
        campaign_manifest=campaign,
        readiness=readiness,
        run_id="run",
        source_git_commit="commit",
    )
    assert first["deterministic_expected_ordering"] == [
        row["job_id"] for row in jobs
    ]
    assert first["compatibility_identity"] == second["compatibility_identity"]
    changed = build_selector_compute_manifest(
        jobs=jobs,
        campaign_manifest=campaign,
        readiness={"logical_checksum": "PLAN-B"},
        run_id="run",
        source_git_commit="commit",
    )
    assert changed["compatibility_identity"] != first["compatibility_identity"]
    historical_jobs = [dict(row, campaign_identity="OTHER-CAMPAIGN") for row in jobs]
    historical = _campaign(historical_jobs, identity="OTHER-CAMPAIGN")
    assert build_selector_compute_manifest(
        jobs=historical_jobs,
        campaign_manifest=historical,
        readiness=readiness,
        run_id="run",
        source_git_commit="commit",
    )["compatibility_identity"] != first["compatibility_identity"]


class _Process:
    pid = os.getpid()
    returncode = 0

    def poll(self):
        return self.returncode

    def communicate(self):
        return "", ""

    def terminate(self):
        self.returncode = -15


def test_component_requires_model_hook_and_synthetic_hook_allows_complete(tmp_path):
    job = _job()
    jobs = [job]
    campaign = _campaign(jobs)

    def execute(hook, run_id):
        report = tmp_path / run_id / "component.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({"status": "COMPLETE", "manifest_path": "owned.json"}),
            encoding="utf-8",
        )
        adapter = SelectorComputeExecution(
            jobs=jobs,
            campaign_manifest=campaign,
            readiness={"logical_checksum": "PLAN"},
            run_id=run_id,
            source_git_commit="commit",
            runs_root=tmp_path / "runs",
            lease_ledger_path=tmp_path / f"{run_id}-ledger.json",
            registry_path=tmp_path / "registry.json",
            available_memory=lambda: 32 * GIB,
            model_package_hook=hook,
        )
        result = adapter.execute_component(
            job=job,
            command=["synthetic-selector"],
            environment={},
            report_path=report,
            transcript_path=tmp_path / run_id / "transcript.txt",
            popen=lambda *args, **kwargs: _Process(),
        )
        adapter.close()
        return result

    blocked = execute(None, "blocked")
    assert blocked["status"] == "INCOMPLETE"
    assert blocked["model_package"]["completion_status"] == "BLOCKED"

    complete = execute(
        lambda **kwargs: {
            "completion_status": "COMPLETE",
            "artifact_identity": "MODEL",
            "package_checksum": "CHECKSUM",
            "preprocessing_identity": "PREPROCESSING",
            "feature_order_identity": "FEATURES",
            "prediction_binding_identity": "BINDING",
        },
        "complete",
    )
    assert complete["status"] == "COMPLETE"
    assert complete["model_package"]["artifact_identity"] == "MODEL"


def test_unknown_model_and_duplicate_items_fail_closed():
    with pytest.raises(ValueError, match="No selector compute resource profile"):
        selector_resource_request(
            _job("random_forest"),
            run_id="run",
            attempt_identity="attempt",
        )
    jobs = [_job(), _job()]
    with pytest.raises(ValueError, match="Duplicate"):
        build_selector_compute_manifest(
            jobs=jobs,
            campaign_manifest=_campaign(jobs),
            readiness={"logical_checksum": "PLAN"},
            run_id="run",
            source_git_commit="commit",
        )


def test_waiting_component_does_not_launch_and_coordinator_does_not_reserve_children(
    tmp_path,
):
    profile = dell_i5_10500_profile(source_git_commit="commit")
    ledger_path = tmp_path / "ledger.json"
    ledger = ResourceLeaseLedger(
        profile=profile,
        path=ledger_path,
        available_memory=lambda: 32 * GIB,
    )
    ledger.initialise_ledger()
    ledger.request_persisted_lease(ResourceRequest(
        pipeline="fixture",
        stage="fixture",
        job_id="existing",
        run_id="other",
        resource_class="SMALL",
        estimated_peak_ram_bytes=GIB,
        cpu_weight=2,
        inner_threads=1,
        gpu_required=False,
        concurrency_group="fixture",
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity="fixture",
        attempt_identity="existing-attempt",
    ))
    job = _job()
    jobs = [job]
    adapter = SelectorComputeExecution(
        jobs=jobs,
        campaign_manifest=_campaign(jobs),
        readiness={"logical_checksum": "PLAN"},
        run_id="waiting",
        source_git_commit="commit",
        runs_root=tmp_path / "runs",
        lease_ledger_path=ledger_path,
        registry_path=tmp_path / "registry.json",
        available_memory=lambda: 32 * GIB,
    )
    launched = False

    def launch(*args, **kwargs):
        nonlocal launched
        launched = True
        return _Process()

    result = adapter.execute_component(
        job=job,
        command=["synthetic"],
        environment={},
        report_path=tmp_path / "unused.json",
        transcript_path=tmp_path / "unused.txt",
        popen=launch,
    )
    adapter.close()
    assert result["status"] == "WAITING_FOR_RESOURCES"
    assert launched is False
    status = ledger.read_ledger_status()
    assert status["active_reserved_ram_bytes"] == GIB


def test_valid_compatible_skip_precedes_expensive_lease_and_subprocess(tmp_path):
    job = _job()
    jobs = [job]
    launched = False
    adapter = SelectorComputeExecution(
        jobs=jobs,
        campaign_manifest=_campaign(jobs),
        readiness={"logical_checksum": "PLAN"},
        run_id="skip",
        source_git_commit="commit",
        runs_root=tmp_path / "runs",
        lease_ledger_path=tmp_path / "ledger.json",
        registry_path=tmp_path / "registry.json",
        available_memory=lambda: 32 * GIB,
        compatible_skip_hook=lambda **kwargs: {
            "completion_status": "COMPLETE",
            "artifact_identity": "MODEL",
            "package_checksum": "CHECKSUM",
            "preprocessing_identity": "PREPROCESSING",
            "feature_order_identity": "FEATURES",
            "prediction_binding_identity": "BINDING",
        },
    )

    def launch(*args, **kwargs):
        nonlocal launched
        launched = True
        return _Process()

    result = adapter.execute_component(
        job=job,
        command=["synthetic"],
        environment={},
        report_path=tmp_path / "unused.json",
        transcript_path=tmp_path / "unused.txt",
        popen=launch,
    )
    adapter.close()
    assert result["status"] == "SKIPPED_COMPATIBLE"
    assert launched is False


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (OSError("startup"), "FAILED_TO_START"),
        (subprocess.TimeoutExpired("synthetic", 1), "FAILED"),
        (KeyboardInterrupt(), "CANCELLED"),
    ],
)
def test_execution_failures_restore_component_capacity(
    tmp_path, failure, expected_status
):
    job = _job()
    jobs = [job]
    report = tmp_path / "component.json"
    report.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")

    class FailingProcess(_Process):
        returncode = None

        def communicate(self):
            raise failure

    def launch(*args, **kwargs):
        if isinstance(failure, OSError):
            raise failure
        return FailingProcess()

    adapter = SelectorComputeExecution(
        jobs=jobs,
        campaign_manifest=_campaign(jobs),
        readiness={"logical_checksum": "PLAN"},
        run_id=f"failure-{expected_status.lower()}",
        source_git_commit="commit",
        runs_root=tmp_path / "runs",
        lease_ledger_path=tmp_path / f"{expected_status}.json",
        registry_path=tmp_path / "registry.json",
        available_memory=lambda: 32 * GIB,
        telemetry_interval_seconds=1,
    )
    with pytest.raises(BaseException):
        adapter.execute_component(
            job=job,
            command=["synthetic"],
            environment={},
            report_path=report,
            transcript_path=tmp_path / "transcript.txt",
            popen=launch,
        )
    adapter.close(reason="BATCH_FAILURE")
    status = adapter.ledger.read_ledger_status()
    assert status["active_reserved_ram_bytes"] == 0
    assert any(
        row["status"] == expected_status for row in status["recent_releases"]
    )


def test_nonzero_subprocess_failure_stops_telemetry_and_is_retryable(tmp_path):
    job = _job()
    jobs = [job]

    class FailedProcess(_Process):
        returncode = 9

    adapter = SelectorComputeExecution(
        jobs=jobs,
        campaign_manifest=_campaign(jobs),
        readiness={"logical_checksum": "PLAN"},
        run_id="subprocess-failure",
        source_git_commit="commit",
        runs_root=tmp_path / "runs",
        lease_ledger_path=tmp_path / "ledger.json",
        registry_path=tmp_path / "registry.json",
        available_memory=lambda: 32 * GIB,
        telemetry_interval_seconds=1,
    )
    with pytest.raises(RuntimeError, match="failed \\(9\\)"):
        adapter.execute_component(
            job=job,
            command=["synthetic"],
            environment={},
            report_path=tmp_path / "unused.json",
            transcript_path=tmp_path / "transcript.txt",
            popen=lambda *args, **kwargs: FailedProcess(),
        )
    adapter.close(reason="BATCH_FAILURE")
    status = adapter.ledger.read_ledger_status()
    assert status["active_reserved_ram_bytes"] == 0
    assert any(row["status"] == "FAILED" for row in status["recent_releases"])
