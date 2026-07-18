from __future__ import annotations

import threading
import time
import json
import subprocess
from pathlib import Path

import pytest

from core.research.ml.selector_component_batch import run_stage10_component_batch
from core.research.ml.selector_component_readiness import READINESS_CONTRACT
from core.research.ml.selector_component_scheduler import (
    run_component_jobs,
    validate_component_plan,
)


MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")
DATES = tuple(f"2025-01-{day:02d}" for day in range(1, 6))


def _jobs():
    return [
        {
            "job_id": f"selector:{date}:{model}",
            "model_id": model,
            "prediction_date": date,
            "horizon_id": None,
            "selector_dataset_root": "dataset",
            "authoritative_output_root": f"components/{model}/{date}",
            "feature_schema": f"{model}.json",
            "target_contract": "forward_return_10d",
            "expected_parent_gate_checksum": "GATE",
            "expected_dataset_checksum": "DATASET",
            "dependency_state": "MISSING",
            "overwrite_policy": "never_overwrite_complete_component",
            "resume_policy": "resume_only_incomplete_owned_component",
            "logical_checksum": f"CHECKSUM-{date}-{model}",
        }
        for date in DATES for model in MODELS
    ]


def _inventory(jobs):
    return {
        "packages": [
            {
                "job_id": job["job_id"],
                "training_rows_path": f"training/{job['job_id']}.json",
                "prediction_rows_path": f"prediction/{job['job_id']}.json",
            }
            for job in jobs
        ]
    }


def _readiness(jobs):
    return {
        "readiness_contract_version": READINESS_CONTRACT,
        "logical_checksum": "READINESS",
        "production_plan": jobs,
    }


def test_exact_plan_validation_rejects_count_job_and_owner_duplicates():
    jobs = _jobs()
    assert validate_component_plan(jobs) == jobs
    with pytest.raises(ValueError, match="exactly 15"):
        validate_component_plan(jobs[:-1])
    duplicate_id = [dict(job) for job in jobs]
    duplicate_id[-1]["job_id"] = duplicate_id[0]["job_id"]
    with pytest.raises(ValueError, match="Duplicate Stage-10 job ID"):
        validate_component_plan(duplicate_id)
    duplicate_owner = [dict(job) for job in jobs]
    duplicate_owner[-1].update(
        model_id=duplicate_owner[0]["model_id"],
        prediction_date=duplicate_owner[0]["prediction_date"],
        job_id=f"selector:{duplicate_owner[0]['prediction_date']}:{duplicate_owner[0]['model_id']}:other",
    )
    with pytest.raises(ValueError, match="Duplicate Stage-10 component owner"):
        validate_component_plan(duplicate_owner)


def test_weighted_capacity_and_report_order_are_deterministic():
    jobs = _jobs()
    lock = threading.Lock()
    active_weight = 0
    peak_weight = 0

    def runner(job):
        nonlocal active_weight, peak_weight
        weight = 2 if job["model_id"] == "ordered_logit_ranker" else 1
        with lock:
            active_weight += weight
            peak_weight = max(peak_weight, active_weight)
        time.sleep(0.002 if job["model_id"] == "ridge" else 0.001)
        with lock:
            active_weight -= weight
        return {"status": "COMPLETED"}

    result = run_component_jobs(
        jobs, runner=runner, max_component_workers=3, capacity=4
    )
    assert peak_weight <= 4
    assert [row["job_id"] for row in result] == [job["job_id"] for job in jobs]
    assert all(row["status"] == "COMPLETED" for row in result)


def test_batch_forwards_jobs_records_skip_and_does_not_change_environment(
    tmp_path, monkeypatch
):
    jobs = _jobs()
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    observed = []

    def runner(job, package):
        observed.append((dict(job), dict(package)))
        return {
            "status": (
                "SKIPPED_COMPATIBLE"
                if job["job_id"] == jobs[0]["job_id"] else "COMPLETE"
            )
        }

    report = run_stage10_component_batch(
        readiness=_readiness(jobs),
        input_inventory=_inventory(jobs),
        parent_gate_path=tmp_path / "gate.json",
        ledger_path=tmp_path / "ledger.jsonl",
        output_root=tmp_path / "batch",
        runner=runner,
    )
    assert observed[0][0] == jobs[0]
    assert observed[0][1]["training_rows_path"].startswith("training/")
    assert report["jobs"][0]["status"] == "SKIPPED_COMPATIBLE"
    assert report["inner_model_threads"] == 1
    assert report["status"] == "COMPLETED"
    assert __import__("os").environ["OMP_NUM_THREADS"] == "9"


def test_default_runner_delegates_to_single_component_command_with_private_env(
    tmp_path, monkeypatch
):
    jobs = _jobs()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        report_path = Path(command[command.index("--verification-output") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"status": "COMPLETED", "manifest_path": "manifest.json"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "core.research.ml.selector_component_batch.subprocess.run", fake_run
    )
    report = run_stage10_component_batch(
        readiness=_readiness(jobs),
        input_inventory=_inventory(jobs),
        parent_gate_path=tmp_path / "gate.json",
        ledger_path=tmp_path / "ledger.jsonl",
        output_root=tmp_path / "batch",
    )
    assert report["status"] == "COMPLETED"
    assert len(calls) == 15
    command, kwargs = calls[0]
    assert command[command.index("--mode") + 1] == "ml-selector-component-publish"
    assert command[command.index("--training-rows-json") + 1].startswith("training/")
    assert all(kwargs["env"][name] == "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ))


def test_failure_stops_safe_dispatch_and_retry_uses_existing_runner(tmp_path):
    jobs = _jobs()
    attempts = []

    def failing(job, package):
        attempts.append(job["job_id"])
        if job["job_id"] == jobs[0]["job_id"]:
            raise RuntimeError("synthetic component failure")
        time.sleep(0.005)
        return {"status": "COMPLETE"}

    first = run_stage10_component_batch(
        readiness=_readiness(jobs),
        input_inventory=_inventory(jobs),
        parent_gate_path=tmp_path / "gate.json",
        ledger_path=tmp_path / "ledger.jsonl",
        output_root=tmp_path / "first",
        runner=failing,
    )
    assert first["status"] == "FAILED"
    assert first["jobs"][0]["status"] == "FAILED"
    assert any(row["status"] == "NOT_STARTED" for row in first["jobs"])

    retried = []
    second = run_stage10_component_batch(
        readiness=_readiness(jobs),
        input_inventory=_inventory(jobs),
        parent_gate_path=tmp_path / "gate.json",
        ledger_path=tmp_path / "ledger.jsonl",
        output_root=tmp_path / "second",
        runner=lambda job, package: retried.append(job["job_id"]) or {
            "status": "SKIPPED_COMPATIBLE"
            if job["job_id"] in attempts[1:] else "COMPLETE"
        },
    )
    assert second["status"] == "COMPLETED"
    assert len(retried) == 15
    assert set(retried) == {job["job_id"] for job in jobs}
    assert any(row["status"] == "SKIPPED_COMPATIBLE" for row in second["jobs"])
