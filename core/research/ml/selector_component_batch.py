from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from core.research.ml.selector_component_readiness import READINESS_CONTRACT
from core.research.ml.selector_component_scheduler import run_component_jobs


BATCH_CONTRACT = "selector_stage10_component_batch.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def run_stage10_component_batch(
    *,
    readiness: Mapping[str, Any],
    input_inventory: Mapping[str, Any],
    parent_gate_path: Path,
    ledger_path: Path,
    output_root: Path,
    max_component_workers: int = 3,
    weighted_capacity: int = 4,
    runner: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
    campaign_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if readiness.get("readiness_contract_version") != READINESS_CONTRACT:
        raise ValueError("Stage-10 readiness contract mismatch")
    jobs = list(readiness.get("production_plan") or [])
    packages = _packages(
        input_inventory, expected_job_ids={str(job.get("job_id")) for job in jobs}
    )
    if set(packages) != {str(job.get("job_id")) for job in jobs}:
        raise ValueError("Stage-10 input-package coverage mismatch")
    if runner is None and any(
        not package.get("package_manifest_path")
        for package in packages.values()
    ):
        raise ValueError(
            "Default component dispatch requires package_manifest_path"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_path = output_root / "campaign_manifest.json"
    if campaign_manifest is None:
        raise ValueError("A frozen selector campaign manifest is required")
    _atomic_json(campaign_path, campaign_manifest)
    campaign_runners = {
        str(row["job_id"]): str(row.get("component_runner") or "")
        for row in campaign_manifest.get("fitted_component_matrix", ())
    }
    invoke = runner or _subprocess_runner(
        parent_gate_path=parent_gate_path,
        ledger_path=ledger_path,
        output_root=output_root,
        campaign_manifest_path=campaign_path,
        campaign_identity=str(campaign_manifest.get("campaign_identity") or ""),
        component_runners=campaign_runners,
    )

    evidence = run_component_jobs(
        jobs,
        runner=lambda job: invoke(job, packages[str(job["job_id"])]),
        max_component_workers=max_component_workers,
        capacity=weighted_capacity,
        campaign_manifest=campaign_manifest,
    )
    report = {
        "batch_contract_version": BATCH_CONTRACT,
        "readiness_logical_checksum": readiness.get("logical_checksum"),
        "job_count": len(jobs),
        "max_component_workers": max_component_workers,
        "weighted_capacity": weighted_capacity,
        "inner_model_threads": 1,
        "campaign_identity": (
            campaign_manifest.get("campaign_identity")
            if campaign_manifest else None
        ),
        "status": "FAILED" if any(row["status"] == "FAILED" for row in evidence)
        else "COMPLETED",
        "jobs": evidence,
    }
    _atomic_json(output_root / "batch_report.json", report)
    return report


def _packages(
    inventory: Mapping[str, Any],
    *,
    expected_job_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    rows = list(inventory.get("packages") or [])
    if len(rows) != len(expected_job_ids):
        if len(expected_job_ids) == 15:
            raise ValueError(
                "Stage-10 input inventory must contain exactly 15 packages"
            )
        raise ValueError(
            "Stage-10 input inventory count differs from component plan"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        if not job_id or job_id in result:
            raise ValueError(f"Duplicate or missing input-package owner: {job_id}")
        for field in ("training_rows_path", "prediction_rows_path"):
            if not row.get(field):
                raise ValueError(f"Input package missing {field}: {job_id}")
        result[job_id] = row
    return result


def _subprocess_runner(
    *,
    parent_gate_path: Path,
    ledger_path: Path,
    output_root: Path,
    campaign_manifest_path: Path,
    campaign_identity: str,
    component_runners: Mapping[str, str],
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
    def invoke(
        job: Mapping[str, Any], package: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        safe_id = str(job["job_id"]).replace(":", "_")
        job_path = output_root / "jobs" / f"{safe_id}.json"
        report_path = output_root / "components" / f"{safe_id}.json"
        transcript_path = output_root / "transcripts" / f"{safe_id}.txt"
        _atomic_json(job_path, dict(job))
        command = [
            sys.executable, "main.py", "--mode", "ml-selector-component-publish",
            "--production-plan-job", str(job_path),
            "--campaign-manifest", str(campaign_manifest_path),
            "--campaign-identity", campaign_identity,
            "--plan-job-identity", str(job["job_id"]),
            "--component-runner", component_runners.get(
                str(job["job_id"]), ""
            ),
            "--operational-input-package", str(package["package_manifest_path"]),
            "--parent-gate", str(parent_gate_path),
            "--training-rows-json", str(package["training_rows_path"]),
            "--prediction-rows-json", str(package["prediction_rows_path"]),
            "--experiment-ledger", str(ledger_path),
            "--verification-output", str(report_path),
        ]
        environment = os.environ.copy()
        environment.update({name: "1" for name in THREAD_VARIABLES})
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(
                f"Component command failed ({result.returncode}): {job['job_id']}"
            )
        return json.loads(report_path.read_text(encoding="utf-8"))

    return invoke


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)
