from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260716T091011Z"
EXPECTED_PARTITIONS = 5654
DATES = ("2024-03-15", "2024-09-16", "2025-03-17", "2025-09-15", "2026-03-16")
MODELS = ("ridge", "elastic_net", "ordered_logit_ranker")
EVALUATION_MODELS = ("momentum_120d", *MODELS)
STATE_VERSION = "post_finaliser_pipeline_state.v1"
VALID_STATES = {
    "WAITING", "READY", "RUNNING", "COMPLETED", "SKIPPED_COMPATIBLE",
    "BLOCKED", "FAILED_RETRYABLE", "FAILED_TERMINAL",
}
FORBIDDEN_MODELS = {
    "huber", "contextual_elastic_net", "multi_horizon_ridge",
    "multi_horizon_elastic_net", "multi_horizon_ordered_logit",
    "rank_xendcg", "lambdarank",
}
ARCHIVE_COMMAND = (
    "python scripts/finalize_alpaca_5m_symbol_year_archive.py "
    "--config config/config.alpaca_5m_symbol_year_finalizer_production.yaml --validate-archive"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def validate_progress(progress: Mapping[str, Any]) -> list[str]:
    reasons = []
    expected = {
        "planned_partitions": EXPECTED_PARTITIONS,
        "completed_partitions": EXPECTED_PARTITIONS,
        "pending_partitions": 0,
        "failed_partitions": 0,
        "invalid_rows": 0,
    }
    for key, value in expected.items():
        if int(progress.get(key, -1)) != value:
            reasons.append(f"{key.upper()}:{progress.get(key)}")
    for key in ("conflicting_duplicates", "temporary_files", "temporary_files_left_behind"):
        value = progress.get(key, 0)
        count = len(value) if isinstance(value, list) else int(value or 0)
        if count:
            reasons.append(f"{key.upper()}:{count}")
    return reasons


def validate_archive(report: Mapping[str, Any]) -> list[str]:
    aliases = {
        "planned_partitions": ("planned_partitions", "partition_count"),
        "completed_partitions": ("completed_partitions", "partition_count"),
        "validated_partitions": ("validated_partitions", "partition_count"),
    }
    reasons = []
    for label, names in aliases.items():
        value = next((report.get(name) for name in names if report.get(name) is not None), -1)
        if int(value) != EXPECTED_PARTITIONS:
            reasons.append(f"{label.upper()}:{value}")
    zero_fields = (
        "missing_partitions", "failed_partitions", "invalid_rows",
        "duplicate_partition_owners",
    )
    for key in zero_fields:
        if int(report.get(key, 0) or 0):
            reasons.append(f"{key.upper()}:{report.get(key)}")
    temporary = report.get("temporary_files_left_behind", report.get("temporary_files", []))
    if len(temporary) if isinstance(temporary, list) else int(temporary or 0):
        reasons.append("TEMPORARY_FILES")
    valid = report.get("archive_valid", report.get("valid"))
    if valid is not True:
        reasons.append("ARCHIVE_VALID_FALSE")
    # Newer validator contracts must carry these; legacy output is accepted only when
    # the committed validator has not yet added the fields.
    for key in ("expected_symbol_year_coverage", "archive_inventory_checksum"):
        if key in report and not report.get(key):
            reasons.append(f"{key.upper()}_MISSING")
    return reasons


def validate_component_plan(readiness: Mapping[str, Any]) -> list[str]:
    reasons = []
    if readiness.get("campaign") != "base":
        reasons.append("CAMPAIGN_NOT_BASE")
    if tuple(readiness.get("required_models", ())) != MODELS:
        reasons.append("MODEL_ROSTER_MISMATCH")
    if tuple(readiness.get("required_dates", ())) != DATES:
        reasons.append("DATE_ROSTER_MISMATCH")
    if int(readiness.get("expected_component_count", -1)) != 15:
        reasons.append("EXPECTED_COMPONENT_COUNT_MISMATCH")
    plan = list(readiness.get("production_plan") or [])
    planned_models = {str(row.get("model_id")) for row in plan}
    if planned_models & FORBIDDEN_MODELS or any(model not in MODELS for model in planned_models):
        reasons.append("CHALLENGER_IN_BASE_PLAN")
    if len({str(row.get("job_id")) for row in plan}) != len(plan):
        reasons.append("DUPLICATE_COMPONENT_JOB")
    return reasons


def validate_final_components(readiness: Mapping[str, Any]) -> list[str]:
    reasons = validate_component_plan(readiness)
    checks = {
        "ready_component_count": 15,
        "missing_component_count": 0,
        "invalid_component_count": 0,
    }
    for key, expected in checks.items():
        if int(readiness.get(key, -1)) != expected:
            reasons.append(f"{key.upper()}:{readiness.get(key)}")
    matched = list(readiness.get("matched_population_results") or [])
    if len(matched) != 5 or any(row.get("status") != "READY" for row in matched):
        reasons.append("MATCHED_DATE_PANELS_NOT_READY")
    if readiness.get("overall_status") != "READY":
        reasons.append("COMPONENT_READINESS_NOT_READY")
    return reasons


def stage_mapping(runbook: Path) -> list[dict[str, Any]]:
    text = runbook.read_text(encoding="utf-8-sig")
    rows = []
    import re
    for number, name in re.findall(r"@\{number=(\d+);\s*name='([^']+)'", text):
        rows.append({"stage_number": int(number), "name": name})
    if [row["stage_number"] for row in rows] != list(range(1, 17)):
        raise ValueError("Runbook does not expose the exact 16-stage mapping")
    return rows


def next_incomplete_stage(run_state: Mapping[str, Any]) -> int:
    if run_state.get("run_id") != RUN_ID:
        raise ValueError("SELECTOR_RUN_ID_MISMATCH")
    if run_state.get("run_state_version") != "selector_parent_publication_run_state_v2":
        raise ValueError("SELECTOR_STATE_SCHEMA_MISMATCH")
    stages = list(run_state.get("stages") or [])
    if [int(row.get("stage_number", -1)) for row in stages] != list(range(1, 17)):
        raise ValueError("SELECTOR_STAGE_MAPPING_MISMATCH")
    return next((int(row["stage_number"]) for row in stages[:10] if row.get("status") != "complete"), 11)


def new_state(root: Path, args: argparse.Namespace, mapping: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = [
        ("DOWNSTREAM-REPLAY", "Replay lineage", "BLOCKED_IMPLEMENTATION"),
        ("DOWNSTREAM-POLICY", "Base policy/cost evaluation", "BLOCKED_REPLAY_LINEAGE"),
        ("DOWNSTREAM-WAVE4", "Wave 4 campaign", "BLOCKED_CAMPAIGN_FREEZE"),
        ("DOWNSTREAM-LGBM", "LightGBM challengers", "BLOCKED_COMPONENT_ADAPTER"),
        ("DOWNSTREAM-STATS", "Statistical gate", "BLOCKED_PORTFOLIO_RESULTS"),
        ("DOWNSTREAM-WAVE6", "Wave 6 decision", "BLOCKED_PROMOTION_GATE"),
    ]
    jobs = [
        _job("PHASE-0", "Wait for safe finaliser quiescence", []),
        _job("PHASE-1", "Validate complete archive", ["PHASE-0"]),
        _job("PHASE-2", "Operational readiness", ["PHASE-1"]),
        _job("PHASE-3", "Resume selector parent publication", ["PHASE-2"]),
        _job("PHASE-4", "Validate base component plan", ["PHASE-3"]),
        _job("PHASE-5", "Build or verify operational input inventory", ["PHASE-4"]),
        _job("PHASE-6", "Publish base components serially", ["PHASE-5"]),
        _job("PHASE-7", "Validate all base components", ["PHASE-6"]),
        _job("PHASE-8", "Freeze operational panel", ["PHASE-7"]),
        _job("PHASE-9", "Evaluate selector components", ["PHASE-8"]),
    ]
    jobs.extend({**_job(job_id, title, ["PHASE-9"]), "state": "BLOCKED", "blocker": blocker,
                 "retryability": False} for job_id, title, blocker in blocked)
    state = {
        "contract_version": STATE_VERSION, "controller_run_id": args.controller_run_id,
        "selector_run_id": RUN_ID, "created_at": utc_now(), "updated_at": utc_now(),
        "repository": str(REPO_ROOT), "runbook_mapping": list(mapping),
        "parameters": {
            "finaliser_manifest": str(args.finaliser_manifest),
            "poll_seconds": args.poll_seconds, "finaliser_process_id": args.finaliser_process_id,
        },
        "inventory_bootstrap": {
            "strategy": args.inventory_strategy,
            "plan_path": None, "dataset_path": None, "parent_gate_path": None,
            "input_output_root": str(args.operational_inputs_output_root) if args.operational_inputs_output_root else None,
            "evaluation_cutoff": args.evaluation_cutoff,
            "build_command": None,
            "inventory_path": str(args.component_input_inventory) if args.component_input_inventory else None,
            "inventory_logical_checksum": None, "build_attempt_count": 0,
            "verification_status": "WAITING_FOR_STAGE_10",
        },
        "jobs": jobs, "attempt_history": [], "commands_resolved": {},
    }
    state["logical_checksum"] = state_checksum(state)
    return state


def _job(job_id: str, title: str, dependencies: list[str]) -> dict[str, Any]:
    return {
        "job_id": job_id, "title": title, "dependencies": dependencies, "state": "WAITING",
        "attempts": 0, "started_at": None, "completed_at": None, "command": None,
        "transcript_path": None, "report_path": None, "logical_checksum": None,
        "exit_code": None, "blocker": None, "retryability": True, "next_legal_command": None,
    }


def state_checksum(state: Mapping[str, Any]) -> str:
    return canonical_hash({key: value for key, value in state.items() if key != "logical_checksum"})


class Pipeline:
    def __init__(self, args: argparse.Namespace, executor: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        configure_inventory_strategy(args)
        self.args, self.executor = args, executor
        self.state_path = args.state_path
        self.run_root = self.state_path.parent
        mapping = stage_mapping(REPO_ROOT / "scripts/selector_parent_publication_runbook.ps1")
        if self.state_path.exists():
            if not args.resume:
                raise ValueError(f"Existing controller state requires -Resume: {self.state_path}")
            self.state = read_json(self.state_path)
            if self.state.get("contract_version") != STATE_VERSION:
                raise ValueError("INCOMPATIBLE_CONTROLLER_STATE")
            if self.state.get("logical_checksum") != state_checksum(self.state):
                raise ValueError("CONTROLLER_STATE_CHECKSUM_MISMATCH")
            prior = self.state.get("inventory_bootstrap", {})
            if prior.get("strategy") != args.inventory_strategy or prior.get("evaluation_cutoff") != args.evaluation_cutoff:
                raise ValueError("INCOMPATIBLE_INVENTORY_STRATEGY_OR_EVALUATION_CUTOFF")
        else:
            self.state = new_state(self.run_root, args, mapping)
            self.save()

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["logical_checksum"] = state_checksum(self.state)
        atomic_write(self.state_path, self.state)

    def job(self, job_id: str) -> dict[str, Any]:
        return next(row for row in self.state["jobs"] if row["job_id"] == job_id)

    def complete(self, job_id: str, *, command: str | None = None, report: Path | None = None,
                 transcript: Path | None = None, skipped: bool = False) -> None:
        row = self.job(job_id)
        row.update(state="SKIPPED_COMPATIBLE" if skipped else "COMPLETED", completed_at=utc_now(),
                   exit_code=0, command=command, report_path=str(report) if report else None,
                   transcript_path=str(transcript) if transcript else None, blocker=None)
        row["logical_checksum"] = canonical_hash({k: v for k, v in row.items() if k != "logical_checksum"})
        self.save()

    def run_command(self, job_id: str, command: Sequence[str], transcript: Path, report: Path | None = None,
                    capture_peak_memory: bool = False) -> None:
        row = self.job(job_id)
        row.update(state="RUNNING", attempts=int(row["attempts"]) + 1, started_at=utc_now(),
                   command=subprocess.list2cmdline(list(command)), transcript_path=str(transcript),
                   report_path=str(report) if report else None)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        self.save()
        if capture_peak_memory and self.executor is subprocess.run:
            result, peak = self._run_with_peak_memory(command)
            peak_report = self.run_root / "reports/selector_resume_peak_memory.json"
            atomic_write(peak_report, {
                "contract_version": "post_finaliser_peak_memory.v1",
                "command": row["command"], "peak_working_set_bytes": peak,
                "scope": "runbook_process_tree", "captured_at": utc_now(),
            })
            row["peak_memory_report"] = str(peak_report)
        else:
            result = self.executor(list(command), cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        transcript.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        self.state["attempt_history"].append({
            "job_id": job_id, "attempt": row["attempts"], "started_at": row["started_at"],
            "completed_at": utc_now(), "command": row["command"], "transcript_path": str(transcript),
            "exit_code": result.returncode,
        })
        if result.returncode:
            row.update(state="FAILED_RETRYABLE", completed_at=utc_now(), exit_code=result.returncode,
                       blocker=f"COMMAND_EXIT:{result.returncode}",
                       next_legal_command=self.resume_command())
            self.save()
            raise RuntimeError(f"{job_id} failed; see {transcript}")
        self.complete(job_id, command=row["command"], report=report, transcript=transcript)

    def _run_with_peak_memory(self, command: Sequence[str]) -> tuple[subprocess.CompletedProcess, int]:
        process = subprocess.Popen(
            list(command), cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        peak = 0
        while process.poll() is None:
            query = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "$root={0}; $all=@(Get-CimInstance Win32_Process); "
                 "$ids=@($root); do {{$before=$ids.Count; "
                 "$ids += @($all | Where-Object {{$ids -contains $_.ParentProcessId}} | "
                 "Select-Object -ExpandProperty ProcessId); $ids=@($ids | Sort-Object -Unique)}} "
                 "while($ids.Count -gt $before); "
                 "(@(Get-Process -Id $ids -ErrorAction SilentlyContinue) | "
                 "Measure-Object WorkingSet64 -Sum).Sum".format(process.pid)],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )
            try:
                peak = max(peak, int((query.stdout or "0").strip() or 0))
            except ValueError:
                pass
            time.sleep(0.5)
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr), peak

    def resume_command(self) -> str:
        command = (
            f'powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_post_finaliser_pipeline.ps1 '
            f'-Resume -ControllerRunId "{self.args.controller_run_id}" -WaitForFinaliser '
            f'-FinaliserManifest "{self.args.finaliser_manifest}"'
        )
        if self.args.inventory_strategy == "prebuilt":
            command += f' -ComponentInputInventory "{self.args.component_input_inventory}"'
        else:
            command += (
                f' -OperationalInputsOutputRoot "{self.args.operational_inputs_output_root}"'
                f' -EvaluationCutoff "{self.args.evaluation_cutoff}"'
            )
        return command

    def wait_for_finaliser(self) -> None:
        if self.job("PHASE-0")["state"] in {"COMPLETED", "SKIPPED_COMPATIBLE"}:
            return
        if not self.args.wait_for_finaliser:
            raise RuntimeError("Finaliser is not yet proven quiescent; use -WaitForFinaliser")
        previous = None
        while True:
            progress = read_json(self.args.finaliser_manifest)
            digest = canonical_hash(progress)
            reasons = validate_progress(progress)
            active = self._finaliser_active()
            temporary = self._temporary_outputs()
            if not reasons and not active and not temporary and digest == previous:
                self.complete("PHASE-0", report=self.args.finaliser_manifest)
                return
            previous = digest
            if reasons and any(value.startswith(("FAILED_PARTITIONS:", "INVALID_ROWS:")) for value in reasons):
                raise RuntimeError(f"Finaliser failure: {reasons}")
            time.sleep(self.args.poll_seconds)

    def _finaliser_active(self) -> bool:
        if self.args.finaliser_process_id:
            result = self.executor(
                ["powershell", "-NoProfile", "-Command",
                 f"Get-Process -Id {self.args.finaliser_process_id} -ErrorAction SilentlyContinue"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        result = self.executor(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
             "'finalize_alpaca_5m_symbol_year_archive.py.*--execute' } | Select-Object -ExpandProperty ProcessId"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        return bool(result.stdout.strip())

    def _temporary_outputs(self) -> list[Path]:
        root = self.args.archive_root
        return list(root.rglob("*.tmp")) + list(root.rglob("*.partial")) if root.exists() else []

    def execute(self) -> None:
        self.wait_for_finaliser()
        validation = self.args.archive_validation
        if self.job("PHASE-1")["state"] not in {"COMPLETED", "SKIPPED_COMPATIBLE"}:
            transcript = self.run_root / "transcripts/archive_validation_attempt_1.txt"
            self.run_command("PHASE-1", shlex.split(ARCHIVE_COMMAND), transcript, validation)
            reasons = validate_archive(read_json(validation))
            if reasons:
                row = self.job("PHASE-1")
                row.update(state="FAILED_TERMINAL", blocker=";".join(reasons), retryability=False)
                self.save()
                raise RuntimeError(f"Archive validation gate failed: {reasons}")
        if self._stop("archive_validation"): return
        readiness = self.run_root / "reports/operational_readiness.json"
        readiness_cmd = [
            sys.executable, "scripts/check_post_finaliser_job_readiness.py", "--json",
            "--selector-run-id", RUN_ID, "--progress", str(self.args.finaliser_manifest),
            "--archive-validation", str(validation), "--selector-state", str(self.args.selector_state),
            "--component-readiness", str(self.args.component_readiness),
        ]
        if self.job("PHASE-2")["state"] not in {"COMPLETED", "SKIPPED_COMPATIBLE"}:
            transcript = self.run_root / "transcripts/readiness_attempt_1.txt"
            self.run_command("PHASE-2", readiness_cmd, transcript)
            payload = json.loads(transcript.read_text(encoding="utf-8"))
            atomic_write(readiness, payload)
            if not payload.get("ready_to_resume"):
                raise RuntimeError("Operational readiness is not READY")
            self.complete("PHASE-2", command=subprocess.list2cmdline(readiness_cmd), report=readiness, transcript=transcript)
        run_state = read_json(self.args.selector_state)
        start = next_incomplete_stage(run_state)
        if start <= 10 and self.job("PHASE-3")["state"] not in {"COMPLETED", "SKIPPED_COMPATIBLE"}:
            transcript = self.run_root / "transcripts/selector_resume.txt"
            command = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                "scripts/selector_parent_publication_runbook.ps1", "-Resume",
                "-FromStage", str(start), "-ThroughStage", "10", "-RunId", RUN_ID,
                "-TranscriptPath", str(transcript),
            ]
            self.run_command(
                "PHASE-3", command, transcript, self.args.selector_state,
                capture_peak_memory=True,
            )
            if next_incomplete_stage(read_json(self.args.selector_state)) <= 10:
                raise RuntimeError("Selector stages were not proven complete by run state")
        elif start == 11:
            self.complete("PHASE-3", report=self.args.selector_state, skipped=True)
        if self._stop("selector_stage_10"): return
        component = read_json(self.args.component_readiness)
        reasons = validate_component_plan(component)
        if reasons:
            raise RuntimeError(f"Component plan rejected: {reasons}")
        self.complete("PHASE-4", report=self.args.component_readiness)
        self.bootstrap_inventory(component)
        if self._stop("input_inventory"): return
        self.publish_components(readiness)
        if self._stop("component_publication"): return
        final = read_json(self.args.component_readiness)
        reasons = validate_final_components(final)
        if reasons:
            raise RuntimeError(f"Final component gate failed: {reasons}")
        self.complete("PHASE-7", report=self.args.component_readiness)
        panel = self.freeze_operational_panel(final)
        self.complete("PHASE-8", report=panel)
        evaluation = self.evaluate(final, panel)
        self.complete("PHASE-9", report=evaluation)
        for row in self.state["jobs"]:
            if row["job_id"].startswith("DOWNSTREAM-"):
                row["next_legal_command"] = None
        self.save()

    def _stop(self, phase: str) -> bool:
        return self.args.stop_after_phase == phase

    def bootstrap_inventory(self, readiness: Mapping[str, Any]) -> Path:
        from core.research.ml.selector_operational_inputs import validate_inventory
        run_state = read_json(self.args.selector_state)
        stages = {int(row["stage_number"]): row for row in run_state.get("stages", [])}
        if stages.get(10, {}).get("status") != "complete":
            raise RuntimeError("WAITING_FOR_STAGE_10")
        artifacts = dict(run_state.get("artifacts") or {})
        plan_path = Path(str(artifacts.get("component_preflight", "")))
        dataset_manifest = Path(str(artifacts.get("dataset_manifest", "")))
        parent_gate = Path(str(artifacts.get("parent_gate", "")))
        if plan_path.resolve() != self.args.component_readiness.resolve():
            raise RuntimeError("STAGE_10_PLAN_PATH_MISMATCH")
        for label, path in (("plan", plan_path), ("dataset", dataset_manifest), ("parent_gate", parent_gate)):
            if not path.is_file(): raise RuntimeError(f"VERIFIED_STAGE_ARTIFACT_MISSING:{label}:{path}")
        bootstrap = self.state["inventory_bootstrap"]
        bootstrap.update(plan_path=str(plan_path), dataset_path=str(dataset_manifest.parent), parent_gate_path=str(parent_gate))
        if self.args.inventory_strategy == "automatic":
            inventory = self.args.operational_inputs_output_root / "inventory.json"
            command = [
                sys.executable, "scripts/build_selector_operational_inputs.py",
                "--plan", str(plan_path), "--selector-dataset", str(dataset_manifest.parent),
                "--parent-gate", str(parent_gate), "--output-root", str(self.args.operational_inputs_output_root),
                "--evaluation-cutoff", self.args.evaluation_cutoff, "--selector-run-id", RUN_ID,
            ]
            bootstrap.update(build_command=subprocess.list2cmdline(command), inventory_path=str(inventory))
            if not inventory.exists():
                bootstrap["verification_status"] = "BUILDING"
                bootstrap["build_attempt_count"] = int(bootstrap["build_attempt_count"]) + 1
                self.save()
                self.run_command("PHASE-5", command, self.run_root / "transcripts/operational_inputs_build.txt", inventory)
        else:
            inventory = self.args.component_input_inventory
        dataset = read_json(dataset_manifest); gate = read_json(parent_gate)
        result = validate_inventory(
            inventory, readiness=readiness, expected_run_id=RUN_ID,
            expected_dataset_id=dataset.get("dataset_id"), expected_dataset_checksum=dataset.get("dataset_checksum"),
            expected_parent_gate_checksum=gate.get("logical_checksum"),
            expected_evaluation_cutoff=self.args.evaluation_cutoff,
        )
        if result["status"] != "READY":
            bootstrap["verification_status"] = "FAILED_TERMINAL"
            self.save(); raise RuntimeError(f"Operational input inventory rejected: {result['reasons']}")
        self.args.component_input_inventory = inventory
        bootstrap.update(verification_status="VERIFIED", inventory_logical_checksum=result["inventory"]["logical_checksum"])
        self.complete("PHASE-5", report=inventory, skipped=self.args.inventory_strategy == "prebuilt" or bootstrap["build_attempt_count"] == 0)
        return inventory

    def publish_components(self, readiness_report: Path) -> None:
        from core.research.ml.selector_operational_inputs import validate_inventory
        validated = validate_inventory(self.args.component_input_inventory)
        if validated["status"] != "READY":
            raise RuntimeError(f"Component input inventory rejected: {validated['reasons']}")
        inventory = validated["inventory"]
        owners = {str(row["job_id"]): row for row in inventory["packages"]}
        if self.job("PHASE-6")["state"] in {"COMPLETED", "SKIPPED_COMPATIBLE"}:
            return
        while True:
            before = read_json(self.args.component_readiness)
            plan = list(before.get("production_plan") or [])
            if not plan:
                self.complete("PHASE-6", report=self.args.component_readiness,
                              skipped=int(before.get("ready_component_count", 0)) == 15)
                return
            job = plan[0]
            job_id = str(job["job_id"])
            if str(job.get("model_id")) not in MODELS:
                raise RuntimeError("Challenger attempted in base campaign")
            owner = owners.get(job_id)
            if owner is None:
                raise RuntimeError(f"Inventory does not own planned job: {job_id}")
            package_manifest = read_json(Path(owner["package_manifest_path"]))
            if package_manifest.get("production_plan_job_checksum") != job.get("logical_checksum"):
                raise RuntimeError(f"Inventory plan checksum mismatch for {job_id}")
            training = Path(owner["training_rows_path"])
            prediction = Path(owner["prediction_rows_path"])
            if not training.is_file() or not prediction.is_file():
                raise RuntimeError(f"Required input owner missing for {job_id}: {training}, {prediction}")
            selected = self.run_root / "selected_jobs" / f"{job_id.replace(':', '_')}.json"
            atomic_write(selected, job)
            transcript = self.run_root / "transcripts" / f"component_{job_id.replace(':', '_')}.txt"
            report = self.run_root / "reports" / f"component_{job_id.replace(':', '_')}.json"
            command = [
                sys.executable, "main.py", "--mode", "ml-selector-component-publish",
                "--production-plan-job", str(selected), "--parent-gate", str(self.args.parent_gate),
                "--training-rows-json", str(training), "--prediction-rows-json", str(prediction),
                "--experiment-ledger", str(self.run_root / "experiment_ledger.jsonl"),
                "--verification-output", str(report),
            ]
            self.run_command("PHASE-6", command, transcript, report)
            revalidate = [
                sys.executable, "main.py", "--mode", "ml-selector-component-preflight",
                "--parent-gate", str(self.args.parent_gate),
                "--selector-dataset-root", str(self.args.selector_dataset_root),
                "--component-output-root", str(self.args.component_root),
                "--approved-component-root", str(self.args.component_root),
                "--config", "config/config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml",
                "--verification-output", str(self.args.component_readiness),
            ]
            result = self.executor(revalidate, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(f"Component readiness revalidation failed after {job_id}")
            after = read_json(self.args.component_readiness)
            if int(after.get("ready_component_count", 0)) <= int(before.get("ready_component_count", 0)):
                raise RuntimeError(f"Component readiness did not advance after {job_id}")
            self.job("PHASE-6")["state"] = "RUNNING"
            self.save()

    def freeze_operational_panel(self, readiness: Mapping[str, Any]) -> Path:
        from core.research.ml.registries import load_registry_bundle
        from core.research.ml.selector_panel_preflight import (
            CHALLENGERS, PRIMARY_MODEL, discover_selector_components, freeze_panel,
            resolve_authoritative_panel,
        )
        components = discover_selector_components(
            [self.args.component_root], models=(PRIMARY_MODEL, *CHALLENGERS)
        )
        bundle = load_registry_bundle()
        panel = resolve_authoritative_panel(
            panel_name="selector_operational_panel_v1", panel_version="v1",
            requested_dates=DATES, components=components, primary_model=PRIMARY_MODEL,
            challengers=CHALLENGERS, registry_set_hash=bundle.registry_set_hash,
            policy_registry_hash=bundle.documents["portfolio_policies"].registry_hash,
        )
        if panel.get("status") != "READY" or tuple(panel.get("resolved_dates", ())) != DATES:
            raise RuntimeError("Operational panel freeze gate failed")
        output = self.run_root / "frozen/selector_operational_panel_v1.v1.json"
        freeze_panel(output, panel)
        return output

    def evaluate(self, readiness: Mapping[str, Any], panel: Path) -> Path:
        from core.research.ml.selector_component_evaluation import evaluate_selector_components
        from core.research.ml.selector_operational_inputs import validate_inventory
        inventory = validate_inventory(self.args.component_input_inventory)["inventory"]
        manifests = sorted(self.args.component_root.glob("model=*/date=*/manifest.json"))
        output = self.run_root / "evaluation"
        result = evaluate_selector_components(
            readiness_path=self.args.component_readiness, component_manifests=manifests,
            outcome_path=Path(inventory["mature_outcome_path"]), output_root=output,
            ledger_path=self.run_root / "experiment_ledger.jsonl",
            panel_id=read_json(panel)["panel_id"], evaluation_cutoff=inventory["evaluation_cutoff"],
            required_models=EVALUATION_MODELS, required_dates=DATES,
        )
        if result.get("evaluation_status") != "READY":
            raise RuntimeError("Selector component evaluation failed")
        return output / "evaluation.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--controller-run-id", required=True)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--wait-for-finaliser", action="store_true")
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--finaliser-manifest", type=Path, required=True)
    value.add_argument("--finaliser-process-id", type=int)
    value.add_argument("--archive-root", type=Path, default=Path("data/processed/market_data/alpaca_5m_symbol_year"))
    value.add_argument("--archive-validation", type=Path, default=Path("reports/data_quality/alpaca_5m_symbol_year_finalisation/archive_validation.json"))
    value.add_argument("--selector-state", type=Path, default=Path(f"reports/ml/readiness/selector_evaluation_1c_e/runs/{RUN_ID}/run_state.json"))
    value.add_argument("--parent-gate", type=Path, default=Path(f"reports/ml/readiness/selector_evaluation_1c_e/runs/{RUN_ID}/selector_parent_gate.json"))
    value.add_argument("--component-readiness", type=Path, default=Path(f"reports/ml/readiness/selector_evaluation_1c_e/runs/{RUN_ID}/component_preflight_v2.json"))
    value.add_argument("--selector-dataset-root", type=Path, default=Path(f"reports/ml/readiness/canonical_v2_selector_dataset_v2/run={RUN_ID}/frozen"))
    value.add_argument("--component-root", type=Path, default=Path(f"reports/ml/selector_components/operational_v2/run={RUN_ID}"))
    value.add_argument("--component-input-inventory", type=Path)
    value.add_argument("--operational-inputs-output-root", type=Path)
    value.add_argument("--evaluation-cutoff")
    value.add_argument("--stop-after-phase", choices=(
        "archive_validation", "selector_stage_10", "input_inventory",
        "component_publication", "selector_evaluation",
    ))
    value.add_argument("--state-path", type=Path)
    return value


def configure_inventory_strategy(args: argparse.Namespace) -> None:
    prebuilt = getattr(args, "component_input_inventory", None)
    output = getattr(args, "operational_inputs_output_root", None)
    cutoff = getattr(args, "evaluation_cutoff", None)
    if prebuilt and (output or cutoff):
        raise ValueError("Conflicting inventory strategies")
    if not prebuilt and not (output and cutoff):
        raise ValueError("Exactly one inventory strategy is required")
    if prebuilt:
        args.inventory_strategy = "prebuilt"
        payload = read_json(prebuilt)
        args.evaluation_cutoff = payload.get("evaluation_cutoff")
        if not args.evaluation_cutoff:
            raise ValueError("Prebuilt inventory has no explicit evaluation cutoff")
    else:
        args.inventory_strategy = "automatic"
        approved = (REPO_ROOT / f"reports/ml/readiness/selector_evaluation_1c_e/runs/{RUN_ID}").resolve()
        candidate = (REPO_ROOT / output).resolve() if not output.is_absolute() else output.resolve()
        try: candidate.relative_to(approved)
        except ValueError as exc: raise ValueError("Operational input output root is outside run ownership") from exc
        args.operational_inputs_output_root = candidate
    try:
        datetime.fromisoformat(str(args.evaluation_cutoff).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Evaluation cutoff must be an explicit ISO timestamp") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.poll_seconds < 60:
        raise SystemExit("-PollSeconds must be at least 60")
    if args.state_path is None:
        args.state_path = Path("reports/operations/post_finaliser_pipeline") / args.controller_run_id / "state.json"
    pipeline = Pipeline(args)
    pipeline.execute()
    print(json.dumps({"status": "COMPLETED", "state": str(args.state_path),
                      "resume_command": pipeline.resume_command()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
