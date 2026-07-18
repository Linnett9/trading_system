from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.check_selector_operational_readiness import evaluate_daily_readiness
from scripts.selector_operational_pipeline import (
    STAGES,
    DirectSelectorPipeline,
)


def _args(tmp_path, *, resume=False, stop_after=None):
    return Namespace(
        run_id="synthetic-run",
        outcome_maturity_cutoff="2027-01-01T00:00:00Z",
        evaluation_cutoff=None,
        selector_run_root=tmp_path / "parent-run",
        operational_inputs_root=tmp_path / "inputs",
        report_root=tmp_path / "reports",
        component_root=tmp_path / "components",
        state_path=tmp_path / "pipeline-state.json",
        panel_config=tmp_path / "panel-config.json",
        frozen_panel=tmp_path / "panels" / "operational.v1.json",
        selector_config=tmp_path / "selector.yaml",
        resume=resume,
        operational_input_workers=4,
        max_component_workers=3,
        weighted_capacity=4,
        stop_after=stop_after,
    )


def _parent_state(args):
    path = args.selector_run_root / "run_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "component_preflight": "preflight.json",
                    "dataset_manifest": "dataset/manifest.json",
                    "parent_gate": "parent_gate.json",
                }
            }
        ),
        encoding="utf-8",
    )


def _successful_executor(calls):
    def execute(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    return execute


def test_stage_order_and_authoritative_delegation_without_real_subprocess(tmp_path):
    args = _args(tmp_path)
    _parent_state(args)
    calls = []
    DirectSelectorPipeline(args, executor=_successful_executor(calls)).run()
    modes = [
        command[command.index("--mode") + 1]
        for command, _ in calls if "--mode" in command
    ]
    assert [row[0] for row in calls][0][0] == "powershell"
    assert any(
        "scripts/build_selector_operational_inputs.py" in command
        for command, _ in calls
    )
    assert any(
        "scripts/run_selector_component_batch.py" in command
        for command, _ in calls
    )
    assert modes == [
        "ml-selector-component-preflight",
        "ml-selector-panel-resolve",
        "ml-selector-evaluation-preflight",
    ]
    state = json.loads(args.state_path.read_text(encoding="utf-8"))
    assert state["stage_order"] == list(STAGES)
    assert all(status == "COMPLETED" for status in state["stages"].values())
    assert state["production_completion_claimed"] is False


def test_completed_skip_and_failed_stage_retry(tmp_path):
    args = _args(tmp_path, stop_after="operational_inputs")
    _parent_state(args)
    calls = []
    failures = {"remaining": 1}

    def execute(command, **kwargs):
        calls.append(list(command))
        if "scripts/build_selector_operational_inputs.py" in command:
            if failures["remaining"]:
                failures["remaining"] -= 1
                return subprocess.CompletedProcess(command, 7, "", "synthetic")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="operational_inputs failed"):
        DirectSelectorPipeline(args, executor=execute).run()
    first_state = json.loads(args.state_path.read_text())
    assert first_state["stages"]["parent_stages_1_10"] == "COMPLETED"
    assert first_state["stages"]["operational_inputs"] == "FAILED_RETRYABLE"

    resumed = _args(tmp_path, resume=True, stop_after="operational_inputs")
    DirectSelectorPipeline(resumed, executor=execute).run()
    assert sum(command[0] == "powershell" for command in calls) == 1
    assert sum(
        "scripts/build_selector_operational_inputs.py" in command
        for command in calls
    ) == 2


def test_interrupted_stage_is_recovered_as_retryable(tmp_path):
    args = _args(tmp_path, stop_after="parent_stages_1_10")
    pipeline = DirectSelectorPipeline(
        args, executor=_successful_executor([])
    )
    pipeline.state["stages"]["parent_stages_1_10"] = "RUNNING"
    pipeline.state["attempts"].append(
        {
            "stage": "parent_stages_1_10",
            "command": "interrupted",
            "started_at": "2027-01-01T00:00:00Z",
            "exit_code": None,
            "transcript": "interrupted.txt",
        }
    )
    args.state_path.write_text(json.dumps(pipeline.state), encoding="utf-8")
    calls = []
    resumed = _args(tmp_path, resume=True, stop_after="parent_stages_1_10")
    DirectSelectorPipeline(
        resumed, executor=_successful_executor(calls)
    ).run()
    state = json.loads(args.state_path.read_text())
    assert state["stages"]["parent_stages_1_10"] == "COMPLETED"
    assert len(calls) == 1
    assert "-Resume" in calls[0][0]


def test_readiness_distinguishes_code_from_production_completion():
    parent = {
        "run_state_version": "selector_parent_publication_run_state_v2",
        "stages": [
            {"stage_number": number, "status": "complete"}
            for number in range(1, 11)
        ],
    }
    daily = {"status": "READY", "whole_table_to_pylist_used": False}
    code_only = evaluate_daily_readiness(
        selector_state=parent, daily_spine_readiness=daily
    )
    assert code_only["code_readiness_status"] == "READY"
    assert code_only["production_completion_status"] == "INCOMPLETE"
    complete = evaluate_daily_readiness(
        selector_state=parent,
        daily_spine_readiness=daily,
        pipeline_state={
            "contract_version": "selector_operational_pipeline_state.v2",
            "stage_order": list(STAGES),
            "stages": {stage: "COMPLETED" for stage in STAGES},
        },
    )
    assert complete["production_completion_status"] == "COMPLETE"
    assert complete["synthetic_benchmark_evidence_used"] is False


def test_pipeline_has_no_inline_scheduler_or_performance_dependencies():
    source = Path("scripts/selector_operational_pipeline.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "run_component_jobs",
        "selector_performance",
        "research_job_performance",
        "benchmark_selector_daily_pipeline",
        "benchmark_research_job_chain",
        "evaluate_selector_components",
    ):
        assert forbidden not in source
