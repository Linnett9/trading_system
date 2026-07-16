from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.post_finaliser_pipeline import (
    DATES, MODELS, RUN_ID, Pipeline, atomic_write, new_state, next_incomplete_stage,
    stage_mapping, state_checksum, validate_archive, validate_component_plan,
    validate_final_components, validate_progress,
)


def progress(**changes):
    value = dict(planned_partitions=5654, completed_partitions=5654, pending_partitions=0,
                 failed_partitions=0, invalid_rows=0, temporary_files=0)
    value.update(changes)
    return value


def archive(**changes):
    value = dict(partition_count=5654, invalid_rows=0, temporary_files_left_behind=[], valid=True)
    value.update(changes)
    return value


def readiness(**changes):
    value = dict(campaign="base", required_models=list(MODELS), required_dates=list(DATES),
                 expected_component_count=15, ready_component_count=15,
                 missing_component_count=0, invalid_component_count=0,
                 production_plan=[], overall_status="READY",
                 matched_population_results=[{"status": "READY"} for _ in DATES])
    value.update(changes)
    return value


@pytest.mark.parametrize("payload,reason", [
    (progress(completed_partitions=5653), "COMPLETED_PARTITIONS"),
    (progress(failed_partitions=1), "FAILED_PARTITIONS"),
    (progress(temporary_files=1), "TEMPORARY_FILES"),
])
def test_progress_rejections(payload, reason):
    assert any(item.startswith(reason) for item in validate_progress(payload))


def test_full_progress_accepts():
    assert validate_progress(progress()) == []


def test_partial_660_archive_rejected():
    assert validate_archive(archive(partition_count=660))


def test_full_archive_accepted():
    assert validate_archive(archive()) == []


@pytest.mark.parametrize("change", [
    {"campaign": "wave4_challengers"},
    {"required_models": ["ridge", "elastic_net", "lambdarank"]},
    {"expected_component_count": 16},
    {"production_plan": [{"job_id": "x", "model_id": "rank_xendcg"}]},
])
def test_base_plan_rejections(change):
    assert validate_component_plan(readiness(**change))


def test_exact_fifteen_plan():
    plan = [{"job_id": f"{date}:{model}", "model_id": model} for date in DATES for model in MODELS]
    assert validate_component_plan(readiness(production_plan=plan)) == []


@pytest.mark.parametrize("change", [
    {"ready_component_count": 14},
    {"missing_component_count": 1},
    {"invalid_component_count": 1},
    {"matched_population_results": [{"status": "READY"}] * 4},
])
def test_final_component_gate(change):
    assert validate_final_components(readiness(**change))


def test_runbook_has_exact_sixteen_stages():
    rows = stage_mapping(Path("scripts/selector_parent_publication_runbook.ps1"))
    assert [row["stage_number"] for row in rows] == list(range(1, 17))


def test_selector_resume_preserves_completed_stages():
    state = {"run_id": RUN_ID, "run_state_version": "selector_parent_publication_run_state_v2",
             "stages": [{"stage_number": n, "status": "complete" if n < 7 else "pending"} for n in range(1, 17)]}
    assert next_incomplete_stage(state) == 7


def test_selector_run_id_is_exact():
    state = {"run_id": "wrong", "run_state_version": "selector_parent_publication_run_state_v2", "stages": []}
    with pytest.raises(ValueError, match="RUN_ID"):
        next_incomplete_stage(state)


def args(tmp_path, resume=False):
    return Namespace(
        controller_run_id="test", resume=resume, wait_for_finaliser=True, poll_seconds=60,
        finaliser_manifest=tmp_path / "progress.json", finaliser_process_id=None,
        archive_root=tmp_path / "archive", archive_validation=tmp_path / "validation.json",
        selector_state=tmp_path / "selector.json", parent_gate=tmp_path / "gate.json",
        component_readiness=tmp_path / "components.json", selector_dataset_root=tmp_path / "dataset",
        component_root=tmp_path / "owners", component_input_root=tmp_path / "inputs",
        outcome_path=tmp_path / "outcomes.csv", evaluation_cutoff="2026-07-16",
        state_path=tmp_path / "state.json",
    )


def test_existing_state_requires_resume(tmp_path):
    a = args(tmp_path)
    atomic_write(a.state_path, new_state(tmp_path, a, stage_mapping(Path("scripts/selector_parent_publication_runbook.ps1"))))
    with pytest.raises(ValueError, match="requires -Resume"):
        Pipeline(a)


def test_atomic_state_has_no_temporary_leftover(tmp_path):
    path = tmp_path / "state.json"
    atomic_write(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_state_checksum_detects_change(tmp_path):
    a = args(tmp_path)
    state = new_state(tmp_path, a, stage_mapping(Path("scripts/selector_parent_publication_runbook.ps1")))
    supplied = state["logical_checksum"]
    state["selector_run_id"] = "changed"
    assert supplied != state_checksum(state)


def test_downstream_jobs_are_blocked_not_failed(tmp_path):
    a = args(tmp_path)
    state = new_state(tmp_path, a, stage_mapping(Path("scripts/selector_parent_publication_runbook.ps1")))
    downstream = [row for row in state["jobs"] if row["job_id"].startswith("DOWNSTREAM")]
    assert {row["state"] for row in downstream} == {"BLOCKED"}
    assert {row["blocker"] for row in downstream} == {
        "BLOCKED_IMPLEMENTATION", "BLOCKED_REPLAY_LINEAGE", "BLOCKED_CAMPAIGN_FREEZE",
        "BLOCKED_COMPONENT_ADAPTER", "BLOCKED_PORTFOLIO_RESULTS", "BLOCKED_PROMOTION_GATE",
    }


def test_resume_command_is_deterministic(tmp_path):
    pipeline = Pipeline(args(tmp_path))
    assert pipeline.resume_command() == pipeline.resume_command()
    assert "-Resume" in pipeline.resume_command()


def test_single_component_command_is_explicit_in_source():
    text = Path("scripts/post_finaliser_pipeline.py").read_text(encoding="utf-8")
    assert "--production-plan-job" in text
    assert "plan[0]" in text


def test_revalidation_occurs_after_publication_in_source():
    text = Path("scripts/post_finaliser_pipeline.py").read_text(encoding="utf-8")
    assert text.index("--production-plan-job") < text.index("Component readiness revalidation failed")


@pytest.mark.parametrize("forbidden", ["portfolio replay", "policy sweep", "exposure construction"])
def test_forbidden_workflows_are_not_commands(forbidden):
    text = Path("scripts/post_finaliser_pipeline.py").read_text(encoding="utf-8").lower()
    assert f"--mode {forbidden}" not in text


def test_no_production_command_executes_in_tests(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("production executor called")
    monkeypatch.setattr("subprocess.run", fail)
    assert validate_progress(progress()) == []


def test_default_concurrency_is_one():
    text = Path("scripts/post_finaliser_pipeline.py").read_text(encoding="utf-8")
    assert "--sklearn-n-jobs 1" in Path("core/research/ml/selector_component_readiness.py").read_text(encoding="utf-8")
    assert "plan[0]" in text
