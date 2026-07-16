import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.research.ml.selector_component_readiness import audit_component_commands
from core.research.ml.selector_execution_harness import (
    STAGES, load_run_state, new_run_state, validate_fresh_component_preflight,
    validate_stage_request, write_run_state_atomic,
)


def _jobs(root, *, count=15):
    return [{"action": "fit", "command": f'python main.py --selector-dataset-root "{root}" --model-allowlist ridge'} for _ in range(count)]


def _payload(root):
    return {"preflight_schema_version": "selector_component_production_preflight_v1", "status": "READY", "component_count": 15, "fitting_performed": False, "prediction_performed": False, "blocking_reasons": [], "dataset_identity": "dataset-v2", "daily_spine_identity": "spine", "symbol_registry_identity": "registry", "daily_feature_store_identity": "features", "jobs": _jobs(root)}


def test_default_stage_request_stops_at_ten_and_stage_eleven_requires_approval():
    validate_stage_request()
    assert STAGES[9].number == 10 and STAGES[10].number == 11
    with pytest.raises(PermissionError): validate_stage_request(1, 11)
    validate_stage_request(1, 11, allow_selector_fits=True)


@pytest.mark.parametrize("start,end", [(0, 10), (2, 1), (1, 17)])
def test_invalid_stage_ranges_fail_before_work(start, end):
    with pytest.raises(ValueError): validate_stage_request(start, end)


def test_stale_or_wrong_preflight_path_is_rejected(tmp_path):
    expected = tmp_path / "fresh.json"; stale = tmp_path / "component_preflight.json"
    stale.write_text(json.dumps(_payload(tmp_path / "dataset")))
    old = datetime.now(timezone.utc) + timedelta(seconds=1)
    result = validate_fresh_component_preflight(_payload(tmp_path / "dataset"), expected_path=expected, actual_path=stale, stage_started_at=old, expected_dataset_root=tmp_path / "dataset")
    assert "AUTHORITATIVE_PREFLIGHT_PATH_MISMATCH" in result["blocking_reasons"]
    assert "STALE_COMPONENT_PREFLIGHT" in result["blocking_reasons"]


def test_fresh_run_owned_preflight_is_accepted(tmp_path):
    path = tmp_path / "runs/run-a/component_preflight_v2.json"; path.parent.mkdir(parents=True)
    start = datetime.now(timezone.utc) - timedelta(seconds=1); payload = _payload(tmp_path / "dataset")
    path.write_text(json.dumps(payload))
    result = validate_fresh_component_preflight(payload, expected_path=path, actual_path=path, stage_started_at=start, expected_dataset_root=tmp_path / "dataset")
    assert result["status"] == "READY"
    assert result["command_binding_audit"]["explicit_override_jobs"] == 15


def test_missing_override_legacy_and_multiple_roots_block():
    jobs = _jobs("dataset-v2")
    jobs[0]["command"] = "python main.py --model-allowlist ridge"
    jobs[1]["command"] = 'python main.py --selector-dataset-root "canonical_v2_selector_dataset_v1/frozen"'
    jobs[2]["command"] = 'python main.py --selector-dataset-root "other-v2"'
    result = audit_component_commands(jobs, expected_dataset_root=__import__("pathlib").Path("dataset-v2"))
    assert set(result["blocking_reasons"]) == {"COMPONENT_DATASET_OVERRIDE_MISSING", "LEGACY_DATASET_ROOT_REFERENCED", "MULTIPLE_OR_UNVERIFIED_DATASET_ROOTS"}


def test_atomic_run_state_round_trip_and_resume_metadata(tmp_path):
    state = new_run_state(run_id="run-a", repository_path=tmp_path, source_commit="abc", from_stage=1, through_stage=10, allow_selector_fits=False)
    path = tmp_path / "run_state.json"; write_run_state_atomic(path, state)
    loaded = load_run_state(path)
    assert loaded["run_id"] == "run-a" and loaded["requested_stage_range"]["through"] == 10
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_run_state_write_failure_does_not_create_destination(tmp_path, monkeypatch):
    path = tmp_path / "run_state.json"
    state = new_run_state(run_id="run-a", repository_path=tmp_path, source_commit="abc", from_stage=1, through_stage=10, allow_selector_fits=False)

    def fail_write(*args, **kwargs):
        raise OSError("injected initialization failure")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="injected initialization failure"):
        write_run_state_atomic(path, state)
    assert not path.exists()


def test_stage_state_schema_has_distinct_expected_and_produced_outputs(tmp_path):
    state = new_run_state(run_id="run-a", repository_path=tmp_path, source_commit="abc", from_stage=1, through_stage=10, allow_selector_fits=False)
    assert [stage["number"] for stage in state["stages"]] == list(range(1, 17))
    assert all("expected_outputs" in stage and "produced_outputs" in stage for stage in state["stages"])
    assert all("outputs" not in stage and "inputs" not in stage for stage in state["stages"])
    state["stages"][0]["produced_outputs"].append("one.json")
    assert state["stages"][1]["produced_outputs"] == []
    round_trip = json.loads(json.dumps(state, sort_keys=True))
    assert round_trip == state


def test_powershell_bounded_initialization_creates_readable_atomic_state(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "reports/ml/readiness/selector_evaluation_1c_d/selector_parent_publication_runbook.ps1"
    run_id = f"pytest-init-{tmp_path.name}"
    run_root = repo / "reports/ml/readiness/selector_evaluation_1c_e/runs" / run_id
    state_path = run_root / "run_state.json"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-FromStage", "1", "-ThroughStage", "10", "-RunId", run_id, "-InitializeOnly"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    try:
        assert result.returncode == 0, result.stderr + result.stdout
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        assert len(state["stages"]) == 16
        assert [stage["number"] for stage in state["stages"]] == list(range(1, 17))
        assert all(stage["status"] == "pending" for stage in state["stages"])
        assert all(stage["expected_outputs"] for stage in state["stages"])
        assert all(stage["produced_outputs"] == [] for stage in state["stages"])
        assert not (run_root / "transcript.txt").exists()
    finally:
        if run_root.exists():
            import shutil
            shutil.rmtree(run_root)


def test_powershell_stage_eleven_without_approval_creates_no_state(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "reports/ml/readiness/selector_evaluation_1c_d/selector_parent_publication_runbook.ps1"
    run_id = f"pytest-fit-gate-{tmp_path.name}"
    state_path = repo / "reports/ml/readiness/selector_evaluation_1c_e/runs" / run_id / "run_state.json"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-FromStage", "11", "-ThroughStage", "11", "-RunId", run_id, "-InitializeOnly"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "require -AllowSelectorFits" in result.stderr + result.stdout
    assert not state_path.exists()


def test_parent_identity_change_blocks_fresh_stage_ten(tmp_path):
    path = tmp_path / "component_preflight_v2.json"; payload = _payload(tmp_path / "old-dataset"); path.write_text(json.dumps(payload))
    result = validate_fresh_component_preflight(payload, expected_path=path, actual_path=path, stage_started_at=datetime.now(timezone.utc) - timedelta(seconds=1), expected_dataset_root=tmp_path / "new-dataset")
    assert "MULTIPLE_OR_UNVERIFIED_DATASET_ROOTS" in result["blocking_reasons"]
