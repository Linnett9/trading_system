from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.check_post_finaliser_job_readiness import (
    canonical_hash,
    canonical_json,
    evaluate_readiness,
    load_json,
    validate_ledger,
)


LEDGER_PATH = Path("config/operations/post_finaliser_job_ledger_v1.json")


def _ledger():
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _progress(**changes):
    value = {
        "planned_partitions": 5654, "completed_partitions": 5654,
        "pending_partitions": 0, "failed_partitions": 0, "invalid_rows": 0,
        "conflicting_duplicates": 0, "temporary_files": 0,
    }
    value.update(changes)
    return value


def _validation(**changes):
    value = {
        "valid": True, "partition_count": 5654, "invalid_rows": 0,
        "temporary_files_left_behind": [],
    }
    value.update(changes)
    return value


def _selector(stage4="failed", stage10="pending"):
    stages = []
    for number in range(1, 17):
        status = "complete" if number <= 3 else "pending"
        if number == 4:
            status = stage4
        if number == 10:
            status = stage10
        stages.append({"stage_number": number, "status": status})
    return {
        "run_state_version": "selector_parent_publication_run_state_v2",
        "run_id": "20260716T091011Z", "stages": stages,
    }


def _ops(ready=True):
    return {"status": "READY" if ready else "BLOCKED", "whole_table_to_pylist_used": False}


def _evaluate(**changes):
    values = {
        "progress": _progress(), "archive_validation": _validation(),
        "selector_state": _selector(), "ops_4a": _ops(),
        "component_readiness": None, "component_inventory": None,
        "finaliser_active": False, "free_memory_bytes": 16 * 1024**3,
        "free_disk_bytes": 100 * 1024**3, "input_errors": {},
    }
    values.update(changes)
    return evaluate_readiness(_ledger(), **values)


def test_finaliser_active_and_incomplete_state():
    result = _evaluate(
        progress=_progress(completed_partitions=5000, pending_partitions=654),
        archive_validation=None, finaliser_active=True,
    )
    assert result["job_statuses"]["JOB-001"] == "ACTIVE"
    assert result["job_statuses"]["JOB-002"] == "WAITING_DEPENDENCY"
    assert result["next_job"] == "JOB-001"


def test_incomplete_idle_finaliser_is_retryable():
    result = _evaluate(
        progress=_progress(completed_partitions=5000, pending_partitions=654),
        archive_validation=None,
    )
    assert result["job_statuses"]["JOB-001"] == "FAILED_RETRYABLE"


def test_failed_partition_blocks_validation():
    result = _evaluate(progress=_progress(failed_partitions=1, completed_partitions=5653))
    assert result["job_statuses"]["JOB-001"] == "FAILED_RETRYABLE"
    assert "FAILED_PARTITIONS:1" in result["job_blockers"]["JOB-001"]
    assert result["job_statuses"]["JOB-002"] == "WAITING_DEPENDENCY"


def test_complete_archive_awaiting_validation():
    result = _evaluate(archive_validation=None)
    assert result["job_statuses"]["JOB-001"] == "COMPLETE"
    assert result["job_statuses"]["JOB-002"] == "READY"
    assert result["next_job"] == "JOB-002"


def test_partial_660_validation_never_satisfies_full_gate():
    result = _evaluate(archive_validation=_validation(partition_count=660))
    assert result["job_statuses"]["JOB-002"] == "BLOCKED"
    assert result["archive"]["partial_660_validation_rejected"] is True


def test_valid_archive_and_retryable_selector_make_preflight_ready():
    result = _evaluate()
    assert result["job_statuses"]["JOB-003"] == "READY"
    assert result["resume_outcome"] == "READY_TO_RESUME"
    assert " -Resume " in f" {result['resume_command']} "
    assert result["job_statuses"]["JOB-004"] == "READY"


def test_ops_readiness_missing_and_invalid_block():
    missing = _evaluate(ops_4a=None, input_errors={"ops_4a_readiness": "MISSING_FILE"})
    assert missing["job_statuses"]["JOB-003"] == "BLOCKED"
    invalid = _evaluate(ops_4a=_ops(False))
    assert "OPS_4A_NOT_READY" in invalid["job_blockers"]["JOB-003"]


def test_low_memory_is_resource_wait_not_dependency_leakage():
    result = _evaluate(free_memory_bytes=1)
    assert result["job_statuses"]["JOB-003"] == "WAITING_RESOURCES"
    assert result["job_statuses"]["JOB-004"] == "WAITING_DEPENDENCY"
    assert result["job_statuses"]["JOB-005"] == "WAITING_DEPENDENCY"


def test_stage10_ready_advances_to_component_job():
    state = _selector(stage4="complete", stage10="complete")
    for row in state["stages"]:
        if row["stage_number"] <= 10:
            row["status"] = "complete"
    result = _evaluate(
        selector_state=state,
        component_readiness={"status": "READY"},
    )
    assert result["job_statuses"]["JOB-004"] == "COMPLETE"
    assert result["job_statuses"]["JOB-005"] == "READY"


def test_exactly_15_components_complete_roster():
    ledger = _ledger()
    components = [
        {"date": date, "model": model, "status": "READY"}
        for date in ledger["policy"]["component_dates"]
        for model in ledger["policy"]["component_models"]
    ]
    state = _selector(stage4="complete", stage10="complete")
    for row in state["stages"]:
        if row["stage_number"] <= 10:
            row["status"] = "complete"
    result = _evaluate(
        selector_state=state, component_readiness={"status": "READY"},
        component_inventory={"components": components},
    )
    assert result["components"] == {"expected_count": 15, "ready_count": 15, "complete_roster": True}
    assert result["job_statuses"]["JOB-005"] == "COMPLETE"
    assert result["job_statuses"]["JOB-006"] == "READY"


def test_incomplete_component_roster_does_not_advance():
    result = _evaluate(component_inventory={"components": [{"date": "2024-03-15", "model": "ridge", "status": "READY"}]})
    assert result["components"]["complete_roster"] is False
    assert result["job_statuses"]["JOB-006"] == "WAITING_DEPENDENCY"


def test_selector_identity_mismatch_and_malformed_files(tmp_path):
    state = _selector()
    state["run_id"] = "wrong"
    result = _evaluate(selector_state=state)
    assert "SELECTOR_RUN_ID_MISMATCH" in result["job_blockers"]["JOB-003"]
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad")
    assert load_json(malformed)[1].startswith("MALFORMED_FILE:")


def test_dependency_order_and_no_downstream_readiness_leakage():
    result = _evaluate(
        progress=_progress(completed_partitions=0, pending_partitions=5654),
        archive_validation=None,
    )
    for number in range(2, 12):
        assert result["job_statuses"][f"JOB-{number:03d}"] != "READY"


def test_ledger_and_canonical_json_are_stable():
    ledger = _ledger()
    assert not validate_ledger(ledger)
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_hash(ledger) == canonical_hash(copy.deepcopy(ledger))


def test_checker_contract_has_no_parquet_paths_or_access():
    ledger = _ledger()
    serialized = canonical_json(ledger).lower()
    assert ".parquet" not in serialized
    source = Path("scripts/check_post_finaliser_job_readiness.py").read_text(encoding="utf-8").lower()
    assert "pyarrow" not in source
    assert "read_parquet" not in source
