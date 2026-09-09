from __future__ import annotations

import json
from pathlib import Path

from scripts.local import ds24_full_family_readiness_audit as audit


def test_r41_matrix_assigns_allowed_state_to_each_unfinished_family() -> None:
    matrix = audit.build_matrix()
    rows = {
        row["family"]: row
        for row in matrix["families"]
        if row["family"] in audit.UNFINISHED_FAMILIES
    }

    assert set(rows) == set(audit.UNFINISHED_FAMILIES)
    assert all(row["state"] in audit.STATE_VALUES for row in rows.values())
    assert rows["elastic_net"]["state"] == "FORWARD_METRICS_CONTRACT_REQUIRED"
    assert rows["Temporal Fusion Transformer"]["state"] == "CONFIGURATION_AUTHORITY_REQUIRED"
    assert matrix["safety"]["holdout_accessed"] is False
    assert matrix["safety"]["paper_orders"] == 0
    assert matrix["safety"]["live_orders"] == 0


def test_random_forest_storage_hold_fatal_remains_recoverable_when_checkpoint_is_valid() -> None:
    state, next_action, blocker = audit.classify_state(
        "random_forest",
        {"worker_script_exists": True, "family_resolves": True},
        {"state": "PASS"},
        {"state": "STALE_RECOVERABLE"},
        {
            "fatal": True,
            "final_exception": "RuntimeError: STORAGE_EMERGENCY_HOLD_ACTIVE:C:/stage/STORAGE_EMERGENCY_HOLD.json",
        },
        {"state": "PASS"},
        {
            "holdout_accessed": False,
            "full_prediction_files_in_metrics_namespaces": 0,
            "paper_orders": 0,
            "live_orders": 0,
        },
    )

    assert state == "READY_TO_LAUNCH"
    assert next_action == "admit when supervisor slot and resource gates open"
    assert blocker == ""


def test_publish_writes_ready_and_repair_queue_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit, "OUTPUT_JSON", tmp_path / "matrix.json")
    monkeypatch.setattr(audit, "OUTPUT_CSV", tmp_path / "matrix.csv")
    monkeypatch.setattr(audit, "REPAIR_QUEUE_JSON", tmp_path / "repair.json")
    monkeypatch.setattr(audit, "READY_QUEUE_JSON", tmp_path / "ready.json")

    matrix = {
        "ticket": "TEST",
        "generated_at_utc": "2026-09-09T00:00:00+00:00",
        "ready_to_launch_families": ["random_forest"],
        "recommended_supervisor_queue_when_slot_opens": ["random_forest"],
        "repair_queue": [
            {
                "family": "elastic_net",
                "state": "FORWARD_METRICS_CONTRACT_REQUIRED",
                "priority": 20,
                "next_action": "publish extended forward-metrics capability sidecar",
                "blocker": "writer_initialised_before_first_prediction",
                "evidence_paths": [],
            }
        ],
        "families": [
            {
                "family": "random_forest",
                "state": "READY_TO_LAUNCH",
                "implementation": {"state": "PASS"},
                "scientific_contract": {"state": "PASS"},
                "v3_forward_metrics": {"state": "PASS"},
                "checkpoint": {"state": "READABLE"},
                "namespace": {"state": "STALE_RECOVERABLE"},
                "stderr": {"state": "FATAL_REPAIR_REQUIRED"},
                "bounded_replay": {"state": "PASS"},
                "resource_profile": {"classification": "heavy"},
                "holdout_guard": "PASS",
                "full_prediction_guard": "PASS",
                "current_cursor": "2016-04-13T20:00:00+00:00",
                "metrics_rows": 3887,
                "next_action": "admit when supervisor slot and resource gates open",
                "blocker": "",
                "evidence_paths": [],
            }
        ],
    }

    audit.publish(matrix)

    assert json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))["ready_family_queue"] == ["random_forest"]
    repair = json.loads((tmp_path / "repair.json").read_text(encoding="utf-8"))
    assert repair["repair_queue"][0]["family"] == "elastic_net"
    assert (tmp_path / "matrix.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(audit.CSV_COLUMNS)
