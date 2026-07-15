from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from core.research.ml.experiment_ledger import (
    append_ledger_event, experiment_spec_hash, latest_run_states,
    new_experiment_run_id, read_ledger,
)


def _append(path: Path, run_id: str, status: str):
    return append_ledger_event(
        path, experiment_spec_hash_value="A" * 64, experiment_run_id=run_id,
        event_status=status, artifact_kind="MODEL_EXPERIMENT",
        canonical_model_id="ridge", requested_model_id="ridge",
        registry_hashes={"entry":"B" * 64}, source_commit="commit",
        error_summary="kept" if status == "FAILED" else None,
        rejection_summary="kept" if status == "REJECTED" else None,
    )


def test_append_history_latest_state_and_distinct_attempt_identity(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; spec = experiment_spec_hash({"model":"ridge"})
    first, second = new_experiment_run_id(spec), new_experiment_run_id(spec)
    assert first != second
    _append(path, first, "STARTED"); _append(path, first, "FAILED")
    _append(path, second, "STARTED"); _append(path, second, "REJECTED")
    _append(path, "skip", "SKIPPED_COMPLETE")
    events = read_ledger(path)
    assert [event["event_status"] for event in events] == ["STARTED", "FAILED", "STARTED", "REJECTED", "SKIPPED_COMPLETE"]
    states = latest_run_states(events)
    assert states[first]["event_status"] == "FAILED"
    assert states[second]["rejection_summary"] == "kept"


def test_concurrent_appends_are_valid_json_lines(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: _append(path, f"run-{index}", "COMPLETED"), range(20)))
    assert len(read_ledger(path)) == 20


def test_missing_final_newline_is_read_and_malformed_line_fails(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"; _append(path, "run", "COMPLETED")
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert len(read_ledger(path)) == 1
    path.write_text(path.read_text() + "\nnot-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed ledger line"):
        read_ledger(path)


def test_diagnostic_cannot_masquerade_as_completed_model_experiment(tmp_path: Path):
    with pytest.raises(ValueError, match="diagnostic"):
        append_ledger_event(
            tmp_path / "x.jsonl", experiment_spec_hash_value="x", experiment_run_id="run",
            event_status="COMPLETED", artifact_kind="RESEARCH_DIAGNOSTIC",
            canonical_model_id=None, requested_model_id=None, registry_hashes={}, source_commit=None,
        )
