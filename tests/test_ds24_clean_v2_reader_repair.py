from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.clean_v2_data import (
    CleanV2DataError,
    _join_repaired_features,
)
from core.research.ml.ds24.clean_v2_features import (
    REPAIRED_FEATURES,
    SOURCE_TIMESTAMP_SUFFIX,
)
from core.research.ml.ds24.clean_v2_runtime import REFIT_POLICY_ID, RUN_ID
from scripts.local import ds24_clean_v2_family_worker as family_worker
from scripts.local import ds24_clean_v2_monitor as monitor
from scripts.local.ds24_clean_v2_reader_preflight import run as reader_preflight
from scripts.local.ds24_clean_v2_reconcile_failures import (
    reconcile_failure_state,
)


def _reader_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = list(
        pd.date_range("2024-01-02T14:35:00Z", periods=6, freq="5min")
    )
    timestamps = [*regular, pd.Timestamp("2024-01-03T00:00:00Z")]
    session_types = ["REGULAR"] * 6 + ["AFTER_HOURS"]
    base = pd.DataFrame(
        {
            "asset_id": ["asset_AAA"] * len(timestamps),
            "decision_timestamp": timestamps,
            "session_date": ["2024-01-02"] * len(timestamps),
            "session_type": session_types,
            "ret_5m": np.arange(len(timestamps), dtype=float),
            **{
                feature: [999.0] * len(timestamps)
                for feature in REPAIRED_FEATURES
            },
        }
    )
    sidecar = base[["asset_id", "decision_timestamp"]].copy()
    regular_required = {
        "minutes_since_open",
        "minutes_until_close",
        "session_progress",
        "early_close_session_flag",
        "opening_period_flag",
    }
    for index, feature in enumerate(REPAIRED_FEATURES):
        values = [np.nan] * len(timestamps)
        values[5] = float(index)
        provenance = [pd.NaT] * len(timestamps)
        provenance[5] = regular[5]
        if feature in regular_required:
            values[:6] = [float(index)] * 6
            provenance[:6] = regular
        sidecar[feature] = values
        sidecar[f"{feature}{SOURCE_TIMESTAMP_SUFFIX}"] = provenance
    return base, sidecar.iloc[::-1].reset_index(drop=True)


def _join(
    base: pd.DataFrame,
    sidecar: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return _join_repaired_features(
        base,
        sidecar,
        repaired=REPAIRED_FEATURES,
        source_timestamp_suffix=SOURCE_TIMESTAMP_SUFFIX,
        partition_label="AAA/2024",
    )


def test_reader_uses_identity_presence_not_permitted_repair_nulls() -> None:
    base, sidecar = _reader_frames()

    result, report = _join(base, sidecar)

    assert result[["asset_id", "decision_timestamp"]].to_dict("records") == base[
        ["asset_id", "decision_timestamp"]
    ].to_dict("records")
    assert report["base_rows"] == 7
    assert report["matched_rows"] == 7
    assert report["unmatched_base_key_count"] == 0
    assert report["regular_session_rows"] == 6
    assert report["extended_hours_rows"] == 1
    assert report["opening_window_rows"] == 5
    assert report["opening_window_non_null_counts"] == {
        "opening_range_position": 0,
        "session_return_30m": 0,
        "opening_return_30m": 0,
    }
    assert report["matched_rows_with_any_null_repair_value"] == 6
    nullable_at_open = [
        "overnight_gap",
        "previous_session_return",
        "two_session_return",
        "opening_range_position",
        "session_return_30m",
        "opening_return_30m",
    ]
    assert result.loc[:4, nullable_at_open].isna().all().all()
    assert result.loc[6, list(REPAIRED_FEATURES)].isna().all()
    assert result.loc[5, "overnight_gap"] == 0.0
    assert not (result[list(REPAIRED_FEATURES)] == 999.0).any().any()


def test_reader_still_rejects_a_genuinely_missing_identity_key() -> None:
    base, sidecar = _reader_frames()
    missing_timestamp = base.iloc[-1]["decision_timestamp"]
    sidecar = sidecar[sidecar["decision_timestamp"] != missing_timestamp]

    with pytest.raises(CleanV2DataError, match="unmatched_keys=1"):
        _join(base, sidecar)


@pytest.mark.parametrize("defect", ["base_duplicate", "sidecar_duplicate", "null"])
def test_reader_rejects_duplicate_and_null_identity_keys(defect: str) -> None:
    base, sidecar = _reader_frames()
    if defect == "base_duplicate":
        base = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    elif defect == "sidecar_duplicate":
        sidecar = pd.concat([sidecar, sidecar.iloc[[0]]], ignore_index=True)
    else:
        sidecar.loc[0, "decision_timestamp"] = pd.NaT

    with pytest.raises(CleanV2DataError, match="duplicate identity|duplicates|null identity"):
        _join(base, sidecar)


def test_reader_rejects_asset_misalignment_without_fuzzy_matching() -> None:
    base, sidecar = _reader_frames()
    sidecar["asset_id"] = "asset_BBB"

    with pytest.raises(CleanV2DataError, match="asset identity mismatch"):
        _join(base, sidecar)


def test_actual_first_aapl_2016_production_reader_path_is_bounded_and_passes() -> None:
    report = reader_preflight(family="random_forest", asset_id="AAPL", year=2016)
    diagnostics = report["diagnostics"]

    assert report["model_fit_performed"] is False
    assert report["data_written"] is False
    assert report["refit_ordinal"] == 0
    assert report["refit_timestamp"] == "2016-02-02T14:35:00+00:00"
    assert diagnostics["base_rows_requested"] == 4226
    assert diagnostics["legacy_sidecar_rows_selected"] == 4225
    assert diagnostics["legacy_unmatched_base_key_count"] == 1
    assert diagnostics["sidecar_rows_selected"] == 4226
    assert diagnostics["unmatched_base_key_count"] == 0
    assert diagnostics["matched_rows_with_any_null_repair_value"] == 2548
    assert diagnostics["production_reader_passed"] is True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_worker_failure_updates_current_attempt_and_preserves_cursors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(family_worker, "ROOT", tmp_path)
    family_root = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / "family=random_forest"
    )
    _write_json(
        family_root / "resume_state.json",
        {
            "run_id": RUN_ID,
            "family": "random_forest",
            "owner_host": "dell",
            "attempt_generation": 7,
            "attempt_id": "prior",
            "terminal_state": "RUNNING",
            "completed_refits": ["one"],
            "metrics_cursor": 12,
        },
    )
    args = argparse.Namespace(
        family="random_forest", host="dell", resume_generation=7
    )

    failure = family_worker._record_worker_failure(args, RuntimeError("reader"))
    state = json.loads((family_root / "resume_state.json").read_text())

    assert failure["current_failure_pointer_updated"] is True
    assert state["terminal_state"] == "FAILED_CLOSED"
    assert state["terminal_failure_classification"] == (
        "DS24_CLEAN_V2_WORKER_FAILED_CLOSED"
    )
    assert state["error"] == "RuntimeError:reader"
    assert state["failure_log_path"].endswith("logs/dell/random_forest.log")
    assert state["completed_refits"] == ["one"]
    assert state["metrics_cursor"] == 12
    assert state["attempt_generation"] == 7


def test_stale_worker_failure_cannot_override_a_later_completed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(family_worker, "ROOT", tmp_path)
    family_root = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / "family=random_forest"
    )
    completed = {
        "run_id": RUN_ID,
        "family": "random_forest",
        "owner_host": "dell",
        "attempt_generation": 9,
        "attempt_id": "later",
        "terminal_state": "COMPLETE",
        "completed_refits": ["one"],
        "metrics_cursor": 12,
    }
    _write_json(family_root / "resume_state.json", completed)
    args = argparse.Namespace(
        family="random_forest", host="dell", resume_generation=8
    )

    failure = family_worker._record_worker_failure(args, RuntimeError("stale"))

    assert failure["current_failure_pointer_updated"] is False
    assert json.loads((family_root / "resume_state.json").read_text()) == completed
    assert not (family_root / "worker_failure.json").exists()
    assert (family_root / "worker_failure_attempt=8.json").is_file()


def test_legacy_failure_reconciliation_is_identity_checked_and_idempotent(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / "family=random_forest"
    log_path = run_root / "logs" / "dell" / "random_forest.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("preserved", encoding="utf-8")
    _write_json(
        family_root / "resume_state.json",
        {
            "run_id": RUN_ID,
            "family": "random_forest",
            "owner_host": "dell",
            "terminal_state": "RUNNING",
            "heartbeat_utc": "2026-09-26T20:48:45+00:00",
            "completed_refits": [],
            "metrics_cursor": 0,
            "error": None,
        },
    )
    _write_json(
        family_root / "worker_failure.json",
        {
            "run_id": RUN_ID,
            "family": "random_forest",
            "host": "dell",
            "classification": "DS24_CLEAN_V2_WORKER_FAILED_CLOSED",
            "error": "CleanV2DataError:coverage",
            "failed_at_utc": "2026-09-26T20:48:46+00:00",
        },
    )
    _write_json(
        run_root / "supervisor_status_dell.json",
        {
            "run_id": RUN_ID,
            "host_role": "dell",
            "active_workers": [],
            "failed_families": {"random_forest": 1},
        },
    )

    first = reconcile_failure_state(
        family="random_forest", host="dell", repository_root=tmp_path
    )
    second = reconcile_failure_state(
        family="random_forest", host="dell", repository_root=tmp_path
    )
    state = json.loads((family_root / "resume_state.json").read_text())

    assert first == second
    assert state["terminal_state"] == "FAILED_CLOSED"
    assert state["error"] == "CleanV2DataError:coverage"
    assert state["metrics_cursor"] == 0
    assert state["attempt_id"] == first["attempt_id"]


def test_monitor_ignores_generic_status_retired_families_and_stale_failures(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / "family=random_forest"
    _write_json(
        family_root / "resume_state.json",
        {
            "run_id": RUN_ID,
            "family": "random_forest",
            "owner_host": "dell",
            "attempt_generation": 10,
            "attempt_id": "old",
            "terminal_state": "FAILED_CLOSED",
            "completed_refits": [],
            "metrics_cursor": 0,
            "error": "old failure",
        },
    )
    _write_json(
        family_root / "worker_failure.json",
        {
            "run_id": RUN_ID,
            "family": "random_forest",
            "host": "dell",
            "attempt_generation": 10,
            "attempt_id": "old",
            "terminal_state": "FAILED_CLOSED",
            "error": "old failure",
        },
    )
    _write_json(
        run_root / "supervisor_status.json",
        {"classification": "STALE", "failed_families": {"itransformer": 1}},
    )
    _write_json(
        run_root / "supervisor_status_dell.json",
        {
            "run_id": RUN_ID,
            "host_role": "dell",
            "classification": "DS24_CLEAN_V2_TOURNAMENT_RUNNING",
            "attempt_generation": 20,
            "active_workers": [
                {
                    "family": "random_forest",
                    "pid": 123,
                    "attempt_generation": 20,
                },
                {"family": "itransformer", "pid": 456, "attempt_generation": 20},
            ],
            "queued_families": ["itransformer"],
            "complete_families": [],
            "failed_families": {"itransformer": 1},
        },
    )

    report = monitor.build_report(
        "dell", repository_root=tmp_path, run_root=run_root
    )

    assert report["legacy_generic_supervisor_status_ignored"] is True
    assert report["classification"] == "DS24_CLEAN_V2_TOURNAMENT_RUNNING"
    assert report["families"][0]["state"] == "RUNNING"
    assert report["families"][0]["error"] is None
    assert [row["family"] for row in report["active_workers"]] == [
        "random_forest"
    ]
    assert report["queued_families"] == []
    assert report["failed_families"] == {}


def test_worker_admission_requires_the_current_clean_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(family_worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        family_worker, "authority_bundle", lambda: {"bundle_sha256": "bundle"}
    )
    monkeypatch.setattr(family_worker, "clean_source_hash", lambda: "source")
    admission_path = (
        tmp_path
        / "research_runs"
        / "ds24_clean_v2"
        / RUN_ID
        / "manual_admission_dell.json"
    )
    admission_core = {
        "run_id": RUN_ID,
        "host_role": "dell",
        "refit_policy": REFIT_POLICY_ID,
        "static_authority_bundle_sha256": "bundle",
        "launch_families": ["random_forest"],
    }
    _write_json(
        admission_path,
        {**admission_core, "admission_token": stable_hash(admission_core)},
    )
    monkeypatch.setenv(
        "DS24_CLEAN_V2_ADMISSION_TOKEN", stable_hash(admission_core)
    )

    with pytest.raises(family_worker.CleanV2WorkerError, match="source hash"):
        family_worker._validate_worker_admission("dell")

    admission_core["clean_source_hash"] = "source"
    token = stable_hash(admission_core)
    _write_json(admission_path, {**admission_core, "admission_token": token})
    monkeypatch.setenv("DS24_CLEAN_V2_ADMISSION_TOKEN", token)
    assert family_worker._validate_worker_admission("dell")[
        "clean_source_hash"
    ] == "source"
