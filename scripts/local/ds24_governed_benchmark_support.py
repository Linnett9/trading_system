"""Shared fail-closed runtime support for supervisor-owned DS24 benchmarks."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd

from core.research.ml.ds24.clean_v2_resources import (
    DELL_WORKER_THREAD_ENVIRONMENT,
    FileReservationClient,
    RESOURCE_POLICY,
    WorkerReservationOwner,
    process_identity,
    write_json_object_atomic,
)


class GovernedRequestView(Protocol):
    family: str
    output_root: Path
    production_run_root: Path


def current_process_is_in_windows_job() -> bool:
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    result = wintypes.BOOL()
    current = kernel32.GetCurrentProcess()
    if not kernel32.IsProcessInJob(current, None, ctypes.byref(result)):
        raise OSError(ctypes.get_last_error(), "IsProcessInJob failed")
    return bool(result.value)


def reservation_client(request: GovernedRequestView) -> FileReservationClient:
    root = os.environ.get("DS24_CLEAN_V2_RESERVATION_ROOT", "")
    generation = int(
        os.environ.get("DS24_CLEAN_V2_BENCHMARK_GENERATION", "0") or 0
    )
    if not root or generation <= 0:
        raise RuntimeError("Governed benchmark reservation identity is missing")
    if os.environ.get("DS24_CLEAN_V2_RESOURCE_POLICY_ID") != RESOURCE_POLICY.policy_id:
        raise RuntimeError("Governed benchmark resource policy mismatch")
    invalid_threads = [
        name
        for name, expected in DELL_WORKER_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    ]
    if invalid_threads:
        raise RuntimeError(
            "Governed benchmark thread policy mismatch:" + ",".join(invalid_threads)
        )
    identity = process_identity(os.getpid())
    if not identity.alive or not identity.creation_time_utc:
        raise RuntimeError("Governed benchmark process identity is unavailable")

    def publish_wait_state(payload: Mapping[str, Any]) -> None:
        status_path = request.output_root / "canary_stage_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_object_atomic(status_path, payload)

    return FileReservationClient(
        Path(root),
        WorkerReservationOwner(
            family=request.family,
            attempt_generation=generation,
            pid=os.getpid(),
            process_creation_time_utc=identity.creation_time_utc,
        ),
        retry_capacity_denials=True,
        wait_observer=publish_wait_state,
    )


def production_progress_snapshot(
    request: GovernedRequestView,
) -> dict[str, Any]:
    """Read production progress without acquiring any writer capability."""

    snapshot: dict[str, Any] = {}
    for family in ("random_forest", "huber", "elastic_net_C5"):
        path = request.production_run_root / f"family={family}" / "resume_state.json"
        if not path.is_file():
            snapshot[family] = {"available": False}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot[family] = {
            "available": True,
            "metrics_cursor": int(payload.get("metrics_cursor", 0) or 0),
            "latest_completed_refit": payload.get("latest_completed_refit"),
            "latest_scored_decision": payload.get("latest_scored_decision"),
            "heartbeat_utc": payload.get("heartbeat_utc"),
            "error": payload.get("error"),
        }
    return snapshot


def prediction_parity(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_rows = list(baseline["bounded_parity_payload"]["predictions"])
    candidate_rows = list(candidate["bounded_parity_payload"]["predictions"])
    key_columns = ("family", "decision_timestamp", "asset_id")
    baseline_by_key = {
        tuple(str(row[column]) for column in key_columns): float(row["prediction"])
        for row in baseline_rows
    }
    candidate_by_key = {
        tuple(str(row[column]) for column in key_columns): float(row["prediction"])
        for row in candidate_rows
    }
    keys_equal = baseline_by_key.keys() == candidate_by_key.keys()
    common = baseline_by_key.keys() & candidate_by_key.keys()
    absolute: list[float] = []
    relative: list[float] = []
    for key in common:
        difference = abs(baseline_by_key[key] - candidate_by_key[key])
        absolute.append(difference)
        scale = max(abs(baseline_by_key[key]), abs(candidate_by_key[key]))
        if scale > 0.0:
            relative.append(difference / scale)

    def ranks(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str, str], int]:
        frame = pd.DataFrame(rows)
        frame["rank"] = frame.groupby("decision_timestamp")["prediction"].rank(
            method="first", ascending=False
        )
        return {
            tuple(str(row[column]) for column in key_columns): int(row["rank"])
            for row in frame.to_dict(orient="records")
        }

    baseline_ranks = ranks(baseline_rows)
    candidate_ranks = ranks(candidate_rows)
    common_rank_keys = baseline_ranks.keys() & candidate_ranks.keys()
    rank_disagreements = sum(
        baseline_ranks[key] != candidate_ranks[key] for key in common_rank_keys
    )
    evaluator_equal = (
        baseline["scientific_fingerprints"]["evaluator_logs"]
        == candidate["scientific_fingerprints"]["evaluator_logs"]
    )
    baseline_summaries = baseline["scientific_fingerprints"].get(
        "evaluator_numeric_summaries", {}
    )
    candidate_summaries = candidate["scientific_fingerprints"].get(
        "evaluator_numeric_summaries", {}
    )
    summary_deltas: list[float] = []
    if baseline_summaries.keys() == candidate_summaries.keys():
        for stem, columns in baseline_summaries.items():
            candidate_columns = candidate_summaries[stem]
            if columns.keys() != candidate_columns.keys():
                summary_deltas = []
                evaluator_equal = False
                break
            for column, summary in columns.items():
                candidate_summary = candidate_columns[column]
                for field in ("sum", "minimum", "maximum"):
                    summary_deltas.append(
                        abs(float(summary[field]) - float(candidate_summary[field]))
                    )
    return {
        "output_keys_equal": keys_equal,
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "maximum_absolute_prediction_delta": max(absolute, default=0.0),
        "maximum_relative_prediction_delta": max(relative, default=0.0),
        "rank_fingerprint_equal": (
            baseline["scientific_fingerprints"]["ranks"]
            == candidate["scientific_fingerprints"]["ranks"]
        ),
        "rank_disagreements": rank_disagreements,
        "evaluator_fingerprints_equal": evaluator_equal,
        "portfolio_sleeve_disagreements": 0 if evaluator_equal else None,
        "maximum_evaluator_numeric_summary_delta": (
            max(summary_deltas, default=0.0) if evaluator_equal else None
        ),
        "training_rows_equal": (
            baseline["bounded_scope"]["training_rows"]
            == candidate["bounded_scope"]["training_rows"]
        ),
        "scoring_rows_equal": (
            baseline["bounded_scope"]["scoring_rows"]
            == candidate["bounded_scope"]["scoring_rows"]
        ),
        "panel_fingerprint_equal": (
            baseline["scientific_fingerprints"]["panel"]
            == candidate["scientific_fingerprints"]["panel"]
        ),
        "feature_order_equal": (
            baseline["scientific_fingerprints"]["feature_order"]
            == candidate["scientific_fingerprints"]["feature_order"]
        ),
        "training_model_inputs_equal": (
            baseline["scientific_fingerprints"]["training_model_inputs"]
            == candidate["scientific_fingerprints"]["training_model_inputs"]
        ),
        "scoring_model_inputs_equal": (
            baseline["scientific_fingerprints"]["scoring_model_inputs"]
            == candidate["scientific_fingerprints"]["scoring_model_inputs"]
        ),
    }
