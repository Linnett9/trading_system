from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import (  # noqa: E402
    load_contract,
    stable_hash,
)
from core.research.ml.ds24.clean_v2_runtime import RUN_ID  # noqa: E402


class FailureReconciliationError(RuntimeError):
    """Raised when a recorded failure cannot safely become current state."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FailureReconciliationError(f"Required failure-state file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FailureReconciliationError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FailureReconciliationError(f"Invalid {label}: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise FailureReconciliationError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def reconcile_failure_state(
    *,
    family: str,
    host: str,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Reconcile one preserved pre-attempt-identity failure, failing closed."""

    ownership = load_contract("cross_host_ownership.json")
    if family not in ownership["hosts"][host]:
        raise FailureReconciliationError(
            f"Family {family!r} is not in the current {host!r} ownership authority"
        )
    run_root = repository_root / "research_runs" / "ds24_clean_v2" / RUN_ID
    family_root = run_root / f"family={family}"
    state_path = family_root / "resume_state.json"
    failure_path = family_root / "worker_failure.json"
    state = _read_json(state_path)
    failure = _read_json(failure_path)

    expected_state_identity = {
        "run_id": RUN_ID,
        "family": family,
        "owner_host": host,
    }
    for key, expected in expected_state_identity.items():
        if state.get(key) != expected:
            raise FailureReconciliationError(
                f"Resume-state identity mismatch for {key}: {state.get(key)!r}"
            )
    expected_failure_identity = {
        "run_id": RUN_ID,
        "family": family,
        "host": host,
    }
    for key, expected in expected_failure_identity.items():
        if failure.get(key) != expected:
            raise FailureReconciliationError(
                f"Worker-failure identity mismatch for {key}: {failure.get(key)!r}"
            )
    if state.get("terminal_state") == "COMPLETE":
        raise FailureReconciliationError("Refusing to override a completed attempt")

    failed_at = _timestamp(failure.get("failed_at_utc"), label="failed_at_utc")
    heartbeat = _timestamp(state.get("heartbeat_utc"), label="heartbeat_utc")
    if failed_at < heartbeat:
        raise FailureReconciliationError(
            "Recorded failure predates the current resume-state heartbeat"
        )
    status_path = run_root / f"supervisor_status_{host}.json"
    status = _read_json(status_path)
    if status.get("run_id") != RUN_ID or status.get("host_role") != host:
        raise FailureReconciliationError("Host-specific supervisor status is out of scope")
    active_families = {
        str(row.get("family"))
        for row in status.get("active_workers", [])
        if isinstance(row, dict)
    }
    if family in active_families:
        raise FailureReconciliationError(
            "Refusing to reconcile state while this family has an active worker"
        )
    if family not in status.get("failed_families", {}):
        raise FailureReconciliationError(
            "Host-specific supervisor status does not record this family as failed"
        )

    state_generation = int(state.get("attempt_generation", 0) or 0)
    failure_generation = int(failure.get("attempt_generation", 0) or 0)
    if state_generation and failure_generation and state_generation != failure_generation:
        raise FailureReconciliationError("Attempt generation mismatch")
    attempt_generation = failure_generation or state_generation or int(failed_at.value)
    existing_attempt_id = failure.get("attempt_id") or state.get("attempt_id")
    attempt_id = str(
        existing_attempt_id
        or stable_hash(
            {
                "run_id": RUN_ID,
                "family": family,
                "host": host,
                "attempt_generation": attempt_generation,
                "failed_at_utc": failed_at.isoformat(),
                "error": failure.get("error"),
            }
        )
    )
    log_path = failure.get("log_path") or (
        Path("research_runs")
        / "ds24_clean_v2"
        / RUN_ID
        / "logs"
        / host
        / f"{family}.log"
    ).as_posix()
    if not (repository_root / str(log_path)).is_file():
        raise FailureReconciliationError(f"Preserved worker log is missing: {log_path}")

    completed_refits = list(state.get("completed_refits", []))
    metrics_cursor = int(state.get("metrics_cursor", 0) or 0)
    classification = str(
        failure.get("classification") or "DS24_CLEAN_V2_WORKER_FAILED_CLOSED"
    )
    updated_failure = {
        **failure,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "classification": classification,
        "terminal_state": "FAILED_CLOSED",
        "log_path": str(log_path),
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
    }
    updated_state = {
        **state,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "terminal_state": "FAILED_CLOSED",
        "terminal_failure_classification": classification,
        "error": failure.get("error"),
        "failure_timestamp": failed_at.isoformat(),
        "failure_log_path": str(log_path),
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
        "heartbeat_utc": failed_at.isoformat(),
    }
    _write_json_atomic(
        family_root / f"worker_failure_attempt={attempt_generation}.json",
        updated_failure,
    )
    _write_json_atomic(failure_path, updated_failure)
    _write_json_atomic(state_path, updated_state)
    return {
        "classification": "DS24_CLEAN_V2_FAILURE_STATE_RECONCILED",
        "run_id": RUN_ID,
        "family": family,
        "host": host,
        "attempt_generation": attempt_generation,
        "attempt_id": attempt_id,
        "terminal_state": "FAILED_CLOSED",
        "error": updated_state["error"],
        "failure_timestamp": updated_state["failure_timestamp"],
        "failure_log_path": updated_state["failure_log_path"],
        "completed_refits": completed_refits,
        "metrics_cursor": metrics_cursor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile one preserved CLEAN V2 worker failure into resume state."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--host", choices=("dell", "mac"), required=True)
    args = parser.parse_args()
    report = reconcile_failure_state(family=args.family, host=args.host)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
