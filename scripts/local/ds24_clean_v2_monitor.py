from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "research_runs/ds24_clean_v2/DS24_CLEAN_V2_TOURNAMENT_R1_20260926"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml.ds24.clean_v2_contracts import load_contract  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _matching_failure(
    family_root: Path,
    state: dict[str, Any],
    *,
    host: str,
    current_attempt_generation: int,
) -> dict[str, Any]:
    failure = _read_json(family_root / "worker_failure.json")
    if not failure:
        return {}
    if (
        failure.get("run_id") != state.get("run_id")
        or failure.get("family") != state.get("family")
        or failure.get("host") != host
    ):
        return {}
    state_generation = int(state.get("attempt_generation", 0) or 0)
    failure_generation = int(failure.get("attempt_generation", 0) or 0)
    if current_attempt_generation > state_generation:
        return {}
    if state_generation or failure_generation:
        return failure if state_generation == failure_generation else {}
    if state.get("terminal_state") == "COMPLETE":
        return {}
    failed_at = str(failure.get("failed_at_utc") or "")
    heartbeat = str(state.get("heartbeat_utc") or "")
    return failure if failed_at >= heartbeat else {}


def build_report(
    host: str,
    *,
    repository_root: Path = ROOT,
    run_root: Path = RUN_ROOT,
) -> dict[str, Any]:
    """Build one read-only report from current host-scoped authority."""

    status_path = run_root / f"supervisor_status_{host}.json"
    status = _read_json(status_path)
    ownership = load_contract("cross_host_ownership.json")
    admitted_families = set(ownership["hosts"][host])
    current_attempt_generation = int(status.get("attempt_generation", 0) or 0)
    active_workers = [
        row
        for row in status.get("active_workers", [])
        if isinstance(row, dict) and row.get("family") in admitted_families
    ]
    active_generations = {
        str(row["family"]): int(
            row.get("attempt_generation", current_attempt_generation) or 0
        )
        for row in active_workers
    }
    queued_families = [
        str(family)
        for family in status.get("queued_families", [])
        if family in admitted_families
    ]
    complete_families = [
        str(family)
        for family in status.get("complete_families", [])
        if family in admitted_families
    ]
    disk = shutil.disk_usage(repository_root.anchor or repository_root)
    rows = []
    for family_path in sorted(run_root.glob("family=*/resume_state.json")):
        state = _read_json(family_path)
        if state.get("owner_host") != host:
            continue
        if state.get("family") not in admitted_families:
            continue
        family = str(state["family"])
        state_generation = int(state.get("attempt_generation", 0) or 0)
        failure = _matching_failure(
            family_path.parent,
            state,
            host=host,
            current_attempt_generation=current_attempt_generation,
        )
        terminal_state = state.get("terminal_state")
        error = failure.get("error", state.get("error"))
        failure_timestamp = failure.get(
            "failed_at_utc", state.get("failure_timestamp")
        )
        failure_log_path = failure.get(
            "log_path", state.get("failure_log_path")
        )
        if active_generations.get(family, 0) > state_generation:
            terminal_state = "RUNNING"
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif family in queued_families and current_attempt_generation > state_generation:
            terminal_state = "QUEUED"
            error = None
            failure_timestamp = None
            failure_log_path = None
        elif failure and terminal_state != "COMPLETE":
            terminal_state = failure.get("terminal_state", "FAILED_CLOSED")
        rows.append(
            {
                "family": family,
                "state": terminal_state,
                "attempt_generation": state.get("attempt_generation"),
                "attempt_id": state.get("attempt_id"),
                "current_historical_date": state.get("latest_scored_decision"),
                "completed_refits": len(state.get("completed_refits", [])),
                "expected_refits": state.get("expected_refits"),
                "score_timestamps": state.get("metrics_cursor", 0),
                "current_rank_ic": state.get("current_rank_ic"),
                "top_n_spread": state.get("top_n_spread"),
                "net_10bps": state.get("net_10bps"),
                "error": error,
                "failure_timestamp": failure_timestamp,
                "failure_log_path": failure_log_path,
                "causal_authority_hash": state.get("feature_authority_hash"),
            }
        )
    state_failures = {
        str(row["family"]): 1
        for row in rows
        if row["state"] == "FAILED_CLOSED"
    }
    status_failures = {
        str(family): exit_code
        for family, exit_code in dict(status.get("failed_families", {})).items()
        if family in admitted_families
    }
    failed_families = {
        **status_failures,
        **state_failures,
    }
    active_family_ids = {str(row["family"]) for row in active_workers}
    queued_families = [
        family
        for family in queued_families
        if family not in failed_families
        and family not in complete_families
        and family not in active_family_ids
    ]
    classification = status.get("classification", "NOT_STARTED")
    if state_failures:
        classification = "DS24_CLEAN_V2_TOURNAMENT_FAILED_CLOSED"
    return {
        "run_id": "DS24_CLEAN_V2_TOURNAMENT_R1_20260926",
        "host_role": host,
        "classification": classification,
        "status_authority_path": status_path.relative_to(repository_root).as_posix(),
        "legacy_generic_supervisor_status_ignored": (
            run_root / "supervisor_status.json"
        ).is_file(),
        "attempt_generation": status.get("attempt_generation"),
        "supervisor_pid": status.get("supervisor_pid"),
        "active_workers": active_workers,
        "queued_families": queued_families,
        "complete_families": complete_families,
        "failed_families": failed_families,
        "families": rows,
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "ram_available_bytes": status.get("ram_available_bytes"),
        "blocking_reasons": status.get("blocking_reasons", []),
        "feature_authority_hash": status.get("feature_authority_hash"),
        "target_authority_hash": status.get("target_authority_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor one DS24 clean-V2 host queue.")
    parser.add_argument("--host", choices=("dell", "mac"), default="dell")
    args = parser.parse_args()
    report = build_report(args.host)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
