#!/usr/bin/env python3
"""Terminal lifecycle entrypoint for HF-DATA-CUTOVER-R1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from application.services.hf_data_cutover_controller import (
    CutoverController,
    clear_control_requests,
    controller_status,
    request_control,
)


DEFAULT_AUDIT_ROOT = REPOSITORY_ROOT / "docs/audits/HF_DATA_CUTOVER_R1_20261025T154506Z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "status", "monitor", "pause", "stop", "prepare"))
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def _render_status(payload: dict[str, object]) -> str:
    child = payload.get("child") if isinstance(payload.get("child"), dict) else {}
    transport = payload.get("transport_progress") if isinstance(payload.get("transport_progress"), dict) else {}
    lines = [
        f"HF-DATA-CUTOVER-R1 {payload.get('status', 'UNKNOWN')}",
        f"run_state: {payload.get('run_state', 'UNKNOWN')}",
        f"controller_live: {str(payload.get('controller_live', False)).lower()}",
        f"stage: {payload.get('current_stage') or '-'}",
        f"heartbeat: {payload.get('heartbeat_at_utc') or '-'}",
    ]
    if child:
        lines.append(f"child_pid: {child.get('pid')} adopted={str(child.get('adopted', False)).lower()}")
    if payload.get("free_bytes") is not None:
        lines.append(f"free_bytes: {payload['free_bytes']}")
    if transport:
        lines.extend(
            [
                f"raw_partitions: {transport.get('completed_partitions')}/{transport.get('planned_partitions')}",
                f"raw_unresolved: {transport.get('unresolved_partitions')}",
                f"acquisition_mode: {transport.get('acquisition_mode')}",
                f"transport_workers: {transport.get('workers')}",
                f"recent_partitions_per_hour: {transport.get('recent_validated_partitions_per_hour')}",
                f"recent_tls_failures: {transport.get('recent_tls_failures')}",
                f"recent_retries: {transport.get('recent_retries')}",
            ]
        )
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    lines.extend(
        [
            f"state: {payload.get('audit_root', DEFAULT_AUDIT_ROOT)}/controller/CONTROLLER_STATE.json",
            f"events: {payload.get('event_log', DEFAULT_AUDIT_ROOT / 'controller/CONTROLLER_EVENTS.jsonl')}",
            f"logs: {payload.get('stage_log_root', DEFAULT_AUDIT_ROOT / 'controller/stage_logs')}",
        ]
    )
    return "\n".join(lines)


def _monitor(audit_root: Path, poll_seconds: float) -> int:
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    event_path = audit_root / "controller/CONTROLLER_EVENTS.jsonl"
    event_offset = 0
    last_status = ""
    try:
        while True:
            rendered = _render_status(controller_status(audit_root))
            if rendered != last_status:
                print(rendered, flush=True)
                last_status = rendered
            if event_path.exists():
                with event_path.open("r", encoding="utf-8") as handle:
                    handle.seek(event_offset)
                    for line in handle:
                        print(f"event: {line.rstrip()}", flush=True)
                    event_offset = handle.tell()
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("Monitor closed; controller and stage worker were not signalled.")
        return 0


def main() -> int:
    arguments = build_parser().parse_args()
    repository_root = arguments.repository_root.resolve()
    audit_root = arguments.audit_root.resolve()
    if arguments.action == "worker":
        return CutoverController(repository_root, audit_root).run()
    if arguments.action == "status":
        print(_render_status(controller_status(audit_root)))
        return 0
    if arguments.action == "monitor":
        return _monitor(audit_root, arguments.poll_seconds)
    if arguments.action in {"pause", "stop"}:
        path = request_control(audit_root, arguments.action)
        print(json.dumps({"requested": arguments.action.upper(), "request_path": str(path)}, indent=2))
        return 0
    if clear_control_requests(audit_root):
        print("READY_TO_START")
        return 0
    print("CONTROLLER_ALREADY_ACTIVE")
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
