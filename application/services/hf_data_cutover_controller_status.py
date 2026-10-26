"""Read-only transport progress projection for the HF data-cutover controller."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def transport_progress(audit_root: Path) -> dict[str, Any]:
    five = _optional_json(audit_root / "IEX_RAW_5M_MANIFEST.json")
    daily = _optional_json(audit_root / "IEX_DAILY_MANIFEST.json")
    completed = sum(int(row.get("completed_partition_count", 0)) for row in (five, daily))
    planned = sum(int(row.get("planned_partition_count", 0)) for row in (five, daily))
    acquisition = _optional_json(audit_root / "RUN_MANIFEST.json").get("iex_raw_backfill") or {}
    selection = _optional_json(audit_root / "PERF1_TRANSPORT_SELECTION.json")
    live_progress = _optional_json(audit_root / "PERF1_LIVE_PROGRESS.json")
    completed = max(completed, int(live_progress.get("completed_partition_count", 0)))
    progress: dict[str, Any] = {
        "completed_partitions": completed,
        "planned_partitions": planned,
        "unresolved_partitions": max(0, planned - completed),
        "acquisition_mode": acquisition.get("last_invocation_kind") or "BACKLOG",
        "workers": selection.get("selected_workers") or acquisition.get("max_in_flight_requests"),
        "request_ceiling_per_minute": acquisition.get("requests_per_minute"),
        "pacing_requests_per_minute": acquisition.get("pacing_requests_per_minute"),
        "recent_validated_partitions_per_hour": None,
        "recent_tls_failures": None,
        "recent_retries": None,
        "last_completed_partition_time": None,
    }
    log_path = audit_root / "controller/stage_logs/raw_iex.stdout.log"
    log_tail = _tail_text(log_path, 256_000)
    ordinary = re.findall(
        r"completed=(\d+)/(\d+).*?partitions_per_hour=([0-9.]+).*?requests=(\d+).*?http_429=(\d+)",
        log_tail,
    )
    ordinary_reliability = re.findall(
        r"IEX RAW BACKFILL .*?requests=(\d+).*?retries=(\d+).*?tls_eof=(\d+).*?http_429=(\d+)",
        log_tail,
    )
    perf1 = re.findall(
        r"IEX PERF1 BACKFILL .*?assembled=(\d+)/(\d+).*?published=(\d+).*?requests=(\d+).*?retries=(\d+).*?tls_eof=(\d+)",
        log_tail,
    )
    if ordinary:
        latest = ordinary[-1]
        live_completed = int(latest[0])
        progress.update({"live_completed_partitions": live_completed, "recent_requests": int(latest[3])})
        if live_completed > completed:
            progress["completed_partitions"] = live_completed
            progress["unresolved_partitions"] = max(0, planned - live_completed)
        perf1_start = _optional_json(audit_root / "PERF1_START_STATE.json")
        if perf1_start:
            captured = datetime.fromisoformat(str(perf1_start["created_at_utc"]).replace("Z", "+00:00"))
            elapsed_hours = max((datetime.now(timezone.utc) - captured).total_seconds() / 3600.0, 1e-9)
            progress["recent_validated_partitions_per_hour"] = max(
                0.0,
                (live_completed - int(perf1_start["completed_partitions"])) / elapsed_hours,
            )
        else:
            progress["recent_validated_partitions_per_hour"] = float(latest[2])
    if perf1:
        latest = perf1[-1]
        live_published = int(latest[2])
        progress.update(
            {
                "live_assembled_partitions": int(latest[0]),
                "live_published_partitions": live_published,
                "recent_requests": int(latest[3]),
                "recent_retries": int(latest[4]),
                "recent_tls_failures": int(latest[5]),
            }
        )
        if live_published:
            progress["completed_partitions"] = min(planned, completed + live_published)
            progress["unresolved_partitions"] = max(0, planned - int(progress["completed_partitions"]))
    if ordinary_reliability:
        latest = ordinary_reliability[-1]
        progress.update(
            {
                "recent_requests": int(latest[0]),
                "recent_retries": int(latest[1]),
                "recent_tls_failures": int(latest[2]),
            }
        )
    if log_path.exists():
        progress["last_completed_partition_time"] = datetime.fromtimestamp(
            log_path.stat().st_mtime,
            timezone.utc,
        ).isoformat().replace("+00:00", "Z")
    return progress


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _tail_text(path: Path, maximum_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - maximum_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = ["transport_progress"]
