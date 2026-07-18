from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .lease_storage import atomic_write_json, exclusive_file_lock
from .run_contracts import checksum

RUN_REGISTRY_CONTRACT = "compute_run_registry.v1"
DEFAULT_REGISTRY_PATH = Path("reports/runs/run_registry.json")


class StaleRegistryRevision(RuntimeError):
    pass


class CorruptRunRegistry(RuntimeError):
    pass


def update_run_registry(
    record: Mapping[str, Any], *, path: Path = DEFAULT_REGISTRY_PATH,
    expected_revision: int | None = None, lock_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    with exclusive_file_lock(
        path.with_suffix(".lock"), timeout_seconds=lock_timeout_seconds
    ):
        registry = _load(path) if path.exists() else _empty()
        revision = int(registry["revision"])
        if expected_revision is not None and expected_revision != revision:
            raise StaleRegistryRevision(
                f"Expected registry revision {expected_revision}, found {revision}"
            )
        required = (
            "run_identity", "run_id", "pipeline", "stage", "status",
            "source_git_commit", "machine_profile_identity",
            "run_root_relative_path", "latest_status_revision",
        )
        if any(record.get(field) in (None, "") for field in required):
            raise ValueError("Run registry record is incomplete")
        rows = {row["run_identity"]: dict(row) for row in registry["runs"]}
        existing = rows.get(record["run_identity"])
        if existing and (
            existing["run_id"] != record["run_id"]
            or existing["run_root_relative_path"] != record["run_root_relative_path"]
        ):
            raise ValueError("Run identity collision")
        if any(
            row["run_id"] == record["run_id"]
            and identity != record["run_identity"]
            for identity, row in rows.items()
        ):
            raise ValueError("Run ID collision")
        rows[str(record["run_identity"])] = {
            "start_timestamp": None, "end_timestamp": None,
            "active_count": 0, "waiting_count": 0, "completed_count": 0,
            "failed_count": 0, "blocked_count": 0, "reserved_ram_bytes": None,
            "measured_peak_ram_bytes": None, "active_cpu_weight": None,
            "summary_path": None, "result_path": None, "health_status": "HEALTHY",
            **dict(record),
            "latest_update_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        registry["runs"] = sorted(rows.values(), key=lambda row: (
            row["pipeline"], row["stage"], row["run_id"]
        ))
        registry["revision"] = revision + 1
        registry["logical_checksum"] = _registry_checksum(registry)
        atomic_write_json(path, registry)
        return registry


def read_run_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    with exclusive_file_lock(path.with_suffix(".lock")):
        return _load(path) if path.exists() else _empty()


def registry_record(
    manifest: Mapping[str, Any], status: Mapping[str, Any],
    *, summary_path: str | None = None, result_path: str | None = None,
    health_status: str = "HEALTHY",
) -> dict[str, Any]:
    counts = status.get("counts", {})
    return {
        "run_identity": manifest["run_identity"], "run_id": manifest["run_id"],
        "pipeline": manifest["pipeline"], "stage": manifest["stage"],
        "status": status["current_status"],
        "start_timestamp": status.get("started_timestamp"),
        "end_timestamp": status.get("completed_timestamp"),
        "source_git_commit": manifest["source_git_commit"],
        "machine_profile_identity": manifest["machine_profile_identity"],
        "run_root_relative_path": manifest["run_root_relative_path"],
        "active_count": counts.get("running", 0),
        "waiting_count": counts.get("waiting", 0),
        "completed_count": counts.get("completed", 0),
        "failed_count": counts.get("failed", 0),
        "blocked_count": counts.get("blocked", 0),
        "reserved_ram_bytes": status.get("reserved_ram_bytes"),
        "measured_peak_ram_bytes": status.get("measured_peak_ram_bytes"),
        "active_cpu_weight": status.get("active_cpu_weight"),
        "latest_status_revision": status["state_revision"],
        "summary_path": summary_path, "result_path": result_path,
        "health_status": health_status,
    }


def _empty() -> dict[str, Any]:
    payload = {
        "contract_version": RUN_REGISTRY_CONTRACT,
        "revision": 0, "runs": [], "logical_checksum": "",
    }
    payload["logical_checksum"] = _registry_checksum(payload)
    return payload


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorruptRunRegistry(f"CORRUPT run registry: {exc}") from exc
    if (
        payload.get("contract_version") != RUN_REGISTRY_CONTRACT
        or not isinstance(payload.get("runs"), list)
        or payload.get("logical_checksum") != _registry_checksum(payload)
        or int(payload.get("revision", -1)) < 0
    ):
        raise CorruptRunRegistry("CORRUPT run registry contract or checksum")
    identities = [row.get("run_identity") for row in payload["runs"]]
    if len(identities) != len(set(identities)):
        raise CorruptRunRegistry("CORRUPT duplicate run identity")
    return payload


def _registry_checksum(payload: Mapping[str, Any]) -> str:
    logical = dict(payload)
    logical.pop("logical_checksum", None)
    return checksum(logical)
