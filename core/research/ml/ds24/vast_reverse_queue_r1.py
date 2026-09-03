from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.research.ml.ds24 import remote_family_queue


TICKET_ID = "DREAM_SYSTEM_DS24_VAST_REVERSE_ORDER_NON_OVERLAPPING_FAMILY_QUEUE_R1"
QUEUE_ID = "DS24_VAST_REVERSE_NINE_FAMILY_R1"
MAC_QUEUE_ID = "DS24_MAC_AUX_NINE_FAMILY_R1"
DELL_QUEUE_ID = "DS24_DELL_FULL_TOURNAMENT_R40"
SOURCE_REMOTE_QUEUE_ID = remote_family_queue.QUEUE_ID
EXTERNAL_STATUS_SCHEMA_VERSION = "ds24_external_family_status_snapshot.v1"
CLAIM_SCHEMA_VERSION = "ds24_vast_family_claim.v1"
QUEUE_STATE_SCHEMA_VERSION = "ds24_vast_reverse_queue_state.v1"
TERMINAL_CLASSIFICATION = (
    "DS24_VAST_REVERSE_NINE_FAMILY_QUEUE_READY_NOT_LAUNCHED_"
    "EXTERNAL_COORDINATION_TRANSPORT_PENDING"
)

STAGE_ROOT_REL = Path(
    "docs/dream_system/components/DS-24_independent_five_minute_selector/"
    "stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
)
DEFAULT_AUTHORITY_ROOT_REL = STAGE_ROOT_REL / "r7_r49_vast_reverse_nine_family_queue_r1"
REMOTE_FAMILY_QUEUE_REL = Path("core/research/ml/ds24/remote_family_queue.py")
MAC_AUX_QUEUE_REL = Path("core/research/ml/ds24/mac_aux_queue_r44f2.py")

EXPECTED_VAST_ORDER = tuple(remote_family_queue.REMOTE_QUEUE_ORDER)
EXPECTED_MAC_ORDER = tuple(reversed(EXPECTED_VAST_ORDER))
STALE_AFTER_SECONDS = 30 * 60
DEFAULT_CLAIM_LEASE_SECONDS = 60 * 60

ACTIVE_OWNERSHIP_STATES = {
    "RESERVED",
    "CLAIMED",
    "LIVE_CLAIMED",
    "RUNNING",
    "FAST_FORWARDING",
    "STARTING",
    "INITIALIZING",
}
COMPLETE_STATES = {"COMPLETE", "SUCCESS", "TERMINAL_COMPLETE", "VERIFIED_COMPLETE"}
IDLE_STATES = {"", "PENDING", "IDLE", "QUEUE_MEMBER_IDLE", "NOT_STARTED", "ABSENT"}
TERMINAL_LOCAL_CLAIM_STATES = {"RELEASED_TEST_ONLY", "EXPIRED_RECOVERY_CONFIRMED", "ABANDONED_TEST_ONLY"}
CONFIGURATION_BLOCKING_STATES = {
    "CONFIGURATION_AUTHORITY_REQUIRED",
    "IMPLEMENTATION_BLOCKED",
    "V3_ADAPTER_REQUIRED",
    "OOF_ADAPTER_REQUIRED",
    "LINUX_CUDA_PORTABILITY_REQUIRED",
}


class VastReverseQueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueEntry:
    ordinal: int
    mac_ordinal: int
    canonical_family_id: str
    display_name: str
    family_class: str
    adapter_authority: str
    configuration_authority: str
    configuration_state: str
    model_class: str
    predictor_contract_hash: str
    target_contract_hash: str
    data_partition_authority_hash: str
    model_configuration_hash: str
    evaluator_version: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if value is None:
        return ""
    return str(value)


def display_name(family_id: str) -> str:
    return {
        "temporal_fusion_transformer": "Temporal Fusion Transformer",
        "market_context_encoder": "Market Context Encoder",
        "momentum_transformer": "Momentum Transformer",
        "itransformer": "iTransformer",
        "transformer": "Transformer",
        "patchtst": "PatchTST",
        "dlinear": "DLinear",
        "lightgbm_lambdarank": "lightgbm_lambdarank",
        "lightgbm_rank_xendcg": "lightgbm_rank_xendcg",
    }.get(family_id, family_id)


def family_class(family_id: str) -> str:
    if family_id in remote_family_queue.GPU_SEQUENCE_FAMILIES:
        return "GPU_SEQUENCE"
    if family_id in remote_family_queue.CPU_RANKING_FAMILIES:
        return "CPU_RANKING"
    raise VastReverseQueueError(f"DS24_VAST_UNKNOWN_NINE_FAMILY:{family_id}")


def _read_authority_rows(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    authority = remote_family_queue.family_configuration_authority(Path(repo_root))
    return {
        str(row.get("family")): row
        for row in authority.get("families", [])
        if isinstance(row, Mapping) and row.get("family")
    }


def build_queue_definition(
    repo_root: Path,
    *,
    accepted_order: Sequence[str] | None = None,
    configuration_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    source_registry = repo_root / REMOTE_FAMILY_QUEUE_REL
    accepted = tuple(accepted_order or EXPECTED_VAST_ORDER)
    validate_accepted_nine(accepted)
    mac_order = tuple(reversed(accepted))
    authority_rows = _read_authority_rows(repo_root)
    overrides = dict(configuration_overrides or {})
    rows: list[dict[str, Any]] = []
    for ordinal, family_id in enumerate(tuple(reversed(mac_order)), start=1):
        authority = dict(authority_rows.get(family_id, {}))
        configuration_state = overrides.get(family_id, str(authority.get("certification_state") or "CONFIGURATION_AUTHORITY_REQUIRED"))
        row = QueueEntry(
            ordinal=ordinal,
            mac_ordinal=mac_order.index(family_id) + 1,
            canonical_family_id=family_id,
            display_name=display_name(family_id),
            family_class=family_class(family_id),
            adapter_authority=str(authority.get("model_class") or remote_family_queue.MODEL_CLASSES.get(family_id, "")),
            configuration_authority=str(authority.get("configuration_source") or ""),
            configuration_state=configuration_state,
            model_class=str(authority.get("model_class") or remote_family_queue.MODEL_CLASSES.get(family_id, "")),
            predictor_contract_hash=str(authority.get("predictor_contract_hash") or remote_family_queue.PREDICTOR_CONTRACT_HASH),
            target_contract_hash=str(authority.get("target_contract_hash") or stable_hash({"target_contract": remote_family_queue.TARGET_CONTRACT_ID})),
            data_partition_authority_hash=stable_hash(
                {
                    "source_registry_content_hash": file_sha256(source_registry) if source_registry.exists() else "",
                    "queue_family": family_id,
                    "target_contract": remote_family_queue.TARGET_CONTRACT_ID,
                }
            ),
            model_configuration_hash=str(authority.get("configuration_hash") or ""),
            evaluator_version="v3",
        )
        rows.append(row.__dict__)
    payload = {
        "ticket_id": TICKET_ID,
        "queue_id": QUEUE_ID,
        "source_queue_id": SOURCE_REMOTE_QUEUE_ID,
        "mac_queue_id": MAC_QUEUE_ID,
        "vast_order_is_reverse_of_mac_order": tuple(row["canonical_family_id"] for row in rows) == tuple(reversed(mac_order)),
        "canonical_family_count": len(rows),
        "canonical_family_order": [row["canonical_family_id"] for row in rows],
        "mac_top_down_order": list(mac_order),
        "source_registry_path": REMOTE_FAMILY_QUEUE_REL.as_posix(),
        "source_registry_content_hash": file_sha256(source_registry) if source_registry.exists() else "",
        "mac_aux_queue_module_path": MAC_AUX_QUEUE_REL.as_posix(),
        "mac_aux_queue_module_present_in_checkout": (repo_root / MAC_AUX_QUEUE_REL).exists(),
        "derived_from_existing_remote_queue_authority": True,
        "entries": rows,
        "transport_implemented": False,
        "model_execution_wired": False,
        "cloud_operation_required": False,
    }
    payload["queue_definition_hash"] = stable_hash(payload)
    return payload


def validate_accepted_nine(families: Sequence[str]) -> None:
    names = [str(value) for value in families]
    if len(names) != 9:
        raise VastReverseQueueError(f"DS24_VAST_NINE_FAMILY_COUNT_MISMATCH:{len(names)}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise VastReverseQueueError("DS24_VAST_DUPLICATE_FAMILY:" + ",".join(duplicates))
    unexpected = sorted(set(names) - set(EXPECTED_VAST_ORDER))
    missing = sorted(set(EXPECTED_VAST_ORDER) - set(names))
    if unexpected or missing:
        raise VastReverseQueueError(
            "DS24_VAST_NINE_FAMILY_SET_MISMATCH:"
            f"missing={','.join(missing)}:unexpected={','.join(unexpected)}"
        )


def queue_entry(queue_definition: Mapping[str, Any], family_id: str) -> Mapping[str, Any]:
    for row in queue_definition.get("entries", []):
        if isinstance(row, Mapping) and row.get("canonical_family_id") == family_id:
            return row
    raise VastReverseQueueError(f"DS24_VAST_QUEUE_ENTRY_MISSING:{family_id}")


def external_family_status_schema() -> dict[str, Any]:
    row_required = [
        "family_id",
        "queue_membership",
        "family_state",
        "ownership_state",
        "active_owner",
        "pid",
        "pid_alive",
        "run_trial_identity",
        "checkpoint_cursor",
        "evaluator_version",
        "predictor_contract_hash",
        "target_contract_hash",
        "data_partition_authority_hash",
        "model_configuration_hash",
        "result_artifact_manifest_hash",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": EXTERNAL_STATUS_SCHEMA_VERSION,
        "title": "DS24 external family status snapshot",
        "type": "object",
        "required": [
            "schema_version",
            "source_machine",
            "source_queue_identity",
            "generated_at_utc",
            "source_state",
            "families",
        ],
        "properties": {
            "schema_version": {"const": EXTERNAL_STATUS_SCHEMA_VERSION},
            "source_machine": {"type": "string"},
            "source_queue_identity": {"type": "string"},
            "generated_at_utc": {"type": "string"},
            "source_state": {
                "type": "object",
                "required": ["path_or_command", "content_hash"],
                "properties": {
                    "path_or_command": {"type": "string"},
                    "content_hash": {"type": "string"},
                },
            },
            "families": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": row_required,
                    "properties": {name: {"type": ["string", "number", "boolean", "null"]} for name in row_required},
                },
            },
        },
    }


def vast_family_claim_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "queue_id",
        "vast_host_identity",
        "canonical_family_id",
        "claim_timestamp_utc",
        "claim_expiry_utc",
        "lease_duration_seconds",
        "scientific_contract_hashes",
        "queue_definition_hash",
        "external_snapshot_hashes",
        "state_generation",
        "previous_state_hash",
        "claim_status",
        "claim_id",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": CLAIM_SCHEMA_VERSION,
        "title": "DS24 Vast family claim",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array"]} for name in required},
    }


def family_contract_hashes(entry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "predictor_contract_hash": str(entry.get("predictor_contract_hash") or ""),
        "target_contract_hash": str(entry.get("target_contract_hash") or ""),
        "data_partition_authority_hash": str(entry.get("data_partition_authority_hash") or ""),
        "model_configuration_hash": str(entry.get("model_configuration_hash") or ""),
        "evaluator_version": str(entry.get("evaluator_version") or "v3"),
    }


def synthetic_external_status_fixture(queue_definition: Mapping[str, Any], *, now_utc: str | None = None) -> dict[str, Any]:
    now = now_utc or utc_now()
    snapshots = [
        neutral_external_snapshot(
            queue_definition,
            source_machine="dell",
            source_queue_identity=DELL_QUEUE_ID,
            generated_at_utc=now,
        ),
        neutral_external_snapshot(
            queue_definition,
            source_machine="mac",
            source_queue_identity=MAC_QUEUE_ID,
            generated_at_utc=now,
        ),
    ]
    return {
        "fixture_id": "DS24_VAST_R1_SYNTHETIC_NEUTRAL_EXTERNAL_STATUS",
        "schema_version": EXTERNAL_STATUS_SCHEMA_VERSION,
        "snapshots": snapshots,
        "expected_first_reverse_candidate": queue_definition["canonical_family_order"][0],
        "contains_credentials": False,
        "contacts_cloud_transport": False,
    }


def neutral_external_snapshot(
    queue_definition: Mapping[str, Any],
    *,
    source_machine: str,
    source_queue_identity: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    rows = []
    for entry in queue_definition.get("entries", []):
        hashes = family_contract_hashes(entry)
        rows.append(
            {
                "family_id": entry["canonical_family_id"],
                "queue_membership": True,
                "family_state": "PENDING",
                "ownership_state": "QUEUE_MEMBER_IDLE",
                "active_owner": "",
                "pid": 0,
                "pid_alive": False,
                "liveness_evidence": "neutral synthetic snapshot; no live claim",
                "run_trial_identity": "",
                "checkpoint_cursor": "",
                "evaluator_version": hashes["evaluator_version"],
                "predictor_contract_hash": hashes["predictor_contract_hash"],
                "target_contract_hash": hashes["target_contract_hash"],
                "data_partition_authority_hash": hashes["data_partition_authority_hash"],
                "model_configuration_hash": hashes["model_configuration_hash"],
                "result_artifact_manifest_hash": "",
            }
        )
    source_state = {
        "path_or_command": f"synthetic://{source_machine}/neutral",
        "content_hash": stable_hash(rows),
    }
    snapshot = {
        "schema_version": EXTERNAL_STATUS_SCHEMA_VERSION,
        "source_machine": source_machine,
        "source_queue_identity": source_queue_identity,
        "generated_at_utc": generated_at_utc,
        "source_state": source_state,
        "families": rows,
    }
    snapshot["snapshot_hash"] = snapshot_hash(snapshot)
    return snapshot


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    payload = dict(snapshot)
    payload.pop("snapshot_hash", None)
    return stable_hash(payload)


def validate_external_status_snapshot(
    snapshot: Mapping[str, Any],
    queue_definition: Mapping[str, Any],
    *,
    now_utc: str | None = None,
    max_age_seconds: int = STALE_AFTER_SECONDS,
    allow_stale: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if snapshot.get("schema_version") != EXTERNAL_STATUS_SCHEMA_VERSION:
        errors.append("SCHEMA_VERSION_MISMATCH")
    source_machine = str(snapshot.get("source_machine") or "")
    if not source_machine:
        errors.append("SOURCE_MACHINE_MISSING")
    if not snapshot.get("source_queue_identity"):
        errors.append("SOURCE_QUEUE_IDENTITY_MISSING")
    source_state = snapshot.get("source_state")
    if not isinstance(source_state, Mapping) or not source_state.get("path_or_command") or not source_state.get("content_hash"):
        errors.append("SOURCE_STATE_HASH_MISSING")
    generated = parse_utc(snapshot.get("generated_at_utc"))
    now = parse_utc(now_utc or utc_now())
    age_seconds: float | None = None
    if generated is None:
        errors.append("GENERATED_AT_UTC_INVALID")
    elif now is not None:
        age_seconds = max(0.0, (now - generated).total_seconds())
        if age_seconds > max_age_seconds and not allow_stale:
            errors.append("SNAPSHOT_STALE_FAIL_CLOSED")
    families = snapshot.get("families")
    if not isinstance(families, list):
        errors.append("FAMILIES_NOT_ARRAY")
        families = []
    queue_ids = set(queue_definition.get("canonical_family_order", []))
    seen: set[str] = set()
    row_errors: list[dict[str, Any]] = []
    required = set(external_family_status_schema()["properties"]["families"]["items"]["required"])
    for index, row in enumerate(families):
        if not isinstance(row, Mapping):
            row_errors.append({"row_index": index, "error": "ROW_NOT_OBJECT"})
            continue
        missing = sorted(required - set(row))
        family_id = str(row.get("family_id") or "")
        if missing:
            row_errors.append({"row_index": index, "family_id": family_id, "error": "ROW_REQUIRED_FIELDS_MISSING", "missing": missing})
        if not family_id:
            row_errors.append({"row_index": index, "error": "FAMILY_ID_MISSING"})
        elif family_id in seen:
            row_errors.append({"row_index": index, "family_id": family_id, "error": "DUPLICATE_FAMILY_STATUS"})
        seen.add(family_id)
        if row.get("queue_membership") is True and family_id not in queue_ids:
            row_errors.append({"row_index": index, "family_id": family_id, "error": "UNEXPECTED_QUEUE_MEMBER"})
        if row.get("evaluator_version") not in ("", None, "v3", "V3"):
            row_errors.append({"row_index": index, "family_id": family_id, "error": "EVALUATOR_VERSION_NOT_V3"})
    if row_errors:
        errors.append("MALFORMED_FAMILY_ROWS")
    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "classification": "SNAPSHOT_VALID" if status == "PASS" else "SNAPSHOT_MALFORMED_OR_STALE_FAIL_CLOSED",
        "source_machine": source_machine,
        "source_queue_identity": str(snapshot.get("source_queue_identity") or ""),
        "age_seconds": age_seconds,
        "errors": errors,
        "row_errors": row_errors,
        "snapshot_hash": snapshot_hash(snapshot),
    }


def compatible_hashes(row: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    expected = family_contract_hashes(entry)
    return all(str(row.get(key) or "") == value for key, value in expected.items())


def classify_external_family_row(row: Mapping[str, Any], entry: Mapping[str, Any], source_machine: str) -> dict[str, Any]:
    ownership = str(row.get("ownership_state") or "").upper()
    state = str(row.get("family_state") or "").upper()
    pid = safe_int(row.get("pid"))
    pid_alive = bool(row.get("pid_alive"))
    hashes_compatible = compatible_hashes(row, entry)
    family_id = str(row.get("family_id") or "")
    if ownership in ACTIVE_OWNERSHIP_STATES or state in ACTIVE_OWNERSHIP_STATES:
        if pid > 0 and not pid_alive:
            classification = "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED"
        elif hashes_compatible:
            classification = "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED"
        else:
            classification = "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION"
    elif ownership in COMPLETE_STATES or state in COMPLETE_STATES:
        if hashes_compatible and row.get("result_artifact_manifest_hash"):
            classification = "SKIPPED_EXTERNAL_VERIFIED"
        else:
            classification = "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED"
    elif ownership in IDLE_STATES or state in IDLE_STATES:
        classification = "QUEUE_MEMBER_IDLE_NOT_OWNED"
    else:
        classification = "STALE_OR_MALFORMED_EVIDENCE_FAIL_CLOSED"
    return {
        "family_id": family_id,
        "source_machine": source_machine,
        "source_queue_identity": str(row.get("source_queue_identity") or ""),
        "classification": classification,
        "hashes_compatible": hashes_compatible,
        "ownership_state": ownership,
        "family_state": state,
        "active_owner": str(row.get("active_owner") or ""),
        "pid": pid,
        "pid_alive": pid_alive,
        "run_trial_identity": str(row.get("run_trial_identity") or ""),
        "checkpoint_cursor": str(row.get("checkpoint_cursor") or ""),
        "result_artifact_manifest_hash": str(row.get("result_artifact_manifest_hash") or ""),
    }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def evaluate_external_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    queue_definition: Mapping[str, Any],
    *,
    now_utc: str | None = None,
    required_sources: Sequence[str] = ("dell", "mac"),
    allow_missing_snapshots: bool = False,
    allow_stale: bool = False,
) -> dict[str, Any]:
    validations = [
        validate_external_status_snapshot(snapshot, queue_definition, now_utc=now_utc, allow_stale=allow_stale)
        for snapshot in snapshots
    ]
    failures = [row for row in validations if row["status"] != "PASS"]
    by_source = {str(snapshot.get("source_machine") or ""): snapshot for snapshot in snapshots}
    missing_sources = sorted(set(required_sources) - set(by_source))
    if missing_sources and not allow_missing_snapshots:
        failures.append(
            {
                "status": "FAIL",
                "classification": "MISSING_REQUIRED_EXTERNAL_SNAPSHOT_FAIL_CLOSED",
                "missing_sources": missing_sources,
            }
        )
    family_evidence: dict[str, list[dict[str, Any]]] = {family: [] for family in queue_definition["canonical_family_order"]}
    for snapshot in snapshots:
        source_machine = str(snapshot.get("source_machine") or "")
        for row in snapshot.get("families", []):
            if not isinstance(row, Mapping):
                continue
            family_id = str(row.get("family_id") or "")
            if family_id not in family_evidence:
                continue
            enriched = dict(row)
            enriched["source_queue_identity"] = str(snapshot.get("source_queue_identity") or "")
            family_evidence[family_id].append(classify_external_family_row(enriched, queue_entry(queue_definition, family_id), source_machine))
    contradictions = contradictory_ownership(family_evidence)
    if contradictions:
        failures.append(
            {
                "status": "FAIL",
                "classification": "CONTRADICTORY_EXTERNAL_OWNERSHIP_FAIL_CLOSED",
                "contradictions": contradictions,
            }
        )
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "classification": "EXTERNAL_COORDINATION_ACCEPTED" if status == "PASS" else "EXTERNAL_COORDINATION_FAIL_CLOSED",
        "validations": validations,
        "failures": failures,
        "missing_sources": missing_sources,
        "contradictions": contradictions,
        "family_evidence": family_evidence,
        "external_snapshot_hashes": {
            str(snapshot.get("source_machine") or f"snapshot_{index}"): snapshot_hash(snapshot)
            for index, snapshot in enumerate(snapshots)
        },
    }


def contradictory_ownership(family_evidence: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    active_classes = {
        "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED",
        "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION",
        "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED",
    }
    for family_id, rows in family_evidence.items():
        active = [row for row in rows if row.get("classification") in active_classes]
        sources = sorted({str(row.get("source_machine") or "") for row in active})
        if len(sources) > 1:
            out.append({"family_id": family_id, "sources": sources, "classifications": [row["classification"] for row in active]})
    return out


def initial_queue_state(queue_definition: Mapping[str, Any], *, now_utc: str | None = None) -> dict[str, Any]:
    now = now_utc or utc_now()
    ledger = [
        {
            "ordinal": entry["ordinal"],
            "canonical_family_id": entry["canonical_family_id"],
            "status": "PENDING",
            "skip_reason": "",
            "block_reason": "",
            "claim_id": "",
            "last_update_utc": "",
        }
        for entry in queue_definition.get("entries", [])
    ]
    state = {
        "schema_version": QUEUE_STATE_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "generation": 0,
        "previous_state_hash": "",
        "current_cursor": "",
        "ledger": ledger,
        "skipped_families": [],
        "blocked_families": [],
        "external_ownership_evidence": [],
        "vast_claims": [],
        "terminal_entries": [],
        "retryable_failure_state": [],
        "last_heartbeat_utc": now,
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }
    state["state_hash"] = queue_state_hash(state)
    return state


def safety_guards() -> dict[str, Any]:
    return {
        "queue_only": True,
        "model_execution_started": False,
        "vast_cloud_operation_performed": False,
        "backblaze_or_cloud_transport_performed": False,
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
        "full_prediction_output_written": False,
        "live_dell_mac_queue_state_mutated": False,
    }


def queue_state_hash(state: Mapping[str, Any]) -> str:
    payload = dict(state)
    payload.pop("state_hash", None)
    return stable_hash(payload)


def validate_queue_state_payload(state: Mapping[str, Any], queue_definition: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if state.get("schema_version") != QUEUE_STATE_SCHEMA_VERSION:
        errors.append("STATE_SCHEMA_VERSION_MISMATCH")
    if state.get("queue_id") != QUEUE_ID:
        errors.append("QUEUE_ID_MISMATCH")
    if state.get("queue_definition_hash") != queue_definition.get("queue_definition_hash"):
        errors.append("QUEUE_DEFINITION_MISMATCH")
    if state.get("state_hash") != queue_state_hash(state):
        errors.append("STATE_HASH_MISMATCH")
    ledger = state.get("ledger")
    if not isinstance(ledger, list):
        errors.append("LEDGER_NOT_ARRAY")
    else:
        families = [str(row.get("canonical_family_id") or "") for row in ledger if isinstance(row, Mapping)]
        if families != list(queue_definition.get("canonical_family_order", [])):
            errors.append("LEDGER_FAMILY_ORDER_MISMATCH")
    return {
        "status": "PASS" if not errors else "FAIL",
        "classification": "QUEUE_STATE_VALID" if not errors else "QUEUE_STATE_CORRUPT_OR_INCOMPATIBLE",
        "errors": errors,
    }


def queue_state_path(queue_root: Path) -> Path:
    return Path(queue_root) / "queue_state.json"


def load_queue_state(queue_root: Path, queue_definition: Mapping[str, Any], *, initialise: bool = True) -> dict[str, Any]:
    path = queue_state_path(queue_root)
    if not path.exists():
        if not initialise:
            raise VastReverseQueueError(f"DS24_VAST_QUEUE_STATE_MISSING:{path}")
        state = initial_queue_state(queue_definition)
        write_queue_state(queue_root, state)
        return state
    state = read_json(path)
    if not isinstance(state, Mapping):
        raise VastReverseQueueError("DS24_VAST_QUEUE_STATE_NOT_OBJECT")
    validation = validate_queue_state_payload(state, queue_definition)
    if validation["status"] != "PASS":
        raise VastReverseQueueError("DS24_VAST_QUEUE_STATE_INVALID:" + ",".join(validation["errors"]))
    return dict(state)


def write_queue_state(queue_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    payload["state_hash"] = queue_state_hash(payload)
    write_json_atomic(queue_state_path(queue_root), payload)
    return payload


def _active_local_claims(state: Mapping[str, Any], *, now_utc: str | None = None) -> list[dict[str, Any]]:
    now = parse_utc(now_utc or utc_now())
    out: list[dict[str, Any]] = []
    for row in state.get("vast_claims", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("claim_status") or "")
        if status in TERMINAL_LOCAL_CLAIM_STATES:
            continue
        expiry = parse_utc(row.get("claim_expiry_utc"))
        claim = dict(row)
        claim["expired"] = bool(now and expiry and expiry <= now)
        out.append(claim)
    return out


def _class_for_evidence(rows: Sequence[Mapping[str, Any]], classes: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("classification") in classes]


def meeting_boundary(
    queue_definition: Mapping[str, Any],
    family_evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    next_ordinal: int | None = None,
) -> dict[str, Any]:
    mac_boundary_classes = {
        "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED",
        "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION",
        "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED",
        "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED",
    }
    positions = {
        str(entry["canonical_family_id"]): int(entry["ordinal"])
        for entry in queue_definition.get("entries", [])
    }
    owned = []
    for family_id, rows in family_evidence.items():
        for row in rows:
            if row.get("source_machine") == "mac" and row.get("classification") in mac_boundary_classes:
                owned.append(
                    {
                        "family_id": family_id,
                        "ordinal": positions[family_id],
                        "classification": row["classification"],
                        "source_machine": "mac",
                    }
                )
    if not owned:
        return {
            "nearest_external_owned_boundary": "",
            "nearest_external_owned_boundary_ordinal": None,
            "queues_met": False,
            "boundary_classification": "NO_EXTERNAL_BOUNDARY",
        }
    nearest = min(owned, key=lambda row: int(row["ordinal"]))
    queues_met = next_ordinal is not None and int(next_ordinal) >= int(nearest["ordinal"])
    return {
        "nearest_external_owned_boundary": nearest["family_id"],
        "nearest_external_owned_boundary_ordinal": nearest["ordinal"],
        "queues_met": queues_met,
        "boundary_classification": "BOUNDARY_REACHED_EXTERNAL_OWNER" if queues_met else "BOUNDARY_AHEAD_EXTERNAL_OWNER",
    }


def dry_run_plan(
    queue_root: Path,
    queue_definition: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    now_utc: str | None = None,
    allow_missing_snapshots: bool = False,
    allow_stale: bool = False,
    skip_configuration_blockers: bool = True,
) -> dict[str, Any]:
    state = load_queue_state(queue_root, queue_definition)
    return plan_from_state(
        state,
        queue_definition,
        snapshots,
        now_utc=now_utc,
        allow_missing_snapshots=allow_missing_snapshots,
        allow_stale=allow_stale,
        skip_configuration_blockers=skip_configuration_blockers,
    )


def plan_from_state(
    state: Mapping[str, Any],
    queue_definition: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    now_utc: str | None = None,
    allow_missing_snapshots: bool = False,
    allow_stale: bool = False,
    skip_configuration_blockers: bool = True,
) -> dict[str, Any]:
    validation = validate_queue_state_payload(state, queue_definition)
    if validation["status"] != "PASS":
        return {
            "status": "FAIL_CLOSED",
            "admission_status": validation["classification"],
            "next_vast_eligible_family": "",
            "diagnostics": validation,
            "model_executor_invoked": False,
        }
    external = evaluate_external_snapshots(
        snapshots,
        queue_definition,
        now_utc=now_utc,
        allow_missing_snapshots=allow_missing_snapshots,
        allow_stale=allow_stale,
    )
    if external["status"] != "PASS":
        return {
            "status": "FAIL_CLOSED",
            "admission_status": external["classification"],
            "next_vast_eligible_family": "",
            "diagnostics": external,
            "model_executor_invoked": False,
            "guards": safety_guards(),
        }
    active_claims = _active_local_claims(state, now_utc=now_utc)
    expired_claims = [row for row in active_claims if row.get("expired")]
    if expired_claims:
        return {
            "status": "FAIL_CLOSED",
            "admission_status": "LOCAL_CLAIM_STALE_RECOVERY_REQUIRED",
            "next_vast_eligible_family": "",
            "stale_claims": expired_claims,
            "model_executor_invoked": False,
            "guards": safety_guards(),
        }
    if active_claims:
        claim = active_claims[0]
        return {
            "status": "RESERVED",
            "admission_status": "LOCAL_VAST_CLAIM_ALREADY_ACTIVE",
            "next_vast_eligible_family": claim["canonical_family_id"],
            "reserved_for_vast": [claim["canonical_family_id"]],
            "claim": claim,
            "model_executor_invoked": False,
            "guards": safety_guards(),
        }
    ledger_by_family = {
        str(row.get("canonical_family_id")): row
        for row in state.get("ledger", [])
        if isinstance(row, Mapping)
    }
    decisions: list[dict[str, Any]] = []
    skipped_external_verified: list[str] = []
    blocked_by_live_external: list[str] = []
    blocked_families: list[dict[str, str]] = []
    next_ordinal: int | None = None
    next_family = ""
    for entry in queue_definition.get("entries", []):
        family_id = str(entry["canonical_family_id"])
        ordinal = int(entry["ordinal"])
        local_status = str(ledger_by_family.get(family_id, {}).get("status") or "PENDING")
        if local_status in {"COMPLETE", "SKIPPED_EXTERNAL_VERIFIED", "RELEASED_TEST_ONLY"}:
            decisions.append({"family_id": family_id, "decision": "LOCAL_TERMINAL_ALREADY_RECORDED", "ordinal": ordinal})
            continue
        if local_status in {"RESERVED_FOR_VAST", "RUNNING"}:
            next_family = family_id
            next_ordinal = ordinal
            decisions.append({"family_id": family_id, "decision": "LOCAL_RESERVED_FOR_VAST", "ordinal": ordinal})
            break
        if str(entry.get("configuration_state") or "") in CONFIGURATION_BLOCKING_STATES:
            decision = {
                "family_id": family_id,
                "decision": "CONFIGURATION_AUTHORITY_REQUIRED",
                "ordinal": ordinal,
                "blocker": str(entry.get("configuration_state") or ""),
            }
            decisions.append(decision)
            blocked_families.append({"family_id": family_id, "reason": decision["blocker"]})
            if not skip_configuration_blockers:
                next_ordinal = ordinal
                break
            continue
        rows = list(external["family_evidence"].get(family_id, []))
        compatible_live = _class_for_evidence(
            rows,
            {
                "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED",
                "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION",
            },
        )
        ambiguous = _class_for_evidence(rows, {"DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED"})
        incompatible_complete = _class_for_evidence(rows, {"EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED"})
        verified_complete = _class_for_evidence(rows, {"SKIPPED_EXTERNAL_VERIFIED"})
        if compatible_live:
            blocked_by_live_external.append(family_id)
            next_ordinal = ordinal
            decisions.append(
                {
                    "family_id": family_id,
                    "decision": "BOUNDARY_REACHED_EXTERNAL_OWNER"
                    if any(row.get("source_machine") == "mac" for row in compatible_live)
                    else "BLOCKED_EXTERNAL_OWNER",
                    "ordinal": ordinal,
                    "evidence": compatible_live,
                }
            )
            break
        if ambiguous:
            next_ordinal = ordinal
            decisions.append({"family_id": family_id, "decision": "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED", "ordinal": ordinal, "evidence": ambiguous})
            break
        if incompatible_complete:
            next_ordinal = ordinal
            decisions.append({"family_id": family_id, "decision": "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED", "ordinal": ordinal, "evidence": incompatible_complete})
            break
        if verified_complete:
            skipped_external_verified.append(family_id)
            decisions.append({"family_id": family_id, "decision": "SKIPPED_EXTERNAL_VERIFIED", "ordinal": ordinal, "evidence": verified_complete})
            continue
        next_family = family_id
        next_ordinal = ordinal
        decisions.append({"family_id": family_id, "decision": "CLAIMABLE", "ordinal": ordinal})
        break
    boundary = meeting_boundary(queue_definition, external["family_evidence"], next_ordinal=next_ordinal)
    if decisions and decisions[-1]["decision"] in {
        "BOUNDARY_REACHED_EXTERNAL_OWNER",
        "BLOCKED_EXTERNAL_OWNER",
        "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED",
        "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED",
        "CONFIGURATION_AUTHORITY_REQUIRED",
    }:
        status = "PAUSED"
        admission = decisions[-1]["decision"]
    elif next_family:
        status = "READY"
        admission = "CLAIMABLE"
    else:
        status = "COMPLETE_OR_NO_USEFUL_WORK"
        admission = "NO_USEFUL_NON_OVERLAPPING_WORK_REMAINS"
    if boundary["queues_met"] and admission == "CLAIMABLE":
        status = "PAUSED"
        admission = "BOUNDARY_REACHED_EXTERNAL_OWNER"
        next_family = ""
    return {
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "status": status,
        "admission_status": admission,
        "next_vast_eligible_family": next_family,
        "next_vast_eligible_ordinal": next_ordinal,
        "nearest_externally_owned_boundary": boundary["nearest_external_owned_boundary"],
        "nearest_externally_owned_boundary_ordinal": boundary["nearest_external_owned_boundary_ordinal"],
        "skipped_external_verified": skipped_external_verified,
        "blocked_by_live_external": blocked_by_live_external,
        "reserved_for_vast": [claim["canonical_family_id"] for claim in active_claims],
        "blocked_families": blocked_families,
        "useful_non_overlapping_work_remains": bool(next_family),
        "queues_met": bool(boundary["queues_met"] or admission == "BOUNDARY_REACHED_EXTERNAL_OWNER"),
        "candidate_decisions": decisions,
        "external_snapshot_hashes": external["external_snapshot_hashes"],
        "external_validation": external,
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }


def build_claim_payload(
    *,
    family_id: str,
    queue_definition: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    now_utc: str,
    lease_seconds: int,
    vast_host_identity: str = "VAST_HOST_PLACEHOLDER_NOT_RENTED",
) -> dict[str, Any]:
    now = parse_utc(now_utc)
    if now is None:
        raise VastReverseQueueError("DS24_VAST_CLAIM_TIMESTAMP_INVALID")
    entry = queue_entry(queue_definition, family_id)
    generation = int(state.get("generation", 0)) + 1
    expiry = format_utc(now + timedelta(seconds=int(lease_seconds)))
    payload = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "vast_host_identity": vast_host_identity,
        "canonical_family_id": family_id,
        "claim_timestamp_utc": format_utc(now),
        "claim_expiry_utc": expiry,
        "lease_duration_seconds": int(lease_seconds),
        "scientific_contract_hashes": family_contract_hashes(entry),
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "external_snapshot_hashes": dict(plan.get("external_snapshot_hashes") or {}),
        "state_generation": generation,
        "previous_state_hash": str(state.get("state_hash") or ""),
        "claim_status": "RESERVED_LOCAL_ONLY_NOT_LAUNCHED",
        "claim_id": "",
        "model_executor_invoked": False,
    }
    payload["claim_id"] = stable_hash(payload)
    return payload


def record_local_vast_claim(
    queue_root: Path,
    queue_definition: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    now_utc: str | None = None,
    lease_seconds: int = DEFAULT_CLAIM_LEASE_SECONDS,
    family_id: str | None = None,
    test_only: bool = True,
    allow_missing_snapshots: bool = False,
    allow_stale: bool = False,
) -> dict[str, Any]:
    if not test_only:
        raise VastReverseQueueError("DS24_VAST_REAL_RUNNABLE_CLAIM_REQUIRES_FUTURE_TRANSPORT_TICKET")
    now = now_utc or utc_now()
    state = load_queue_state(queue_root, queue_definition)
    active_claims = _active_local_claims(state, now_utc=now)
    expired = [row for row in active_claims if row.get("expired")]
    if expired:
        raise VastReverseQueueError("DS24_VAST_LOCAL_CLAIM_STALE_RECOVERY_REQUIRED")
    if active_claims:
        return {
            "status": "CLAIM_ALREADY_ACTIVE",
            "claim": active_claims[0],
            "claim_count": len(active_claims),
            "model_executor_invoked": False,
        }
    plan = plan_from_state(
        state,
        queue_definition,
        snapshots,
        now_utc=now,
        allow_missing_snapshots=allow_missing_snapshots,
        allow_stale=allow_stale,
    )
    claim_family = family_id or str(plan.get("next_vast_eligible_family") or "")
    if plan.get("admission_status") != "CLAIMABLE" or not claim_family:
        raise VastReverseQueueError(f"DS24_VAST_CLAIM_REFUSED:{plan.get('admission_status')}")
    if claim_family != plan.get("next_vast_eligible_family"):
        raise VastReverseQueueError("DS24_VAST_CLAIM_FAMILY_NOT_NEXT_ELIGIBLE")
    claim = build_claim_payload(
        family_id=claim_family,
        queue_definition=queue_definition,
        plan=plan,
        state=state,
        now_utc=now,
        lease_seconds=lease_seconds,
    )
    claim_path = Path(queue_root) / "claims" / f"{claim_family}.claim.json"
    if claim_path.exists():
        existing = read_json(claim_path)
        if isinstance(existing, Mapping) and existing.get("claim_status") not in TERMINAL_LOCAL_CLAIM_STATES:
            raise VastReverseQueueError("DS24_VAST_ACTIVE_CLAIM_FILE_EXISTS")
    write_json_atomic(claim_path, claim)
    updated = dict(state)
    updated["generation"] = int(state.get("generation", 0)) + 1
    updated["previous_state_hash"] = str(state.get("state_hash") or "")
    updated["last_heartbeat_utc"] = now
    updated["vast_claims"] = [*list(state.get("vast_claims", [])), claim]
    updated["external_ownership_evidence"] = [dict(plan.get("external_snapshot_hashes") or {})]
    external_hashes = dict(plan.get("external_snapshot_hashes") or {})
    skipped = set(str(family) for family in plan.get("skipped_external_verified", []))
    existing_skipped = {
        str(row.get("family_id") or "")
        for row in state.get("skipped_families", [])
        if isinstance(row, Mapping)
    }
    updated["skipped_families"] = [
        *list(state.get("skipped_families", [])),
        *[
            {
                "family_id": family,
                "reason": "SKIPPED_EXTERNAL_VERIFIED",
                "recorded_at_utc": now,
                "external_snapshot_hashes": external_hashes,
            }
            for family in sorted(skipped)
            if family not in existing_skipped
        ],
    ]
    blockers = {
        str(row.get("family_id") or ""): str(row.get("reason") or "")
        for row in plan.get("blocked_families", [])
        if isinstance(row, Mapping)
    }
    existing_blockers = {
        (str(row.get("family_id") or ""), str(row.get("reason") or ""))
        for row in state.get("blocked_families", [])
        if isinstance(row, Mapping)
    }
    updated["blocked_families"] = [
        *list(state.get("blocked_families", [])),
        *[
            {
                "family_id": family,
                "reason": reason,
                "recorded_at_utc": now,
                "external_snapshot_hashes": external_hashes,
            }
            for family, reason in sorted(blockers.items())
            if (family, reason) not in existing_blockers
        ],
    ]
    ledger = []
    for row in state.get("ledger", []):
        row = dict(row)
        family = str(row.get("canonical_family_id") or "")
        if family in skipped:
            row.update(
                {
                    "status": "SKIPPED_EXTERNAL_VERIFIED",
                    "skip_reason": "compatible external completion verified",
                    "last_update_utc": now,
                }
            )
        if family in blockers:
            row.update(
                {
                    "status": "BLOCKED_CONFIGURATION_SKIPPED",
                    "block_reason": blockers[family],
                    "last_update_utc": now,
                }
            )
        if row.get("canonical_family_id") == claim_family:
            row.update(
                {
                    "status": "RESERVED_FOR_VAST",
                    "claim_id": claim["claim_id"],
                    "last_update_utc": now,
                }
            )
        ledger.append(row)
    updated["ledger"] = ledger
    updated["current_cursor"] = claim_family
    saved = write_queue_state(queue_root, updated)
    return {
        "status": "CLAIM_RECORDED_TEST_ONLY",
        "claim": claim,
        "claim_artifact": str(claim_path),
        "queue_state_hash": saved["state_hash"],
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }


def release_local_vast_claim(
    queue_root: Path,
    queue_definition: Mapping[str, Any],
    *,
    family_id: str,
    now_utc: str | None = None,
    test_only: bool = True,
    recover_stale: bool = False,
) -> dict[str, Any]:
    if not test_only:
        raise VastReverseQueueError("DS24_VAST_RELEASE_ONLY_AVAILABLE_AS_TEST_RECOVERY_IN_R1")
    now = now_utc or utc_now()
    state = load_queue_state(queue_root, queue_definition)
    claims = [dict(row) for row in state.get("vast_claims", []) if isinstance(row, Mapping)]
    changed = False
    for claim in claims:
        if claim.get("canonical_family_id") != family_id or claim.get("claim_status") in TERMINAL_LOCAL_CLAIM_STATES:
            continue
        expired = parse_utc(claim.get("claim_expiry_utc")) and parse_utc(claim.get("claim_expiry_utc")) <= parse_utc(now)
        if expired and not recover_stale:
            raise VastReverseQueueError("DS24_VAST_STALE_CLAIM_RELEASE_REQUIRES_EXPLICIT_RECOVERY")
        claim["claim_status"] = "EXPIRED_RECOVERY_CONFIRMED" if expired else "RELEASED_TEST_ONLY"
        claim["released_at_utc"] = now
        changed = True
        claim_path = Path(queue_root) / "claims" / f"{family_id}.claim.json"
        write_json_atomic(claim_path, claim)
    if not changed:
        raise VastReverseQueueError(f"DS24_VAST_NO_ACTIVE_CLAIM_TO_RELEASE:{family_id}")
    updated = dict(state)
    updated["generation"] = int(state.get("generation", 0)) + 1
    updated["previous_state_hash"] = str(state.get("state_hash") or "")
    updated["last_heartbeat_utc"] = now
    updated["vast_claims"] = claims
    ledger = []
    for row in state.get("ledger", []):
        row = dict(row)
        if row.get("canonical_family_id") == family_id:
            row.update({"status": "RELEASED_TEST_ONLY", "last_update_utc": now})
        ledger.append(row)
    updated["ledger"] = ledger
    saved = write_queue_state(queue_root, updated)
    return {"status": "CLAIM_RELEASED_TEST_ONLY", "queue_state_hash": saved["state_hash"], "model_executor_invoked": False}


def load_snapshots(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    snapshots = []
    for path in paths:
        payload = read_json(Path(path))
        if not isinstance(payload, Mapping):
            raise VastReverseQueueError(f"DS24_VAST_EXTERNAL_SNAPSHOT_NOT_OBJECT:{path}")
        snapshots.append(dict(payload))
    return snapshots


def shallow_process_snapshot(repo_root: Path) -> dict[str, Any]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$rows=Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -match 'python|powershell' -or ($_.CommandLine -match 'ds24|ds26') } | "
                "Select-Object ProcessId,Name,CommandLine; "
                "$rows | ConvertTo-Json -Depth 3"
            ),
        ]
    else:
        command = ["ps", "-eo", "pid=,comm=,args="]
    try:
        completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, timeout=20, check=False)
    except Exception as exc:
        return {"taken_at_utc": utc_now(), "status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "protected_processes": []}
    rows = _parse_process_rows(completed.stdout, windows=os.name == "nt")
    protected = [
        {
            "pid": row["pid"],
            "name": row["name"],
            "role": protected_process_role(row["command_line"]),
            "command_hash": stable_hash(row["command_line"]),
        }
        for row in rows
        if protected_process_role(row["command_line"])
    ]
    return {
        "taken_at_utc": utc_now(),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "process_count": len(rows),
        "protected_process_count": len(protected),
        "protected_processes": protected,
    }


def _parse_process_rows(stdout: str, *, windows: bool) -> list[dict[str, Any]]:
    if not stdout.strip():
        return []
    if windows:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        return [
            {
                "pid": safe_int(row.get("ProcessId")),
                "name": str(row.get("Name") or ""),
                "command_line": str(row.get("CommandLine") or ""),
            }
            for row in items
            if isinstance(row, Mapping)
        ]
    out = []
    for line in stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        out.append({"pid": safe_int(parts[0]), "name": parts[1], "command_line": parts[2] if len(parts) > 2 else ""})
    return out


def protected_process_role(command_line: str) -> str:
    text = command_line.lower()
    if "ds24_p8_r14_e3g_c2_r7_policy_queue_supervisor.py" in text:
        return "DS24_DELL_SUPERVISOR"
    if "ds24_p8_r14_e3g_c2_r7_r14_policy_worker.py" in text:
        return "DS24_DELL_POLICY_WORKER"
    if "ds26_prospective_capture_worker.py" in text:
        return "DS26_PROSPECTIVE_CAPTURE"
    if "rclone" in text and "ds24" in text:
        return "UNRELATED_BACKBLAZE_UPLOAD_LEFT_UNINSPECTED"
    return ""


def process_signature(snapshot: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    return sorted(
        (
            safe_int(row.get("pid")),
            str(row.get("role") or ""),
            str(row.get("command_hash") or ""),
        )
        for row in snapshot.get("protected_processes", [])
        if isinstance(row, Mapping)
    )


def collect_process_table(repo_root: Path) -> dict[str, Any]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$rows=Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -match 'python|powershell' -or ($_.CommandLine -match 'ds24|ds26') } | "
                "Select-Object ProcessId,Name,CommandLine; "
                "$rows | ConvertTo-Json -Depth 3"
            ),
        ]
    else:
        command = ["ps", "-eo", "pid=,comm=,args="]
    try:
        completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, timeout=20, check=False)
    except Exception as exc:
        raise VastReverseQueueError(f"DS24_EXTERNAL_SNAPSHOT_PROCESS_TABLE_FAILED:{type(exc).__name__}:{exc}") from exc
    if completed.returncode != 0:
        raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_PROCESS_TABLE_FAILED:" + completed.stderr.strip())
    return {
        "status": "PASS",
        "captured_at_utc": utc_now(),
        "path_or_command": " ".join(command),
        "processes": _parse_process_rows(completed.stdout, windows=os.name == "nt"),
    }


def process_rows_from_snapshot(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        if payload.get("status") not in ("PASS", None, ""):
            raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_PROCESS_STATE_FAILED")
        source = payload.get("processes") or payload.get("rows") or []
    elif isinstance(payload, list):
        source = payload
    else:
        raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_PROCESS_STATE_NOT_ARRAY")
    rows = []
    for row in source:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "pid": safe_int(row.get("pid") or row.get("ProcessId")),
                "name": str(row.get("name") or row.get("Name") or ""),
                "command_line": str(row.get("command_line") or row.get("CommandLine") or row.get("command") or ""),
                "pid_alive": bool(row.get("pid_alive", True)),
            }
        )
    return rows


def load_process_rows(repo_root: Path, process_snapshot: str | Path | None = None) -> tuple[list[dict[str, Any]], str]:
    if process_snapshot:
        path = Path(process_snapshot)
        payload = read_json(path)
        rows = process_rows_from_snapshot(payload)
        return rows, str(path)
    payload = collect_process_table(repo_root)
    return process_rows_from_snapshot(payload), str(payload.get("path_or_command") or "process-table")


def _command_mentions_family(command_line: str, family_id: str) -> bool:
    text = re.sub(r"['\"`]", " ", command_line.lower())
    family = family_id.lower()
    return any(
        marker in text
        for marker in (
            f"--family {family}",
            f"--family={family}",
            f"family={family}",
            f"family {family}",
        )
    )


def process_rows_by_family(process_rows: Sequence[Mapping[str, Any]], queue_definition: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in queue_definition["canonical_family_order"]}
    for row in process_rows:
        if not isinstance(row, Mapping):
            continue
        command_line = str(row.get("command_line") or "")
        if "ds24" not in command_line.lower():
            continue
        for family_id in by_family:
            if _command_mentions_family(command_line, family_id):
                by_family[family_id].append(dict(row))
    ambiguous = {family: rows for family, rows in by_family.items() if len(rows) > 1}
    if ambiguous:
        raise VastReverseQueueError(
            "DS24_EXTERNAL_SNAPSHOT_AMBIGUOUS_MULTIPLE_PROCESSES:"
            + ",".join(sorted(ambiguous))
        )
    return {family: rows[0] for family, rows in by_family.items() if rows}


def _ledger_rows_by_family(state: Mapping[str, Any], queue_definition: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = [row for row in state.get("ledger", []) if isinstance(row, Mapping)]
    by_family = {str(row.get("canonical_family_id") or ""): row for row in rows}
    expected = list(queue_definition.get("canonical_family_order", []))
    if [str(row.get("canonical_family_id") or "") for row in rows] != expected:
        raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_LEDGER_FAMILY_ORDER_MISMATCH")
    return by_family


def _mac_aux_state_candidates(queue_root: Path) -> list[Path]:
    preferred = [
        queue_root / "queue_state.json",
        queue_root / "status.json",
        queue_root / "state.json",
        queue_root / "mac_aux_queue_state.json",
    ]
    existing = [path for path in preferred if path.exists()]
    if existing:
        return existing
    return sorted(path for path in queue_root.glob("*.json") if path.is_file())


def _mac_aux_family_rows(payload: Mapping[str, Any], queue_definition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("ledger", "families", "family_status", "statuses", "queue", "entries", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    expected = set(queue_definition.get("canonical_family_order", []))
    keyed_rows = []
    for key, value in payload.items():
        if key in expected and isinstance(value, Mapping):
            row = dict(value)
            row.setdefault("canonical_family_id", key)
            keyed_rows.append(row)
    return keyed_rows


def _row_family_id(row: Mapping[str, Any]) -> str:
    for key in ("canonical_family_id", "family_id", "family", "model_family", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _mac_aux_status(row: Mapping[str, Any]) -> str:
    raw = ""
    for key in ("status", "family_state", "state", "ownership_state", "lifecycle_state", "run_state"):
        raw = str(row.get(key) or "").strip()
        if raw:
            break
    status = raw.upper().replace("-", "_").replace(" ", "_")
    if not status:
        if row.get("completed") is True:
            return "COMPLETE"
        if row.get("running") is True or row.get("pid"):
            return "RUNNING"
        return "PENDING"
    if status in COMPLETE_STATES or status in {"DONE", "COMPLETED", "FINISHED"}:
        return "COMPLETE"
    if status in {"RUNNING", "ACTIVE", "IN_PROGRESS", "TRAINING", "STARTED"}:
        return "RUNNING"
    if status in {"RESERVED", "CLAIMED", "LIVE_CLAIMED", "LEASED", "OWNED"}:
        return "CLAIMED"
    if status in IDLE_STATES or status in {"QUEUED", "WAITING"}:
        return "PENDING"
    if status in CONFIGURATION_BLOCKING_STATES or status == "BLOCKED_CONFIGURATION_SKIPPED":
        return status
    raise VastReverseQueueError(f"DS24_MAC_AUX_QUEUE_STATUS_INCOMPATIBLE:{_row_family_id(row)}:{status}")


def _row_first_string(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _translate_mac_aux_state(
    queue_root: Path,
    queue_definition: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    candidates = _mac_aux_state_candidates(queue_root)
    if not candidates:
        raise VastReverseQueueError(f"DS24_MAC_AUX_QUEUE_STATE_MISSING:{queue_root}")
    if len(candidates) != 1:
        raise VastReverseQueueError(
            "DS24_MAC_AUX_QUEUE_STATE_AMBIGUOUS:" + ",".join(path.name for path in candidates)
        )
    source = candidates[0]
    payload = read_json(source)
    if not isinstance(payload, Mapping):
        raise VastReverseQueueError("DS24_MAC_AUX_QUEUE_STATE_NOT_OBJECT")
    identity = str(
        payload.get("queue_id")
        or payload.get("source_queue_identity")
        or queue_root.name.replace("queue=", "")
        or ""
    )
    if identity != MAC_QUEUE_ID:
        raise VastReverseQueueError(f"DS24_MAC_AUX_QUEUE_ID_INCOMPATIBLE:{identity}")
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in _mac_aux_family_rows(payload, queue_definition):
        family_id = _row_family_id(row)
        if family_id in by_family:
            raise VastReverseQueueError(f"DS24_MAC_AUX_QUEUE_DUPLICATE_FAMILY:{family_id}")
        if family_id:
            by_family[family_id] = row
    expected = list(queue_definition.get("canonical_family_order", []))
    missing = [family_id for family_id in expected if family_id not in by_family]
    extra = sorted(family_id for family_id in by_family if family_id not in expected)
    if missing or extra:
        raise VastReverseQueueError(
            "DS24_MAC_AUX_QUEUE_FAMILY_SET_INCOMPATIBLE:"
            f"missing={','.join(missing)}:extra={','.join(extra)}"
        )
    ledger = []
    for entry in queue_definition.get("entries", []):
        family_id = str(entry["canonical_family_id"])
        row = by_family[family_id]
        ledger.append(
            {
                "ordinal": entry["ordinal"],
                "canonical_family_id": family_id,
                "status": _mac_aux_status(row),
                "claim_id": _row_first_string(row, ("claim_id", "lease_id", "run_id", "worker_id")),
                "checkpoint_cursor": _row_first_string(row, ("checkpoint_cursor", "cursor", "latest_checkpoint", "checkpoint_path")),
                "result_artifact_manifest_hash": _row_first_string(
                    row,
                    (
                        "result_artifact_manifest_hash",
                        "manifest_hash",
                        "output_manifest_hash",
                        "artifact_manifest_hash",
                        "result_hash",
                        "completion_hash",
                    ),
                ),
                "last_update_utc": _row_first_string(row, ("last_update_utc", "updated_at_utc", "timestamp_utc")),
            }
        )
    state = {
        "schema_version": QUEUE_STATE_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "generation": safe_int(payload.get("generation") or payload.get("state_generation")),
        "previous_state_hash": str(payload.get("previous_state_hash") or ""),
        "current_cursor": _row_first_string(
            payload,
            ("current_cursor", "cursor", "active_family", "running_family", "current_family", "next_family"),
        ),
        "ledger": ledger,
        "skipped_families": [],
        "blocked_families": [],
        "external_ownership_evidence": [],
        "vast_claims": [],
        "terminal_entries": [],
        "retryable_failure_state": [],
        "last_heartbeat_utc": str(payload.get("last_heartbeat_utc") or payload.get("updated_at_utc") or ""),
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }
    state["state_hash"] = queue_state_hash(state)
    validation = validate_queue_state_payload(state, queue_definition)
    if validation["status"] != "PASS":
        raise VastReverseQueueError("DS24_MAC_AUX_QUEUE_TRANSLATION_INVALID:" + ",".join(validation["errors"]))
    return state, f"{source} translated_from_mac_aux_queue"


def _load_export_queue_state(
    queue_root: Path,
    queue_definition: Mapping[str, Any],
    *,
    machine: str,
) -> tuple[dict[str, Any], str]:
    try:
        return load_queue_state(queue_root, queue_definition, initialise=False), str(queue_state_path(queue_root))
    except VastReverseQueueError as exc:
        if machine != "mac":
            raise
        try:
            return _translate_mac_aux_state(queue_root, queue_definition)
        except VastReverseQueueError as translate_exc:
            raise VastReverseQueueError(
                f"DS24_MAC_AUX_QUEUE_INCOMPATIBLE:{queue_root}:{translate_exc}"
            ) from translate_exc


def _terminal_result_hash(state: Mapping[str, Any], ledger_row: Mapping[str, Any], family_id: str) -> str:
    for key in ("result_artifact_manifest_hash", "manifest_hash", "output_manifest_hash"):
        value = str(ledger_row.get(key) or "")
        if value:
            return value
    for row in state.get("terminal_entries", []):
        if isinstance(row, Mapping) and row.get("canonical_family_id") == family_id:
            for key in ("result_artifact_manifest_hash", "manifest_hash", "output_manifest_hash"):
                value = str(row.get(key) or "")
                if value:
                    return value
    return ""


def _row_cursor(state: Mapping[str, Any], ledger_row: Mapping[str, Any], family_id: str) -> str:
    for key in ("checkpoint_cursor", "cursor", "latest_checkpoint", "checkpoint_path"):
        value = str(ledger_row.get(key) or "")
        if value:
            return value
    if state.get("current_cursor") == family_id:
        return str(state.get("current_cursor") or "")
    return ""


def _export_family_row(
    *,
    entry: Mapping[str, Any],
    ledger_row: Mapping[str, Any],
    state: Mapping[str, Any],
    machine: str,
    process_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    family_id = str(entry["canonical_family_id"])
    ledger_status = str(ledger_row.get("status") or "PENDING").upper()
    pid = safe_int(process_row.get("pid")) if isinstance(process_row, Mapping) else 0
    pid_alive = bool(process_row.get("pid_alive", True)) if isinstance(process_row, Mapping) else False
    hashes = family_contract_hashes(entry)
    result_hash = _terminal_result_hash(state, ledger_row, family_id)
    active_without_process_states = {"RESERVED", "CLAIMED", "LIVE_CLAIMED", "RESERVED_FOR_VAST"}
    running_states = ACTIVE_OWNERSHIP_STATES - active_without_process_states
    complete_states = COMPLETE_STATES | {"SKIPPED_EXTERNAL_VERIFIED"}
    if process_row is not None:
        family_state = "RUNNING"
        ownership_state = "RUNNING"
        active_owner = machine
    elif ledger_status in running_states:
        raise VastReverseQueueError(f"DS24_EXTERNAL_SNAPSHOT_ACTIVE_STATE_WITHOUT_LIVE_PROCESS:{family_id}:{ledger_status}")
    elif ledger_status in active_without_process_states:
        family_state = ledger_status
        ownership_state = "CLAIMED"
        active_owner = machine
    elif ledger_status in complete_states:
        if not result_hash:
            raise VastReverseQueueError(f"DS24_EXTERNAL_SNAPSHOT_COMPLETE_WITHOUT_RESULT_MANIFEST:{family_id}")
        family_state = "COMPLETE"
        ownership_state = "COMPLETE"
        active_owner = machine
    elif ledger_status in IDLE_STATES:
        family_state = "PENDING"
        ownership_state = "QUEUE_MEMBER_IDLE"
        active_owner = ""
    elif ledger_status in CONFIGURATION_BLOCKING_STATES or ledger_status == "BLOCKED_CONFIGURATION_SKIPPED":
        family_state = ledger_status
        ownership_state = ledger_status
        active_owner = machine
    else:
        raise VastReverseQueueError(f"DS24_EXTERNAL_SNAPSHOT_AMBIGUOUS_FAMILY_STATE:{family_id}:{ledger_status}")
    return {
        "family_id": family_id,
        "queue_membership": True,
        "family_state": family_state,
        "ownership_state": ownership_state,
        "active_owner": active_owner,
        "pid": pid,
        "pid_alive": pid_alive,
        "liveness_evidence": "process_table_match" if process_row is not None else "queue_state_no_live_process",
        "run_trial_identity": str(ledger_row.get("run_trial_identity") or ledger_row.get("claim_id") or ""),
        "checkpoint_cursor": _row_cursor(state, ledger_row, family_id),
        "evaluator_version": hashes["evaluator_version"],
        "predictor_contract_hash": hashes["predictor_contract_hash"],
        "target_contract_hash": hashes["target_contract_hash"],
        "data_partition_authority_hash": hashes["data_partition_authority_hash"],
        "model_configuration_hash": hashes["model_configuration_hash"],
        "result_artifact_manifest_hash": result_hash,
    }


def export_external_snapshot(
    repo_root: Path,
    queue_root: Path,
    *,
    machine: str,
    output: str | Path | None = None,
    process_snapshot: str | Path | None = None,
    now_utc: str | None = None,
) -> dict[str, Any]:
    machine = str(machine or "").lower()
    if machine not in {"dell", "mac"}:
        raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_MACHINE_MUST_BE_DELL_OR_MAC")
    repo_root = Path(repo_root).resolve()
    queue_root = Path(queue_root)
    if not queue_root.is_absolute():
        queue_root = repo_root / queue_root
    queue_definition = build_queue_definition(repo_root)
    state, queue_state_source = _load_export_queue_state(queue_root, queue_definition, machine=machine)
    process_rows, process_source = load_process_rows(repo_root, process_snapshot)
    process_by_family = process_rows_by_family(process_rows, queue_definition)
    ledger_by_family = _ledger_rows_by_family(state, queue_definition)
    rows = [
        _export_family_row(
            entry=entry,
            ledger_row=ledger_by_family[str(entry["canonical_family_id"])],
            state=state,
            machine=machine,
            process_row=process_by_family.get(str(entry["canonical_family_id"])),
        )
        for entry in queue_definition.get("entries", [])
    ]
    process_hashes = [
        {"pid": safe_int(row.get("pid")), "command_hash": stable_hash(str(row.get("command_line") or ""))}
        for row in process_rows
        if isinstance(row, Mapping) and str(row.get("command_line") or "")
    ]
    snapshot = {
        "schema_version": EXTERNAL_STATUS_SCHEMA_VERSION,
        "source_machine": machine,
        "source_queue_identity": DELL_QUEUE_ID if machine == "dell" else MAC_QUEUE_ID,
        "generated_at_utc": now_utc or utc_now(),
        "current_cursor": str(state.get("current_cursor") or ""),
        "state_generation": int(state.get("generation") or 0),
        "queue_state_hash": str(state.get("state_hash") or ""),
        "scientific_contract_hashes": {
            str(entry["canonical_family_id"]): family_contract_hashes(entry)
            for entry in queue_definition.get("entries", [])
        },
        "source_state": {
            "path_or_command": f"{queue_state_source} + {process_source}",
            "content_hash": stable_hash(
                {
                    "queue_state_hash": state.get("state_hash", ""),
                    "process_hashes": process_hashes,
                    "families": rows,
                }
            ),
        },
        "families": rows,
    }
    snapshot["snapshot_hash"] = snapshot_hash(snapshot)
    validation = validate_external_status_snapshot(snapshot, queue_definition, now_utc=snapshot["generated_at_utc"])
    if validation["status"] != "PASS":
        raise VastReverseQueueError("DS24_EXTERNAL_SNAPSHOT_EXPORT_INVALID:" + ",".join(validation["errors"]))
    if output:
        write_json_atomic(Path(output), snapshot)
    return snapshot


def validate_queue_definition(queue_definition: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    order = [str(row.get("canonical_family_id") or "") for row in queue_definition.get("entries", []) if isinstance(row, Mapping)]
    try:
        validate_accepted_nine(order)
    except VastReverseQueueError as exc:
        errors.append(str(exc))
    if order != list(queue_definition.get("canonical_family_order", [])):
        errors.append("CANONICAL_ORDER_MISMATCH")
    if queue_definition.get("queue_id") != QUEUE_ID:
        errors.append("QUEUE_ID_MISMATCH")
    observed_hash = dict(queue_definition)
    expected_hash = str(observed_hash.pop("queue_definition_hash", ""))
    if stable_hash(observed_hash) != expected_hash:
        errors.append("QUEUE_DEFINITION_HASH_MISMATCH")
    return {
        "status": "PASS" if not errors else "FAIL",
        "classification": "QUEUE_DEFINITION_VALID" if not errors else "QUEUE_DEFINITION_INVALID",
        "errors": errors,
        "queue_definition_hash": queue_definition.get("queue_definition_hash", ""),
        "canonical_family_order": order,
    }


def artifact_inventory(root: Path, filenames: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for name in filenames:
        path = Path(root) / name
        if path.is_file():
            rows.append({"path": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return rows


def write_authority_package(repo_root: Path, authority_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root)
    authority_root = Path(authority_root or (repo_root / DEFAULT_AUTHORITY_ROOT_REL))
    if not authority_root.is_absolute():
        authority_root = repo_root / authority_root
    assert_authority_root_safe(repo_root, authority_root)
    before = shallow_process_snapshot(repo_root)
    queue_definition = build_queue_definition(repo_root)
    queue_root = authority_root / "queue_state"
    state = load_queue_state(queue_root, queue_definition)
    fixture = synthetic_external_status_fixture(queue_definition, now_utc=utc_now())
    plan = dry_run_plan(
        queue_root,
        queue_definition,
        fixture["snapshots"],
        now_utc=fixture["snapshots"][0]["generated_at_utc"],
    )
    validation = {
        "ticket_id": TICKET_ID,
        "queue_definition": validate_queue_definition(queue_definition),
        "queue_state": validate_queue_state_payload(state, queue_definition),
        "external_snapshot_fixture": [
            validate_external_status_snapshot(snapshot, queue_definition, now_utc=fixture["snapshots"][0]["generated_at_utc"])
            for snapshot in fixture["snapshots"]
        ],
        "dry_run_plan_status": plan["status"],
        "status": "PASS" if plan["next_vast_eligible_family"] == queue_definition["canonical_family_order"][0] else "FAIL",
    }
    test_evidence = {
        "ticket_id": TICKET_ID,
        "status": "PENDING_EXTERNAL_TEST_RUN",
        "focused_tests": "",
        "adjacent_ds24_tests": "",
        "architecture_conformance": "",
        "diff_whitespace": "",
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }
    limitations = {
        "terminal_classification": TERMINAL_CLASSIFICATION,
        "vast_execution_ready": False,
        "cloud_transfer_complete": False,
        "environment_ready": False,
        "model_certification_claimed": False,
        "cross_host_atomic_ownership": False,
        "transport_limitation": "Dell/Mac/Vast snapshot and claim transport is a file/schema interface only in this ticket.",
        "mac_aux_module_observed": queue_definition["mac_aux_queue_module_present_in_checkout"],
        "mac_aux_module_note": "Named mac_aux_queue_r44f2 module was not present; canonical nine-family IDs were derived from remote_family_queue authority.",
        "no_backblaze_upload_inspection": True,
    }
    readme = readme_text(queue_definition, plan)
    files = {
        "vast_reverse_queue_definition.json": queue_definition,
        "external_family_status.schema.json": external_family_status_schema(),
        "synthetic_external_status_fixture.json": fixture,
        "vast_family_claim.schema.json": vast_family_claim_schema(),
        "dry_run_plan.json": plan,
        "queue_validation.json": validation,
        "test_evidence.json": test_evidence,
        "limitations.json": limitations,
        "process_snapshot_before.json": before,
    }
    authority_root.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        write_json_atomic(authority_root / name, payload)
    write_text_atomic(authority_root / "README.md", readme)
    after = shallow_process_snapshot(repo_root)
    write_json_atomic(authority_root / "process_snapshot_after.json", after)
    manifest_files = [
        "vast_reverse_queue_definition.json",
        "external_family_status.schema.json",
        "synthetic_external_status_fixture.json",
        "vast_family_claim.schema.json",
        "dry_run_plan.json",
        "queue_validation.json",
        "test_evidence.json",
        "README.md",
        "limitations.json",
        "process_snapshot_before.json",
        "process_snapshot_after.json",
        "queue_state/queue_state.json",
    ]
    manifest = {
        "ticket_id": TICKET_ID,
        "queue_id": QUEUE_ID,
        "terminal_classification": TERMINAL_CLASSIFICATION,
        "created_at_utc": utc_now(),
        "authority_root": repo_rel(repo_root, authority_root),
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "external_status_schema_version": EXTERNAL_STATUS_SCHEMA_VERSION,
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "dry_run_next_family": plan["next_vast_eligible_family"],
        "canonical_family_order": queue_definition["canonical_family_order"],
        "artifact_inventory": artifact_inventory(authority_root, manifest_files),
        "safety": {
            **safety_guards(),
            "process_signature_unchanged": process_signature(before) == process_signature(after),
            "before_protected_process_count": before["protected_process_count"],
            "after_protected_process_count": after["protected_process_count"],
        },
        "model_process_started_or_stopped": False,
        "cloud_or_vast_operation_occurred": False,
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json_atomic(authority_root / "manifest.json", manifest)
    return manifest


def assert_authority_root_safe(repo_root: Path, authority_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    target = Path(authority_root).resolve()
    forbidden = [
        repo_root / STAGE_ROOT_REL / "r7_r14_policy_workers",
        repo_root / STAGE_ROOT_REL / "r7_r44f_vast_morning_launch_readiness" / "transfer",
    ]
    for root in forbidden:
        try:
            target.relative_to(root.resolve())
        except ValueError:
            continue
        raise VastReverseQueueError(f"DS24_VAST_AUTHORITY_ROOT_INSIDE_PROTECTED_NAMESPACE:{target}")


def readme_text(queue_definition: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    order = "\n".join(
        f"{index}. `{family}`"
        for index, family in enumerate(queue_definition.get("canonical_family_order", []), start=1)
    )
    return f"""# DS24 Vast Reverse Nine-Family Queue R1

Queue ID: `{QUEUE_ID}`

Terminal classification: `{TERMINAL_CLASSIFICATION}`

This is a queue-only authority package. It does not rent Vast hardware, connect
to cloud storage, install dependencies, launch an executor, fit a model, score a
model, open a holdout, write full predictions, or place paper/live orders.

Vast runs the nine-family lane bottom-to-top so a future Mac queue can continue
top-to-bottom. The canonical family IDs are derived from
`{REMOTE_FAMILY_QUEUE_REL.as_posix()}` and the planned Vast order is:

{order}

Dell and Mac ownership must be supplied later as files matching
`external_family_status.schema.json`. A mere queue membership row is not a live
claim. A compatible live claim blocks Vast admission; a compatible verified
completion is skipped as `SKIPPED_EXTERNAL_VERIFIED`; stale, missing,
malformed, contradictory, dead-PID or incompatible evidence fails closed.

Local Vast claims are deterministic JSON artifacts under `queue_state/claims/`.
They are atomic local writes and remain `RESERVED_LOCAL_ONLY_NOT_LAUNCHED`.
Expired claims require explicit recovery and are not silently replaced.

Dry-run next family from neutral fresh snapshots:
`{plan.get("next_vast_eligible_family", "")}`.

Cross-host atomicity is not implemented here. The file/schema interface is ready
for a later transport ticket.

Future Vast launcher inputs:

1. A queue root containing `queue_state/queue_state.json`.
2. Fresh Dell and Mac snapshots that validate against
   `external_family_status.schema.json`.
3. A non-expired claim artifact matching `vast_family_claim.schema.json`.
4. A later-ticket executor adapter for the claimed canonical family ID.
5. Vast host identity and environment evidence supplied outside this package.
"""


def record_test_evidence(
    authority_root: Path,
    *,
    focused_tests: str,
    adjacent_ds24_tests: str,
    architecture_conformance: str,
    diff_whitespace: str,
) -> dict[str, Any]:
    authority_root = Path(authority_root)
    path = authority_root / "test_evidence.json"
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        payload = {}
    updated = {
        **dict(payload),
        "status": "PASS"
        if all(text.startswith("PASS") for text in [focused_tests, adjacent_ds24_tests, architecture_conformance, diff_whitespace])
        else "FAIL",
        "focused_tests": focused_tests,
        "adjacent_ds24_tests": adjacent_ds24_tests,
        "architecture_conformance": architecture_conformance,
        "diff_whitespace": diff_whitespace,
        "updated_at_utc": utc_now(),
        "model_executor_invoked": False,
        "guards": safety_guards(),
    }
    updated["test_evidence_hash"] = stable_hash(updated)
    write_json_atomic(path, updated)
    manifest_path = authority_root / "manifest.json"
    manifest = read_json(manifest_path)
    if isinstance(manifest, Mapping) and manifest:
        manifest = dict(manifest)
        manifest["test_evidence_status"] = updated["status"]
        manifest["test_evidence_hash"] = updated["test_evidence_hash"]
        manifest["artifact_inventory"] = artifact_inventory(
            authority_root,
            [
                "vast_reverse_queue_definition.json",
                "external_family_status.schema.json",
                "synthetic_external_status_fixture.json",
                "vast_family_claim.schema.json",
                "dry_run_plan.json",
                "queue_validation.json",
                "test_evidence.json",
                "README.md",
                "limitations.json",
                "process_snapshot_before.json",
                "process_snapshot_after.json",
                "queue_state/queue_state.json",
            ],
        )
        manifest["manifest_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
        write_json_atomic(manifest_path, manifest)
    return updated


def status_payload(queue_root: Path, queue_definition: Mapping[str, Any]) -> dict[str, Any]:
    state = load_queue_state(queue_root, queue_definition)
    return {
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition["queue_definition_hash"],
        "state_validation": validate_queue_state_payload(state, queue_definition),
        "current_cursor": state.get("current_cursor", ""),
        "generation": state.get("generation", 0),
        "skipped_families": state.get("skipped_families", []),
        "blocked_families": state.get("blocked_families", []),
        "external_ownership_evidence": state.get("external_ownership_evidence", []),
        "reserved_for_vast": [
            row.get("canonical_family_id")
            for row in _active_local_claims(state)
            if not row.get("expired")
        ],
        "ledger": state.get("ledger", []),
        "guards": safety_guards(),
        "model_executor_invoked": False,
    }


def print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 Vast reverse nine-family queue R1")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT_REL))

    validate_snapshot = sub.add_parser("validate-snapshot")
    validate_snapshot.add_argument("--repo-root", default=".")
    validate_snapshot.add_argument("--snapshot", required=True)
    validate_snapshot.add_argument("--now-utc", default="")
    validate_snapshot.add_argument("--allow-stale", action="store_true")

    export_snapshot = sub.add_parser("export-external-snapshot")
    export_snapshot.add_argument("--repo-root", default=".")
    export_snapshot.add_argument("--queue-root", required=True)
    export_snapshot.add_argument("--machine", required=True, choices=["dell", "mac"])
    export_snapshot.add_argument("--output", required=True)
    export_snapshot.add_argument("--process-snapshot", default="")
    export_snapshot.add_argument("--now-utc", default="")

    for name in ("dry-run-plan", "next", "boundary"):
        command = sub.add_parser(name)
        command.add_argument("--repo-root", default=".")
        command.add_argument("--queue-root", default="")
        command.add_argument("--external-snapshot", action="append", default=[])
        command.add_argument("--now-utc", default="")
        command.add_argument("--allow-neutral-synthetic", action="store_true")
        command.add_argument("--allow-stale", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--repo-root", default=".")
    status.add_argument("--queue-root", default="")

    claim = sub.add_parser("claim")
    claim.add_argument("--repo-root", default=".")
    claim.add_argument("--queue-root", default="")
    claim.add_argument("--external-snapshot", action="append", default=[])
    claim.add_argument("--now-utc", default="")
    claim.add_argument("--lease-seconds", type=int, default=DEFAULT_CLAIM_LEASE_SECONDS)
    claim.add_argument("--family", default="")
    claim.add_argument("--test-only", action="store_true")
    claim.add_argument("--allow-neutral-synthetic", action="store_true")

    release = sub.add_parser("release-claim")
    release.add_argument("--repo-root", default=".")
    release.add_argument("--queue-root", default="")
    release.add_argument("--family", required=True)
    release.add_argument("--now-utc", default="")
    release.add_argument("--test-only", action="store_true")
    release.add_argument("--recover-stale", action="store_true")

    validate_state = sub.add_parser("validate-state")
    validate_state.add_argument("--repo-root", default=".")
    validate_state.add_argument("--queue-root", default="")

    record = sub.add_parser("record-test-evidence")
    record.add_argument("--authority-root", required=True)
    record.add_argument("--focused-tests", required=True)
    record.add_argument("--adjacent-ds24-tests", required=True)
    record.add_argument("--architecture-conformance", required=True)
    record.add_argument("--diff-whitespace", required=True)

    args = parser.parse_args(argv)
    repo_root = Path(getattr(args, "repo_root", ".")).resolve()
    queue_definition = build_queue_definition(repo_root) if getattr(args, "command", "") != "record-test-evidence" else {}

    if args.command == "prepare":
        print_json(write_authority_package(repo_root, Path(args.authority_root)))
        return 0
    if args.command == "record-test-evidence":
        print_json(
            record_test_evidence(
                Path(args.authority_root),
                focused_tests=args.focused_tests,
                adjacent_ds24_tests=args.adjacent_ds24_tests,
                architecture_conformance=args.architecture_conformance,
                diff_whitespace=args.diff_whitespace,
            )
        )
        return 0
    if args.command == "validate-snapshot":
        print_json(
            validate_external_status_snapshot(
                read_json(Path(args.snapshot)),
                queue_definition,
                now_utc=args.now_utc or None,
                allow_stale=args.allow_stale,
            )
        )
        return 0
    if args.command == "export-external-snapshot":
        snapshot = export_external_snapshot(
            repo_root,
            Path(args.queue_root),
            machine=args.machine,
            output=Path(args.output),
            process_snapshot=args.process_snapshot or None,
            now_utc=args.now_utc or None,
        )
        print_json(
            {
                "status": "PASS",
                "classification": "EXTERNAL_SNAPSHOT_EXPORTED",
                "source_machine": snapshot["source_machine"],
                "source_queue_identity": snapshot["source_queue_identity"],
                "output": str(Path(args.output)),
                "snapshot_hash": snapshot["snapshot_hash"],
                "family_count": len(snapshot["families"]),
            }
        )
        return 0
    queue_root = Path(getattr(args, "queue_root", "") or (repo_root / DEFAULT_AUTHORITY_ROOT_REL / "queue_state"))
    if not queue_root.is_absolute():
        queue_root = repo_root / queue_root
    if args.command == "status":
        print_json(status_payload(queue_root, queue_definition))
        return 0
    if args.command == "validate-state":
        state = load_queue_state(queue_root, queue_definition, initialise=False)
        print_json(validate_queue_state_payload(state, queue_definition))
        return 0
    snapshots = load_snapshots(getattr(args, "external_snapshot", []))
    if getattr(args, "allow_neutral_synthetic", False) and not snapshots:
        snapshots = synthetic_external_status_fixture(queue_definition, now_utc=args.now_utc or utc_now())["snapshots"]
    if args.command in {"dry-run-plan", "next", "boundary"}:
        plan = dry_run_plan(
            queue_root,
            queue_definition,
            snapshots,
            now_utc=args.now_utc or None,
            allow_missing_snapshots=getattr(args, "allow_neutral_synthetic", False),
            allow_stale=getattr(args, "allow_stale", False),
        )
        if args.command == "boundary":
            print_json(
                {
                    "queue_id": QUEUE_ID,
                    "nearest_externally_owned_boundary": plan.get("nearest_externally_owned_boundary", ""),
                    "queues_met": plan.get("queues_met", False),
                    "admission_status": plan.get("admission_status", ""),
                    "model_executor_invoked": False,
                }
            )
        elif args.command == "next":
            print_json(
                {
                    "queue_id": QUEUE_ID,
                    "next_vast_eligible_family": plan.get("next_vast_eligible_family", ""),
                    "admission_status": plan.get("admission_status", ""),
                    "model_executor_invoked": False,
                }
            )
        else:
            print_json(plan)
        return 0
    if args.command == "claim":
        print_json(
            record_local_vast_claim(
                queue_root,
                queue_definition,
                snapshots,
                now_utc=args.now_utc or None,
                lease_seconds=args.lease_seconds,
                family_id=args.family or None,
                test_only=bool(args.test_only),
                allow_missing_snapshots=bool(args.allow_neutral_synthetic),
            )
        )
        return 0
    if args.command == "release-claim":
        print_json(
            release_local_vast_claim(
                queue_root,
                queue_definition,
                family_id=args.family,
                now_utc=args.now_utc or None,
                test_only=bool(args.test_only),
                recover_stale=bool(args.recover_stale),
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
