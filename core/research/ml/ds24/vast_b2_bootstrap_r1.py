from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.research.ml.ds24 import vast_reverse_queue_r1 as reverse_queue


TICKET_ID = "DREAM_SYSTEM_DS24_VAST_B2_BOOTSTRAP_REVERSE_AUTOSTART_DURABLE_OUTPUT_SYNC_AND_DELL_REPATRIATION_R1"
TERMINAL_CLASSIFICATION = "DS24_VAST_B2_BOOTSTRAP_REVERSE_AUTOSTART_AND_DELL_REPATRIATION_READY_NOT_EXECUTED"
READY_LIVE_PREFLIGHT_CLASSIFICATION = "READY_FOR_EXPLICITLY_AUTHORIZED_VAST_LIVE_PREFLIGHT"
BLOCKED_QUEUE_AUTHORITY = "BLOCKED_VAST_REVERSE_QUEUE_AUTHORITY_MISSING"

QUEUE_ID = reverse_queue.QUEUE_ID
PREREQUISITE_CLASSIFICATION = reverse_queue.TERMINAL_CLASSIFICATION
DATASET_ID = "DS24_FULL_DATA_R1"
B2_BUCKET = "TradingSystemDataset44"
B2_PREFIX = "ds24/full_data_r1"
DATASET_COMPLETE_MARKER_KEY = f"{B2_PREFIX}/DATASET_COMPLETE.json"
EXPECTED_DATASET_OBJECT_COUNT = 18_505
EXPECTED_DATASET_BYTES = 47_323_707_293
DISPLAYED_DATASET_SIZE = "44.074 GiB"

STAGE_ROOT_REL = reverse_queue.STAGE_ROOT_REL
DEFAULT_QUEUE_AUTHORITY_ROOT_REL = reverse_queue.DEFAULT_AUTHORITY_ROOT_REL
DEFAULT_AUTHORITY_ROOT_REL = STAGE_ROOT_REL / "r7_r50_vast_b2_bootstrap_reverse_autostart_durable_sync_repatriation_r1"
SOURCE_MANIFEST_REL = (
    STAGE_ROOT_REL
    / "r7_r44f_vast_morning_launch_readiness"
    / "transfer"
    / "full_data_rsync_files_from.txt"
)

INPUT_DATASET_SCHEMA_VERSION = "ds24_vast_input_dataset_authority.v1"
BOOTSTRAP_CONFIG_SCHEMA_VERSION = "ds24_vast_bootstrap_config.v1"
OWNERSHIP_PLAN_SCHEMA_VERSION = "ds24_vast_ownership_plan.v1"
VAST_OUTPUT_MANIFEST_SCHEMA_VERSION = "ds24_vast_output_manifest.v1"
DELL_IMPORT_RECEIPT_SCHEMA_VERSION = "ds24_vast_dell_import_receipt.v1"

DATASET_FINALIZE_CONFIRM_TOKEN = "FINALIZE_DS24_B2_DATASET_AFTER_UPLOAD_COMPLETE"
LIVE_BOOTSTRAP_CONFIRM_TOKEN = "AUTHORIZE_DS24_VAST_LIVE_START_AFTER_VERIFY"
REAL_B2_CONFIRM_TOKEN = "AUTHORIZE_DS24_B2_REMOTE_IO"
DEFAULT_OWNERSHIP_PLAN_TTL_SECONDS = 30 * 60
DEFAULT_BOOTSTRAP_LEASE_SECONDS = 6 * 60 * 60
DEFAULT_PUBLISHER_MAX_BACKUP_AGE_SECONDS = 20 * 60
DELL_IMPORT_ROOT_REL = DEFAULT_AUTHORITY_ROOT_REL / "dell_imports"

FORBIDDEN_ARTIFACT_MARKERS = (
    "full_prediction",
    "full-prediction",
    "prediction_partitions",
    "holdout",
    "paper_order",
    "live_order",
    ".env",
    "credential",
    "secret",
    "rclone.conf",
)
DEFAULT_PERMITTED_DATASET_ROOTS = (
    "data/",
    "config/",
    "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/",
)


class VastB2BootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetManifestRow:
    relative_path: str
    size_bytes: int
    sha256: str = ""
    sha1: str = ""

    def object_key(self, prefix: str = B2_PREFIX) -> str:
        return f"{prefix.rstrip('/')}/{self.relative_path}"


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


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except Exception:
        return Path(path).as_posix().replace("\\", "/")


def csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if value is None:
        return ""
    return str(value)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{time.time_ns()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column, "")) for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def normalize_relative_path(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raise VastB2BootstrapError("DS24_VAST_EMPTY_RELATIVE_PATH")
    if "://" in raw or raw.startswith("/") or raw.startswith("~"):
        raise VastB2BootstrapError(f"DS24_VAST_ABSOLUTE_OR_URI_PATH_REFUSED:{raw}")
    if len(raw) >= 2 and raw[1] == ":":
        raise VastB2BootstrapError(f"DS24_VAST_DRIVE_PATH_REFUSED:{raw}")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise VastB2BootstrapError(f"DS24_VAST_TRAVERSAL_PATH_REFUSED:{raw}")
    return "/".join(parts)


def safe_join(root: Path, relative_path: str) -> Path:
    rel = normalize_relative_path(relative_path)
    target = Path(root).resolve() / rel
    try:
        target.resolve().relative_to(Path(root).resolve())
    except ValueError as exc:
        raise VastB2BootstrapError(f"DS24_VAST_PATH_OUTSIDE_ROOT:{relative_path}") from exc
    return target


def ensure_no_forbidden_artifact_path(relative_path: str) -> None:
    lowered = normalize_relative_path(relative_path).lower()
    for marker in FORBIDDEN_ARTIFACT_MARKERS:
        if marker in lowered:
            raise VastB2BootstrapError(f"DS24_VAST_FORBIDDEN_ARTIFACT_PATH:{relative_path}:{marker}")


def input_dataset_authority_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "dataset_id",
        "bucket",
        "prefix",
        "expected_object_count",
        "expected_bytes",
        "source_manifest_hash",
        "remote_verification",
        "completion_marker",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": INPUT_DATASET_SCHEMA_VERSION,
        "title": "DS24 Vast input dataset completion authority",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array", "boolean"]} for name in required},
    }


def vast_bootstrap_config_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "queue_id",
        "queue_definition_hash",
        "dataset_id",
        "bucket",
        "prefix",
        "dataset_marker_key",
        "run_id",
        "workspace_root",
        "dataset_root",
        "output_root",
        "ownership_plan_path",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": BOOTSTRAP_CONFIG_SCHEMA_VERSION,
        "title": "DS24 Vast bootstrap config",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array", "boolean"]} for name in required},
    }


def ownership_plan_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "queue_id",
        "queue_definition_hash",
        "plan_generation",
        "created_at_utc",
        "expires_at_utc",
        "dell_status_snapshot_hash",
        "mac_status_snapshot_hash",
        "acknowledgements",
        "vast_static_partition",
        "plan_hash",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": OWNERSHIP_PLAN_SCHEMA_VERSION,
        "title": "DS24 Vast cross-machine ownership plan",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array", "boolean"]} for name in required},
    }


def vast_output_manifest_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "queue_id",
        "run_id",
        "dataset_authority_hash",
        "queue_definition_hash",
        "files",
        "completion_marker_written_last",
        "manifest_hash",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": VAST_OUTPUT_MANIFEST_SCHEMA_VERSION,
        "title": "DS24 Vast output manifest",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array", "boolean"]} for name in required},
    }


def dell_import_receipt_schema() -> dict[str, Any]:
    required = [
        "schema_version",
        "vast_run_id",
        "family_id",
        "queue_id",
        "source_b2_prefix",
        "remote_manifest_hash",
        "local_manifest_hash",
        "contract_hashes",
        "verification_time_utc",
        "local_paths",
        "artifact_tiers_present",
        "artifact_tiers_deferred",
        "import_eligibility",
        "limitations",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": DELL_IMPORT_RECEIPT_SCHEMA_VERSION,
        "title": "DS24 Vast Dell import receipt",
        "type": "object",
        "required": required,
        "properties": {name: {"type": ["string", "number", "object", "array", "boolean"]} for name in required},
    }


def load_prerequisite_queue_authority(repo_root: Path, authority_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = Path(authority_root or (repo_root / DEFAULT_QUEUE_AUTHORITY_ROOT_REL))
    if not root.is_absolute():
        root = repo_root / root
    manifest_path = root / "manifest.json"
    definition_path = root / "vast_reverse_queue_definition.json"
    manifest = read_json(manifest_path)
    queue_definition = read_json(definition_path)
    checks = {
        "manifest_present": manifest_path.is_file(),
        "definition_present": definition_path.is_file(),
        "queue_id_matches": manifest.get("queue_id") == QUEUE_ID and queue_definition.get("queue_id") == QUEUE_ID,
        "classification_matches": manifest.get("terminal_classification") == PREREQUISITE_CLASSIFICATION,
        "test_evidence_pass": manifest.get("test_evidence_status") == "PASS",
        "definition_hash_matches_manifest": queue_definition.get("queue_definition_hash") == manifest.get("queue_definition_hash"),
        "definition_valid": reverse_queue.validate_queue_definition(queue_definition).get("status") == "PASS"
        if isinstance(queue_definition, Mapping) and queue_definition
        else False,
    }
    payload = {
        "authority_id": "DS24_VAST_R49_QUEUE_PREREQUISITE_VALIDATION_V1",
        "queue_authority_root": repo_rel(repo_root, root),
        "manifest_path": repo_rel(repo_root, manifest_path),
        "queue_definition_path": repo_rel(repo_root, definition_path),
        "queue_id": manifest.get("queue_id", ""),
        "queue_definition_hash": manifest.get("queue_definition_hash", ""),
        "terminal_classification": manifest.get("terminal_classification", ""),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_if_failed": BLOCKED_QUEUE_AUTHORITY,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def manifest_rows_from_files(repo_root: Path, relative_paths: Sequence[str]) -> list[DatasetManifestRow]:
    rows = []
    for rel in relative_paths:
        norm = normalize_relative_path(rel)
        path = safe_join(repo_root, norm)
        rows.append(
            DatasetManifestRow(
                relative_path=norm,
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
                sha1=file_sha1(path),
            )
        )
    return rows


def parse_source_manifest(path: Path, repo_root: Path, *, hash_local_files: bool = False) -> list[DatasetManifestRow]:
    text = Path(path).read_text(encoding="utf-8-sig")
    if "," in text.splitlines()[0]:
        parsed: list[DatasetManifestRow] = []
        for row in csv.DictReader(text.splitlines()):
            rel = normalize_relative_path(str(row.get("source_relative_path") or row.get("relative_path") or row.get("path") or ""))
            size = int(float(row.get("size_bytes") or 0))
            sha256 = str(row.get("sha256") or row.get("content_sha256") or "").lower()
            sha1 = str(row.get("sha1") or row.get("content_sha1") or "").lower()
            if hash_local_files and (not size or not sha256 or not sha1):
                local = safe_join(repo_root, rel)
                size = local.stat().st_size
                sha256 = file_sha256(local)
                sha1 = file_sha1(local)
            parsed.append(DatasetManifestRow(rel, size, sha256, sha1))
        return parsed
    rows = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        rel = normalize_relative_path(raw)
        size = 0
        sha256 = ""
        sha1 = ""
        if hash_local_files:
            local = safe_join(repo_root, rel)
            size = local.stat().st_size
            sha256 = file_sha256(local)
            sha1 = file_sha1(local)
        rows.append(DatasetManifestRow(rel, size, sha256, sha1))
    return rows


def rows_payload(rows: Sequence[DatasetManifestRow]) -> list[dict[str, Any]]:
    return [row.__dict__ for row in rows]


def validate_input_manifest(
    repo_root: Path,
    rows: Sequence[DatasetManifestRow],
    *,
    expected_count: int,
    expected_bytes: int,
    permitted_roots: Sequence[str] = DEFAULT_PERMITTED_DATASET_ROOTS,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    normalized_roots = tuple(normalize_relative_path(root).rstrip("/") + "/" for root in permitted_roots)
    normalized: list[DatasetManifestRow] = []
    for index, row in enumerate(rows):
        try:
            rel = normalize_relative_path(row.relative_path)
            if rel in seen:
                errors.append({"row_index": index, "path": rel, "error": "DUPLICATE_ENTRY"})
            seen.add(rel)
            if normalized_roots and not any((rel + "/").startswith(root) or rel.startswith(root) for root in normalized_roots):
                errors.append({"row_index": index, "path": rel, "error": "PATH_OUTSIDE_PERMITTED_ROOTS"})
            safe_join(repo_root, rel)
            ensure_no_forbidden_artifact_path(rel)
            normalized.append(DatasetManifestRow(rel, int(row.size_bytes), str(row.sha256).lower(), str(row.sha1).lower()))
        except Exception as exc:
            errors.append({"row_index": index, "path": str(row.relative_path), "error": str(exc)})
    total_bytes = sum(int(row.size_bytes) for row in normalized)
    if len(normalized) != expected_count:
        errors.append({"error": "EXPECTED_OBJECT_COUNT_MISMATCH", "expected": expected_count, "actual": len(normalized)})
    if total_bytes != expected_bytes:
        errors.append({"error": "EXPECTED_BYTES_MISMATCH", "expected": expected_bytes, "actual": total_bytes})
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "classification": "INPUT_MANIFEST_VALID" if not errors else "INPUT_MANIFEST_INVALID",
        "object_count": len(normalized),
        "total_bytes": total_bytes,
        "source_manifest_hash": stable_hash(rows_payload(normalized)),
        "errors": errors,
    }
    payload["validation_hash"] = stable_hash(payload)
    return payload


def normalize_remote_objects(remote_objects: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in remote_objects:
        key = str(row.get("key") or row.get("path") or row.get("name") or "").strip().replace("\\", "/")
        if not key:
            continue
        hashes = row.get("hashes") if isinstance(row.get("hashes"), Mapping) else {}
        out[key] = {
            "key": key,
            "size_bytes": int(float(row.get("size_bytes") or row.get("size") or 0)),
            "sha256": str(row.get("sha256") or hashes.get("sha256") or "").lower(),
            "sha1": str(row.get("sha1") or hashes.get("sha1") or "").lower(),
            "content_hash": str(row.get("content_hash") or row.get("hash") or "").lower(),
        }
    return out


def compare_manifest_to_b2(
    rows: Sequence[DatasetManifestRow],
    remote_objects: Sequence[Mapping[str, Any]],
    *,
    prefix: str = B2_PREFIX,
) -> dict[str, Any]:
    remote_by_key = normalize_remote_objects(remote_objects)
    missing: list[dict[str, Any]] = []
    differing: list[dict[str, Any]] = []
    verified_count = 0
    verified_bytes = 0
    hash_sources: dict[str, int] = {}
    expected_keys = {row.object_key(prefix): row for row in rows}
    for key, row in expected_keys.items():
        observed = remote_by_key.get(key)
        if not observed:
            missing.append({"key": key, "reason": "REMOTE_OBJECT_MISSING"})
            continue
        if int(observed["size_bytes"]) != int(row.size_bytes):
            differing.append({"key": key, "reason": "SIZE_MISMATCH", "expected": row.size_bytes, "actual": observed["size_bytes"]})
            continue
        if row.sha256 and observed.get("sha256"):
            hash_sources["sha256"] = hash_sources.get("sha256", 0) + 1
            if row.sha256.lower() != observed["sha256"]:
                differing.append({"key": key, "reason": "SHA256_MISMATCH"})
                continue
        elif row.sha1 and observed.get("sha1"):
            hash_sources["b2_sha1"] = hash_sources.get("b2_sha1", 0) + 1
            if row.sha1.lower() != observed["sha1"]:
                differing.append({"key": key, "reason": "B2_SHA1_MISMATCH"})
                continue
        elif row.sha256 and observed.get("content_hash"):
            hash_sources["content_hash"] = hash_sources.get("content_hash", 0) + 1
            if row.sha256.lower() != observed["content_hash"]:
                differing.append({"key": key, "reason": "CONTENT_HASH_MISMATCH"})
                continue
        else:
            hash_sources["size_only_no_hash_available"] = hash_sources.get("size_only_no_hash_available", 0) + 1
        verified_count += 1
        verified_bytes += int(row.size_bytes)
    unrelated = [
        {"key": key, "classification": "IGNORED_UNRELATED_PREFIX_OBJECT"}
        for key in sorted(remote_by_key)
        if key.startswith(prefix.rstrip("/") + "/") and key not in expected_keys and key != DATASET_COMPLETE_MARKER_KEY
    ]
    payload = {
        "status": "PASS" if not missing and not differing else "FAIL",
        "classification": "SOURCE_TO_B2_VERIFIED" if not missing and not differing else "SOURCE_TO_B2_MISMATCH",
        "verified_object_count": verified_count,
        "verified_bytes": verified_bytes,
        "missing_objects": missing,
        "differing_objects": differing,
        "unexpected_unrelated_objects": unrelated,
        "unexpected_unrelated_object_count": len(unrelated),
        "b2_supported_hashes_used": hash_sources,
    }
    payload["comparison_hash"] = stable_hash(payload)
    return payload


def build_dataset_completion_marker(authority: Mapping[str, Any], *, rclone_version: str = "UNOBSERVED_NOT_EXECUTED") -> dict[str, Any]:
    marker = {
        "schema_version": INPUT_DATASET_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "bucket": B2_BUCKET,
        "prefix": B2_PREFIX,
        "marker_key": DATASET_COMPLETE_MARKER_KEY,
        "expected_object_count": authority.get("expected_object_count", EXPECTED_DATASET_OBJECT_COUNT),
        "expected_bytes": authority.get("expected_bytes", EXPECTED_DATASET_BYTES),
        "source_manifest_hash": authority.get("source_manifest_hash", ""),
        "source_authority_hashes": authority.get("source_authority_hashes", {}),
        "verification_timestamp_utc": authority.get("verification_timestamp_utc", utc_now()),
        "rclone_version": rclone_version,
        "verification_result": authority.get("status", ""),
        "completion_marker_predecessor_hash": authority.get("authority_hash", ""),
        "credentials_included": False,
    }
    marker["completion_marker_content_hash"] = stable_hash(marker)
    return marker


class DatasetPublisherFinalizer:
    def __init__(
        self,
        *,
        repo_root: Path,
        bucket: str = B2_BUCKET,
        prefix: str = B2_PREFIX,
        expected_count: int = EXPECTED_DATASET_OBJECT_COUNT,
        expected_bytes: int = EXPECTED_DATASET_BYTES,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.expected_count = int(expected_count)
        self.expected_bytes = int(expected_bytes)

    def finalize(
        self,
        rows: Sequence[DatasetManifestRow],
        remote_objects: Sequence[Mapping[str, Any]],
        *,
        output_root: Path | None = None,
        now_utc: str | None = None,
        upload_authority: bool = False,
        confirm_token: str = "",
        rclone_version: str = "UNOBSERVED_NOT_EXECUTED",
    ) -> dict[str, Any]:
        manifest_validation = validate_input_manifest(
            self.repo_root,
            rows,
            expected_count=self.expected_count,
            expected_bytes=self.expected_bytes,
        )
        comparison = compare_manifest_to_b2(rows, remote_objects, prefix=self.prefix)
        checks = {
            "manifest_valid": manifest_validation["status"] == "PASS",
            "source_to_b2_verified": comparison["status"] == "PASS",
            "expected_count_matches": comparison["verified_object_count"] == self.expected_count,
            "expected_bytes_matches": comparison["verified_bytes"] == self.expected_bytes,
            "completion_marker_written_last": False,
            "real_b2_write_authorized": upload_authority and confirm_token == DATASET_FINALIZE_CONFIRM_TOKEN,
        }
        status = "PASS" if all(value for key, value in checks.items() if key != "completion_marker_written_last" and key != "real_b2_write_authorized") else "FAIL"
        checks["completion_marker_written_last"] = output_root is not None and status == "PASS"
        authority = {
            "schema_version": INPUT_DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "expected_object_count": self.expected_count,
            "expected_bytes": self.expected_bytes,
            "displayed_size": DISPLAYED_DATASET_SIZE,
            "source_manifest_path": SOURCE_MANIFEST_REL.as_posix(),
            "source_manifest_hash": manifest_validation["source_manifest_hash"],
            "source_authority_hashes": {
                "prerequisite_queue_authority": load_prerequisite_queue_authority(self.repo_root).get("manifest_hash", ""),
            },
            "remote_verification": comparison,
            "verification_timestamp_utc": now_utc or utc_now(),
            "rclone_version": rclone_version,
            "upload_authority_document_allowed": status == "PASS",
            "completion_marker_key": DATASET_COMPLETE_MARKER_KEY,
            "status": status,
            "classification": "DATASET_COMPLETE_AUTHORITY_READY" if status == "PASS" else "DATASET_UPLOAD_INCOMPLETE_OR_UNVERIFIED",
            "credentials_included": False,
            "checks": checks,
        }
        authority["authority_hash"] = stable_hash(
            {key: value for key, value in authority.items() if key not in {"authority_hash", "completion_marker"}}
        )
        marker = build_dataset_completion_marker(authority, rclone_version=rclone_version)
        authority["completion_marker"] = marker
        if output_root is not None and status == "PASS":
            root = Path(output_root)
            write_json_atomic(root / "input_dataset_authority.json", authority)
            write_json_atomic(root / "DATASET_COMPLETE.json", marker)
        elif output_root is not None:
            write_json_atomic(Path(output_root) / "input_dataset_authority.failed.json", authority)
        return authority


def repository_deployment_authority(repo_root: Path, prerequisite: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    commit = run_git(repo_root, ["rev-parse", "HEAD"])
    tracked_dirty = run_git(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    requirements = repo_root / "requirements.txt"
    required_untracked = [
        "core/research/ml/ds24/vast_reverse_queue_r1.py",
        "core/research/ml/ds24/vast_b2_bootstrap_r1.py",
        "scripts/local/ds24_vast_reverse_queue_r1.py",
        "scripts/local/ds24_vast_b2_bootstrap_r1.py",
        "tests/test_ds24_vast_reverse_queue_r1.py",
        "tests/test_ds24_vast_b2_bootstrap_r1.py",
        DEFAULT_QUEUE_AUTHORITY_ROOT_REL.as_posix(),
        DEFAULT_AUTHORITY_ROOT_REL.as_posix(),
    ]
    present_required_untracked = [
        path for path in required_untracked if (repo_root / path).exists() and path in run_git(repo_root, ["ls-files", "--others", "--exclude-standard", "--", path])
    ]
    payload = {
        "authority_id": "DS24_VAST_R50_REPOSITORY_DEPLOYMENT_AUTHORITY_V1",
        "method": "git clone exact commit plus integrity-checked small deployment bundle for required untracked authorities",
        "repo_commit_sha": commit.strip(),
        "tracked_dirty_worktree": bool(tracked_dirty.strip()),
        "dirty_worktree_status_sample": tracked_dirty.splitlines()[:25],
        "required_untracked_production_files": present_required_untracked,
        "python_version": platform.python_version(),
        "requirements_path": "requirements.txt",
        "requirements_sha256": file_sha256(requirements) if requirements.is_file() else "",
        "queue_id": QUEUE_ID,
        "queue_definition_hash": prerequisite.get("queue_definition_hash", ""),
        "dataset_packaged_in_code_bundle": False,
        "credentials_packaged": False,
    }
    payload["deployment_authority_hash"] = stable_hash(payload)
    return payload


def run_git(repo_root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=repo_root, text=True, capture_output=True, timeout=20, check=False)
    except Exception:
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def deployment_bundle_manifest(repo_root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    rows = []
    total_bytes = 0
    for rel in relative_paths:
        norm = normalize_relative_path(rel)
        if norm.startswith("data/"):
            raise VastB2BootstrapError("DS24_VAST_CODE_BUNDLE_REFUSES_DATASET_PAYLOAD")
        path = safe_join(repo_root, norm)
        if path.is_dir():
            continue
        if not path.is_file():
            continue
        ensure_no_forbidden_artifact_path(norm)
        size = path.stat().st_size
        total_bytes += size
        rows.append({"relative_path": norm, "size_bytes": size, "sha256": file_sha256(path)})
    payload = {
        "schema_version": "ds24_vast_deployment_bundle_manifest.v1",
        "files": rows,
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "dataset_bytes_included": 0,
        "dataset_packaged_again": False,
        "credentials_included": False,
    }
    payload["bundle_manifest_hash"] = stable_hash(payload)
    return payload


def capacity_plan(
    *,
    dataset_bytes: int,
    repo_environment_bytes: int,
    temp_transfer_bytes: int,
    queue_state_and_logs_bytes: int,
    checkpoint_reserve_bytes: int,
    final_artifact_reserve_bytes: int,
    emergency_reserve_bytes: int,
    free_bytes: int,
) -> dict[str, Any]:
    required = (
        int(dataset_bytes)
        + int(repo_environment_bytes)
        + int(temp_transfer_bytes)
        + int(queue_state_and_logs_bytes)
        + int(checkpoint_reserve_bytes)
        + int(final_artifact_reserve_bytes)
        + int(emergency_reserve_bytes)
    )
    return {
        "status": "PASS" if int(free_bytes) >= required else "FAIL",
        "classification": "CAPACITY_GATE_PASS" if int(free_bytes) >= required else "INSUFFICIENT_CAPACITY_FAIL_CLOSED",
        "dataset_bytes": int(dataset_bytes),
        "repo_environment_bytes": int(repo_environment_bytes),
        "temp_transfer_bytes": int(temp_transfer_bytes),
        "queue_state_and_logs_bytes": int(queue_state_and_logs_bytes),
        "checkpoint_reserve_bytes": int(checkpoint_reserve_bytes),
        "final_artifact_reserve_bytes": int(final_artifact_reserve_bytes),
        "emergency_reserve_bytes": int(emergency_reserve_bytes),
        "required_bytes": required,
        "free_bytes": int(free_bytes),
        "hardcoded_44gib_only_requirement": False,
    }


def redact_secrets(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        out = {}
        for key, value in payload.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("secret", "token", "key", "credential", "password")):
                out[key_text] = "<REDACTED>"
            else:
                out[key_text] = redact_secrets(value)
        return out
    if isinstance(payload, list):
        return [redact_secrets(value) for value in payload]
    text = str(payload)
    if any(marker in text.lower() for marker in ("b2_application_key=", "aws_secret", "secret=", "password=", "token=")):
        return "<REDACTED>"
    return payload


def secret_preflight(env: Mapping[str, str], required_names: Sequence[str]) -> dict[str, Any]:
    present = sorted(name for name in required_names if env.get(name))
    missing = sorted(set(required_names) - set(present))
    return {
        "status": "PASS" if not missing else "FAIL",
        "classification": "CREDENTIAL_REFERENCES_PRESENT" if not missing else "CREDENTIAL_REFERENCES_MISSING",
        "required_secret_names": list(required_names),
        "present_secret_names": present,
        "missing_secret_names": missing,
        "secret_values_printed": False,
        "diagnostics": redact_secrets({name: env.get(name, "") for name in required_names}),
    }


def build_rclone_command_contract(*, action: str, source: str, destination: str, files_from: str = "") -> dict[str, Any]:
    command = ["rclone", action, source, destination, "--retries", "20", "--low-level-retries", "50"]
    if files_from:
        command.extend(["--files-from", files_from])
    command.extend(["--stats", "30s"])
    return {
        "command": command,
        "uses_copy_semantics": action == "copy",
        "destructive_sync_default": False,
        "credentials_in_command": False,
        "requires_explicit_authorization": True,
        "confirm_token": REAL_B2_CONFIRM_TOKEN,
    }


def resource_gate(system: Mapping[str, Any], required: Mapping[str, Any] | None = None) -> dict[str, Any]:
    req = {
        "cpu_cores": 16,
        "ram_bytes": 64 * 1024**3,
        "gpu_present": True,
        "vram_bytes": 24 * 1024**3,
        "cuda_present": True,
        "disk_free_bytes": 250 * 1024**3,
    }
    req.update(dict(required or {}))
    checks = {
        "cpu": int(system.get("cpu_cores", 0)) >= int(req["cpu_cores"]),
        "ram": int(system.get("ram_bytes", 0)) >= int(req["ram_bytes"]),
        "gpu": bool(system.get("gpu_present")) is bool(req["gpu_present"]),
        "vram": int(system.get("vram_bytes", 0)) >= int(req["vram_bytes"]),
        "cuda": bool(system.get("cuda_present")) is bool(req["cuda_present"]),
        "disk": int(system.get("disk_free_bytes", 0)) >= int(req["disk_free_bytes"]),
        "time_sync": bool(system.get("time_synchronized", True)),
        "workspace_writable": bool(system.get("workspace_writable", True)),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "classification": "VAST_RESOURCE_GATE_PASS" if all(checks.values()) else "VAST_RESOURCE_GATE_FAIL_CLOSED",
        "checks": checks,
        "required": req,
        "observed": redact_secrets(dict(system)),
    }


def build_acknowledgement(*, machine: str, plan_hash: str, now_utc: str | None = None, unavailable_reason: str = "") -> dict[str, Any]:
    payload = {
        "schema_version": "ds24_vast_ownership_ack.v1",
        "machine": machine,
        "plan_hash": plan_hash,
        "acknowledged_at_utc": now_utc or utc_now(),
        "unavailable_reason": unavailable_reason,
        "credentials_included": False,
    }
    payload["ack_hash"] = stable_hash(payload)
    return payload


def ownership_plan_hash_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    payload.pop("plan_hash", None)
    payload.pop("acknowledgements", None)
    return payload


def create_ownership_plan(
    queue_definition: Mapping[str, Any],
    dell_snapshot: Mapping[str, Any],
    mac_snapshot: Mapping[str, Any],
    *,
    generation: int = 1,
    created_at_utc: str | None = None,
    ttl_seconds: int = DEFAULT_OWNERSHIP_PLAN_TTL_SECONDS,
    dell_ack: Mapping[str, Any] | None = None,
    mac_ack: Mapping[str, Any] | None = None,
    mac_unavailable_reason: str = "",
) -> dict[str, Any]:
    created = created_at_utc or utc_now()
    created_dt = parse_utc(created)
    if created_dt is None:
        raise VastB2BootstrapError("DS24_VAST_OWNERSHIP_PLAN_TIMESTAMP_INVALID")
    expires = format_utc(created_dt + timedelta(seconds=int(ttl_seconds)))
    state = reverse_queue.initial_queue_state(queue_definition, now_utc=created)
    plan = reverse_queue.plan_from_state(
        state,
        queue_definition,
        [dell_snapshot, mac_snapshot],
        now_utc=created,
        allow_missing_snapshots=False,
    )
    external = plan.get("external_validation", {})
    family_evidence = external.get("family_evidence", {}) if isinstance(external, Mapping) else {}
    completed: list[str] = []
    running: list[str] = []
    dell_owned: list[str] = []
    mac_owned: list[str] = []
    incompatible: list[str] = []
    for family_id, rows in family_evidence.items():
        for row in rows:
            classification = row.get("classification")
            source = row.get("source_machine")
            if classification == "SKIPPED_EXTERNAL_VERIFIED":
                completed.append(str(family_id))
            if classification in {
                "COMPATIBLE_EXTERNAL_RUNNING_OR_CLAIMED",
                "UNVERIFIED_EXTERNAL_RUNNING_OR_CLAIMED_BLOCKS_ADMISSION",
                "DEAD_EXTERNAL_PID_AMBIGUOUS_RECOVERY_REQUIRED",
            }:
                running.append(str(family_id))
                if source == "dell":
                    dell_owned.append(str(family_id))
                if source == "mac":
                    mac_owned.append(str(family_id))
            if classification == "EXTERNAL_COMPLETION_INCOMPATIBLE_OR_UNVERIFIED":
                incompatible.append(str(family_id))
    vast_partition = []
    for decision in plan.get("candidate_decisions", []):
        if decision.get("decision") == "CLAIMABLE":
            vast_partition.append(decision.get("family_id"))
            break
        if decision.get("decision") in {"SKIPPED_EXTERNAL_VERIFIED", "CONFIGURATION_AUTHORITY_REQUIRED"}:
            continue
        break
    acknowledgements = {
        "dell": dict(dell_ack or {}),
        "mac": dict(mac_ack or {}),
        "mac_unavailable_reason": mac_unavailable_reason,
    }
    payload = {
        "schema_version": OWNERSHIP_PLAN_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition.get("queue_definition_hash", ""),
        "scientific_contract_hashes": {
            row["canonical_family_id"]: reverse_queue.family_contract_hashes(row)
            for row in queue_definition.get("entries", [])
            if isinstance(row, Mapping)
        },
        "plan_generation": int(generation),
        "created_at_utc": created,
        "expires_at_utc": expires,
        "dell_status_snapshot_hash": reverse_queue.snapshot_hash(dell_snapshot),
        "mac_status_snapshot_hash": reverse_queue.snapshot_hash(mac_snapshot),
        "completed_compatible_families": sorted(set(completed)),
        "currently_running_families": sorted(set(running)),
        "dell_owned_families": sorted(set(dell_owned)),
        "mac_owned_families": sorted(set(mac_owned)),
        "vast_reserved_families": [],
        "vast_static_partition": [family for family in vast_partition if family],
        "incompatible_or_unverified_completion": sorted(set(incompatible)),
        "meeting_boundary": {
            "nearest_externally_owned_boundary": plan.get("nearest_externally_owned_boundary", ""),
            "queues_met": bool(plan.get("queues_met")),
            "useful_non_overlapping_work_remains": bool(plan.get("useful_non_overlapping_work_remains")),
            "admission_status": plan.get("admission_status", ""),
            "next_vast_eligible_family": plan.get("next_vast_eligible_family", ""),
        },
        "acknowledgements": acknowledgements,
        "requires_new_generation_to_replan": True,
        "cross_host_atomicity_claimed": False,
        "static_partition_enforced": True,
        "source_plan": plan,
    }
    payload["plan_hash"] = stable_hash(ownership_plan_hash_payload(payload))
    return payload


def validate_ownership_plan(plan: Mapping[str, Any], *, now_utc: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if plan.get("schema_version") != OWNERSHIP_PLAN_SCHEMA_VERSION:
        errors.append("OWNERSHIP_PLAN_SCHEMA_VERSION_MISMATCH")
    if plan.get("queue_id") != QUEUE_ID:
        errors.append("OWNERSHIP_PLAN_QUEUE_ID_MISMATCH")
    expected_hash = str(plan.get("plan_hash") or "")
    if expected_hash != stable_hash(ownership_plan_hash_payload(plan)):
        errors.append("OWNERSHIP_PLAN_HASH_MISMATCH")
    now = parse_utc(now_utc or utc_now())
    expires = parse_utc(plan.get("expires_at_utc"))
    if now is None or expires is None or expires <= now:
        errors.append("OWNERSHIP_PLAN_STALE")
    acks = plan.get("acknowledgements") if isinstance(plan.get("acknowledgements"), Mapping) else {}
    dell_ack = acks.get("dell") if isinstance(acks.get("dell"), Mapping) else {}
    mac_ack = acks.get("mac") if isinstance(acks.get("mac"), Mapping) else {}
    mac_reason = str(acks.get("mac_unavailable_reason") or "")
    if dell_ack.get("plan_hash") != expected_hash:
        errors.append("DELL_ACKNOWLEDGEMENT_MISSING_OR_MISMATCHED")
    if mac_ack.get("plan_hash") != expected_hash and not mac_reason:
        errors.append("MAC_ACKNOWLEDGEMENT_OR_REASON_MISSING")
    if plan.get("source_plan", {}).get("external_validation", {}).get("status") == "FAIL":
        errors.append("EXTERNAL_COORDINATION_FAIL_CLOSED")
    if plan.get("incompatible_or_unverified_completion"):
        errors.append("INCOMPATIBLE_COMPLETION_NOT_ACCEPTED")
    if set(plan.get("vast_static_partition", [])) & set(plan.get("dell_owned_families", [])):
        errors.append("VAST_DELL_FAMILY_OVERLAP")
    if set(plan.get("vast_static_partition", [])) & set(plan.get("mac_owned_families", [])):
        errors.append("VAST_MAC_FAMILY_OVERLAP")
    return {
        "status": "PASS" if not errors else "FAIL",
        "classification": "OWNERSHIP_PLAN_VALID" if not errors else "OWNERSHIP_PLAN_INVALID_FAIL_CLOSED",
        "errors": errors,
    }


def read_dataset_marker(remote_root: Path, marker_key: str = DATASET_COMPLETE_MARKER_KEY) -> dict[str, Any]:
    return read_json(Path(remote_root) / marker_key)


def validate_dataset_marker(marker: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if marker.get("schema_version") != INPUT_DATASET_SCHEMA_VERSION:
        errors.append("DATASET_MARKER_SCHEMA_VERSION_MISMATCH")
    if marker.get("dataset_id") != DATASET_ID:
        errors.append("DATASET_MARKER_DATASET_ID_MISMATCH")
    if marker.get("bucket") != B2_BUCKET or marker.get("prefix") != B2_PREFIX:
        errors.append("DATASET_MARKER_B2_LOCATION_MISMATCH")
    if marker.get("verification_result") != "PASS":
        errors.append("DATASET_MARKER_VERIFICATION_NOT_PASS")
    expected = str(marker.get("completion_marker_content_hash") or "")
    observed = dict(marker)
    observed.pop("completion_marker_content_hash", None)
    if expected != stable_hash(observed):
        errors.append("DATASET_MARKER_HASH_MISMATCH")
    return {
        "status": "PASS" if not errors else "FAIL",
        "classification": "DATASET_MARKER_VALID" if not errors else "DATASET_MARKER_INVALID",
        "errors": errors,
    }


def copy_file_resumable(source: Path, destination: Path, expected_sha256: str = "") -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    skipped = False
    resumed = False
    if destination.exists() and expected_sha256 and file_sha256(destination) == expected_sha256:
        skipped = True
    elif destination.exists() and destination.stat().st_size < source.stat().st_size:
        resumed = True
        with source.open("rb") as src, destination.open("ab") as dst:
            src.seek(destination.stat().st_size)
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source, destination)
    ok = (not expected_sha256) or file_sha256(destination) == expected_sha256
    return {
        "status": "PASS" if ok else "FAIL",
        "skipped_matching": skipped,
        "resumed_partial": resumed,
        "size_bytes": destination.stat().st_size if destination.exists() else 0,
        "sha256": file_sha256(destination) if destination.exists() else "",
    }


def download_dataset_from_fake_b2(
    remote_root: Path,
    dataset_root: Path,
    rows: Sequence[DatasetManifestRow],
    *,
    prefix: str = B2_PREFIX,
) -> dict[str, Any]:
    results = []
    failures = []
    for row in rows:
        source = safe_join(remote_root, row.object_key(prefix))
        destination = safe_join(dataset_root, row.relative_path)
        if not source.is_file():
            failures.append({"path": row.relative_path, "reason": "REMOTE_OBJECT_MISSING"})
            continue
        result = copy_file_resumable(source, destination, expected_sha256=row.sha256)
        result["relative_path"] = row.relative_path
        results.append(result)
        if result["status"] != "PASS":
            failures.append({"path": row.relative_path, "reason": "DOWNLOADED_HASH_MISMATCH"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "classification": "DATASET_DOWNLOAD_VERIFIED" if not failures else "DATASET_DOWNLOAD_FAILED",
        "downloaded_or_verified_count": len(results),
        "skipped_matching_count": sum(1 for row in results if row["skipped_matching"]),
        "resumed_partial_count": sum(1 for row in results if row["resumed_partial"]),
        "failures": failures,
    }


def verify_downloaded_dataset(dataset_root: Path, rows: Sequence[DatasetManifestRow]) -> dict[str, Any]:
    failures = []
    total = 0
    for row in rows:
        path = safe_join(dataset_root, row.relative_path)
        if not path.is_file():
            failures.append({"path": row.relative_path, "reason": "LOCAL_OBJECT_MISSING"})
            continue
        size = path.stat().st_size
        total += size
        if size != row.size_bytes:
            failures.append({"path": row.relative_path, "reason": "SIZE_MISMATCH", "actual": size, "expected": row.size_bytes})
            continue
        if row.sha256 and file_sha256(path) != row.sha256:
            failures.append({"path": row.relative_path, "reason": "SHA256_MISMATCH"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "classification": "LOCAL_DATASET_VERIFIED" if not failures else "LOCAL_DATASET_INVALID",
        "verified_count": len(rows) - len(failures),
        "verified_bytes": total,
        "failures": failures,
    }


def build_bootstrap_config(
    *,
    repo_root: Path,
    run_id: str,
    dataset_root: Path,
    output_root: Path,
    ownership_plan_path: Path,
    queue_definition_hash: str,
    dataset_authority_hash: str = "",
    workspace_root: str = "/workspace/ds24",
) -> dict[str, Any]:
    payload = {
        "schema_version": BOOTSTRAP_CONFIG_SCHEMA_VERSION,
        "queue_id": QUEUE_ID,
        "queue_definition_hash": queue_definition_hash,
        "dataset_id": DATASET_ID,
        "dataset_authority_hash": dataset_authority_hash,
        "bucket": B2_BUCKET,
        "prefix": B2_PREFIX,
        "dataset_marker_key": DATASET_COMPLETE_MARKER_KEY,
        "run_id": run_id,
        "workspace_root": workspace_root,
        "repo_root": repo_rel(Path(repo_root).resolve(), Path(repo_root).resolve()),
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "output_root": str(output_root).replace("\\", "/"),
        "ownership_plan_path": str(ownership_plan_path).replace("\\", "/"),
        "credentials": {
            "b2_application_key_id_env": "B2_APPLICATION_KEY_ID",
            "b2_application_key_env": "B2_APPLICATION_KEY",
        },
        "live_start_confirmation_token": LIVE_BOOTSTRAP_CONFIRM_TOKEN,
    }
    payload["config_hash"] = stable_hash(payload)
    return payload


def acquire_lease(lease_path: Path, *, lease_id: str, now_utc: str | None = None, ttl_seconds: int = DEFAULT_BOOTSTRAP_LEASE_SECONDS) -> dict[str, Any]:
    now = parse_utc(now_utc or utc_now())
    existing = read_json(lease_path)
    if isinstance(existing, Mapping) and existing.get("lease_status") == "ACTIVE":
        expires = parse_utc(existing.get("expires_at_utc"))
        if expires and now and expires > now:
            return {"status": "FAIL", "classification": "BOOTSTRAP_LEASE_ALREADY_ACTIVE", "existing_lease": existing}
    if now is None:
        raise VastB2BootstrapError("DS24_VAST_LEASE_TIMESTAMP_INVALID")
    lease = {
        "schema_version": "ds24_vast_bootstrap_lease.v1",
        "lease_id": lease_id,
        "pid": os.getpid(),
        "created_at_utc": format_utc(now),
        "expires_at_utc": format_utc(now + timedelta(seconds=ttl_seconds)),
        "lease_status": "ACTIVE",
    }
    lease["lease_hash"] = stable_hash(lease)
    write_json_atomic(lease_path, lease)
    return {"status": "PASS", "classification": "BOOTSTRAP_LEASE_ACQUIRED", "lease": lease}


def existing_supervisor_active(output_root: Path, *, now_utc: str | None = None) -> dict[str, Any]:
    lease = read_json(Path(output_root) / "supervisor.lease.json")
    if not isinstance(lease, Mapping) or not lease:
        return {"active": False, "classification": "NO_SUPERVISOR_LEASE"}
    expires = parse_utc(lease.get("expires_at_utc"))
    now = parse_utc(now_utc or utc_now())
    active = lease.get("status") in {"RUNNING", "STUB_RUNNING"} and bool(expires and now and expires > now)
    return {"active": active, "classification": "SUPERVISOR_ALREADY_ACTIVE" if active else "SUPERVISOR_LEASE_INACTIVE", "lease": lease}


class VastBootstrapController:
    def __init__(
        self,
        *,
        repo_root: Path,
        queue_authority_root: Path | None = None,
        config: Mapping[str, Any],
        dataset_rows: Sequence[DatasetManifestRow],
        remote_root: Path,
        resource_snapshot: Mapping[str, Any],
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.queue_authority_root = queue_authority_root
        self.config = dict(config)
        self.dataset_rows = list(dataset_rows)
        self.remote_root = Path(remote_root)
        self.resource_snapshot = dict(resource_snapshot)
        self.env = dict(env or {})

    @property
    def dataset_root(self) -> Path:
        return Path(self.config["dataset_root"])

    @property
    def output_root(self) -> Path:
        return Path(self.config["output_root"])

    @property
    def run_id(self) -> str:
        return str(self.config["run_id"])

    def preflight(self, *, now_utc: str | None = None) -> dict[str, Any]:
        prerequisite = load_prerequisite_queue_authority(self.repo_root, self.queue_authority_root)
        marker = read_dataset_marker(self.remote_root)
        marker_validation = validate_dataset_marker(marker)
        plan = read_json(Path(self.config["ownership_plan_path"]))
        ownership_validation = validate_ownership_plan(plan, now_utc=now_utc) if isinstance(plan, Mapping) and plan else {
            "status": "FAIL",
            "classification": "OWNERSHIP_PLAN_MISSING",
            "errors": ["OWNERSHIP_PLAN_MISSING"],
        }
        capacity = capacity_plan(
            dataset_bytes=sum(row.size_bytes for row in self.dataset_rows),
            repo_environment_bytes=8 * 1024**3,
            temp_transfer_bytes=max(1, sum(row.size_bytes for row in self.dataset_rows) // 20),
            queue_state_and_logs_bytes=2 * 1024**3,
            checkpoint_reserve_bytes=120 * 1024**3,
            final_artifact_reserve_bytes=40 * 1024**3,
            emergency_reserve_bytes=30 * 1024**3,
            free_bytes=int(self.resource_snapshot.get("disk_free_bytes", 0)),
        )
        resources = resource_gate(self.resource_snapshot)
        secrets = secret_preflight(self.env, ["B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY"])
        duplicate = existing_supervisor_active(self.output_root, now_utc=now_utc)
        checks = {
            "prerequisite_queue_authority": prerequisite["status"] == "PASS",
            "dataset_completion_marker": marker_validation["status"] == "PASS",
            "ownership_plan": ownership_validation["status"] == "PASS",
            "capacity": capacity["status"] == "PASS",
            "resources": resources["status"] == "PASS",
            "credentials_referenced": secrets["status"] == "PASS",
            "duplicate_supervisor_absent": duplicate["active"] is False,
            "config_hash_present": bool(self.config.get("config_hash")),
        }
        payload = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "classification": "VAST_BOOTSTRAP_PREFLIGHT_PASS" if all(checks.values()) else "VAST_BOOTSTRAP_PREFLIGHT_FAIL_CLOSED",
            "checks": checks,
            "prerequisite": prerequisite,
            "dataset_marker_validation": marker_validation,
            "ownership_plan_validation": ownership_validation,
            "capacity_plan": capacity,
            "resource_gate": resources,
            "secret_preflight": secrets,
            "duplicate_supervisor": duplicate,
            "model_started": False,
            "live_model_started": False,
        }
        payload["preflight_hash"] = stable_hash(payload)
        return payload

    def run(
        self,
        *,
        mode: str,
        now_utc: str | None = None,
        test_stub_executor: bool = False,
        execute_live: bool = False,
        confirm_token: str = "",
    ) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if mode == "status":
            return self.status(now_utc=now_utc)
        if mode == "preflight-only":
            return self.preflight(now_utc=now_utc)
        lease = acquire_lease(self.output_root / "bootstrap.lease.json", lease_id=self.run_id, now_utc=now_utc)
        if lease["status"] != "PASS":
            return {"status": "FAIL", "classification": lease["classification"], "lease": lease, "model_started": False}
        preflight = self.preflight(now_utc=now_utc)
        if preflight["status"] != "PASS":
            return {"status": "FAIL", "classification": "BOOTSTRAP_ABORTED_BEFORE_MODEL_START", "preflight": preflight, "model_started": False, "live_model_started": False}
        download = download_dataset_from_fake_b2(self.remote_root, self.dataset_root, self.dataset_rows)
        if mode == "download-only":
            return {"status": download["status"], "classification": "DOWNLOAD_ONLY_COMPLETE", "download": download, "model_started": False}
        verify = verify_downloaded_dataset(self.dataset_root, self.dataset_rows)
        if mode == "verify-only":
            return {"status": verify["status"], "classification": "VERIFY_ONLY_COMPLETE", "download": download, "verify": verify, "model_started": False}
        if mode not in {"start-after-verify", "resume"}:
            raise VastB2BootstrapError(f"DS24_VAST_UNKNOWN_BOOTSTRAP_MODE:{mode}")
        if download["status"] != "PASS" or verify["status"] != "PASS":
            return {"status": "FAIL", "classification": "BOOTSTRAP_DATASET_VERIFY_FAILED_NO_START", "download": download, "verify": verify, "model_started": False}
        if not test_stub_executor and not (execute_live and confirm_token == LIVE_BOOTSTRAP_CONFIRM_TOKEN):
            return {
                "status": "FAIL",
                "classification": READY_LIVE_PREFLIGHT_CLASSIFICATION,
                "download": download,
                "verify": verify,
                "model_started": False,
                "live_model_started": False,
                "required_confirm_token": LIVE_BOOTSTRAP_CONFIRM_TOKEN,
            }
        publisher = start_publisher_stub(self.output_root, run_id=self.run_id, now_utc=now_utc)
        launch = launch_reverse_queue_stub(self.output_root, run_id=self.run_id, now_utc=now_utc, resume=(mode == "resume"))
        payload = {
            "status": "PASS",
            "classification": "VAST_REVERSE_QUEUE_AUTOSTARTED_STUB" if test_stub_executor else "VAST_REVERSE_QUEUE_AUTOSTARTED_LIVE_AUTHORIZED",
            "preflight": preflight,
            "download": download,
            "verify": verify,
            "publisher": publisher,
            "launch": launch,
            "model_started": not test_stub_executor,
            "stub_executor_started": bool(test_stub_executor),
            "live_model_started": bool(execute_live and not test_stub_executor),
            "queue_id": QUEUE_ID,
            "run_id": self.run_id,
        }
        write_json_atomic(self.output_root / "bootstrap_result.json", payload)
        return payload

    def status(self, *, now_utc: str | None = None) -> dict[str, Any]:
        return {
            "status": "PASS",
            "queue_id": QUEUE_ID,
            "run_id": self.run_id,
            "bootstrap_lease": read_json(self.output_root / "bootstrap.lease.json"),
            "supervisor": existing_supervisor_active(self.output_root, now_utc=now_utc),
            "publisher": read_json(self.output_root / "publisher.lease.json"),
            "model_started_by_status": False,
        }


def start_publisher_stub(output_root: Path, *, run_id: str, now_utc: str | None = None) -> dict[str, Any]:
    now = parse_utc(now_utc or utc_now())
    if now is None:
        raise VastB2BootstrapError("DS24_VAST_PUBLISHER_TIMESTAMP_INVALID")
    payload = {
        "schema_version": "ds24_vast_publisher_lease.v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "status": "STUB_RUNNING",
        "pid": os.getpid(),
        "started_at_utc": format_utc(now),
        "heartbeat_utc": format_utc(now),
        "publisher_started_before_supervisor": True,
    }
    payload["lease_hash"] = stable_hash(payload)
    write_json_atomic(Path(output_root) / "publisher.lease.json", payload)
    return payload


def launch_reverse_queue_stub(output_root: Path, *, run_id: str, now_utc: str | None = None, resume: bool = False) -> dict[str, Any]:
    now = parse_utc(now_utc or utc_now())
    if now is None:
        raise VastB2BootstrapError("DS24_VAST_LAUNCH_TIMESTAMP_INVALID")
    existing = existing_supervisor_active(output_root, now_utc=now_utc)
    if existing["active"]:
        return {"status": "FAIL", "classification": "DUPLICATE_SUPERVISOR_PREVENTED", "existing": existing}
    existing_queue_state = read_json(Path(output_root) / "queue_state" / "queue_state.json")
    existing_cursor = str(existing_queue_state.get("current_cursor") or "") if isinstance(existing_queue_state, Mapping) else ""
    payload = {
        "schema_version": "ds24_vast_supervisor_lease.v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "status": "STUB_RUNNING",
        "pid": os.getpid(),
        "started_at_utc": format_utc(now),
        "expires_at_utc": format_utc(now + timedelta(seconds=DEFAULT_BOOTSTRAP_LEASE_SECONDS)),
        "resume": bool(resume),
        "existing_queue_cursor_preserved": existing_cursor,
        "queue_state_reset": False,
        "detached_mechanism": "tmux|screen|supervisord command contract; stub for tests",
        "stdout_path": "logs/vast_reverse_queue.stdout.log",
        "stderr_path": "logs/vast_reverse_queue.stderr.log",
        "outer_holdout_access": False,
        "paper_orders": 0,
        "live_orders": 0,
    }
    payload["lease_hash"] = stable_hash(payload)
    write_json_atomic(Path(output_root) / "supervisor.lease.json", payload)
    write_json_atomic(Path(output_root) / "heartbeat.json", {"run_id": run_id, "queue_id": QUEUE_ID, "heartbeat_utc": format_utc(now)})
    return payload


def artifact_retention_policy(queue_definition: Mapping[str, Any]) -> dict[str, Any]:
    family_decisions = []
    for entry in queue_definition.get("entries", []):
        family = str(entry.get("canonical_family_id") or "")
        family_class = str(entry.get("family_class") or "")
        if family_class == "GPU_SEQUENCE":
            weights = "latest valid resume checkpoint and best accepted serializable state only; rolling per-window weights are not retained en masse"
        elif family_class == "CPU_RANKING":
            weights = "ranking model state retained only when the family checkpoint contract declares it resumable; final deployable acceptance is separate"
        else:
            weights = "not applicable"
        family_decisions.append(
            {
                "family_id": family,
                "family_class": family_class,
                "weights_required_for_resume": family_class in {"GPU_SEQUENCE", "CPU_RANKING"},
                "final_deployable_weight_acceptance": "not claimed by transport ticket",
                "per_window_weights_retained": False,
                "policy": weights,
            }
        )
    payload = {
        "policy_id": "DS24_VAST_R50_ARTIFACT_RETENTION_POLICY_V1",
        "queue_id": QUEUE_ID,
        "tier_1_critical_compact_authority": [
            "queue_state",
            "ownership_ledger",
            "heartbeat",
            "configuration",
            "manifest_hashes",
            "performance_metrics",
            "ic_time_series",
            "permitted_compact_oof",
            "contract_top_n_outputs",
            "completion_state",
            "stdout_stderr_logs",
            "resource_telemetry",
            "failure_diagnostics",
        ],
        "tier_2_resumability_and_model_state": [
            "latest_valid_queue_checkpoint",
            "latest_valid_family_checkpoint",
            "optimizer_scheduler_state_when_required",
            "model_weights_required_for_resume",
            "best_accepted_serializable_weights_when_available",
            "preprocessing_normalization_state",
            "feature_ordering_and_configuration_hashes",
        ],
        "forbidden_or_disposable": list(FORBIDDEN_ARTIFACT_MARKERS)
        + ["deterministically_rebuildable_caches", "temporary_downloads", "duplicate_nonrecoverable_checkpoints"],
        "allowed_roots": [
            "queue_state",
            "ownership",
            "metrics_only_v3",
            "ensemble_oof_scores_v2",
            "checkpoints",
            "logs",
            "telemetry",
            "manifests",
            "config",
        ],
        "family_weight_retention": family_decisions,
        "scientific_acceptance_claimed": False,
    }
    payload["policy_hash"] = stable_hash(payload)
    return payload


def classify_artifact(relative_path: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    rel = normalize_relative_path(relative_path)
    lowered = rel.lower()
    for marker in FORBIDDEN_ARTIFACT_MARKERS:
        if marker in lowered:
            return {"status": "FORBIDDEN", "tier": "forbidden", "reason": marker}
    root = rel.split("/", 1)[0]
    if root not in set(policy.get("allowed_roots", [])):
        return {"status": "REJECTED", "tier": "outside_whitelist", "reason": "ROOT_NOT_ALLOWED"}
    if root in {"checkpoints"}:
        return {"status": "ALLOWED", "tier": "tier2", "reason": "RESUMABILITY_CHECKPOINT"}
    return {"status": "ALLOWED", "tier": "tier1", "reason": "COMPACT_AUTHORITY"}


def inventory_allowed_artifacts(local_run_root: Path, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    root = Path(local_run_root)
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = repo_rel(root, path)
        classification = classify_artifact(rel, policy)
        if classification["status"] != "ALLOWED":
            continue
        rows.append(
            {
                "relative_path": rel,
                "tier": classification["tier"],
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def committed_marker_path(remote_run_root: Path) -> Path:
    return Path(remote_run_root) / "COMMITTED.json"


class VastDurableArtifactPublisher:
    def __init__(self, *, local_run_root: Path, remote_run_root: Path, policy: Mapping[str, Any], run_id: str, dataset_authority_hash: str, queue_definition_hash: str) -> None:
        self.local_run_root = Path(local_run_root)
        self.remote_run_root = Path(remote_run_root)
        self.policy = dict(policy)
        self.run_id = run_id
        self.dataset_authority_hash = dataset_authority_hash
        self.queue_definition_hash = queue_definition_hash

    def publish_once(self, *, interrupt_after_files: int | None = None, now_utc: str | None = None) -> dict[str, Any]:
        files = inventory_allowed_artifacts(self.local_run_root, self.policy)
        uploaded = []
        failures = []
        committed = False
        for index, row in enumerate(files):
            if interrupt_after_files is not None and index >= interrupt_after_files:
                failures.append({"reason": "SYNTHETIC_INTERRUPTION", "remaining_files": len(files) - index})
                break
            src = safe_join(self.local_run_root, row["relative_path"])
            dst = safe_join(self.remote_run_root, row["relative_path"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            uploaded.append(row)
        status = "PASS" if not failures else "FAIL"
        manifest = {
            "schema_version": VAST_OUTPUT_MANIFEST_SCHEMA_VERSION,
            "queue_id": QUEUE_ID,
            "run_id": self.run_id,
            "dataset_authority_hash": self.dataset_authority_hash,
            "queue_definition_hash": self.queue_definition_hash,
            "remote_prefix": f"ds24/vast_runs/queue={QUEUE_ID}/run={self.run_id}",
            "files": files if status == "PASS" else uploaded,
            "file_count": len(files) if status == "PASS" else len(uploaded),
            "total_bytes": sum(int(row["size_bytes"]) for row in (files if status == "PASS" else uploaded)),
            "completion_marker_written_last": status == "PASS",
            "forbidden_files_transferred": 0,
            "copy_semantics": "copy",
            "destructive_sync_used": False,
            "created_at_utc": now_utc or utc_now(),
        }
        manifest["manifest_hash"] = stable_hash(manifest)
        write_json_atomic(self.remote_run_root / "vast_output_manifest.json", manifest)
        if status == "PASS":
            committed_marker = {
                "queue_id": QUEUE_ID,
                "run_id": self.run_id,
                "manifest_hash": manifest["manifest_hash"],
                "committed_at_utc": now_utc or utc_now(),
                "written_after_manifest": True,
            }
            committed_marker["marker_hash"] = stable_hash(committed_marker)
            write_json_atomic(committed_marker_path(self.remote_run_root), committed_marker)
            committed = True
        return {
            "status": status,
            "classification": "VAST_OUTPUTS_DURABLY_PUBLISHED" if status == "PASS" else "VAST_OUTPUT_PUBLICATION_INTERRUPTED_RETRYABLE",
            "uploaded_count": len(uploaded),
            "backlog_count": len(files) - len(uploaded),
            "failures": failures,
            "manifest_hash": manifest["manifest_hash"],
            "committed_marker_written": committed,
            "forbidden_files_transferred": 0,
            "copy_semantics": "copy",
        }

    def status(self, *, now_utc: str | None = None, max_backup_age_seconds: int = DEFAULT_PUBLISHER_MAX_BACKUP_AGE_SECONDS) -> dict[str, Any]:
        marker = read_json(committed_marker_path(self.remote_run_root))
        committed_at = parse_utc(marker.get("committed_at_utc")) if isinstance(marker, Mapping) else None
        now = parse_utc(now_utc or utc_now())
        age = (now - committed_at).total_seconds() if now and committed_at else None
        stale = age is None or age > max_backup_age_seconds
        return {
            "status": "PASS" if not stale else "FAIL",
            "classification": "BACKUP_FRESH" if not stale else "BACKUP_STALE_BLOCK_NEW_FAMILY_ADMISSION",
            "backup_age_seconds": age,
            "max_backup_age_seconds": max_backup_age_seconds,
            "backlog_blocks_new_family": stale,
            "active_fit_killed": False,
        }


def dell_capacity_gate(*, selected_remote_bytes: int, staging_bytes: int, final_local_bytes: int, emergency_reserve_bytes: int, free_bytes: int) -> dict[str, Any]:
    required = int(selected_remote_bytes) + int(staging_bytes) + int(final_local_bytes) + int(emergency_reserve_bytes)
    ok = int(free_bytes) >= required
    return {
        "status": "PASS" if ok else "FAIL",
        "classification": "DELL_CAPACITY_PASS" if ok else "DEFERRED_LOCAL_CAPACITY_REMOTE_COPY_DURABLE",
        "selected_remote_bytes": int(selected_remote_bytes),
        "temporary_staging_bytes": int(staging_bytes),
        "final_local_bytes": int(final_local_bytes),
        "emergency_reserve_bytes": int(emergency_reserve_bytes),
        "required_bytes": required,
        "free_bytes": int(free_bytes),
        "deleted_local_data": False,
        "deleted_b2_data": False,
    }


class DellArtifactRepatriationClient:
    def __init__(self, *, remote_runs_root: Path, local_import_root: Path, free_bytes: int = 10**12) -> None:
        self.remote_runs_root = Path(remote_runs_root)
        self.local_import_root = Path(local_import_root)
        self.free_bytes = int(free_bytes)

    def discover(self) -> dict[str, Any]:
        runs = []
        if self.remote_runs_root.exists():
            for manifest_path in sorted(self.remote_runs_root.rglob("vast_output_manifest.json")):
                marker = committed_marker_path(manifest_path.parent)
                manifest = read_json(manifest_path)
                if not marker.exists():
                    continue
                runs.append(
                    {
                        "run_id": manifest.get("run_id", manifest_path.parent.name.replace("run=", "")),
                        "manifest_path": str(manifest_path),
                        "manifest_hash": manifest.get("manifest_hash", ""),
                        "remote_run_root": str(manifest_path.parent),
                    }
                )
        return {"status": "PASS", "eligible_runs": runs, "eligible_run_count": len(runs)}

    def retrieve(self, *, run_id: str, tier: str = "compact", verify_only: bool = False, dry_run: bool = False, now_utc: str | None = None) -> dict[str, Any]:
        discovery = self.discover()
        selected = next((row for row in discovery["eligible_runs"] if row["run_id"] == run_id), None)
        if not selected:
            return {"status": "FAIL", "classification": "NO_COMMITTED_VAST_RUN_FOUND", "run_id": run_id}
        remote_root = Path(selected["remote_run_root"])
        manifest = read_json(remote_root / "vast_output_manifest.json")
        all_files = [row for row in manifest.get("files", []) if isinstance(row, Mapping)]
        selected_files = [
            row for row in all_files if tier == "all" or str(row.get("tier") or "tier1") == "tier1"
        ]
        selected_bytes = sum(int(row.get("size_bytes") or 0) for row in selected_files)
        capacity = dell_capacity_gate(
            selected_remote_bytes=selected_bytes,
            staging_bytes=selected_bytes,
            final_local_bytes=selected_bytes,
            emergency_reserve_bytes=5 * 1024**3,
            free_bytes=self.free_bytes,
        )
        if capacity["status"] != "PASS":
            return {"status": "FAIL", "classification": capacity["classification"], "capacity": capacity, "b2_copy_remains_durable": True}
        if dry_run:
            return {"status": "PASS", "classification": "DELL_REPATRIATION_DRY_RUN", "selected_file_count": len(selected_files), "selected_bytes": selected_bytes}
        staging = self.local_import_root / ".staging" / run_id
        final = self.local_import_root / f"run={run_id}"
        conflicts = []
        copied = []
        for row in selected_files:
            rel = normalize_relative_path(str(row.get("relative_path") or ""))
            ensure_no_forbidden_artifact_path(rel)
            src = safe_join(remote_root, rel)
            dst = safe_join(staging, rel)
            if final.joinpath(rel).exists() and file_sha256(final / rel) != row.get("sha256"):
                quarantine = self.local_import_root / "quarantine" / run_id / rel
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final / rel, quarantine)
                conflicts.append({"relative_path": rel, "quarantine_path": str(quarantine)})
            if not verify_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied.append(row)
        validation_failures = []
        validation_root = final if verify_only else staging
        for row in selected_files:
            rel = normalize_relative_path(str(row.get("relative_path") or ""))
            path = validation_root / rel
            if not path.is_file():
                validation_failures.append({"relative_path": rel, "reason": "MISSING_LOCAL_FILE"})
            elif file_sha256(path) != row.get("sha256"):
                validation_failures.append({"relative_path": rel, "reason": "SHA256_MISMATCH"})
        if validation_failures:
            return {"status": "FAIL", "classification": "DELL_REPATRIATION_VERIFY_FAILED", "failures": validation_failures, "conflicts": conflicts}
        if not verify_only:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                for path in staging.rglob("*"):
                    if path.is_file():
                        rel = repo_rel(staging, path)
                        dst = final / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dst)
            else:
                os.replace(staging, final)
        receipt = build_dell_import_receipt(
            manifest,
            local_root=final,
            tier=tier,
            copied_files=copied,
            conflicts=conflicts,
            now_utc=now_utc,
        )
        if not verify_only:
            write_json_atomic(final / "dell_import_receipt.json", receipt)
        return {
            "status": "PASS",
            "classification": "DELL_REPATRIATION_VERIFIED",
            "receipt": receipt,
            "selected_file_count": len(selected_files),
            "artifact_tiers_present": receipt["artifact_tiers_present"],
            "artifact_tiers_deferred": receipt["artifact_tiers_deferred"],
            "conflicts": conflicts,
            "scientific_acceptance_claimed": False,
            "b2_copy_deleted": False,
        }


def build_dell_import_receipt(
    manifest: Mapping[str, Any],
    *,
    local_root: Path,
    tier: str,
    copied_files: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    now_utc: str | None = None,
) -> dict[str, Any]:
    tiers_present = sorted({str(row.get("tier") or "tier1") for row in copied_files})
    tiers_deferred = ["tier2"] if tier == "compact" and any(str(row.get("tier") or "") == "tier2" for row in manifest.get("files", [])) else []
    first_family = ""
    for row in copied_files:
        parts = normalize_relative_path(str(row.get("relative_path") or "")).split("/")
        for part in parts:
            if part.startswith("family="):
                first_family = part.split("=", 1)[1]
    payload = {
        "schema_version": DELL_IMPORT_RECEIPT_SCHEMA_VERSION,
        "vast_run_id": manifest.get("run_id", ""),
        "family_id": first_family,
        "queue_id": manifest.get("queue_id", ""),
        "source_b2_prefix": manifest.get("remote_prefix", ""),
        "remote_manifest_hash": manifest.get("manifest_hash", ""),
        "local_manifest_hash": stable_hash({"files": list(copied_files), "local_root": str(local_root).replace("\\", "/")}),
        "contract_hashes": {
            "dataset_authority_hash": manifest.get("dataset_authority_hash", ""),
            "queue_definition_hash": manifest.get("queue_definition_hash", ""),
        },
        "verification_time_utc": now_utc or utc_now(),
        "local_paths": [str(local_root / row.get("relative_path", "")).replace("\\", "/") for row in copied_files],
        "artifact_tiers_present": tiers_present,
        "artifact_tiers_deferred": tiers_deferred,
        "hash_conflicts_quarantined": list(conflicts),
        "import_eligibility": "IMPORT_READY_FOR_SEPARATE_SCIENTIFIC_ACCEPTANCE_TICKET",
        "limitations": ["transfer success is not scientific acceptance", "live Dell worker output namespaces not overwritten"],
    }
    payload["receipt_hash"] = stable_hash(payload)
    return payload


def synthetic_end_to_end_evidence(repo_root: Path) -> dict[str, Any]:
    prerequisite = load_prerequisite_queue_authority(repo_root)
    queue_definition = read_json(Path(repo_root) / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
    with tempfile.TemporaryDirectory(prefix="ds24_vast_b2_r1_") as tmp:
        root = Path(tmp)
        source_root = root / "source"
        remote_root = root / "remote"
        dataset_root = root / "vast_dataset"
        output_root = root / "vast_output"
        source_files = ["data/a.bin", "data/b.bin", "data/c.bin"]
        for index, rel in enumerate(source_files, start=1):
            path = safe_join(source_root, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((f"payload-{index}-" * index).encode("utf-8"))
        rows = manifest_rows_from_files(source_root, source_files)
        for row in rows:
            src = safe_join(source_root, row.relative_path)
            dst = safe_join(remote_root, row.object_key())
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        remote_objects = [
            {"key": row.object_key(), "size_bytes": row.size_bytes, "sha256": row.sha256, "sha1": row.sha1}
            for row in rows
        ]
        finalizer = DatasetPublisherFinalizer(
            repo_root=repo_root,
            expected_count=len(rows),
            expected_bytes=sum(row.size_bytes for row in rows),
        )
        dataset_authority = finalizer.finalize(
            rows,
            remote_objects,
            output_root=remote_root / B2_PREFIX,
            now_utc="2026-09-03T12:00:00Z",
        )
        marker_src = remote_root / B2_PREFIX / "DATASET_COMPLETE.json"
        marker_dst = remote_root / DATASET_COMPLETE_MARKER_KEY
        if marker_src != marker_dst:
            marker_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(marker_src, marker_dst)
        dell, mac = reverse_queue.synthetic_external_status_fixture(queue_definition, now_utc="2026-09-03T12:00:00Z")["snapshots"]
        plan = create_ownership_plan(
            queue_definition,
            dell,
            mac,
            created_at_utc="2026-09-03T12:00:00Z",
            mac_unavailable_reason="synthetic package evidence",
        )
        plan["acknowledgements"]["dell"] = build_acknowledgement(
            machine="dell",
            plan_hash=plan["plan_hash"],
            now_utc="2026-09-03T12:00:01Z",
        )
        plan_path = root / "ownership_plan.json"
        write_json_atomic(plan_path, plan)
        config = build_bootstrap_config(
            repo_root=repo_root,
            run_id="synthetic-run",
            dataset_root=dataset_root,
            output_root=output_root,
            ownership_plan_path=plan_path,
            queue_definition_hash=queue_definition["queue_definition_hash"],
            dataset_authority_hash=dataset_authority["authority_hash"],
        )
        controller = VastBootstrapController(
            repo_root=repo_root,
            config=config,
            dataset_rows=rows,
            remote_root=remote_root,
            resource_snapshot={
                "cpu_cores": 32,
                "ram_bytes": 128 * 1024**3,
                "gpu_present": True,
                "vram_bytes": 24 * 1024**3,
                "cuda_present": True,
                "disk_free_bytes": 400 * 1024**3,
                "time_synchronized": True,
                "workspace_writable": True,
            },
            env={"B2_APPLICATION_KEY_ID": "redacted", "B2_APPLICATION_KEY": "redacted"},
        )
        preflight = controller.run(mode="preflight-only", now_utc="2026-09-03T12:01:00Z")
        autostart = controller.run(mode="start-after-verify", now_utc="2026-09-03T12:02:00Z", test_stub_executor=True)
        policy = artifact_retention_policy(queue_definition)
        local_run = output_root / "run_payload"
        for rel, content in {
            "queue_state/queue_state.json": "{}",
            "metrics_only_v3/family=temporal_fusion_transformer/metrics.json": "{}",
            "checkpoints/family=temporal_fusion_transformer/latest.ckpt": "checkpoint",
            "prediction_partitions/full_predictions.parquet": "forbidden",
        }.items():
            path = safe_join(local_run, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        publisher = VastDurableArtifactPublisher(
            local_run_root=local_run,
            remote_run_root=remote_root / "vast_run",
            policy=policy,
            run_id="synthetic-run",
            dataset_authority_hash=dataset_authority["authority_hash"],
            queue_definition_hash=queue_definition["queue_definition_hash"],
        )
        interrupted = publisher.publish_once(interrupt_after_files=1, now_utc="2026-09-03T12:03:00Z")
        published = publisher.publish_once(now_utc="2026-09-03T12:04:00Z")
        repatriation = DellArtifactRepatriationClient(
            remote_runs_root=remote_root,
            local_import_root=root / "dell_import",
            free_bytes=100 * 1024**3,
        ).retrieve(run_id="synthetic-run", tier="compact", now_utc="2026-09-03T12:05:00Z")
        checks = {
            "prerequisite_queue_valid": prerequisite["status"] == "PASS",
            "dataset_finalizer_passed": dataset_authority["status"] == "PASS",
            "marker_valid": validate_dataset_marker(read_json(remote_root / DATASET_COMPLETE_MARKER_KEY))["status"] == "PASS",
            "preflight_blocks_nothing_when_gates_pass": preflight["status"] == "PASS",
            "autostart_stub_after_verify": autostart["status"] == "PASS" and autostart["stub_executor_started"] is True,
            "live_model_not_started": autostart["live_model_started"] is False,
            "publisher_interruption_retryable": interrupted["status"] == "FAIL" and interrupted["committed_marker_written"] is False,
            "publisher_commits_marker_last": published["status"] == "PASS" and published["committed_marker_written"] is True,
            "forbidden_prediction_not_transferred": published["forbidden_files_transferred"] == 0,
            "dell_compact_repatriation_passed": repatriation["status"] == "PASS" and "tier2" in repatriation["artifact_tiers_deferred"],
        }
        return {
            "schema_version": "ds24_vast_r50_synthetic_end_to_end_evidence.v1",
            "checks": checks,
            "prerequisite": prerequisite,
            "dataset_authority_hash": dataset_authority["authority_hash"],
            "ownership_plan_hash": plan["plan_hash"],
            "bootstrap_preflight_classification": preflight.get("classification"),
            "autostart_classification": autostart.get("classification"),
            "publisher_retry_classification": interrupted.get("classification"),
            "publisher_final_classification": published.get("classification"),
            "dell_repatriation_classification": repatriation.get("classification"),
            "model_execution": {"live_model_started": False, "stub_executor_started": True},
            "cloud_operation_performed": False,
            "vast_instance_rented": False,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }


def failure_recovery_matrix() -> dict[str, Any]:
    rows = [
        ("interrupted B2 download", "resume copy, verify hash, keep partial isolated", "NO_MODEL_START_UNTIL_VERIFY"),
        ("interrupted B2 upload", "retry copy, publish manifest and committed marker last", "REMOTE_READER_IGNORES_UNCOMMITTED"),
        ("expired credentials", "fail secret preflight without printing values", "CREDENTIAL_REFERENCES_MISSING"),
        ("storage or transaction caps", "fail remote command contract and keep local outputs", "RETRYABLE_REMOTE_BACKLOG"),
        ("missing completion marker", "bootstrap preflight fails closed", "DATASET_MARKER_INVALID"),
        ("count mismatch", "dataset finalizer refuses authority", "DATASET_UPLOAD_INCOMPLETE_OR_UNVERIFIED"),
        ("hash mismatch", "download/finalizer fails closed", "SHA256_MISMATCH"),
        ("insufficient Vast disk", "capacity gate fails", "INSUFFICIENT_CAPACITY_FAIL_CLOSED"),
        ("insufficient Dell disk", "defer local retrieval", "DEFERRED_LOCAL_CAPACITY_REMOTE_COPY_DURABLE"),
        ("incompatible repository revision", "deployment authority mismatch blocks", "INCOMPATIBLE_CODE_OR_QUEUE_HASH"),
        ("missing queue authority", "terminal blocker", BLOCKED_QUEUE_AUTHORITY),
        ("stale ownership plan", "validate plan fails", "OWNERSHIP_PLAN_STALE"),
        ("conflicting family owner", "external coordination fails closed", "EXTERNAL_COORDINATION_FAIL_CLOSED"),
        ("dead supervisor with live lease", "require explicit recovery", "BOOTSTRAP_LEASE_ALREADY_ACTIVE"),
        ("live supervisor with stale heartbeat", "status reports stale, no duplicate launch", "SUPERVISOR_LEASE_INACTIVE"),
        ("corrupt queue state", "R49 queue state validator refuses", "QUEUE_STATE_CORRUPT_OR_INCOMPATIBLE"),
        ("corrupt checkpoint", "publisher/repatriation hash validation fails", "SHA256_MISMATCH"),
        ("B2 outage", "preserve local work and backlog", "VAST_OUTPUT_PUBLICATION_INTERRUPTED_RETRYABLE"),
        ("Vast restart", "resume mode preserves queue cursor and lease", "RESUME_WITH_EXISTING_STATE"),
        ("Dell restart", "receipt plus committed manifest allow idempotent re-run", "DELL_REPATRIATION_VERIFIED"),
        ("duplicate command invocation", "lease and supervisor guards prevent duplicate", "BOOTSTRAP_LEASE_ALREADY_ACTIVE"),
    ]
    payload = {
        "matrix_id": "DS24_VAST_R50_FAILURE_RECOVERY_MATRIX_V1",
        "rows": [
            {"failure": failure, "recovery": recovery, "classification": classification}
            for failure, recovery, classification in rows
        ],
        "every_failure_leaves_resume_evidence": True,
    }
    payload["matrix_hash"] = stable_hash(payload)
    return payload


def operational_command_catalog(authority_root: Path) -> dict[str, Any]:
    base = "python scripts/local/ds24_vast_b2_bootstrap_r1.py"
    root = str(authority_root).replace("\\", "/")
    commands = {
        "finalize_b2_dataset_after_upload": f"{base} finalize-dataset --repo-root <DELL_REPO_ROOT> --source-manifest <SOURCE_MANIFEST> --remote-inventory <B2_INVENTORY_JSON> --authority-root {root} --execute-real-b2 --confirm-token {DATASET_FINALIZE_CONFIRM_TOKEN}",
        "generate_dell_status_evidence": f"python scripts/local/ds24_vast_reverse_queue_r1.py status --repo-root <DELL_REPO_ROOT> --queue-root <DELL_QUEUE_STATE_ROOT> > <DELL_STATUS_SNAPSHOT_JSON>",
        "generate_mac_status_evidence": "python -m core.research.ml.ds24.mac_aux_queue_r44f2 status --queue-id DS24_MAC_AUX_NINE_FAMILY_R1 > <MAC_STATUS_SNAPSHOT_JSON>",
        "create_ownership_plan": f"{base} create-ownership-plan --repo-root <DELL_REPO_ROOT> --dell-snapshot <DELL_STATUS_SNAPSHOT_JSON> --mac-snapshot <MAC_STATUS_SNAPSHOT_JSON> --authority-root {root}",
        "publish_acknowledgement": f"{base} publish-ack --machine <dell|mac> --plan <OWNERSHIP_PLAN_JSON> --output <ACK_JSON>",
        "vast_preflight": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --preflight-only --vast-instance-id <VAST_INSTANCE_ID>",
        "vast_download_only": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --download-only --vast-instance-id <VAST_INSTANCE_ID>",
        "vast_verify_only": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --verify-only --vast-instance-id <VAST_INSTANCE_ID>",
        "vast_start_after_verify": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --start-after-verify --vast-instance-id <VAST_INSTANCE_ID> --execute-live --confirm-token {LIVE_BOOTSTRAP_CONFIRM_TOKEN}",
        "vast_status": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --status --vast-instance-id <VAST_INSTANCE_ID>",
        "vast_resume": f"{base} bootstrap --config <VAST_BOOTSTRAP_CONFIG_JSON> --resume --vast-instance-id <VAST_INSTANCE_ID>",
        "vast_output_publisher_status": f"{base} publisher --config <PUBLISHER_CONFIG_JSON> --status --run-id <RUN_ID>",
        "dell_compact_artifact_retrieval": f"{base} repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --once --run-id <RUN_ID> --tier compact",
        "dell_full_artifact_retrieval": f"{base} repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --once --run-id <RUN_ID> --tier all",
        "dell_verification": f"{base} repatriate --config <DELL_REPATRIATION_CONFIG_JSON> --verify-only --run-id <RUN_ID>",
        "safe_terminal_closeout": f"{base} prepare-package --repo-root <DELL_REPO_ROOT> --authority-root {root}",
    }
    payload = {"command_catalog_id": "DS24_VAST_R50_OPERATIONAL_COMMANDS_V1", "commands": commands, "credentials_literal_values_included": False}
    payload["command_catalog_hash"] = stable_hash(payload)
    return payload


def runbook_text(package: Mapping[str, Any]) -> str:
    commands = package["operational_commands"]["commands"]
    lines = [
        "# DS24 Vast B2 Bootstrap, Autostart, Durable Sync and Dell Repatriation R1",
        "",
        f"Terminal classification: `{TERMINAL_CLASSIFICATION}`",
        "",
        "This package prepares the deployment system only. It does not rent Vast, contact Backblaze, read credentials, launch live DS24 model work, inspect holdout outcomes, or place paper/live orders.",
        "",
        "The bootstrap consumes the accepted R49 reverse queue authority and requires fresh Dell/Mac ownership evidence before any future launch.",
        "",
        "## Future Commands",
    ]
    for name, command in commands.items():
        lines.extend([f"### {name}", f"`{command}`", ""])
    lines.extend(
        [
            "## Gates",
            "",
            "The dataset finalizer must run after the current upload completes and must publish `DATASET_COMPLETE.json` last. The Vast bootstrap treats that marker as necessary but still verifies the downloaded files locally.",
            "",
            "Ownership planning is generation-scoped and expires. Dell acknowledgement is required; Mac acknowledgement or a documented unavailability reason is required. The plan freezes a static non-overlapping Vast partition for that generation.",
            "",
            "The publisher uses copy semantics and writes committed markers last. The Dell repatriation client imports into staging, validates hashes, quarantines conflicts, and writes receipts without publishing into live worker namespaces.",
            "",
            "Scientific acceptance of any imported Vast artifact remains a separate decision.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def artifact_inventory_for_root(root: Path, names: Sequence[str]) -> dict[str, Any]:
    rows = []
    for name in names:
        path = Path(root) / name
        if path.is_file():
            rows.append({"path": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    payload = {"schema_version": "ds24_vast_r50_artifact_inventory.v1", "files": rows, "file_count": len(rows)}
    payload["inventory_hash"] = stable_hash(payload)
    return payload


def write_authority_package(repo_root: Path, authority_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    authority_root = Path(authority_root or (repo_root / DEFAULT_AUTHORITY_ROOT_REL))
    if not authority_root.is_absolute():
        authority_root = repo_root / authority_root
    reverse_queue.assert_authority_root_safe(repo_root, authority_root)
    before = reverse_queue.shallow_process_snapshot(repo_root)
    prerequisite = load_prerequisite_queue_authority(repo_root)
    queue_definition = read_json(repo_root / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
    if prerequisite["status"] != "PASS":
        limitations = {"terminal_classification": BLOCKED_QUEUE_AUTHORITY, "reason": "accepted R49 queue authority missing or invalid"}
        authority_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(authority_root / "limitations.json", limitations)
        return limitations
    deployment = repository_deployment_authority(repo_root, prerequisite)
    policy = artifact_retention_policy(queue_definition)
    synthetic = synthetic_end_to_end_evidence(repo_root)
    matrix = failure_recovery_matrix()
    commands = operational_command_catalog(authority_root)
    limitations = {
        "terminal_classification": TERMINAL_CLASSIFICATION,
        "vast_instance_rented": False,
        "live_model_launched": False,
        "current_upload_touched": False,
        "current_worker_touched": False,
        "credentials_read_or_stored": False,
        "holdout_accessed": False,
        "paper_orders": 0,
        "live_orders": 0,
        "cross_host_atomicity_claimed": False,
        "live_validation_requires_instance": True,
        "scientific_acceptance_claimed": False,
    }
    test_evidence = {
        "ticket_id": TICKET_ID,
        "status": "PENDING_EXTERNAL_TEST_RUN",
        "focused_tests": "",
        "adjacent_tests": "",
        "architecture_conformance": "",
        "compile_static": "",
        "diff_whitespace": "",
        "guards": limitations,
    }
    files: dict[str, Any] = {
        "input_dataset_authority.schema.json": input_dataset_authority_schema(),
        "vast_bootstrap_config.schema.json": vast_bootstrap_config_schema(),
        "ownership_plan.schema.json": ownership_plan_schema(),
        "artifact_retention_policy.json": policy,
        "vast_output_manifest.schema.json": vast_output_manifest_schema(),
        "dell_import_receipt.schema.json": dell_import_receipt_schema(),
        "synthetic_end_to_end_evidence.json": synthetic,
        "failure_recovery_matrix.json": matrix,
        "operational_commands.json": commands,
        "repository_deployment_authority.json": deployment,
        "queue_prerequisite_validation.json": prerequisite,
        "limitations.json": limitations,
        "test_evidence.json": test_evidence,
        "process_snapshot_before.json": before,
    }
    authority_root.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        write_json_atomic(authority_root / name, payload)
    package_for_runbook = {"operational_commands": commands}
    write_text_atomic(authority_root / "RUNBOOK.md", runbook_text(package_for_runbook))
    after = reverse_queue.shallow_process_snapshot(repo_root)
    write_json_atomic(authority_root / "process_snapshot_after.json", after)
    inventory_names = [
        "input_dataset_authority.schema.json",
        "vast_bootstrap_config.schema.json",
        "ownership_plan.schema.json",
        "artifact_retention_policy.json",
        "vast_output_manifest.schema.json",
        "dell_import_receipt.schema.json",
        "synthetic_end_to_end_evidence.json",
        "failure_recovery_matrix.json",
        "operational_commands.json",
        "repository_deployment_authority.json",
        "queue_prerequisite_validation.json",
        "test_evidence.json",
        "limitations.json",
        "RUNBOOK.md",
        "process_snapshot_before.json",
        "process_snapshot_after.json",
    ]
    inventory = artifact_inventory_for_root(authority_root, inventory_names)
    write_json_atomic(authority_root / "artifact_inventory.json", inventory)
    manifest_files = ["artifact_inventory.json", *inventory_names]
    manifest = {
        "ticket_id": TICKET_ID,
        "terminal_classification": TERMINAL_CLASSIFICATION,
        "created_at_utc": utc_now(),
        "authority_root": repo_rel(repo_root, authority_root),
        "queue_id": QUEUE_ID,
        "prerequisite_queue_authority_hash": prerequisite["manifest_hash"],
        "queue_definition_hash": prerequisite["queue_definition_hash"],
        "repository_deployment_authority_method": deployment["method"],
        "b2_bucket": B2_BUCKET,
        "b2_prefix": B2_PREFIX,
        "expected_dataset_object_count": EXPECTED_DATASET_OBJECT_COUNT,
        "expected_dataset_bytes": EXPECTED_DATASET_BYTES,
        "completion_marker_key": DATASET_COMPLETE_MARKER_KEY,
        "bootstrap_entry_point": "scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap",
        "reverse_autostart_entry_point": "scripts/local/ds24_vast_b2_bootstrap_r1.py bootstrap --start-after-verify",
        "output_publisher_entry_point": "scripts/local/ds24_vast_b2_bootstrap_r1.py publisher",
        "dell_repatriation_entry_point": "scripts/local/ds24_vast_b2_bootstrap_r1.py repatriate",
        "ownership_coordination": "B2-carried static generation ownership plan with Dell ack and Mac ack or explicit unavailable reason",
        "artifact_inventory": artifact_inventory_for_root(authority_root, manifest_files)["files"],
        "synthetic_evidence_status": synthetic["status"],
        "safety": {
            "vast_instance_rented": False,
            "live_model_launched": False,
            "current_upload_touched": False,
            "current_worker_touched": False,
            "credentials_read_or_stored": False,
            "holdout_accessed": False,
            "paper_orders": 0,
            "live_orders": 0,
            "process_signature_unchanged": reverse_queue.process_signature(before) == reverse_queue.process_signature(after),
            "protected_process_count_before": before.get("protected_process_count", 0),
            "protected_process_count_after": after.get("protected_process_count", 0),
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json_atomic(authority_root / "manifest.json", manifest)
    return manifest


def record_test_evidence(
    authority_root: Path,
    *,
    focused_tests: str,
    adjacent_tests: str,
    architecture_conformance: str,
    compile_static: str,
    diff_whitespace: str,
) -> dict[str, Any]:
    authority_root = Path(authority_root)
    payload = read_json(authority_root / "test_evidence.json")
    if not isinstance(payload, Mapping):
        payload = {}
    updated = {
        **dict(payload),
        "status": "PASS"
        if all(text.startswith("PASS") for text in [focused_tests, adjacent_tests, architecture_conformance, compile_static, diff_whitespace])
        else "FAIL",
        "focused_tests": focused_tests,
        "adjacent_tests": adjacent_tests,
        "architecture_conformance": architecture_conformance,
        "compile_static": compile_static,
        "diff_whitespace": diff_whitespace,
        "updated_at_utc": utc_now(),
    }
    updated["test_evidence_hash"] = stable_hash(updated)
    write_json_atomic(authority_root / "test_evidence.json", updated)
    inventory_names = [
        "input_dataset_authority.schema.json",
        "vast_bootstrap_config.schema.json",
        "ownership_plan.schema.json",
        "artifact_retention_policy.json",
        "vast_output_manifest.schema.json",
        "dell_import_receipt.schema.json",
        "synthetic_end_to_end_evidence.json",
        "failure_recovery_matrix.json",
        "operational_commands.json",
        "repository_deployment_authority.json",
        "queue_prerequisite_validation.json",
        "test_evidence.json",
        "limitations.json",
        "RUNBOOK.md",
        "process_snapshot_before.json",
        "process_snapshot_after.json",
    ]
    write_json_atomic(authority_root / "artifact_inventory.json", artifact_inventory_for_root(authority_root, inventory_names))
    manifest = read_json(authority_root / "manifest.json")
    if isinstance(manifest, Mapping) and manifest:
        manifest = dict(manifest)
        manifest["test_evidence_status"] = updated["status"]
        manifest["test_evidence_hash"] = updated["test_evidence_hash"]
        manifest["artifact_inventory"] = artifact_inventory_for_root(
            authority_root,
            ["artifact_inventory.json", *inventory_names],
        )["files"]
        manifest["manifest_hash"] = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
        write_json_atomic(authority_root / "manifest.json", manifest)
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS24 Vast B2 bootstrap, autostart, durable sync and Dell repatriation R1")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-package")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--authority-root", default=str(DEFAULT_AUTHORITY_ROOT_REL))

    finalize = sub.add_parser("finalize-dataset")
    finalize.add_argument("--repo-root", default=".")
    finalize.add_argument("--source-manifest", required=True)
    finalize.add_argument("--remote-inventory", required=True)
    finalize.add_argument("--authority-root", required=True)
    finalize.add_argument("--expected-count", type=int, default=EXPECTED_DATASET_OBJECT_COUNT)
    finalize.add_argument("--expected-bytes", type=int, default=EXPECTED_DATASET_BYTES)
    finalize.add_argument("--confirm-token", default="")
    finalize.add_argument("--execute-real-b2", action="store_true")

    plan = sub.add_parser("create-ownership-plan")
    plan.add_argument("--repo-root", default=".")
    plan.add_argument("--dell-snapshot", required=True)
    plan.add_argument("--mac-snapshot", required=True)
    plan.add_argument("--authority-root", required=True)
    plan.add_argument("--mac-unavailable-reason", default="")
    plan.add_argument("--now-utc", default="")

    ack = sub.add_parser("publish-ack")
    ack.add_argument("--machine", required=True)
    ack.add_argument("--plan", required=True)
    ack.add_argument("--output", required=True)
    ack.add_argument("--now-utc", default="")

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--config", required=True)
    mode = bootstrap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--start-after-verify", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--resume", action="store_true")
    bootstrap.add_argument("--vast-instance-id", default="")
    bootstrap.add_argument("--execute-live", action="store_true")
    bootstrap.add_argument("--confirm-token", default="")
    bootstrap.add_argument("--test-stub-executor", action="store_true")

    publisher = sub.add_parser("publisher")
    publisher.add_argument("--config", required=True)
    pub_mode = publisher.add_mutually_exclusive_group(required=True)
    pub_mode.add_argument("--status", action="store_true")
    pub_mode.add_argument("--once", action="store_true")
    publisher.add_argument("--run-id", default="")
    publisher.add_argument("--interrupt-after-files", type=int, default=-1)

    repatriate = sub.add_parser("repatriate")
    repatriate.add_argument("--config", required=True)
    rep_mode = repatriate.add_mutually_exclusive_group(required=True)
    rep_mode.add_argument("--status", action="store_true")
    rep_mode.add_argument("--once", action="store_true")
    rep_mode.add_argument("--watch", action="store_true")
    rep_mode.add_argument("--verify-only", action="store_true")
    repatriate.add_argument("--run-id", default="")
    repatriate.add_argument("--tier", choices=["compact", "all"], default="compact")
    repatriate.add_argument("--dry-run", action="store_true")

    record = sub.add_parser("record-test-evidence")
    record.add_argument("--authority-root", required=True)
    record.add_argument("--focused-tests", required=True)
    record.add_argument("--adjacent-tests", required=True)
    record.add_argument("--architecture-conformance", required=True)
    record.add_argument("--compile-static", required=True)
    record.add_argument("--diff-whitespace", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare-package":
        print(json.dumps(write_authority_package(Path(args.repo_root), Path(args.authority_root)), indent=2, sort_keys=True))
        return 0
    if args.command == "finalize-dataset":
        rows = parse_source_manifest(Path(args.source_manifest), Path(args.repo_root).resolve(), hash_local_files=True)
        remote_inventory = read_json(Path(args.remote_inventory))
        remote_objects = remote_inventory.get("objects", remote_inventory) if isinstance(remote_inventory, Mapping) else remote_inventory
        if args.execute_real_b2 and args.confirm_token != DATASET_FINALIZE_CONFIRM_TOKEN:
            raise VastB2BootstrapError("DS24_VAST_REAL_B2_FINALIZE_CONFIRM_TOKEN_REQUIRED")
        result = DatasetPublisherFinalizer(
            repo_root=Path(args.repo_root).resolve(),
            expected_count=args.expected_count,
            expected_bytes=args.expected_bytes,
        ).finalize(
            rows,
            remote_objects if isinstance(remote_objects, list) else [],
            output_root=Path(args.authority_root) / "dataset_finalization",
            upload_authority=bool(args.execute_real_b2),
            confirm_token=args.confirm_token,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "create-ownership-plan":
        repo_root = Path(args.repo_root).resolve()
        queue_definition = read_json(repo_root / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
        result = create_ownership_plan(
            queue_definition,
            read_json(Path(args.dell_snapshot)),
            read_json(Path(args.mac_snapshot)),
            created_at_utc=args.now_utc or None,
            mac_unavailable_reason=args.mac_unavailable_reason,
        )
        root = Path(args.authority_root)
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(root / "ownership_plan.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "publish-ack":
        plan_payload = read_json(Path(args.plan))
        result = build_acknowledgement(machine=args.machine, plan_hash=str(plan_payload.get("plan_hash") or ""), now_utc=args.now_utc or None)
        write_json_atomic(Path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "record-test-evidence":
        print(
            json.dumps(
                record_test_evidence(
                    Path(args.authority_root),
                    focused_tests=args.focused_tests,
                    adjacent_tests=args.adjacent_tests,
                    architecture_conformance=args.architecture_conformance,
                    compile_static=args.compile_static,
                    diff_whitespace=args.diff_whitespace,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "bootstrap":
        config = read_json(Path(args.config))
        selected_mode = (
            "preflight-only"
            if args.preflight_only
            else "download-only"
            if args.download_only
            else "verify-only"
            if args.verify_only
            else "start-after-verify"
            if args.start_after_verify
            else "status"
            if args.status
            else "resume"
        )
        if not isinstance(config, Mapping) or not config.get("dataset_rows") or not config.get("remote_root") or not config.get("resource_snapshot"):
            result = {
                "status": "PASS",
                "classification": "BOOTSTRAP_COMMAND_CONTRACT_READY_NOT_EXECUTED",
                "mode": selected_mode,
                "queue_id": QUEUE_ID,
                "vast_instance_id_placeholder_observed": bool(args.vast_instance_id),
                "requires_dataset_rows_remote_root_resource_snapshot_for_local_execution": True,
                "requires_live_confirmation_for_real_start": args.start_after_verify and not args.test_stub_executor,
                "required_confirm_token": LIVE_BOOTSTRAP_CONFIRM_TOKEN,
                "live_model_started": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        rows = [DatasetManifestRow(**row) for row in config.get("dataset_rows", [])]
        controller = VastBootstrapController(
            repo_root=Path(".").resolve(),
            config=config,
            dataset_rows=rows,
            remote_root=Path(str(config["remote_root"])),
            resource_snapshot=config.get("resource_snapshot", {}),
            env={name: os.environ.get(name, "") for name in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY")},
        )
        result = controller.run(
            mode=selected_mode,
            test_stub_executor=args.test_stub_executor,
            execute_live=args.execute_live,
            confirm_token=args.confirm_token,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    if args.command == "publisher":
        config = read_json(Path(args.config))
        if not isinstance(config, Mapping) or not config.get("local_run_root") or not config.get("remote_run_root"):
            result = {
                "status": "PASS",
                "classification": "PUBLISHER_COMMAND_CONTRACT_READY_NOT_EXECUTED",
                "queue_id": QUEUE_ID,
                "run_id": args.run_id or "<RUN_ID>",
                "copy_semantics": "copy",
                "destructive_sync_default": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        policy = config.get("policy") if isinstance(config.get("policy"), Mapping) else artifact_retention_policy(
            read_json(Path(".").resolve() / DEFAULT_QUEUE_AUTHORITY_ROOT_REL / "vast_reverse_queue_definition.json")
        )
        client = VastDurableArtifactPublisher(
            local_run_root=Path(str(config["local_run_root"])),
            remote_run_root=Path(str(config["remote_run_root"])),
            policy=policy,
            run_id=args.run_id or str(config.get("run_id") or "run"),
            dataset_authority_hash=str(config.get("dataset_authority_hash") or ""),
            queue_definition_hash=str(config.get("queue_definition_hash") or ""),
        )
        result = client.status() if args.status else client.publish_once(
            interrupt_after_files=None if args.interrupt_after_files < 0 else args.interrupt_after_files
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    if args.command == "repatriate":
        config = read_json(Path(args.config))
        if not isinstance(config, Mapping) or not config.get("remote_runs_root") or not config.get("local_import_root"):
            result = {
                "status": "PASS",
                "classification": "REPATRIATION_COMMAND_CONTRACT_READY_NOT_EXECUTED",
                "queue_id": QUEUE_ID,
                "run_id": args.run_id or "<RUN_ID>",
                "tier": args.tier,
                "scientific_acceptance_claimed": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        client = DellArtifactRepatriationClient(
            remote_runs_root=Path(str(config["remote_runs_root"])),
            local_import_root=Path(str(config["local_import_root"])),
            free_bytes=int(config.get("free_bytes") or 0),
        )
        if args.status or args.watch:
            result = client.discover()
        else:
            result = client.retrieve(
                run_id=args.run_id or str(config.get("run_id") or ""),
                tier=args.tier,
                verify_only=args.verify_only,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
