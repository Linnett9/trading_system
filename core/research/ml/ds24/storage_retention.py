from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RETENTION_CONTRACT_ID = "DS24_R38_LIVE_SAFE_CHECKPOINT_RETENTION_AND_LOG_CONTAINMENT_V1"
CHECKPOINT_RECLAMATION_LEDGER_CONTRACT = "DS24_R38_CHECKPOINT_RECLAMATION_LEDGER_V1"
LOG_CONTAINMENT_CONTRACT = "DS24_R38_LOG_CONTAINMENT_V1"
MATERIAL_STATE_LEDGER_CONTRACT = "DS24_R38_MATERIAL_STATE_LEDGER_FILTER_V1"
WARNING_DEDUP_CONTRACT = "DS24_R38_WARNING_DEDUPLICATION_V1"
LOCK_CONTRACT = "DS24_R38_SINGLE_INSTANCE_CLEANUP_LOCK_V1"

TIMESTAMP_PATTERN = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def openable_path(path: Path) -> str:
    resolved = str(Path(path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with open(openable_path(temp), "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(openable_path(temp), openable_path(path))


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(openable_path(temp), "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(openable_path(temp), openable_path(path))


def retention_contract() -> dict[str, Any]:
    return {
        "contract": RETENTION_CONTRACT_ID,
        "applies_to": ["tabular_policy_worker", "lightgbm_ranking_worker", "sequence_policy_worker"],
        "always_retain": [
            "live_lease_checkpoint",
            "newest_successfully_committed_checkpoint",
            "immediately_preceding_checkpoint",
            "unresolved_partial_commit_checkpoint",
            "month_end_anchor_checkpoint",
            "terminal_checkpoint",
            "audit_or_completion_referenced_checkpoint",
            "configuration_policy_feature_schema_hashes",
        ],
        "reclaimable_only_when": [
            "refit_package_completely_committed",
            "no_pending_transaction_references_checkpoint",
            "older_than_newest_two_successful_checkpoints",
            "not_month_end_or_terminal_anchor",
            "active_pid_moved_beyond_checkpoint",
            "deterministic_rebuild_authority_exists",
            "retained_checkpoint_resume_validation_passed",
        ],
        "delete_forbidden": [
            "metrics",
            "top_bottom_n_trace",
            "pending_outcomes",
            "terminal_summaries",
            "leases",
            "progress_files",
            "current_checkpoints",
        ],
        "commit_order": [
            "write_new_checkpoint_atomically",
            "validate_new_checkpoint",
            "commit_metrics_and_progress",
            "update_current_checkpoint_reference",
            "identify_obsolete_checkpoints",
            "append_hash_ledger",
            "remove_only_authorised_obsolete_checkpoints",
        ],
        "log_containment": {
            "warning_policy": "retain_first_signature_per_refit_and_count_duplicates",
            "never_suppress": ["exceptions", "tracebacks", "convergence_failures", "new_warning_signatures"],
            "stale_log_cleanup": "streaming_gzip_with_sha256_round_trip_before_unlink",
        },
    }


def parse_checkpoint_timestamp(path: Path | str) -> str:
    match = TIMESTAMP_PATTERN.search(str(path))
    if not match:
        return ""
    raw = match.group("stamp")
    parsed = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _timestamp_key(value: str) -> tuple[int, str]:
    if not value:
        return (0, "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        return (1, parsed.isoformat())
    except ValueError:
        return (1, value)


@dataclass(frozen=True)
class CheckpointRecord:
    family: str
    path: Path
    timestamp: str
    size_bytes: int
    sha256: str = ""
    configuration_hash: str = ""
    policy_hash: str = ""
    status: str = "SUCCESS"
    partial: bool = False
    terminal: bool = False
    referenced: bool = False

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        family: str,
        timestamp: str | None = None,
        sha256: str = "",
        configuration_hash: str = "",
        policy_hash: str = "",
        status: str = "SUCCESS",
        partial: bool = False,
        terminal: bool = False,
        referenced: bool = False,
    ) -> "CheckpointRecord":
        stat = Path(path).stat()
        return cls(
            family=family,
            path=Path(path),
            timestamp=timestamp or parse_checkpoint_timestamp(path),
            size_bytes=int(stat.st_size),
            sha256=sha256,
            configuration_hash=configuration_hash,
            policy_hash=policy_hash,
            status=status,
            partial=partial,
            terminal=terminal,
            referenced=referenced,
        )

    def with_hash(self) -> "CheckpointRecord":
        if self.sha256:
            return self
        return CheckpointRecord(
            family=self.family,
            path=self.path,
            timestamp=self.timestamp,
            size_bytes=self.size_bytes,
            sha256=sha256_file(self.path),
            configuration_hash=self.configuration_hash,
            policy_hash=self.policy_hash,
            status=self.status,
            partial=self.partial,
            terminal=self.terminal,
            referenced=self.referenced,
        )

    def to_ledger(self, *, reason: str, deletion_timestamp: str, retained_replacement_checkpoint: str) -> dict[str, Any]:
        hashed = self.with_hash()
        return {
            "contract": CHECKPOINT_RECLAMATION_LEDGER_CONTRACT,
            "family": hashed.family,
            "checkpoint_path": str(hashed.path),
            "checkpoint_timestamp": hashed.timestamp,
            "size": hashed.size_bytes,
            "sha256": hashed.sha256,
            "configuration_hash": hashed.configuration_hash,
            "policy_hash": hashed.policy_hash,
            "reason_reclaimable": reason,
            "deletion_timestamp": deletion_timestamp,
            "retained_replacement_checkpoint": retained_replacement_checkpoint,
        }


def _resolved_paths(paths: Iterable[Path | str]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path:
            continue
        try:
            out.add(str(Path(path).resolve()).lower())
        except OSError:
            out.add(str(Path(path)).lower())
    return out


def _month_anchor_paths(records: Sequence[CheckpointRecord]) -> set[str]:
    by_month: dict[str, CheckpointRecord] = {}
    for record in records:
        if not record.timestamp:
            continue
        month = _timestamp_key(record.timestamp)[1][:7]
        current = by_month.get(month)
        if current is None or _timestamp_key(record.timestamp) > _timestamp_key(current.timestamp):
            by_month[month] = record
    return _resolved_paths(record.path for record in by_month.values())


def validate_retained_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"valid": False, "reason": "NO_RETAINED_CHECKPOINT_PATH"}
    candidate = Path(path)
    if not candidate.exists():
        return {"valid": False, "path": str(candidate), "reason": "MISSING"}
    try:
        if candidate.stat().st_size <= 0:
            return {"valid": False, "path": str(candidate), "reason": "EMPTY"}
        digest = sha256_file(candidate)
    except OSError as exc:
        return {"valid": False, "path": str(candidate), "reason": f"{type(exc).__name__}:{exc}"}
    return {"valid": True, "path": str(candidate), "sha256": digest, "bytes": int(candidate.stat().st_size)}


def classify_checkpoints(
    records: Sequence[CheckpointRecord],
    *,
    live_checkpoint_paths: Sequence[Path | str] = (),
    referenced_paths: Sequence[Path | str] = (),
    partial_paths: Sequence[Path | str] = (),
    terminal_timestamp: str = "",
    active_refit_timestamp: str = "",
    deterministic_rebuild_authority: bool = False,
    retained_checkpoint_path: Path | str | None = None,
    retained_checkpoint_valid: bool = False,
    ownership_ambiguous: bool = False,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: (_timestamp_key(record.timestamp), str(record.path)))
    successful = [record for record in ordered if str(record.status).upper() in {"SUCCESS", "FIT_COMPLETE", "COMMITTED"}]
    newest_successful = _resolved_paths(record.path for record in successful[-2:])
    preceding = _resolved_paths(record.path for record in successful[-3:-2])
    month_anchors = _month_anchor_paths(ordered)
    live = _resolved_paths(live_checkpoint_paths)
    referenced = _resolved_paths(referenced_paths)
    partial = _resolved_paths(partial_paths)
    retained_replacement = str(retained_checkpoint_path or "")
    retained: list[dict[str, Any]] = []
    reclaimable: list[dict[str, Any]] = []
    for record in ordered:
        resolved = next(iter(_resolved_paths([record.path])), str(record.path).lower())
        reasons: list[str] = []
        if resolved in live:
            reasons.append("LIVE_LEASE_CHECKPOINT")
        if resolved in newest_successful:
            reasons.append("NEWEST_TWO_SUCCESSFUL")
        if resolved in preceding:
            reasons.append("IMMEDIATELY_PRECEDING_CHECKPOINT")
        if resolved in partial or record.partial:
            reasons.append("UNRESOLVED_PARTIAL_COMMIT")
        if resolved in month_anchors:
            reasons.append("MONTH_END_ANCHOR")
        if record.terminal or (terminal_timestamp and _timestamp_key(record.timestamp) >= _timestamp_key(terminal_timestamp)):
            reasons.append("TERMINAL_CHECKPOINT")
        if resolved in referenced or record.referenced:
            reasons.append("AUDIT_OR_COMPLETION_REFERENCED")
        if ownership_ambiguous:
            reasons.append("OWNERSHIP_AMBIGUOUS")
        if not deterministic_rebuild_authority:
            reasons.append("NO_DETERMINISTIC_REBUILD_AUTHORITY")
        if not retained_checkpoint_valid:
            reasons.append("RETAINED_CHECKPOINT_VALIDATION_NOT_PROVEN")
        if active_refit_timestamp and _timestamp_key(record.timestamp) >= _timestamp_key(active_refit_timestamp):
            reasons.append("ACTIVE_PID_NOT_MOVED_BEYOND_CHECKPOINT")
        elif not active_refit_timestamp:
            reasons.append("ACTIVE_REFIT_CURSOR_UNKNOWN")
        if reasons:
            retained.append(
                {
                    "family": record.family,
                    "path": str(record.path),
                    "timestamp": record.timestamp,
                    "bytes": record.size_bytes,
                    "reasons": reasons,
                }
            )
            continue
        reclaimable.append(
            {
                "family": record.family,
                "path": str(record.path),
                "timestamp": record.timestamp,
                "bytes": record.size_bytes,
                "reason": "REDUNDANT_DAILY_COMPLETED_PACKAGE",
                "retained_replacement_checkpoint": retained_replacement,
            }
        )
    return {
        "contract": RETENTION_CONTRACT_ID,
        "checkpoint_count": len(ordered),
        "retained": retained,
        "reclaimable": reclaimable,
        "retained_count": len(retained),
        "reclaimable_count": len(reclaimable),
        "reclaimable_bytes": int(sum(int(row["bytes"]) for row in reclaimable)),
    }


def write_checkpoint_reclamation_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    existing = ""
    if Path(path).exists():
        existing = Path(path).read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    appended = "".join(json.dumps(dict(row), sort_keys=True, default=str) + "\n" for row in rows)
    write_text_atomic(path, existing + appended)
    return {"path": str(path), "rows": len(rows), "bytes_appended": len(appended.encode("utf-8"))}


def remove_reclaimable_checkpoints(
    *,
    candidates: Sequence[CheckpointRecord],
    ledger_path: Path,
    retained_replacement_checkpoint: Path | str,
    reason: str = "REDUNDANT_DAILY_COMPLETED_PACKAGE",
    dry_run: bool = False,
) -> dict[str, Any]:
    deletion_timestamp = utc_now_iso()
    ledger_rows = [
        candidate.to_ledger(
            reason=reason,
            deletion_timestamp=deletion_timestamp,
            retained_replacement_checkpoint=str(retained_replacement_checkpoint),
        )
        for candidate in candidates
    ]
    write_checkpoint_reclamation_ledger(ledger_path, ledger_rows)
    removed: list[dict[str, Any]] = []
    if not dry_run:
        for row, candidate in zip(ledger_rows, candidates):
            os.unlink(openable_path(candidate.path))
            removed.append({**row, "removed": not candidate.path.exists()})
    return {"ledger": str(ledger_path), "ledger_rows": len(ledger_rows), "removed": removed, "dry_run": dry_run}


def discover_checkpoint_records(checkpoint_dir: Path, *, family: str, pattern: str = "*") -> list[CheckpointRecord]:
    if not checkpoint_dir.exists():
        return []
    records: list[CheckpointRecord] = []
    for path in sorted(checkpoint_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in {".pkl", ".joblib", ".pt", ".pth", ".ckpt"}:
            continue
        records.append(CheckpointRecord.from_path(path, family=family))
    return records


def apply_checkpoint_retention(
    *,
    family_root: Path,
    checkpoint_dir: Path,
    family: str,
    current_checkpoint_path: Path | None,
    active_refit_timestamp: str,
    deterministic_rebuild_authority: bool,
    referenced_paths: Sequence[Path | str] = (),
    partial_paths: Sequence[Path | str] = (),
    terminal_timestamp: str = "",
    pattern: str = "*",
    ledger_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    records = discover_checkpoint_records(checkpoint_dir, family=family, pattern=pattern)
    validation = validate_retained_checkpoint(current_checkpoint_path)
    classification = classify_checkpoints(
        records,
        live_checkpoint_paths=[current_checkpoint_path] if current_checkpoint_path else [],
        referenced_paths=referenced_paths,
        partial_paths=partial_paths,
        terminal_timestamp=terminal_timestamp,
        active_refit_timestamp=active_refit_timestamp,
        deterministic_rebuild_authority=deterministic_rebuild_authority,
        retained_checkpoint_path=current_checkpoint_path,
        retained_checkpoint_valid=bool(validation.get("valid")),
    )
    candidates_by_path = {str(record.path): record for record in records}
    reclaimable_records = [candidates_by_path[row["path"]] for row in classification["reclaimable"]]
    ledger = remove_reclaimable_checkpoints(
        candidates=reclaimable_records,
        ledger_path=ledger_path or (family_root / "checkpoint_retention_ledger.jsonl"),
        retained_replacement_checkpoint=current_checkpoint_path or "",
        dry_run=dry_run,
    )
    result = {
        "contract": RETENTION_CONTRACT_ID,
        "family": family,
        "checkpoint_dir": str(checkpoint_dir),
        "current_checkpoint": validation,
        "classification": classification,
        "ledger": ledger,
        "created_at_utc": utc_now_iso(),
    }
    write_json_atomic(family_root / "checkpoint_retention_state.json", result)
    return result


class CleanupLock:
    def __init__(self, path: Path, *, purpose: str = "r38_cleanup") -> None:
        self.path = Path(path)
        self.purpose = purpose
        self.token = stable_hash({"pid": os.getpid(), "path": str(self.path), "time": time.time_ns()})

    def acquire(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract": LOCK_CONTRACT,
            "pid": os.getpid(),
            "purpose": self.purpose,
            "token": self.token,
            "acquired_at_utc": utc_now_iso(),
        }
        raw = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")
        fd = os.open(openable_path(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)
        return payload

    def release(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("token") != self.token:
            return False
        try:
            os.unlink(openable_path(self.path))
            return True
        except FileNotFoundError:
            return True

    def __enter__(self) -> "CleanupLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def _gzip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_closed_log(
    path: Path,
    *,
    active_log_paths: Sequence[Path | str] = (),
    delete_original: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    active = _resolved_paths(active_log_paths)
    resolved = next(iter(_resolved_paths([path])), str(path).lower())
    if resolved in active:
        raise RuntimeError(f"DS24_R38_ACTIVE_LOG_REFUSED:{path}")
    original_sha = sha256_file(path)
    size_before = int(path.stat().st_size)
    gz_path = path.with_suffix(path.suffix + ".gz")
    temp = gz_path.with_name(f"{gz_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(openable_path(path), "rb") as source, gzip.open(openable_path(temp), "wb", compresslevel=6) as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    compressed_sha = _gzip_sha256(temp)
    if compressed_sha != original_sha:
        try:
            os.unlink(openable_path(temp))
        finally:
            raise RuntimeError(f"DS24_R38_LOG_COMPRESSION_HASH_MISMATCH:{path}")
    os.replace(openable_path(temp), openable_path(gz_path))
    if delete_original:
        os.unlink(openable_path(path))
    return {
        "contract": LOG_CONTAINMENT_CONTRACT,
        "path": str(path),
        "compressed_path": str(gz_path),
        "size_before": size_before,
        "size_after": int(gz_path.stat().st_size),
        "original_sha256": original_sha,
        "verified_gzip_round_trip": True,
        "original_removed": delete_original and not path.exists(),
        "created_at_utc": utc_now_iso(),
    }


class WarningDeduplicator:
    def __init__(
        self,
        *,
        telemetry_path: Path | None = None,
        passthrough: Callable[..., Any] | None = None,
        refit_context: Callable[[], str] | None = None,
    ) -> None:
        self.telemetry_path = telemetry_path
        self.passthrough = passthrough or warnings.showwarning
        self.refit_context = refit_context or (lambda: "")
        self.counts: dict[str, int] = {}
        self.first_seen: dict[str, dict[str, Any]] = {}
        self.flush_error_count = 0
        self.last_flush_error = ""

    def signature(self, message: Any, category: type[Warning], filename: str, lineno: int) -> str:
        return stable_hash(
            {
                "refit": self.refit_context(),
                "category": getattr(category, "__name__", str(category)),
                "message": str(message),
                "filename": Path(filename).name,
                "lineno": int(lineno),
            }
        )

    def _flush(self) -> None:
        if self.telemetry_path is None:
            return
        payload = {
            "contract": WARNING_DEDUP_CONTRACT,
            "duplicate_warning_signatures": {
                key: max(0, count - 1) for key, count in sorted(self.counts.items()) if count > 1
            },
            "first_seen": self.first_seen,
            "updated_at_utc": utc_now_iso(),
        }
        try:
            write_json_atomic(self.telemetry_path, payload)
        except OSError as exc:
            self.flush_error_count += 1
            self.last_flush_error = f"{type(exc).__name__}:{exc}"

    def showwarning(
        self,
        message: Any,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        sig = self.signature(message, category, filename, lineno)
        self.counts[sig] = self.counts.get(sig, 0) + 1
        if self.counts[sig] == 1:
            self.first_seen[sig] = {
                "category": getattr(category, "__name__", str(category)),
                "message": str(message),
                "filename": str(filename),
                "lineno": int(lineno),
                "refit": self.refit_context(),
                "first_seen_utc": utc_now_iso(),
            }
            self._flush()
            self.passthrough(message, category, filename, lineno, file=file, line=line)
            return
        self._flush()


_INSTALLED_DEDUPLICATOR: WarningDeduplicator | None = None


def install_warning_containment(
    telemetry_path: Path | None = None,
    *,
    refit_context: Callable[[], str] | None = None,
) -> WarningDeduplicator:
    global _INSTALLED_DEDUPLICATOR
    if _INSTALLED_DEDUPLICATOR is not None:
        return _INSTALLED_DEDUPLICATOR
    deduplicator = WarningDeduplicator(telemetry_path=telemetry_path, refit_context=refit_context)
    warnings.showwarning = deduplicator.showwarning
    _INSTALLED_DEDUPLICATOR = deduplicator
    return deduplicator


MATERIAL_CHANGE_FIELDS = (
    "state",
    "resource_block",
    "pid",
    "owner_pid",
    "family",
    "progress",
    "cursor",
    "last_completed_T",
    "phase",
)


def material_state_changed(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> bool:
    if not previous:
        return True
    return any(str(previous.get(key, "")) != str(current.get(key, "")) for key in MATERIAL_CHANGE_FIELDS)


def append_material_state_jsonl(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    if path.exists():
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                previous = json.loads(lines[-1])
        except Exception:
            previous = None
    if not material_state_changed(previous, row):
        return {"appended": False, "reason": "NO_MATERIAL_CHANGE", "path": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(openable_path(path), "a", encoding="utf-8") as handle:
        payload = {"contract": MATERIAL_STATE_LEDGER_CONTRACT, **dict(row), "written_at_utc": utc_now_iso()}
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return {"appended": True, "path": str(path)}
