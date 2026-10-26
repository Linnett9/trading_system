from __future__ import annotations

import errno
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ATOMIC_JSON_PUBLICATION_CONTRACT_VERSION = "atomic_json_publication.v1"
DEFAULT_INITIAL_DELAY_SECONDS = 0.05
DEFAULT_MAX_DELAY_SECONDS = 1.0
DEFAULT_MAX_ELAPSED_SECONDS = 15.0
DEFAULT_JSON_FORMAT = "indent2_sort_keys_default_str_allow_nan_false_no_newline.v1"
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class JsonByteFormat:
    """Controls the exact JSON bytes written by `atomic_write_json`."""

    indent: int | None = 2
    sort_keys: bool = True
    default: Callable[[Any], Any] | None = str
    allow_nan: bool = False
    ensure_ascii: bool = True
    separators: tuple[str, str] | None = None
    trailing_newline: bool = False
    encoding: str = "utf-8"
    format_id: str = DEFAULT_JSON_FORMAT

    def encode(self, payload: Any) -> bytes:
        text = json.dumps(
            payload,
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=self.default,
            allow_nan=self.allow_nan,
            ensure_ascii=self.ensure_ascii,
            separators=self.separators,
        )
        if self.trailing_newline:
            text += "\n"
        return text.encode(self.encoding)


TICKET63_JSON_FORMAT = JsonByteFormat()


class AtomicJsonPublicationError(PermissionError):
    def __init__(
        self,
        message: str,
        *,
        path: Path,
        temporary_path: Path,
        attempts: int,
        elapsed_seconds: float,
        original: BaseException,
        failure_artifact_path: Path | None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.temporary_path = temporary_path
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.original = original
        self.failure_artifact_path = failure_artifact_path
        self.winerror = getattr(original, "winerror", None)
        self.errno = getattr(original, "errno", None)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    json_format: JsonByteFormat = TICKET63_JSON_FORMAT,
    fsync: bool = True,
    initial_delay_seconds: float = DEFAULT_INITIAL_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    jitter_ratio: float = 0.20,
    replace_fn: Callable[[Path, Path], None] | None = None,
    temporary_path_fn: Callable[[Path], Path] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    random_fn: Callable[[], float] = random.random,
    contract_version: str = ATOMIC_JSON_PUBLICATION_CONTRACT_VERSION,
    failure_contract_version: str = "atomic_json_publication_failure.v1",
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = json_format.encode(payload)
    temporary = temporary_path_fn(path) if temporary_path_fn else path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    failure_artifact_path: Path | None = None
    attempts = 0
    transient_errors: list[dict[str, Any]] = []
    replace = replace_fn or os.replace
    with _WRITE_LOCK:
        try:
            with temporary.open("wb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                if fsync:
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
            json.loads(temporary.read_text(encoding="utf-8-sig"))
            started = monotonic_fn()
            delay = float(initial_delay_seconds)
            while True:
                attempts += 1
                try:
                    replace(temporary, path)
                    _cleanup_invocation_temps((temporary,))
                    return {
                        "contract_version": contract_version,
                        "status": "PASS",
                        "path": str(path),
                        "attempts": attempts,
                        "elapsed_seconds": monotonic_fn() - started,
                        "transient_error_count": len(transient_errors),
                        "retried": bool(transient_errors),
                        "json_format": json_format.format_id,
                    }
                except OSError as exc:
                    if not _is_transient_replace_error(exc):
                        raise
                    elapsed = monotonic_fn() - started
                    transient_errors.append(_error_record(exc, attempts, elapsed))
                    if elapsed >= max_elapsed_seconds:
                        failure_artifact_path = _write_publication_failure_artifact(
                            path=path,
                            temporary=temporary,
                            attempts=attempts,
                            elapsed_seconds=elapsed,
                            exc=exc,
                            transient_errors=transient_errors,
                            json_format=json_format,
                            contract_version=failure_contract_version,
                        )
                        raise AtomicJsonPublicationError(
                            f"atomic JSON publication retry exhausted for {path}",
                            path=path,
                            temporary_path=temporary,
                            attempts=attempts,
                            elapsed_seconds=elapsed,
                            original=exc,
                            failure_artifact_path=failure_artifact_path,
                        ) from exc
                    if not temporary.exists():
                        raise FileNotFoundError(f"ATOMIC_JSON_TEMP_MISSING_BEFORE_RETRY:{temporary}") from exc
                    if not path.parent.exists():
                        raise FileNotFoundError(f"ATOMIC_JSON_DESTINATION_PARENT_MISSING_BEFORE_RETRY:{path.parent}") from exc
                    if temporary.read_bytes() != payload_bytes:
                        raise RuntimeError(f"ATOMIC_JSON_TEMP_PAYLOAD_MUTATED_BEFORE_RETRY:{temporary}") from exc
                    sleep_for = min(delay, max_elapsed_seconds - elapsed)
                    if jitter_ratio:
                        jitter = sleep_for * float(jitter_ratio) * random_fn()
                        sleep_for = min(max_delay_seconds, sleep_for + jitter)
                    if sleep_for > 0:
                        sleep_fn(sleep_for)
                    delay = min(float(max_delay_seconds), delay * 2.0)
        except AtomicJsonPublicationError:
            raise
        except BaseException:
            _cleanup_invocation_temps((temporary,))
            raise


def cleanup_stale_atomic_json_temps(root: Path, *, target_name: str, dry_run: bool = False) -> dict[str, Any]:
    root = Path(root)
    pattern = f".{target_name}.*.tmp"
    candidates = [
        path
        for path in root.glob(pattern)
        if path.is_file() and path.name.startswith(f".{target_name}.") and path.name.endswith(".tmp")
    ]
    removed = []
    for path in candidates:
        if dry_run:
            removed.append({"path": str(path), "removed": False, "dry_run": True})
            continue
        try:
            path.unlink()
            removed.append({"path": str(path), "removed": True, "dry_run": False})
        except PermissionError as exc:
            removed.append({"path": str(path), "removed": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "contract_version": "stale_atomic_json_temp_cleanup.v1",
        "status": "PASS" if all(row.get("removed") or row.get("dry_run") for row in removed) else "PARTIAL",
        "root": str(root),
        "target_name": target_name,
        "candidate_count": len(candidates),
        "removed_count": sum(1 for row in removed if row.get("removed")),
        "actions": removed,
    }


def atomic_writer_contract() -> dict[str, Any]:
    return {
        "contract_version": ATOMIC_JSON_PUBLICATION_CONTRACT_VERSION,
        "temporary_file_directory": "destination_parent",
        "unique_temporary_name": True,
        "flush_before_replace": True,
        "optional_fsync": True,
        "replace_operation": "os.replace",
        "in_process_lock": True,
        "retry_errors": ["PermissionError", "winerror=5", "winerror=32"],
        "initial_delay_seconds": DEFAULT_INITIAL_DELAY_SECONDS,
        "max_delay_seconds": DEFAULT_MAX_DELAY_SECONDS,
        "max_elapsed_seconds": DEFAULT_MAX_ELAPSED_SECONDS,
        "json_format": DEFAULT_JSON_FORMAT,
        "non_retry_errors": ["invalid_path", "disk_full", "identity_mismatch", "malformed_payload"],
        "failure_artifact": "best_effort_direct_json_without_atomic_replace",
    }


def _is_transient_replace_error(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    if winerror in {5, 32}:
        return True
    if winerror is None and getattr(exc, "errno", None) in {errno.EACCES, errno.EPERM, errno.EBUSY}:
        return True
    return False


def _error_record(exc: OSError, attempts: int, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "attempt": attempts,
        "elapsed_seconds": elapsed_seconds,
        "type": type(exc).__name__,
        "winerror": getattr(exc, "winerror", None),
        "errno": getattr(exc, "errno", None),
        "message": str(exc),
    }


def _cleanup_invocation_temps(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            path.unlink()
        except PermissionError:
            pass


def _write_publication_failure_artifact(
    *,
    path: Path,
    temporary: Path,
    attempts: int,
    elapsed_seconds: float,
    exc: OSError,
    transient_errors: list[dict[str, Any]],
    json_format: JsonByteFormat,
    contract_version: str,
) -> Path | None:
    artifact = path.with_name(f".{path.name}.{uuid.uuid4().hex}.publication_failure.json")
    payload = {
        "contract_version": contract_version,
        "status": "FAILED",
        "path": str(path),
        "temporary_payload_path": str(temporary),
        "temporary_payload_preserved": temporary.exists(),
        "exception_type": type(exc).__name__,
        "winerror": getattr(exc, "winerror", None),
        "errno": getattr(exc, "errno", None),
        "attempt_count": attempts,
        "elapsed_retry_seconds": elapsed_seconds,
        "transient_errors": transient_errors,
        "json_format": json_format.format_id,
    }
    try:
        with artifact.open("wb") as handle:
            handle.write(TICKET63_JSON_FORMAT.encode(payload))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        return artifact
    except OSError:
        return None
