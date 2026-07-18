from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


class LedgerLockTimeout(TimeoutError):
    """Raised when the machine-wide ledger lock cannot be acquired in time."""


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float = 10.0,
    poll_seconds: float = 0.05,
) -> Iterator[None]:
    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("Lock timing values are invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                _lock(handle)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(
                        f"Timed out acquiring resource ledger lock: {path}"
                    )
                time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0)))
        yield
    finally:
        if acquired:
            handle.seek(0)
            _unlock(handle)
        handle.close()


def _lock(handle: object) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]


def _unlock(handle: object) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def quarantine_copy(path: Path, quarantine_dir: Path, *, suffix: str) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{path.name}.{suffix}.corrupt"
    if target.exists():
        raise FileExistsError(f"Quarantine evidence already exists: {target}")
    os.replace(path, target)
    _fsync_directory(quarantine_dir)
    return target


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
