from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any


class AtomicJsonPublishError(RuntimeError):
    pass


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    advisory: bool = False,
    retry_window_seconds: float = 30.0,
    initial_delay_seconds: float = 0.05,
    max_delay_seconds: float = 2.0,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    attempt = 0
    deadline = time.monotonic() + retry_window_seconds
    last_error = ""
    while True:
        attempt += 1
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.{attempt}.tmp")
        try:
            with open(openable_path(tmp), "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(openable_path(tmp), openable_path(path))
            return {"published": True, "attempts": attempt, "tmp_path": str(tmp), "error": ""}
        except PermissionError as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if time.monotonic() >= deadline:
                if advisory:
                    return {"published": False, "attempts": attempt, "tmp_path": str(tmp), "error": last_error}
                raise AtomicJsonPublishError(last_error) from exc
            delay = min(max_delay_seconds, initial_delay_seconds * (2 ** min(attempt - 1, 8)))
            delay *= 0.75 + random.random() * 0.5
            time.sleep(delay)
        except OSError as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if time.monotonic() >= deadline:
                if advisory:
                    return {"published": False, "attempts": attempt, "tmp_path": str(tmp), "error": last_error}
                raise AtomicJsonPublishError(last_error) from exc
            time.sleep(min(max_delay_seconds, initial_delay_seconds * (2 ** min(attempt - 1, 8))))
