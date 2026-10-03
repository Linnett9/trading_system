from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterable, Mapping
import uuid

import pandas as pd

from core.research.ml.ds24.prediction_ledger_contract import (
    FORBIDDEN_LEDGER_COLUMNS,
    PREDICTION_KEY,
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_SCHEMA_VERSION,
    RESEARCH_OUTPUT_POLICY_VERSION,
    PredictionLedgerContractError,
    validate_prediction_ledger_rows,
)
from core.research.ml.ds24.prediction_ledger_queries import (
    LedgerQuery,
    apply_ledger_query,
    join_certified_outcomes,
    rank_changes,
    rebalance_candidates,
    score_spreads,
)


MANIFEST_SCHEMA = "DS24_RANKED_PREDICTION_LEDGER_MANIFEST_V1"
LOCK_TIMEOUT_SECONDS = 30.0


class PredictionLedgerStorageError(RuntimeError):
    """Raised when immutable prediction-ledger publication cannot be proven safe."""


def _stable_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(_stable_json(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _local_process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            # ERROR_INVALID_PARAMETER is the documented nonexistent-PID result.
            # Access denied or another indeterminate failure must fail closed:
            # never steal a lock from a process merely because it is inaccessible.
            return ctypes.windll.kernel32.GetLastError() != 87  # type: ignore[attr-defined]
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


@contextmanager
def _manifest_lock(root: Path) -> Iterable[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".manifest.lock"
    owner = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = _stable_json(owner)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            break
        except FileExistsError:
            try:
                current_bytes = path.read_bytes()
                current = json.loads(current_bytes)
            except (OSError, ValueError, TypeError):
                current_bytes = b""
                current = {}
            abandoned = (
                current.get("hostname") == socket.gethostname()
                and not _local_process_is_alive(int(current.get("pid", 0) or 0))
            )
            if abandoned:
                try:
                    if path.read_bytes() == current_bytes:
                        path.unlink()
                        continue
                except (FileNotFoundError, OSError):
                    pass
            if time.monotonic() >= deadline:
                raise PredictionLedgerStorageError(
                    f"Timed out acquiring prediction-ledger manifest lock: {path}"
                )
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            if path.read_bytes() == encoded:
                path.unlink()
        except FileNotFoundError:
            pass


def _normalised_for_identity(rows: pd.DataFrame) -> list[dict[str, Any]]:
    work = rows.loc[:, PREDICTION_LEDGER_COLUMNS].copy()
    for column in ("decision_timestamp", "refit_timestamp"):
        work[column] = pd.to_datetime(work[column], utc=True).map(lambda value: value.isoformat())
    work = work.sort_values(list(PREDICTION_KEY), kind="mergesort")
    return work.to_dict(orient="records")


class PredictionLedgerStore:
    """Append-only, restart-safe publisher for immutable monthly family parts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "manifest_schema": MANIFEST_SCHEMA,
            "ledger_schema": PREDICTION_LEDGER_SCHEMA_VERSION,
            "research_output_policy": RESEARCH_OUTPUT_POLICY_VERSION,
            "partitioning": ["family", "decision_month"],
            "parts": [],
            "row_count": 0,
        }

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._empty_manifest()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = self._empty_manifest()
        for key in ("manifest_schema", "ledger_schema", "research_output_policy", "partitioning"):
            if payload.get(key) != expected[key]:
                raise PredictionLedgerStorageError(f"Unsupported prediction-ledger manifest {key}")
        return payload

    def append(self, rows: pd.DataFrame) -> dict[str, Any]:
        unexpected = sorted(set(rows.columns) - set(PREDICTION_LEDGER_COLUMNS))
        if unexpected:
            raise PredictionLedgerContractError(f"Unexpected ledger columns: {unexpected}")
        validate_prediction_ledger_rows(rows)
        work = rows.loc[:, PREDICTION_LEDGER_COLUMNS].copy()
        work["decision_timestamp"] = pd.to_datetime(work["decision_timestamp"], utc=True)
        work["refit_timestamp"] = pd.to_datetime(work["refit_timestamp"], utc=True)
        work = work.sort_values(list(PREDICTION_KEY), kind="mergesort").reset_index(drop=True)
        work["_decision_month"] = work["decision_timestamp"].dt.strftime("%Y-%m")

        publications: list[dict[str, Any]] = []
        with _manifest_lock(self.root):
            for (family, month), group in work.groupby(
                ["family", "_decision_month"], sort=True
            ):
                publications.append(
                    self._append_partition(
                        str(family), str(month), group.drop(columns="_decision_month")
                    )
                )
        return {
            "appended_parts": sum(bool(item["appended"]) for item in publications),
            "rows": sum(int(item["rows"]) for item in publications if item["appended"]),
            "idempotent_parts": sum(not bool(item["appended"]) for item in publications),
            "publications": publications,
        }

    def _append_partition(self, family: str, month: str, rows: pd.DataFrame) -> dict[str, Any]:
        manifest = self._manifest()
        logical_hash = _sha256_bytes(_stable_json(_normalised_for_identity(rows)))
        timestamp_values = pd.to_datetime(rows["decision_timestamp"], utc=True)
        first = timestamp_values.min().strftime("%Y%m%dT%H%M%SZ")
        last = timestamp_values.max().strftime("%Y%m%dT%H%M%SZ")
        relative = Path(f"family={family}") / f"period={month}" / f"part-{first}-{last}-{logical_hash[:16]}.parquet"
        destination = self.root / relative

        existing_part = next(
            (part for part in manifest["parts"] if part["logical_sha256"] == logical_hash), None
        )
        if existing_part is not None:
            existing_path = self.root / existing_part["path"]
            if not existing_path.is_file() or _file_sha256(existing_path) != existing_part["file_sha256"]:
                raise PredictionLedgerStorageError("Idempotent part identity exists but its file is invalid")
            return {"appended": False, "rows": len(rows), "path": existing_part["path"]}

        candidate_keys = rows.loc[:, PREDICTION_KEY]
        for part in manifest["parts"]:
            if part["family"] != family or part["decision_month"] != month:
                continue
            existing = pd.read_parquet(self.root / part["path"], columns=list(PREDICTION_KEY))
            overlap = candidate_keys.merge(existing, on=list(PREDICTION_KEY), how="inner")
            if not overlap.empty:
                raise PredictionLedgerStorageError(
                    "Duplicate (family, decision_timestamp, asset_id) keys are forbidden"
                )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        if destination.exists():
            recovered = pd.read_parquet(destination)
            if _sha256_bytes(_stable_json(_normalised_for_identity(recovered))) != logical_hash:
                raise PredictionLedgerStorageError("Conflicting orphan partition blocks recovery")
        else:
            rows.loc[:, PREDICTION_LEDGER_COLUMNS].to_parquet(
                temporary, index=False, compression="zstd"
            )
            os.replace(temporary, destination)
        file_hash = _file_sha256(destination)
        part = {
            "path": relative.as_posix(),
            "family": family,
            "decision_month": month,
            "first_decision_timestamp": timestamp_values.min().isoformat(),
            "last_decision_timestamp": timestamp_values.max().isoformat(),
            "row_count": int(len(rows)),
            "logical_sha256": logical_hash,
            "file_sha256": file_hash,
        }
        manifest["parts"] = sorted(
            [*manifest["parts"], part], key=lambda item: (item["family"], item["decision_month"], item["path"])
        )
        manifest["row_count"] = sum(int(item["row_count"]) for item in manifest["parts"])
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, manifest)
        return {"appended": True, "rows": len(rows), "path": relative.as_posix()}


class PredictionLedgerReader:
    """Read-only research API over validated prediction-ledger partitions."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store = PredictionLedgerStore(root)

    def read(self, query: LedgerQuery | None = None) -> pd.DataFrame:
        query = query or LedgerQuery()
        manifest = self.store._manifest()
        requested = set(query.families or [])
        parts = [part for part in manifest["parts"] if not requested or part["family"] in requested]
        frames: list[pd.DataFrame] = []
        start = pd.Timestamp(query.start).tz_convert("UTC") if query.start is not None else None
        end = pd.Timestamp(query.end).tz_convert("UTC") if query.end is not None else None
        for part in parts:
            part_start = pd.Timestamp(part["first_decision_timestamp"])
            part_end = pd.Timestamp(part["last_decision_timestamp"])
            if start is not None and part_end < start:
                continue
            if end is not None and part_start > end:
                continue
            path = self.root / part["path"]
            if _file_sha256(path) != part["file_sha256"]:
                raise PredictionLedgerStorageError(f"Prediction-ledger part hash mismatch: {part['path']}")
            frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS)
        rows = pd.concat(frames, ignore_index=True)
        return apply_ledger_query(rows, query)

    @staticmethod
    def rank_changes(rows: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
        return rank_changes(rows, periods=periods)

    @staticmethod
    def score_spreads(rows: pd.DataFrame) -> pd.DataFrame:
        return score_spreads(rows)

    @staticmethod
    def rebalance_candidates(rows: pd.DataFrame, *, minimum_rank_change: float) -> pd.DataFrame:
        return rebalance_candidates(rows, minimum_rank_change=minimum_rank_change)

    @staticmethod
    def join_certified_outcomes(
        rows: pd.DataFrame,
        outcomes: pd.DataFrame,
        *,
        authority_id: str,
        holding_period: str,
        outcome_column: str,
    ) -> pd.DataFrame:
        return join_certified_outcomes(
            rows,
            outcomes,
            authority_id=authority_id,
            holding_period=holding_period,
            outcome_column=outcome_column,
        )
