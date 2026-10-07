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
RETRY_EQUIVALENCE_COLUMNS = tuple(
    column for column in PREDICTION_LEDGER_COLUMNS if column != "source_hash"
)


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


def _canonical_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["family"]),
        pd.Timestamp(row["decision_timestamp"]).tz_convert("UTC").isoformat(),
        str(row["asset_id"]),
    )


def _retry_payloads(rows: pd.DataFrame) -> dict[tuple[str, str, str], bytes]:
    """Return restart-stable scientific payloads keyed by ledger identity.

    ``source_hash`` identifies the executing operational source and therefore
    legitimately changes when a publication/resume defect is repaired.  It is
    retained on the first committed row, while all scientific, model, ranking,
    authority and run fields must remain byte-equivalent on a retry.
    """

    work = rows.loc[:, RETRY_EQUIVALENCE_COLUMNS].copy()
    for column in ("decision_timestamp", "refit_timestamp"):
        work[column] = pd.to_datetime(work[column], utc=True).map(
            lambda value: value.isoformat()
        )
    return {
        _canonical_key(row): _stable_json(row)
        for row in work.to_dict(orient="records")
    }


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

    def record_history_coverage(
        self,
        *,
        family: str,
        pre_ledger_refit_count: int,
        pre_ledger_score_count: int,
        pre_ledger_latest_decision_timestamp: str | None,
    ) -> dict[str, Any]:
        """Record the immutable per-family boundary between legacy and ledger output."""

        if pre_ledger_refit_count < 0 or pre_ledger_score_count < 0:
            raise PredictionLedgerStorageError("Pre-ledger counts cannot be negative")
        if not family or any(character in family for character in "/\\"):
            raise PredictionLedgerStorageError("Invalid history-coverage family")
        history_incomplete = pre_ledger_score_count > 0
        coverage = {
            "family": family,
            "ledger_history_complete": not history_incomplete,
            "pre_ledger_refit_count": int(pre_ledger_refit_count),
            "pre_ledger_score_count": int(pre_ledger_score_count),
            "pre_ledger_latest_decision_timestamp": (
                pd.to_datetime(
                    pre_ledger_latest_decision_timestamp,
                    utc=True,
                    errors="raise",
                ).isoformat()
                if pre_ledger_latest_decision_timestamp
                else None
            ),
            "first_durable_ledger_timestamp": None,
            "historical_reconstruction": (
                "MODEL_RETRAIN_REQUIRED" if history_incomplete else "NOT_REQUIRED"
            ),
        }
        with _manifest_lock(self.root):
            manifest = self._manifest()
            histories = dict(manifest.get("history_coverage", {}))
            existing = histories.get(family)
            if existing is not None:
                return dict(existing)
            family_parts = [
                part for part in manifest["parts"] if part["family"] == family
            ]
            if family_parts:
                coverage["first_durable_ledger_timestamp"] = min(
                    str(part["first_decision_timestamp"]) for part in family_parts
                )
            histories[family] = coverage
            manifest["history_coverage"] = dict(sorted(histories.items()))
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(self.manifest_path, manifest)
        return coverage

    def persisted_burn_in_timestamps(self, *, family: str) -> set[str]:
        """Read only bounded timestamp/eligibility columns, one part at a time."""

        manifest = self._manifest()
        timestamps: set[str] = set()
        for part in manifest["parts"]:
            if part["family"] != family:
                continue
            path = self.root / part["path"]
            if _file_sha256(path) != part["file_sha256"]:
                raise PredictionLedgerStorageError(
                    f"Prediction-ledger part hash mismatch: {part['path']}"
                )
            rows = pd.read_parquet(
                path, columns=["decision_timestamp", "evaluation_eligible"]
            )
            burn_in = rows[~rows["evaluation_eligible"].astype(bool)]
            timestamps.update(
                pd.to_datetime(burn_in["decision_timestamp"], utc=True)
                .drop_duplicates()
                .map(lambda timestamp: timestamp.isoformat())
            )
        return timestamps

    def read_timestamp_populations(
        self,
        *,
        family: str,
        decision_timestamps: Iterable[str | pd.Timestamp],
    ) -> pd.DataFrame:
        """Read hash-verified durable rows for a bounded family/timestamp set.

        This is the ledger-first recovery read boundary.  It deliberately
        returns partial populations so the application layer can distinguish
        them from absent and complete populations and fail with the correct
        recovery classification.
        """

        requested = {
            pd.Timestamp(value).tz_convert("UTC")
            for value in decision_timestamps
        }
        if not requested:
            return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS)
        months = {value.strftime("%Y-%m") for value in requested}
        requested_first = min(requested)
        requested_last = max(requested)
        frames: list[pd.DataFrame] = []
        with _manifest_lock(self.root):
            manifest = self._recover_orphan_parts(
                self._manifest(), family=family, months=months
            )
            for part in manifest["parts"]:
                if part["family"] != family or part["decision_month"] not in months:
                    continue
                part_first = pd.Timestamp(part["first_decision_timestamp"]).tz_convert(
                    "UTC"
                )
                part_last = pd.Timestamp(part["last_decision_timestamp"]).tz_convert(
                    "UTC"
                )
                if part_last < requested_first or part_first > requested_last:
                    continue
                path = self.root / part["path"]
                if not path.is_file() or _file_sha256(path) != part["file_sha256"]:
                    raise PredictionLedgerStorageError(
                        f"Prediction-ledger part hash mismatch: {part['path']}"
                    )
                rows = pd.read_parquet(path, columns=list(PREDICTION_LEDGER_COLUMNS))
                decisions = pd.to_datetime(
                    rows["decision_timestamp"], utc=True, errors="raise"
                )
                selected = rows.loc[decisions.isin(requested)].copy()
                if not selected.empty:
                    selected["decision_timestamp"] = decisions.loc[selected.index]
                    selected["refit_timestamp"] = pd.to_datetime(
                        selected["refit_timestamp"], utc=True, errors="raise"
                    )
                    frames.append(selected)
        if not frames:
            return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        if result.duplicated(list(PREDICTION_KEY)).any():
            raise PredictionLedgerStorageError(
                "Duplicate durable prediction keys block ledger-first recovery"
            )
        return result.sort_values(list(PREDICTION_KEY), kind="mergesort").reset_index(
            drop=True
        )

    def _recover_orphan_parts(
        self,
        manifest: dict[str, Any],
        *,
        family: str,
        months: set[str],
    ) -> dict[str, Any]:
        """Adopt only self-validating atomic parts omitted by a crashed manifest write."""

        known_paths = {str(part["path"]) for part in manifest["parts"]}
        orphan_paths: list[tuple[str, Path]] = []
        for month in sorted(months):
            directory = self.root / f"family={family}" / f"period={month}"
            if not directory.is_dir():
                continue
            orphan_paths.extend(
                (month, path)
                for path in sorted(directory.glob("part-*.parquet"))
                if path.relative_to(self.root).as_posix() not in known_paths
            )
        if not orphan_paths:
            return manifest

        existing_keys: set[tuple[str, str, str]] = set()
        for part in manifest["parts"]:
            if part["family"] != family or part["decision_month"] not in months:
                continue
            path = self.root / part["path"]
            if not path.is_file() or _file_sha256(path) != part["file_sha256"]:
                raise PredictionLedgerStorageError(
                    f"Prediction-ledger part hash mismatch: {part['path']}"
                )
            keys = pd.read_parquet(path, columns=list(PREDICTION_KEY))
            existing_keys.update(
                _canonical_key(row) for row in keys.to_dict(orient="records")
            )

        recovered_parts: list[dict[str, Any]] = []
        recovered_keys: set[tuple[str, str, str]] = set()
        for month, path in orphan_paths:
            relative = path.relative_to(self.root).as_posix()
            rows = pd.read_parquet(path, columns=list(PREDICTION_LEDGER_COLUMNS))
            validate_prediction_ledger_rows(rows)
            if set(rows["family"].astype(str)) != {family}:
                raise PredictionLedgerStorageError(
                    f"Orphan partition family mismatch: {relative}"
                )
            timestamps = pd.to_datetime(
                rows["decision_timestamp"], utc=True, errors="raise"
            )
            if set(timestamps.dt.strftime("%Y-%m")) != {month}:
                raise PredictionLedgerStorageError(
                    f"Orphan partition month mismatch: {relative}"
                )
            logical_hash = _sha256_bytes(
                _stable_json(_normalised_for_identity(rows))
            )
            expected_name = (
                f"part-{timestamps.min().strftime('%Y%m%dT%H%M%SZ')}-"
                f"{timestamps.max().strftime('%Y%m%dT%H%M%SZ')}-"
                f"{logical_hash[:16]}.parquet"
            )
            if path.name != expected_name:
                raise PredictionLedgerStorageError(
                    f"Orphan partition identity mismatch: {relative}"
                )
            candidate_keys = {
                _canonical_key(row) for row in rows.to_dict(orient="records")
            }
            overlap = candidate_keys & (existing_keys | recovered_keys)
            if overlap:
                raise PredictionLedgerStorageError(
                    "Conflicting orphan partition keys block recovery: "
                    f"count={len(overlap)}, path={relative}"
                )
            recovered_keys.update(candidate_keys)
            recovered_parts.append(
                {
                    "path": relative,
                    "family": family,
                    "decision_month": month,
                    "first_decision_timestamp": timestamps.min().isoformat(),
                    "last_decision_timestamp": timestamps.max().isoformat(),
                    "row_count": int(len(rows)),
                    "logical_sha256": logical_hash,
                    "file_sha256": _file_sha256(path),
                }
            )
        if not recovered_parts:
            return manifest
        updated = dict(manifest)
        updated["parts"] = sorted(
            [*manifest["parts"], *recovered_parts],
            key=lambda item: (item["family"], item["decision_month"], item["path"]),
        )
        histories = dict(updated.get("history_coverage", {}))
        coverage = histories.get(family)
        if coverage is not None:
            coverage = dict(coverage)
            recovered_first = min(
                str(part["first_decision_timestamp"]) for part in recovered_parts
            )
            current_first = coverage.get("first_durable_ledger_timestamp")
            coverage["first_durable_ledger_timestamp"] = min(
                value for value in (current_first, recovered_first) if value
            )
            histories[family] = coverage
            updated["history_coverage"] = dict(sorted(histories.items()))
        updated["row_count"] = sum(int(item["row_count"]) for item in updated["parts"])
        updated["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, updated)
        return updated

    def _append_partition(self, family: str, month: str, rows: pd.DataFrame) -> dict[str, Any]:
        manifest = self._manifest()
        incoming_row_count = len(rows)
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
            return {
                "appended": False,
                "rows": len(rows),
                "idempotent_rows": len(rows),
                "path": existing_part["path"],
            }

        candidate_payloads = _retry_payloads(rows)
        committed_keys: set[tuple[str, str, str]] = set()
        committed_paths: list[str] = []
        source_hash_mismatch_count = 0
        for part in manifest["parts"]:
            if part["family"] != family or part["decision_month"] != month:
                continue
            existing_path = self.root / part["path"]
            if (
                not existing_path.is_file()
                or _file_sha256(existing_path) != part["file_sha256"]
            ):
                raise PredictionLedgerStorageError(
                    f"Prediction-ledger part hash mismatch: {part['path']}"
                )
            existing_keys = pd.read_parquet(
                existing_path, columns=list(PREDICTION_KEY)
            )
            overlapping_keys = {
                _canonical_key(row)
                for row in existing_keys.to_dict(orient="records")
                if _canonical_key(row) in candidate_payloads
            }
            if not overlapping_keys:
                continue
            existing = pd.read_parquet(
                existing_path, columns=list(PREDICTION_LEDGER_COLUMNS)
            )
            existing_payloads = _retry_payloads(existing)
            conflicts = sorted(
                key
                for key in overlapping_keys
                if existing_payloads.get(key) != candidate_payloads[key]
            )
            if conflicts:
                raise PredictionLedgerStorageError(
                    "Conflicting duplicate (family, decision_timestamp, asset_id) "
                    f"keys are forbidden: count={len(conflicts)}, examples={conflicts[:3]}"
                )
            existing_sources = {
                _canonical_key(row): str(row["source_hash"])
                for row in existing.to_dict(orient="records")
            }
            incoming_sources = {
                _canonical_key(row): str(row["source_hash"])
                for row in rows.to_dict(orient="records")
            }
            source_hash_mismatch_count += sum(
                existing_sources[key] != incoming_sources[key]
                for key in overlapping_keys
            )
            committed_keys.update(overlapping_keys)
            committed_paths.append(str(part["path"]))

        if committed_keys:
            rows = rows[
                ~rows.apply(
                    lambda row: _canonical_key(row) in committed_keys,
                    axis=1,
                )
            ].copy()
            if rows.empty:
                return {
                    "appended": False,
                    "rows": incoming_row_count,
                    "idempotent_rows": incoming_row_count,
                    "source_hash_mismatch_rows": source_hash_mismatch_count,
                    "path": committed_paths[0],
                    "committed_paths": sorted(committed_paths),
                }

            logical_hash = _sha256_bytes(
                _stable_json(_normalised_for_identity(rows))
            )
            timestamp_values = pd.to_datetime(rows["decision_timestamp"], utc=True)
            first = timestamp_values.min().strftime("%Y%m%dT%H%M%SZ")
            last = timestamp_values.max().strftime("%Y%m%dT%H%M%SZ")
            relative = Path(f"family={family}") / f"period={month}" / f"part-{first}-{last}-{logical_hash[:16]}.parquet"
            destination = self.root / relative

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
        histories = dict(manifest.get("history_coverage", {}))
        coverage = histories.get(family)
        if coverage is not None:
            coverage = dict(coverage)
            first_timestamp = timestamp_values.min().isoformat()
            current_first = coverage.get("first_durable_ledger_timestamp")
            coverage["first_durable_ledger_timestamp"] = min(
                value for value in (current_first, first_timestamp) if value
            )
            histories[family] = coverage
            manifest["history_coverage"] = dict(sorted(histories.items()))
        manifest["parts"] = sorted(
            [*manifest["parts"], part], key=lambda item: (item["family"], item["decision_month"], item["path"])
        )
        manifest["row_count"] = sum(int(item["row_count"]) for item in manifest["parts"])
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.manifest_path, manifest)
        return {
            "appended": True,
            "rows": len(rows),
            "idempotent_rows": len(committed_keys),
            "source_hash_mismatch_rows": source_hash_mismatch_count,
            "path": relative.as_posix(),
        }


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
