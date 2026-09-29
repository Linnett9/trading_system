from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

import core.research.ml.target_authority as _target_authority_module
from core.research.ml.target_authority import (
    INELIGIBLE_SOURCE_BAR,
    MATURED_VALID,
    MISSING_SOURCE_BAR,
    MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
    NOT_YET_MATURE,
    PRICE_AUTHORITY_VERSION,
    QUARANTINED_SOURCE_BAR,
    RIGHT_CENSORED,
    SESSION_BOUNDARY_CONFLICT,
    TARGET_CATALOGUE_VERSION,
    TARGET_RESOLUTION_POLICY_VERSION,
    UNKNOWN_SOURCE_GAP,
    build_target_manifest,
    calculate_target,
    canonical_hash,
    file_sha256,
    normalise_target_bars,
    resolve_target_contract,
    source_worktree_provenance,
    target_catalogue_payload,
    validate_target_availability,
)
from infrastructure.data.calendar_authority import ExchangeCalendarAuthority, default_calendar_authority
from infrastructure.data.market_sessions import (
    AFTER_HOURS_SESSION,
    CLOSED_SESSION,
    EASTERN,
    HALTED_SESSION,
    PRE_MARKET_SESSION,
    REGULAR_SESSION,
    MarketSessionContext,
)


FIVE_MINUTE_TARGET_DATASET_FOUNDATION_VERSION = "five_minute_target_dataset_foundation.v1"
FIVE_MINUTE_TARGET_DATASET_VERSION = "five_minute_target_dataset.v1"
PERMITTED_USE_FIXTURE_ONLY = "FIXTURE_OR_MECHANICAL_VALIDATION_ONLY"

DEFAULT_TARGET_IDS = (
    "forward_return_30m__decision_5m",
    "forward_return_60m__decision_5m",
    "forward_return_to_close__decision_5m",
    "forward_return_next_open__decision_5m",
)

ELIGIBLE_DECISION_ROW = "ELIGIBLE_DECISION_ROW"
EXCLUDED_NON_REGULAR_SESSION = "EXCLUDED_NON_REGULAR_SESSION"
EXCLUDED_INVALID_TIMESTAMP = "EXCLUDED_INVALID_TIMESTAMP"
EXCLUDED_DUPLICATE_SOURCE_BAR = "EXCLUDED_DUPLICATE_SOURCE_BAR"
EXCLUDED_INVALID_PRICE = "EXCLUDED_INVALID_PRICE"
EXCLUDED_UNRESOLVED_IDENTITY = "EXCLUDED_UNRESOLVED_IDENTITY"
EXCLUDED_SOURCE_BAR_NOT_FINAL = "EXCLUDED_SOURCE_BAR_NOT_FINAL"
EXCLUDED_TARGET_NOT_APPLICABLE = "EXCLUDED_TARGET_NOT_APPLICABLE"

DECISION_ROW_CLASSIFICATIONS = {
    ELIGIBLE_DECISION_ROW,
    EXCLUDED_NON_REGULAR_SESSION,
    EXCLUDED_INVALID_TIMESTAMP,
    EXCLUDED_DUPLICATE_SOURCE_BAR,
    EXCLUDED_INVALID_PRICE,
    EXCLUDED_UNRESOLVED_IDENTITY,
    EXCLUDED_SOURCE_BAR_NOT_FINAL,
    EXCLUDED_TARGET_NOT_APPLICABLE,
}

TARGET_ROW_FIELDS = (
    "asset_id",
    "canonical_symbol",
    "decision_timestamp",
    "decision_session",
    "decision_source_bar_id",
    "decision_source_row_hash",
    "target_id",
    "target_contract_version",
    "target_code_hash",
    "target_resolution_policy_version",
    "target_start_timestamp",
    "target_end_timestamp",
    "target_available_timestamp",
    "target_value",
    "target_is_mature",
    "target_is_realised",
    "target_is_trainable",
    "target_resolution_classification",
    "target_resolution_reason",
    "target_entry_bar_id",
    "target_exit_bar_id",
    "target_source_bar_ids",
    "target_source_bar_count",
    "source_cutoff",
    "training_cutoff",
    "calendar_authority_version",
    "price_authority_version",
    "source_dataset_version",
    "source_partition",
    "source_partition_hash",
)

SOURCE_COLUMNS = (
    "asset_id",
    "canonical_symbol",
    "provider_symbol",
    "timestamp_utc",
    "session_date",
    "session_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "provider",
    "feed",
    "timeframe",
    "adjustment_policy",
    "raw_chunk_id",
    "source_row_hash",
    "dataset_version",
)

REQUIRED_SOURCE_COLUMNS = (
    "asset_id",
    "canonical_symbol",
    "timestamp_utc",
    "session_date",
    "session_type",
    "open",
    "high",
    "low",
    "close",
    "source_row_hash",
    "dataset_version",
)


class FiveMinuteTargetDatasetError(RuntimeError):
    """Raised when the five-minute target dataset foundation cannot build safely."""


class SourceSchemaError(FiveMinuteTargetDatasetError):
    """Raised when a bounded source partition lacks required fields."""


@dataclass(frozen=True)
class SourcePartition:
    symbol: str
    year: int
    path: Path
    exists: bool
    sha256: str | None
    row_count: int
    filtered_row_count: int
    schema_columns: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "year": self.year,
            "path": _rel(self.path),
            "exists": self.exists,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "filtered_row_count": self.filtered_row_count,
            "schema_columns": list(self.schema_columns),
        }


@dataclass(frozen=True)
class SourceReadResult:
    rows_by_symbol: Mapping[str, tuple[dict[str, Any], ...]]
    partitions: tuple[SourcePartition, ...]
    decision_sessions: tuple[date, ...]
    read_sessions: tuple[date, ...]
    filters_applied: bool
    whole_archive_scanned: bool

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for symbol in sorted(self.rows_by_symbol):
            rows.extend(self.rows_by_symbol[symbol])
        return tuple(rows)


@dataclass(frozen=True)
class BuildResult:
    output_root: Path
    target_rows: tuple[dict[str, Any], ...]
    decision_rows: tuple[dict[str, Any], ...]
    source_read: SourceReadResult
    content_hash: str
    schema_hash: str
    dataset_manifest: Mapping[str, Any]
    quality_report: Mapping[str, Any]

    @property
    def target_row_count(self) -> int:
        return len(self.target_rows)


def build_five_minute_target_dataset(
    *,
    source_root: Path | str,
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    target_ids: Sequence[str],
    output_root: Path | str,
    source_cutoff: datetime | date | str | None,
    training_cutoff: datetime | date | str | None = None,
    calendar_authority: ExchangeCalendarAuthority | None = None,
    max_workers: int = 1,
    overwrite: bool = False,
    run_id: str | None = None,
    permitted_use: str = PERMITTED_USE_FIXTURE_ONLY,
    protected_process_active: bool = False,
    max_real_symbols: int | None = None,
    max_real_sessions_per_symbol: int | None = None,
    allow_large_build: bool = False,
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]] = (),
) -> BuildResult:
    """Build a bounded, manifest-backed five-minute target dataset.

    This function deliberately owns only dataset mechanics: bounded source reads,
    decision-row classification, target-row shaping, manifests, hashes, and atomic
    publication. Target mathematics are delegated to Ticket 71 target authority.
    """

    _install_cached_exchange_session_context()
    authority = _CachedCalendarAuthority(calendar_authority or default_calendar_authority())
    resolved_symbols = _normalise_symbols(symbols)
    resolved_start = _coerce_date(start_date)
    resolved_end = _coerce_date(end_date)
    resolved_target_ids = _normalise_target_ids(target_ids)
    resolved_source_cutoff = _coerce_timestamp(source_cutoff) if source_cutoff is not None else None
    resolved_training_cutoff = (
        _coerce_timestamp(training_cutoff)
        if training_cutoff is not None
        else resolved_source_cutoff
    )
    _validate_resource_limits(
        symbols=resolved_symbols,
        start_date=resolved_start,
        end_date=resolved_end,
        max_workers=max_workers,
        protected_process_active=protected_process_active,
        max_real_symbols=max_real_symbols,
        max_real_sessions_per_symbol=max_real_sessions_per_symbol,
        allow_large_build=allow_large_build,
    )
    source_read = read_five_minute_source_slice(
        source_root=Path(source_root),
        symbols=resolved_symbols,
        start_date=resolved_start,
        end_date=resolved_end,
        target_ids=resolved_target_ids,
        calendar_authority=authority,
    )
    if resolved_source_cutoff is None:
        resolved_source_cutoff = _max_bar_final_timestamp(source_read.rows) or _session_end(authority, resolved_end)
    if resolved_training_cutoff is None:
        resolved_training_cutoff = resolved_source_cutoff

    decision_rows, rows_for_authority = classify_decision_rows(
        source_read.rows,
        start_date=resolved_start,
        end_date=resolved_end,
        source_cutoff=resolved_source_cutoff,
    )
    target_rows = _build_target_rows(
        rows_for_authority=rows_for_authority,
        decision_rows=decision_rows,
        target_ids=resolved_target_ids,
        source_cutoff=resolved_source_cutoff,
        training_cutoff=resolved_training_cutoff,
        calendar_authority=authority,
        market_halts=market_halts,
    )
    target_rows = tuple(sorted(target_rows, key=_target_row_sort_key))
    schema = target_row_schema()
    schema_hash = canonical_hash(schema)
    content_hash = target_rows_content_hash(target_rows)

    output_path = Path(output_root)
    run = run_id or deterministic_run_id(
        source_root=Path(source_root),
        symbols=resolved_symbols,
        start_date=resolved_start,
        end_date=resolved_end,
        target_ids=resolved_target_ids,
        source_cutoff=resolved_source_cutoff,
        training_cutoff=resolved_training_cutoff,
        content_hash=content_hash,
    )
    temp_token = canonical_hash({"output_path": str(output_path), "run_id": run})[:16].lower()
    temp_root = output_path.parent / f".tmp_{temp_token}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    try:
        artifacts = _write_dataset_artifacts(
            output_root=temp_root,
            source_root=Path(source_root),
            symbols=resolved_symbols,
            start_date=resolved_start,
            end_date=resolved_end,
            target_ids=resolved_target_ids,
            source_cutoff=resolved_source_cutoff,
            training_cutoff=resolved_training_cutoff,
            source_read=source_read,
            decision_rows=decision_rows,
            target_rows=target_rows,
            calendar_authority=authority,
            schema=schema,
            schema_hash=schema_hash,
            content_hash=content_hash,
            run_id=run,
            permitted_use=permitted_use,
            max_workers=max_workers,
        )
        _publish_atomic(temp_root, output_path, overwrite=overwrite)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return BuildResult(
        output_root=output_path,
        target_rows=target_rows,
        decision_rows=decision_rows,
        source_read=source_read,
        content_hash=content_hash,
        schema_hash=schema_hash,
        dataset_manifest=artifacts["dataset_manifest"],
        quality_report=artifacts["quality_report"],
    )


def read_five_minute_source_slice(
    *,
    source_root: Path | str,
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    target_ids: Sequence[str],
    calendar_authority: ExchangeCalendarAuthority | None = None,
) -> SourceReadResult:
    authority = _CachedCalendarAuthority(calendar_authority or default_calendar_authority())
    root = Path(source_root)
    resolved_symbols = _normalise_symbols(symbols)
    resolved_start = _coerce_date(start_date)
    resolved_end = _coerce_date(end_date)
    decision_sessions = tuple(authority.sessions(resolved_start, resolved_end))
    read_sessions = _read_sessions_for_targets(
        authority=authority,
        decision_sessions=decision_sessions,
        target_ids=target_ids,
    )
    if not read_sessions:
        read_sessions = decision_sessions
    wanted = tuple(day.isoformat() for day in read_sessions)
    years = tuple(range(min(read_sessions).year, max(read_sessions).year + 1)) if read_sessions else ()
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in resolved_symbols}
    partitions: list[SourcePartition] = []
    for symbol in resolved_symbols:
        for year in years:
            path = root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
            if not path.exists():
                partitions.append(
                    SourcePartition(
                        symbol=symbol,
                        year=year,
                        path=path,
                        exists=False,
                        sha256=None,
                        row_count=0,
                        filtered_row_count=0,
                        schema_columns=(),
                    )
                )
                continue
            parquet = pq.ParquetFile(path)
            schema_columns = tuple(parquet.schema_arrow.names)
            missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in schema_columns]
            if missing:
                raise SourceSchemaError(f"{path} missing required source columns: {missing}")
            selected_columns = [column for column in SOURCE_COLUMNS if column in schema_columns]
            table = pq.read_table(
                path,
                columns=selected_columns,
                filters=[("session_date", "in", list(wanted))],
            )
            partition_hash = file_sha256(path)
            rows = [
                _normalise_source_row(
                    row,
                    symbol=symbol,
                    source_partition=path,
                    source_partition_hash=partition_hash or "",
                )
                for row in table.to_pylist()
            ]
            rows_by_symbol[symbol].extend(rows)
            partitions.append(
                SourcePartition(
                    symbol=symbol,
                    year=year,
                    path=path,
                    exists=True,
                    sha256=partition_hash,
                    row_count=parquet.metadata.num_rows,
                    filtered_row_count=len(rows),
                    schema_columns=schema_columns,
                )
            )
    return SourceReadResult(
        rows_by_symbol={
            symbol: tuple(sorted(rows, key=_source_row_sort_key))
            for symbol, rows in sorted(rows_by_symbol.items())
        },
        partitions=tuple(partitions),
        decision_sessions=decision_sessions,
        read_sessions=read_sessions,
        filters_applied=True,
        whole_archive_scanned=False,
    )


def classify_decision_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_date: date,
    end_date: date,
    source_cutoff: datetime,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    duplicate_keys = _duplicate_source_keys(rows)
    decision_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    for row in sorted((dict(item) for item in rows), key=_source_row_sort_key):
        classified = _classify_source_row_for_decision(
            row,
            start_date=start_date,
            end_date=end_date,
            source_cutoff=source_cutoff,
            duplicate_keys=duplicate_keys,
        )
        decision_rows.append(classified)
        authority_row = dict(row)
        if classified["decision_row_classification"] == EXCLUDED_INVALID_PRICE:
            authority_row["bar_status"] = "INELIGIBLE"
        elif str(row.get("bar_status") or "").strip():
            authority_row["bar_status"] = row["bar_status"]
        else:
            authority_row["bar_status"] = "OK"
        if classified["decision_row_classification"] == EXCLUDED_DUPLICATE_SOURCE_BAR:
            continue
        authority_rows.append(authority_row)
    return tuple(decision_rows), tuple(authority_rows)


def target_rows_content_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    semantic_rows = [
        {field: row.get(field, "") for field in TARGET_ROW_FIELDS}
        for row in sorted(rows, key=_target_row_sort_key)
    ]
    return canonical_hash(semantic_rows)


def target_row_schema() -> dict[str, Any]:
    bool_fields = {"target_is_mature", "target_is_realised", "target_is_trainable"}
    float_fields = {"target_value"}
    integer_fields = {"target_source_bar_count"}
    fields = []
    for field in TARGET_ROW_FIELDS:
        if field in bool_fields:
            kind = "bool"
        elif field in float_fields:
            kind = "float64_nullable"
        elif field in integer_fields:
            kind = "int64"
        else:
            kind = "string"
        fields.append({"name": field, "type": kind, "required": True})
    return {
        "schema_version": "five_minute_target_rows.schema.v1",
        "fields": fields,
        "stable_list_serialisation": {
            "target_source_bar_ids": "JSON array string with sorted source-path order from target authority",
        },
    }


def deterministic_run_id(
    *,
    source_root: Path,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    target_ids: Sequence[str],
    source_cutoff: datetime,
    training_cutoff: datetime,
    content_hash: str,
) -> str:
    payload = {
        "source_root": _rel(source_root),
        "symbols": list(symbols),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "target_ids": list(target_ids),
        "source_cutoff": _format_timestamp(source_cutoff),
        "training_cutoff": _format_timestamp(training_cutoff),
        "content_hash": content_hash,
    }
    return "run_" + canonical_hash(payload)[:16].lower()


def target_authority_reuse_map() -> dict[str, Any]:
    return {
        "reuse_map_version": "ticket_71b_f_target_authority_reuse_map.v1",
        "foundation_module": "core.research.ml.five_minute_target_dataset",
        "owner_module": "core.research.ml.target_authority",
        "reused_functions": [
            {
                "function": "resolve_target_contract",
                "module": "core.research.ml.target_authority",
                "use": "strict canonical target contract resolution for requested target IDs",
            },
            {
                "function": "normalise_target_bars",
                "module": "core.research.ml.target_authority",
                "use": "normalise source rows into authority bar objects with start/final timestamps",
            },
            {
                "function": "calculate_target",
                "module": "core.research.ml.target_authority",
                "use": "all target timing, value, maturity, trainability and source-path calculations",
            },
            {
                "function": "build_target_manifest",
                "module": "core.research.ml.target_authority",
                "use": "authority-level target manifest fields and resolution counts",
            },
            {
                "function": "target_catalogue_payload",
                "module": "core.research.ml.target_authority",
                "use": "catalogue identity, content hash and contract listing",
            },
            {
                "function": "validate_target_availability",
                "module": "core.research.ml.target_authority",
                "use": "PIT and trainability validation over generated target rows",
            },
        ],
        "not_reimplemented": [
            "target boundary math",
            "next-open calendar selection",
            "near-close conflict handling",
            "complete expected five-minute source path checks",
            "target value calculation",
        ],
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})



def write_missing_bar_analysis_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write the large per-partition missing-bar diagnostic compactly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("canonical_symbol", pa.string()),
            ("target_id", pa.string()),
            ("decision_timestamp", pa.string()),
            ("target_resolution_classification", pa.string()),
            ("target_resolution_reason", pa.string()),
            ("target_source_bar_count", pa.int64()),
        ]
    )

    normalized = []

    for row in rows:
        count = row.get("target_source_bar_count")

        normalized.append(
            {
                "canonical_symbol": (
                    None
                    if row.get("canonical_symbol") in (None, "")
                    else str(row.get("canonical_symbol"))
                ),
                "target_id": (
                    None
                    if row.get("target_id") in (None, "")
                    else str(row.get("target_id"))
                ),
                "decision_timestamp": (
                    None
                    if row.get("decision_timestamp") in (None, "")
                    else str(row.get("decision_timestamp"))
                ),
                "target_resolution_classification": (
                    None
                    if row.get("target_resolution_classification") in (None, "")
                    else str(row.get("target_resolution_classification"))
                ),
                "target_resolution_reason": (
                    None
                    if row.get("target_resolution_reason") in (None, "")
                    else str(row.get("target_resolution_reason"))
                ),
                "target_source_bar_count": (
                    None
                    if count in (None, "")
                    else int(count)
                ),
            }
        )

    table = pa.Table.from_pylist(
        normalized,
        schema=schema,
    )

    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
    )


def audit_repository_provenance() -> dict[str, Any]:
    status = _git("status", "--short").splitlines()
    return {
        "audit_id": "ticket_71b_five_minute_target_foundation",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "head_subject": _git("log", "-1", "--pretty=%s"),
        "head_author_date": _git("log", "-1", "--pretty=%aI"),
        "head_committer_date": _git("log", "-1", "--pretty=%cI"),
        "dirty_tree_entry_count": len(status),
        "dirty_tree_entries": status,
        "model_training_performed": False,
        "portfolio_replay_performed": False,
        "promotion_performed": False,
    }


def audit_active_processes() -> dict[str, Any]:
    pid4276 = _active_pid_4276_fallback()
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -eq 4276 -or $_.CommandLine -match "
        "'ticket_57|ticket_57f|target_resolution|frozen-selector' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine | "
        "ConvertTo-Json -Depth 4"
    )
    stdout = _run(["powershell", "-NoProfile", "-Command", script])
    raw = []
    if stdout:
        try:
            payload = json.loads(stdout)
            raw = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            raw = []
    processes = []
    for item in raw:
        process_id = int(item.get("ProcessId") or 0)
        name = str(item.get("Name") or "")
        command_line = str(item.get("CommandLine") or "")
        is_ticket57 = process_id == 4276 or (
            name.lower().startswith("python")
            and any(token in command_line.lower() for token in ("ticket_57", "ticket_57f", "target_resolution"))
        )
        processes.append(
            {
                "process_id": process_id,
                "parent_process_id": item.get("ParentProcessId"),
                "name": name,
                "creation_date": item.get("CreationDate"),
                "executable_path": item.get("ExecutablePath"),
                "command_line": command_line,
                "classification": "ACTIVE_TICKET_57_PROCESS" if is_ticket57 else "MATCHED_QUERY_PROCESS",
            }
        )
    if pid4276 and not any(item["process_id"] == 4276 for item in processes):
        processes.insert(0, pid4276)
    active = any(item["classification"] == "ACTIVE_TICKET_57_PROCESS" for item in processes)
    return {
        "active_ticket57_process_detected": active,
        "matched_processes": processes,
        "ticket_71b_f_resource_restrictions_applied": active,
    }


def _active_pid_4276_fallback() -> dict[str, Any] | None:
    stdout = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process -Id 4276 -ErrorAction SilentlyContinue | "
            "Select-Object Id,ProcessName,StartTime,Path | ConvertTo-Json -Depth 3",
        ]
    )
    if not stdout:
        return None
    try:
        item = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "process_id": 4276,
            "parent_process_id": None,
            "name": "unknown",
            "creation_date": "",
            "executable_path": "",
            "command_line": "PID 4276 present; command line unavailable from fallback",
            "classification": "ACTIVE_TICKET_57_PROCESS",
        }
    if isinstance(item, list):
        item = item[0] if item else {}
    if not item:
        return None
    return {
        "process_id": 4276,
        "parent_process_id": None,
        "name": item.get("ProcessName"),
        "creation_date": item.get("StartTime"),
        "executable_path": item.get("Path"),
        "command_line": "PID 4276 present; see command-line process query when available",
        "classification": "ACTIVE_TICKET_57_PROCESS",
    }


def audit_available_memory() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return {
            "source": "psutil",
            "free_physical_memory_kib": int(memory.available / 1024),
            "total_visible_memory_kib": int(memory.total / 1024),
            "free_physical_memory_gib_estimate": round(float(memory.available) / 1024**3, 2),
            "total_visible_memory_gib_estimate": round(float(memory.total) / 1024**3, 2),
            "probe_failed": False,
        }
    except Exception as psutil_exc:
        psutil_error = repr(psutil_exc)
    stdout = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_OperatingSystem | "
            "Select-Object FreePhysicalMemory,TotalVisibleMemorySize | ConvertTo-Json",
        ]
    )
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
    free_kib = payload.get("FreePhysicalMemory")
    total_kib = payload.get("TotalVisibleMemorySize")
    if free_kib and total_kib:
        return {
            "source": "cim",
            "free_physical_memory_kib": free_kib,
            "total_visible_memory_kib": total_kib,
            "free_physical_memory_gib_estimate": round(float(free_kib) / 1024 / 1024, 2),
            "total_visible_memory_gib_estimate": round(float(total_kib) / 1024 / 1024, 2),
            "probe_failed": False,
        }
    native = _windows_memory_status()
    if native:
        return native
    return {
        "source": "unavailable",
        "free_physical_memory_kib": free_kib,
        "total_visible_memory_kib": total_kib,
        "free_physical_memory_gib_estimate": None,
        "total_visible_memory_gib_estimate": None,
        "probe_failed": True,
        "psutil_error": psutil_error,
    }


def _windows_memory_status() -> dict[str, Any] | None:
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except Exception:
        return None
    if not ok:
        return None
    return {
        "source": "kernel32_GlobalMemoryStatusEx",
        "free_physical_memory_kib": int(status.ullAvailPhys / 1024),
        "total_visible_memory_kib": int(status.ullTotalPhys / 1024),
        "free_physical_memory_gib_estimate": round(float(status.ullAvailPhys) / 1024**3, 2),
        "total_visible_memory_gib_estimate": round(float(status.ullTotalPhys) / 1024**3, 2),
        "probe_failed": False,
    }


def source_schema_audit(source_root: Path, symbols: Sequence[str] = ("AAPL", "SPY"), year: int = 2024) -> dict[str, Any]:
    partitions = []
    for symbol in symbols:
        path = source_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        if not path.exists():
            partitions.append({"symbol": symbol, "path": _rel(path), "exists": False})
            continue
        parquet = pq.ParquetFile(path)
        partitions.append(
            {
                "symbol": symbol,
                "path": _rel(path),
                "exists": True,
                "row_count_from_footer": parquet.metadata.num_rows,
                "row_group_count": parquet.metadata.num_row_groups,
                "schema": [
                    {"name": field.name, "type": str(field.type)}
                    for field in parquet.schema_arrow
                ],
                "required_columns_present": all(
                    column in parquet.schema_arrow.names for column in REQUIRED_SOURCE_COLUMNS
                ),
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_audit_version": "ticket_71b_f_source_schema_audit.v1",
        "source_root": _rel(source_root),
        "bounded_footer_read_only": True,
        "whole_archive_scanned": False,
        "partitions": partitions,
    }


def _build_target_rows(
    *,
    rows_for_authority: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    target_ids: Sequence[str],
    source_cutoff: datetime,
    training_cutoff: datetime,
    calendar_authority: ExchangeCalendarAuthority,
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]],
) -> tuple[dict[str, Any], ...]:
    rows_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_for_authority:
        rows_by_symbol[str(row.get("canonical_symbol") or row.get("asset_id") or "").upper()].append(dict(row))
    indexes_by_symbol = {
        symbol: _build_bar_index(rows)
        for symbol, rows in rows_by_symbol.items()
    }
    contracts = {target_id: resolve_target_contract(target_id) for target_id in target_ids}
    output: list[dict[str, Any]] = []
    eligible_decisions = [
        row for row in decision_rows
        if row.get("decision_row_classification") == ELIGIBLE_DECISION_ROW
    ]
    for decision in sorted(eligible_decisions, key=_decision_row_sort_key):
        symbol = str(decision["canonical_symbol"])
        for target_id in target_ids:
            contract = contracts[target_id]
            bar_window = _bar_window_for_contract(
                indexes_by_symbol.get(symbol, {}),
                decision_row=decision,
                target_contract=contract,
                calendar_authority=calendar_authority,
            )
            normalised_window = normalise_target_bars(
                bar_window,
                asset_id=symbol,
                timeframe="5m",
                calendar_authority=calendar_authority,
            )
            result = calculate_target(
                asset_id=symbol,
                decision_timestamp=decision["decision_timestamp"],
                bar_source=normalised_window,
                target_contract=contract,
                calendar_authority=calendar_authority,
                source_cutoff=source_cutoff,
                training_cutoff=training_cutoff,
                market_halts=market_halts,
            ).payload()
            target_row = _canonical_target_row(
                result,
                decision_row=decision,
                source_cutoff=source_cutoff,
                training_cutoff=training_cutoff,
            )
            output.append(target_row)
    return tuple(output)


class _CachedCalendarAuthority:
    def __init__(self, wrapped: ExchangeCalendarAuthority):
        self._wrapped = wrapped
        self._session_cache: dict[tuple[str, str], Any] = {}
        self._sessions_cache: dict[tuple[str, str, str], list[date]] = {}
        self._next_cache: dict[tuple[str, str], date | None] = {}
        self._identity_cache: dict[tuple[str | None, str | None, str], dict[str, Any]] = {}

    def session(self, day: date | str, *, exchange: str | None = None) -> Any:
        key = (_coerce_date(day).isoformat(), str(exchange or ""))
        if key not in self._session_cache:
            self._session_cache[key] = self._wrapped.session(day, exchange=exchange)
        return self._session_cache[key]

    def sessions(self, start: date | str, end: date | str, *, exchange: str | None = None) -> list[date]:
        key = (_coerce_date(start).isoformat(), _coerce_date(end).isoformat(), str(exchange or ""))
        if key not in self._sessions_cache:
            self._sessions_cache[key] = self._wrapped.sessions(start, end, exchange=exchange)
        return list(self._sessions_cache[key])

    def next_session(self, day: date | str, *, exchange: str | None = None) -> date | None:
        key = (_coerce_date(day).isoformat(), str(exchange or ""))
        if key not in self._next_cache:
            self._next_cache[key] = self._wrapped.next_session(day, exchange=exchange)
        return self._next_cache[key]

    def previous_session(self, day: date | str, *, exchange: str | None = None) -> date | None:
        return self._wrapped.previous_session(day, exchange=exchange)

    def identity(
        self,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        exchange: str | None = None,
    ) -> dict[str, Any]:
        key = (
            _coerce_date(start).isoformat() if start is not None else None,
            _coerce_date(end).isoformat() if end is not None else None,
            str(exchange or ""),
        )
        if key not in self._identity_cache:
            self._identity_cache[key] = self._wrapped.identity(start=start, end=end, exchange=exchange)
        return dict(self._identity_cache[key])


_ORIGINAL_EXCHANGE_SESSION_CONTEXT = _target_authority_module.exchange_session_context
_EXCHANGE_CONTEXT_BY_TIMESTAMP: dict[datetime, MarketSessionContext] = {}
_SESSION_RECORD_BY_DAY: dict[date, Any] = {}


def _install_cached_exchange_session_context() -> None:
    if _target_authority_module.exchange_session_context is not _cached_exchange_session_context:
        _target_authority_module.exchange_session_context = _cached_exchange_session_context


def _cached_exchange_session_context(
    timestamp: datetime,
    *,
    exchange: str = "XNYS",
    market_halts: Sequence[Mapping[str, Any] | tuple[Any, Any]] = (),
) -> MarketSessionContext:
    if market_halts:
        return _ORIGINAL_EXCHANGE_SESSION_CONTEXT(
            timestamp,
            exchange=exchange,
            market_halts=market_halts,
        )
    timestamp_utc = _coerce_timestamp(timestamp)
    cached = _EXCHANGE_CONTEXT_BY_TIMESTAMP.get(timestamp_utc)
    if cached is not None:
        return cached
    local = timestamp_utc.astimezone(EASTERN)
    record = _SESSION_RECORD_BY_DAY.get(local.date())
    if record is None:
        record = default_calendar_authority().session(local.date(), exchange=exchange)
        _SESSION_RECORD_BY_DAY[local.date()] = record
    trading_session = CLOSED_SESSION
    if (
        record.is_trading_day
        and record.pre_market_open_timestamp
        and record.open_timestamp
        and record.close_timestamp
        and record.after_hours_close_timestamp
    ):
        pre_open = record.pre_market_open_timestamp.astimezone(EASTERN)
        regular_open = record.open_timestamp.astimezone(EASTERN)
        regular_close = record.close_timestamp.astimezone(EASTERN)
        extended_close = record.after_hours_close_timestamp.astimezone(EASTERN)
        if pre_open <= local < regular_open:
            trading_session = PRE_MARKET_SESSION
        elif regular_open <= local < regular_close:
            trading_session = REGULAR_SESSION
        elif regular_close <= local < extended_close:
            trading_session = AFTER_HOURS_SESSION
    context = MarketSessionContext(
        exchange=record.exchange,
        timezone=record.market_timezone,
        timestamp_utc=timestamp_utc,
        local_timestamp=local,
        session_date=record.session_date,
        trading_session=trading_session if trading_session != HALTED_SESSION else HALTED_SESSION,
        is_trading_day=record.is_trading_day,
        is_early_close=record.early_close,
        regular_open=record.open_timestamp,
        regular_close=record.close_timestamp,
        pre_market_open=record.pre_market_open_timestamp,
        after_hours_close=record.after_hours_close_timestamp,
        halted=False,
        halt_reason="",
        calendar_identity=record.authority_version_identity,
        calendar_authority_version=record.authority_version,
        calendar_authority_version_identity=record.authority_version_identity,
        calendar_source_status=record.source_status,
        calendar_base_status=record.base_status,
        calendar_package=record.package,
        calendar_package_version=record.package_version,
        calendar_schedule_hash=record.schedule_hash,
        calendar_fallback_used=record.fallback_used,
        calendar_closure_reason=record.closure_reason,
    )
    _EXCHANGE_CONTEXT_BY_TIMESTAMP[timestamp_utc] = context
    return context


def _bar_window_for_contract(
    index: Mapping[str, Any],
    *,
    decision_row: Mapping[str, Any],
    target_contract: Any,
    calendar_authority: ExchangeCalendarAuthority,
) -> tuple[dict[str, Any], ...]:
    decision = _coerce_timestamp(decision_row["decision_timestamp"])
    session = _coerce_date(decision_row["decision_session"])
    horizon_unit = str(target_contract.horizon_unit)
    rows_by_final: Mapping[datetime, list[dict[str, Any]]] = index.get("by_final", {})
    rows_by_start: Mapping[datetime, list[dict[str, Any]]] = index.get("by_start", {})
    selected: list[dict[str, Any]] = []
    selected.extend(rows_by_final.get(decision, ()))
    if horizon_unit == "elapsed_market_minutes":
        target_end = decision + timedelta(minutes=int(target_contract.horizon_value or 0))
        current = decision + timedelta(minutes=5)
        while current <= target_end:
            selected.extend(rows_by_final.get(current, ()))
            current += timedelta(minutes=5)
    elif horizon_unit == "to_session_close":
        record = calendar_authority.session(session)
        target_end = record.close_timestamp.astimezone(timezone.utc) if record.close_timestamp else decision
        current = decision + timedelta(minutes=5)
        while current <= target_end:
            selected.extend(rows_by_final.get(current, ()))
            current += timedelta(minutes=5)
    elif horizon_unit == "to_next_session_open":
        next_session = calendar_authority.next_session(session)
        if next_session is None:
            target_end = decision
        else:
            record = calendar_authority.session(next_session)
            target_end = record.open_timestamp.astimezone(timezone.utc) if record.open_timestamp else decision
            selected.extend(rows_by_start.get(target_end, ()))
            selected.extend(rows_by_final.get(target_end + timedelta(minutes=5), ()))
    else:
        target_end = decision
    return tuple(sorted(_dedupe_source_rows(selected), key=_source_row_sort_key))


def _build_bar_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[datetime, list[dict[str, Any]]]]:
    by_final: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    by_start: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        payload = dict(row)
        final = _bar_final_timestamp(payload)
        start = _parse_optional_timestamp(payload.get("timestamp_utc") or payload.get("timestamp"))
        if final is not None:
            by_final[final].append(payload)
        if start is not None:
            by_start[start].append(payload)
    return {"by_final": by_final, "by_start": by_start}


def _dedupe_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("source_bar_id") or row.get("source_row_hash") or _source_key(row))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(row))
    return output


def _canonical_target_row(
    result: Mapping[str, Any],
    *,
    decision_row: Mapping[str, Any],
    source_cutoff: datetime,
    training_cutoff: datetime,
) -> dict[str, Any]:
    source_bar_ids = list(result.get("target_source_bar_ids") or [])
    value = result.get("value")
    target_value = None if value in ("", None) else float(value)
    return {
        "asset_id": str(decision_row["asset_id"]),
        "canonical_symbol": str(decision_row["canonical_symbol"]),
        "decision_timestamp": str(result.get("decision_timestamp") or decision_row["decision_timestamp"]),
        "decision_session": str(decision_row["decision_session"]),
        "decision_source_bar_id": str(decision_row["source_bar_id"]),
        "decision_source_row_hash": str(decision_row["source_row_hash"]),
        "target_id": str(result["target_id"]),
        "target_contract_version": MULTI_TIMEFRAME_TARGET_CONTRACT_VERSION,
        "target_code_hash": str(result.get("target_code_hash") or ""),
        "target_resolution_policy_version": str(result.get("resolution_policy_version") or TARGET_RESOLUTION_POLICY_VERSION),
        "target_start_timestamp": str(result.get("target_start_timestamp") or ""),
        "target_end_timestamp": str(result.get("target_end_timestamp") or ""),
        "target_available_timestamp": str(result.get("target_available_timestamp") or ""),
        "target_value": target_value,
        "target_is_mature": bool(result.get("target_is_mature")),
        "target_is_realised": bool(result.get("target_is_realised")),
        "target_is_trainable": bool(result.get("target_is_trainable")),
        "target_resolution_classification": str(result.get("target_resolution_classification") or ""),
        "target_resolution_reason": str(result.get("target_resolution_reason") or ""),
        "target_entry_bar_id": str(result.get("target_entry_bar_id") or ""),
        "target_exit_bar_id": str(result.get("target_exit_bar_id") or ""),
        "target_source_bar_ids": json.dumps(source_bar_ids, separators=(",", ":"), ensure_ascii=False),
        "target_source_bar_count": len(source_bar_ids),
        "source_cutoff": _format_timestamp(source_cutoff),
        "training_cutoff": _format_timestamp(training_cutoff),
        "calendar_authority_version": str(result.get("calendar_authority_version") or ""),
        "price_authority_version": str(result.get("price_authority_version") or PRICE_AUTHORITY_VERSION),
        "source_dataset_version": str(decision_row.get("dataset_version") or ""),
        "source_partition": str(decision_row.get("source_partition") or ""),
        "source_partition_hash": str(decision_row.get("source_partition_hash") or ""),
    }


def _write_dataset_artifacts(
    *,
    output_root: Path,
    source_root: Path,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    target_ids: Sequence[str],
    source_cutoff: datetime,
    training_cutoff: datetime,
    source_read: SourceReadResult,
    decision_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    calendar_authority: ExchangeCalendarAuthority,
    schema: Mapping[str, Any],
    schema_hash: str,
    content_hash: str,
    run_id: str,
    permitted_use: str,
    max_workers: int,
) -> dict[str, Mapping[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "target_rows.parquet"
    _write_target_rows_parquet(rows_path, target_rows)
    decision_inventory = _decision_row_inventory(decision_rows)
    quality_report = _quality_report(decision_rows, target_rows)
    calendar_identity = calendar_authority.identity(start=start_date, end=end_date)
    availability_validation = validate_target_availability(target_rows)
    pit_validation = {
        "validation_version": "five_minute_target_dataset_pit_validation.v1",
        "authority_validation": availability_validation,
        "pit_violation_count": len(availability_validation.get("violations", [])),
        "passed": not availability_validation.get("violations"),
    }
    catalogue = target_catalogue_payload()
    git = source_worktree_provenance()
    source_manifest = {
        "manifest_version": "five_minute_target_source_manifest.v1",
        "source_root": _rel(source_root),
        "source_partitions": [partition.payload() for partition in source_read.partitions],
        "source_partition_hashes": {
            partition.payload()["path"]: partition.sha256
            for partition in source_read.partitions
            if partition.exists
        },
        "source_row_count": len(source_read.rows),
        "source_rows_loaded_after_filters": len(source_read.rows),
        "filters_applied_before_python_conversion": source_read.filters_applied,
        "whole_archive_scanned": source_read.whole_archive_scanned,
        "decision_sessions": [day.isoformat() for day in source_read.decision_sessions],
        "read_sessions": [day.isoformat() for day in source_read.read_sessions],
    }
    dataset_manifest = {
        "manifest_version": "five_minute_target_dataset_manifest.v1",
        "dataset_id": "canonical_five_minute_targets",
        "dataset_version": FIVE_MINUTE_TARGET_DATASET_VERSION,
        "foundation_version": FIVE_MINUTE_TARGET_DATASET_FOUNDATION_VERSION,
        "run_id": run_id,
        "created_at": _stable_created_at(),
        "permitted_use": permitted_use,
        "source_root": _rel(source_root),
        "source_partitions": source_manifest["source_partitions"],
        "source_partition_hashes": source_manifest["source_partition_hashes"],
        "source_row_count": source_manifest["source_row_count"],
        "decision_row_count": len(decision_rows),
        "eligible_decision_row_count": decision_inventory["classification_counts"].get(ELIGIBLE_DECISION_ROW, 0),
        "target_row_count": len(target_rows),
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "target_ids": list(target_ids),
        "target_catalogue_version": TARGET_CATALOGUE_VERSION,
        "target_catalogue_hash": catalogue["content_hash"],
        "target_code_hashes": _target_code_hashes(target_ids),
        "calendar_authority": calendar_identity,
        "source_cutoff": _format_timestamp(source_cutoff),
        "training_cutoff": _format_timestamp(training_cutoff),
        "schema_hash": schema_hash,
        "content_hash": content_hash,
        "git": git,
        "dirty_tree_or_patch_provenance": git,
        "classification_counts": quality_report["target_resolution_classification_counts"],
        "maturity_counts": quality_report["maturity_counts"],
        "trainability_counts": quality_report["trainability_counts"],
        "decision_classification_counts": decision_inventory["classification_counts"],
        "duplicate_counts": decision_inventory["duplicate_counts"],
        "pit_violations": pit_validation["pit_violation_count"],
        "model_training_performed": False,
        "portfolio_replay_performed": False,
        "promotion_performed": False,
        "max_workers": max_workers,
    }
    target_manifest = build_target_manifest(
        target_rows,
        selected_target=target_ids[0],
        source_cutoff=source_cutoff,
        output_paths=(rows_path,),
        source_paths=[partition.path for partition in source_read.partitions if partition.exists],
        configuration={
            "foundation_version": FIVE_MINUTE_TARGET_DATASET_FOUNDATION_VERSION,
            "symbols": list(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "target_ids": list(target_ids),
        },
        calendar_identity=calendar_identity,
        producer_command="python scripts/ticket_71b_five_minute_target_dataset.py",
        producer_module="core.research.ml.five_minute_target_dataset",
    )
    checksums = _checksums(output_root, extra={"content_hash": content_hash, "schema_hash": schema_hash})
    write_json(output_root / "source_manifest.json", source_manifest)
    write_json(output_root / "dataset_manifest.json", dataset_manifest)
    write_json(output_root / "target_manifest.json", target_manifest)
    write_json(output_root / "schema.json", schema)
    write_json(output_root / "quality_report.json", quality_report)
    write_json(output_root / "pit_validation.json", pit_validation)
    write_json(output_root / "decision_row_inventory.json", decision_inventory)
    write_missing_bar_analysis_parquet(
        output_root / "missing_bar_analysis.parquet",
        _missing_bar_analysis(target_rows),
    )
    write_csv(output_root / "coverage_by_target.csv", _coverage_by_target(target_rows), _coverage_target_fields())
    write_csv(output_root / "coverage_by_symbol.csv", _coverage_by_symbol(target_rows), _coverage_symbol_fields())
    write_csv(output_root / "coverage_by_date.csv", _coverage_by_date(target_rows), _coverage_date_fields())
    checksums = _checksums(output_root, extra={"content_hash": content_hash, "schema_hash": schema_hash})
    write_json(output_root / "checksums.json", checksums)
    return {
        "dataset_manifest": dataset_manifest,
        "quality_report": quality_report,
        "source_manifest": source_manifest,
        "pit_validation": pit_validation,
    }


def _decision_row_inventory(decision_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classifications = Counter(str(row.get("decision_row_classification") or "") for row in decision_rows)
    reasons = Counter(str(row.get("decision_row_reason") or "") for row in decision_rows)
    duplicate_rows = [
        row for row in decision_rows
        if row.get("decision_row_classification") == EXCLUDED_DUPLICATE_SOURCE_BAR
    ]
    return {
        "inventory_version": "five_minute_decision_row_inventory.v1",
        "decision_row_count": len(decision_rows),
        "classification_counts": dict(sorted(classifications.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "duplicate_counts": {
            "duplicate_source_bar_rows": len(duplicate_rows),
            "duplicate_source_bar_keys": len({str(row.get("duplicate_key") or "") for row in duplicate_rows}),
        },
        "opening_session_handling": "first regular-session source bar is eligible by default; no warmup exclusion is applied in the foundation",
        "non_regular_handling": "extended-hours rows are classified and excluded from regular-session decision rows",
        "sample_exclusions": [
            {
                "canonical_symbol": row.get("canonical_symbol"),
                "timestamp_utc": row.get("timestamp_utc"),
                "classification": row.get("decision_row_classification"),
                "reason": row.get("decision_row_reason"),
            }
            for row in decision_rows
            if row.get("decision_row_classification") != ELIGIBLE_DECISION_ROW
        ][:25],
    }


def _quality_report(
    decision_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    classifications = Counter(str(row.get("target_resolution_classification") or "") for row in target_rows)
    reasons = Counter(str(row.get("target_resolution_reason") or "") for row in target_rows)
    mature = sum(bool(row.get("target_is_mature")) for row in target_rows)
    realised = sum(bool(row.get("target_is_realised")) for row in target_rows)
    trainable = sum(bool(row.get("target_is_trainable")) for row in target_rows)
    return {
        "quality_report_version": "five_minute_target_quality_report.v1",
        "decision_row_count": len(decision_rows),
        "target_row_count": len(target_rows),
        "target_resolution_classification_counts": dict(sorted(classifications.items())),
        "target_resolution_reason_counts": dict(sorted(reasons.items())),
        "maturity_counts": {"mature": mature, "not_mature": len(target_rows) - mature},
        "realisation_counts": {"realised": realised, "not_realised": len(target_rows) - realised},
        "trainability_counts": {"trainable": trainable, "not_trainable": len(target_rows) - trainable},
        "model_training_performed": False,
        "portfolio_replay_performed": False,
        "promotion_performed": False,
    }


def _classify_source_row_for_decision(
    row: Mapping[str, Any],
    *,
    start_date: date,
    end_date: date,
    source_cutoff: datetime,
    duplicate_keys: set[str],
) -> dict[str, Any]:
    output = dict(row)
    timestamp = _parse_optional_timestamp(row.get("timestamp_utc"))
    if timestamp is None:
        return _decision_payload(output, EXCLUDED_INVALID_TIMESTAMP, "timestamp_utc_parse_failed")
    final = _bar_final_timestamp(row)
    if final is None:
        return _decision_payload(output, EXCLUDED_INVALID_TIMESTAMP, "bar_final_timestamp_parse_failed")
    session_date = _parse_optional_date(row.get("session_date")) or final.date()
    output["decision_timestamp"] = _format_timestamp(final)
    output["decision_session"] = session_date.isoformat()
    source_key = _source_key(row)
    output["duplicate_key"] = source_key
    if source_key in duplicate_keys:
        return _decision_payload(output, EXCLUDED_DUPLICATE_SOURCE_BAR, "duplicate_asset_timestamp_key")
    if session_date < start_date or session_date > end_date:
        return _decision_payload(output, EXCLUDED_TARGET_NOT_APPLICABLE, "outside_requested_decision_range")
    if not str(row.get("asset_id") or "").strip() or not str(row.get("canonical_symbol") or "").strip():
        return _decision_payload(output, EXCLUDED_UNRESOLVED_IDENTITY, "asset_or_symbol_missing")
    if not str(row.get("source_bar_id") or row.get("source_row_hash") or "").strip():
        return _decision_payload(output, EXCLUDED_UNRESOLVED_IDENTITY, "stable_source_identity_missing")
    if str(row.get("session_type") or "").strip().lower() not in {"rth", "regular"}:
        return _decision_payload(output, EXCLUDED_NON_REGULAR_SESSION, "session_type_not_regular")
    if not _valid_ohlc(row):
        return _decision_payload(output, EXCLUDED_INVALID_PRICE, "ohlc_not_finite_positive_or_consistent")
    if final > source_cutoff:
        return _decision_payload(output, EXCLUDED_SOURCE_BAR_NOT_FINAL, "bar_final_after_source_cutoff")
    return _decision_payload(output, ELIGIBLE_DECISION_ROW, "regular_finalised_source_bar")


def _decision_payload(row: Mapping[str, Any], classification: str, reason: str) -> dict[str, Any]:
    if classification not in DECISION_ROW_CLASSIFICATIONS:
        raise FiveMinuteTargetDatasetError(f"unsupported decision-row classification: {classification}")
    output = dict(row)
    output["decision_row_classification"] = classification
    output["decision_row_reason"] = reason
    output["source_bar_id"] = str(output.get("source_bar_id") or output.get("source_row_hash") or "")
    output["source_row_hash"] = str(output.get("source_row_hash") or output.get("source_bar_id") or "")
    return output


def _normalise_source_row(
    row: Mapping[str, Any],
    *,
    symbol: str,
    source_partition: Path,
    source_partition_hash: str,
) -> dict[str, Any]:
    result = dict(row)
    timestamp = result.get("timestamp_utc")
    if isinstance(timestamp, datetime):
        result["timestamp_utc"] = _format_timestamp(timestamp)
    result["canonical_symbol"] = str(result.get("canonical_symbol") or symbol).upper()
    result["asset_id"] = str(result.get("asset_id") or result["canonical_symbol"])
    result["timeframe"] = str(result.get("timeframe") or "5m")
    result["source_row_hash"] = str(result.get("source_row_hash") or "")
    result["source_bar_id"] = str(
        result.get("source_bar_id")
        or result.get("source_row_hash")
        or f"{result['asset_id']}|5m|{result.get('timestamp_utc')}"
    )
    result["source_partition"] = _rel(source_partition)
    result["source_partition_hash"] = source_partition_hash
    return result


def _normalise_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    output = tuple(sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}))
    if not output:
        raise FiveMinuteTargetDatasetError("at least one symbol is required")
    return output


def _normalise_target_ids(target_ids: Sequence[str]) -> tuple[str, ...]:
    output = tuple(str(target_id).strip() for target_id in target_ids if str(target_id).strip())
    if not output:
        raise FiveMinuteTargetDatasetError("at least one target_id is required")
    for target_id in output:
        contract = resolve_target_contract(target_id, allow_legacy_aliases=False)
        if contract.decision_timeframe != "5m" or contract.source_bar_timeframe != "5m":
            raise FiveMinuteTargetDatasetError(f"target_id is not a canonical five-minute contract: {target_id}")
    return output


def _validate_resource_limits(
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    max_workers: int,
    protected_process_active: bool,
    max_real_symbols: int | None,
    max_real_sessions_per_symbol: int | None,
    allow_large_build: bool,
) -> None:
    if max_workers != 1 and protected_process_active:
        raise FiveMinuteTargetDatasetError("protected Ticket 57 process requires max_workers=1")
    if max_workers < 1:
        raise FiveMinuteTargetDatasetError("max_workers must be >= 1")
    if end_date < start_date:
        raise FiveMinuteTargetDatasetError("end_date must be on or after start_date")
    session_count = len(default_calendar_authority().sessions(start_date, end_date))
    if protected_process_active:
        if max_real_symbols is not None and len(symbols) > max_real_symbols:
            raise FiveMinuteTargetDatasetError("protected Ticket 57 process limits real symbols")
        if max_real_sessions_per_symbol is not None and session_count > max_real_sessions_per_symbol:
            raise FiveMinuteTargetDatasetError("protected Ticket 57 process limits real sessions per symbol")
    if not allow_large_build and (len(symbols) > 50 or session_count > 65):
        raise FiveMinuteTargetDatasetError(
            "large five-minute target builds require allow_large_build=True in a later ticket"
        )


def _read_sessions_for_targets(
    *,
    authority: ExchangeCalendarAuthority,
    decision_sessions: Sequence[date],
    target_ids: Sequence[str],
) -> tuple[date, ...]:
    sessions = set(decision_sessions)
    includes_next_open = any(
        resolve_target_contract(target_id).horizon_unit == "to_next_session_open"
        for target_id in target_ids
    )
    if includes_next_open:
        for day in decision_sessions:
            next_day = authority.next_session(day)
            if next_day is not None:
                sessions.add(next_day)
    return tuple(sorted(sessions))


def _target_code_hashes(target_ids: Sequence[str]) -> dict[str, str]:
    return {
        target_id: resolve_target_contract(target_id).target_code_hash
        for target_id in target_ids
    }


def _write_target_rows_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = [{field: row.get(field) for field in TARGET_ROW_FIELDS} for row in rows]
    table = pa.Table.from_pylist(payload, schema=_arrow_schema())
    pq.write_table(table, path, compression="zstd")


def _arrow_schema() -> pa.Schema:
    fields = []
    for field in TARGET_ROW_FIELDS:
        if field in {"target_is_mature", "target_is_realised", "target_is_trainable"}:
            fields.append(pa.field(field, pa.bool_()))
        elif field == "target_value":
            fields.append(pa.field(field, pa.float64()))
        elif field == "target_source_bar_count":
            fields.append(pa.field(field, pa.int64()))
        else:
            fields.append(pa.field(field, pa.string()))
    return pa.schema(fields)


def _coverage_by_target(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["target_id"])].append(row)
    for target_id, target_rows in sorted(by_target.items()):
        counts = Counter(str(row.get("target_resolution_classification") or "") for row in target_rows)
        output.append(
            {
                "target_id": target_id,
                "target_rows": len(target_rows),
                "matured_rows": sum(bool(row.get("target_is_mature")) for row in target_rows),
                "realised_rows": sum(bool(row.get("target_is_realised")) for row in target_rows),
                "trainable_rows": sum(bool(row.get("target_is_trainable")) for row in target_rows),
                "classification_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            }
        )
    return output


def _coverage_by_symbol(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row["canonical_symbol"])].append(row)
    for symbol, symbol_rows in sorted(by_symbol.items()):
        output.append(
            {
                "canonical_symbol": symbol,
                "target_rows": len(symbol_rows),
                "decision_count": len({str(row["decision_timestamp"]) for row in symbol_rows}),
                "trainable_rows": sum(bool(row.get("target_is_trainable")) for row in symbol_rows),
                "targets": ";".join(sorted({str(row["target_id"]) for row in symbol_rows})),
            }
        )
    return output


def _coverage_by_date(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["decision_session"])].append(row)
    for session, date_rows in sorted(by_date.items()):
        output.append(
            {
                "decision_session": session,
                "target_rows": len(date_rows),
                "symbol_count": len({str(row["canonical_symbol"]) for row in date_rows}),
                "trainable_rows": sum(bool(row.get("target_is_trainable")) for row in date_rows),
            }
        )
    return output


def _missing_bar_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        classification = str(row.get("target_resolution_classification") or "")
        if classification == MATURED_VALID:
            continue
        output.append(
            {
                "canonical_symbol": row.get("canonical_symbol"),
                "target_id": row.get("target_id"),
                "decision_timestamp": row.get("decision_timestamp"),
                "target_resolution_classification": classification,
                "target_resolution_reason": row.get("target_resolution_reason"),
                "target_source_bar_count": row.get("target_source_bar_count"),
            }
        )
    return output


def _coverage_target_fields() -> tuple[str, ...]:
    return ("target_id", "target_rows", "matured_rows", "realised_rows", "trainable_rows", "classification_counts")


def _coverage_symbol_fields() -> tuple[str, ...]:
    return ("canonical_symbol", "target_rows", "decision_count", "trainable_rows", "targets")


def _coverage_date_fields() -> tuple[str, ...]:
    return ("decision_session", "target_rows", "symbol_count", "trainable_rows")


def _missing_bar_fields() -> tuple[str, ...]:
    return (
        "canonical_symbol",
        "target_id",
        "decision_timestamp",
        "target_resolution_classification",
        "target_resolution_reason",
        "target_source_bar_count",
    )


def _checksums(output_root: Path, *, extra: Mapping[str, str]) -> dict[str, Any]:
    files = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "checksums.json":
            files[path.name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return {
        "checksums_version": "five_minute_target_dataset_checksums.v1",
        "files": files,
        **dict(extra),
    }


def _publish_atomic(temp_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FiveMinuteTargetDatasetError(f"output_root already exists: {output_root}")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_root, output_root)


def _duplicate_source_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    counts = Counter(_source_key(row) for row in rows)
    return {key for key, count in counts.items() if key and count > 1}


def _source_key(row: Mapping[str, Any]) -> str:
    symbol = str(row.get("canonical_symbol") or row.get("asset_id") or "").upper()
    timestamp = str(row.get("timestamp_utc") or "")
    return f"{symbol}|{timestamp}"


def _source_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("canonical_symbol") or row.get("asset_id") or ""),
        str(row.get("timestamp_utc") or ""),
        str(row.get("source_bar_id") or row.get("source_row_hash") or ""),
    )


def _decision_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("canonical_symbol") or ""),
        str(row.get("decision_timestamp") or ""),
        str(row.get("source_bar_id") or ""),
    )


def _target_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("canonical_symbol") or ""),
        str(row.get("decision_timestamp") or ""),
        str(row.get("target_id") or ""),
    )


def _valid_ohlc(row: Mapping[str, Any]) -> bool:
    try:
        open_price = float(row.get("open"))
        high = float(row.get("high"))
        low = float(row.get("low"))
        close = float(row.get("close"))
    except (TypeError, ValueError):
        return False
    values = (open_price, high, low, close)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return False
    return high >= low and high >= max(open_price, close) and low <= min(open_price, close)


def _bar_final_timestamp(row: Mapping[str, Any]) -> datetime | None:
    explicit = _parse_optional_timestamp(row.get("bar_finalized_timestamp") or row.get("bar_end_timestamp"))
    if explicit is not None:
        return explicit
    start = _parse_optional_timestamp(row.get("timestamp_utc") or row.get("timestamp"))
    if start is None:
        return None
    return start + timedelta(minutes=5)


def _max_bar_final_timestamp(rows: Sequence[Mapping[str, Any]]) -> datetime | None:
    values = [value for value in (_bar_final_timestamp(row) for row in rows) if value is not None]
    return max(values) if values else None


def _session_end(authority: ExchangeCalendarAuthority, day: date) -> datetime:
    record = authority.session(day)
    if record.close_timestamp is not None:
        return record.close_timestamp.astimezone(timezone.utc)
    return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)


def _parse_optional_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _coerce_timestamp(value)
    except (TypeError, ValueError):
        return None


def _coerce_timestamp(value: datetime | date | str | Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_optional_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return _coerce_date(value)
    except (TypeError, ValueError):
        return None


def _coerce_date(value: date | str | Any) -> date:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _stable_created_at() -> str:
    return "DEVELOPMENT_FIXTURE_STABLE_TIMESTAMP"


def _rel(path: Path) -> str:
    root = Path.cwd().resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _git(*args: str) -> str:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return cp.stdout.strip() if cp.returncode == 0 else ""


def _run(cmd: Sequence[str]) -> str:
    try:
        cp = subprocess.run(
            list(cmd),
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return cp.stdout.strip() if cp.returncode == 0 else ""
