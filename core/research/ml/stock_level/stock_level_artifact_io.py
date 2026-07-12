from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from core.research.framework.data import CsvRowRepository


CANONICAL_ARTIFACT_FORMAT = "parquet"
DEFAULT_PARQUET_COMPRESSION = "zstd"
TIMESTAMP_COLUMNS = {
    "feature_timestamp",
    "feature_data_cutoff_timestamp",
    "decision_timestamp",
    "target_start_timestamp",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_timestamp",
    "benchmark_target_start_timestamp",
    "benchmark_label_start_timestamp",
    "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp",
}


def stock_level_artifact_format(config: Mapping[str, Any]) -> str:
    ml = dict(config.get("ml", {}) or {})
    return str(ml.get("stock_level_artifact_format", CANONICAL_ARTIFACT_FORMAT)).lower()


def stock_level_parquet_compression(config: Mapping[str, Any]) -> str:
    ml = dict(config.get("ml", {}) or {})
    return str(ml.get("stock_level_parquet_compression", DEFAULT_PARQUET_COMPRESSION)).lower()


def canonical_artifact_path(output_dir: Path, stem: str, config: Mapping[str, Any]) -> Path:
    fmt = stock_level_artifact_format(config)
    if fmt == "parquet":
        return output_dir / f"{stem}.parquet"
    if fmt == "csv":
        return output_dir / f"{stem}.csv"
    raise ValueError("ml.stock_level_artifact_format must be parquet or csv")


def write_stock_level_artifact(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    config: Mapping[str, Any],
    inspection_sample_path: Path | None = None,
    inspection_sample_max_rows: int = 100,
    phase_timings: list[dict[str, Any]] | None = None,
    write_phase_name: str | None = None,
    validation_phase_name: str | None = None,
    hash_phase_name: str | None = None,
) -> dict[str, Any]:
    materialized = [
        {name: row.get(name) for name in fieldnames}
        for row in rows
    ]
    fmt = stock_level_artifact_format(config)
    if fmt == "parquet":
        compression = stock_level_parquet_compression(config)
        identity_rows = [_normalize_row(row, fieldnames) for row in materialized]
        _atomic_write_parquet(
            path,
            materialized,
            fieldnames=fieldnames,
            compression=compression,
            phase_timings=phase_timings,
            write_phase_name=write_phase_name,
            validation_phase_name=validation_phase_name,
        )
        _remove_legacy_full_csv(path, inspection_sample_path)
        sample_payload = None
        if inspection_sample_path is not None:
            sample_payload = write_inspection_sample(
                inspection_sample_path,
                materialized[:inspection_sample_max_rows],
                fieldnames=fieldnames,
            )
        return artifact_identity(
            path,
            rows=identity_rows,
            fieldnames=fieldnames,
            artifact_format="parquet",
            compression=compression,
            inspection_sample=sample_payload,
            phase_timings=phase_timings,
            hash_phase_name=hash_phase_name,
        )
    if fmt == "csv":
        if write_phase_name and phase_timings is not None:
            started, start_ts = _phase_start()
            _atomic_write_csv(path, materialized, fieldnames=fieldnames)
            _record_phase(phase_timings, write_phase_name, started, start_ts)
        else:
            _atomic_write_csv(path, materialized, fieldnames=fieldnames)
        return artifact_identity(
            path,
            rows=materialized,
            fieldnames=fieldnames,
            artifact_format="csv",
            compression=None,
            inspection_sample=None,
            phase_timings=phase_timings,
            hash_phase_name=hash_phase_name,
        )
    raise ValueError("ml.stock_level_artifact_format must be parquet or csv")


def read_stock_level_artifact(
    path: Path,
    *,
    expected_format: str = CANONICAL_ARTIFACT_FORMAT,
    required_columns: set[str] | None = None,
    expected_schema_fingerprint: str | None = None,
    allow_csv_fallback: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if expected_format != "parquet":
            raise ValueError(f"Expected {expected_format}, got parquet: {path}")
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise ValueError(f"Could not read Parquet artifact {path}: {exc}") from exc
        schema_fp = schema_fingerprint(table.schema.names, table.schema)
        if expected_schema_fingerprint and schema_fp != expected_schema_fingerprint:
            raise ValueError(
                f"Schema fingerprint mismatch for {path}: "
                f"{schema_fp} != {expected_schema_fingerprint}"
            )
        columns = set(table.schema.names)
        rows = table.to_pylist()
    elif suffix == ".csv" and allow_csv_fallback:
        rows = CsvRowRepository().read(path)
        if rows:
            columns = set(rows[0])
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                columns = set(next(csv.reader(handle), []))
    else:
        raise ValueError(
            f"Refusing to read non-canonical stock-level artifact {path}; "
            "set allow_csv_fallback=True only for explicit legacy fixtures."
        )
    missing = sorted((required_columns or set()) - columns)
    if missing:
        raise ValueError(f"Stock-level artifact {path} missing required columns: {missing}")
    return rows


def artifact_identity(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    fieldnames: Sequence[str] | None = None,
    artifact_format: str | None = None,
    compression: str | None = None,
    inspection_sample: Mapping[str, Any] | None = None,
    phase_timings: list[dict[str, Any]] | None = None,
    hash_phase_name: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        column_order = list(schema.names)
        row_count = parquet.metadata.num_rows
        compression_codecs = sorted({
            str(parquet.metadata.row_group(group).column(column).compression)
            for group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.row_group(group).num_columns)
        })
        schema_fp = schema_fingerprint(column_order, schema)
        parsed_rows = rows
    else:
        column_order = list(fieldnames or (rows[0].keys() if rows else []))
        row_count = len(rows or [])
        compression_codecs = []
        schema_fp = schema_fingerprint(column_order, None)
        parsed_rows = rows
    parsed_rows = list(parsed_rows or [])
    decision_values = [
        row.get("decision_timestamp") or row.get("rebalance_date")
        for row in parsed_rows
        if row.get("decision_timestamp") or row.get("rebalance_date")
    ]
    realized = [
        row for row in parsed_rows
        if row.get("actual_forward_return_10d") not in (None, "")
    ]
    target_contract_versions = sorted({
        str(row.get("target_provenance_contract_version"))
        for row in parsed_rows
        if row.get("target_provenance_contract_version") not in (None, "")
    })
    source_dataset_hashes = sorted({
        str(row.get("source_dataset_hash"))
        for row in parsed_rows
        if row.get("source_dataset_hash") not in (None, "")
    })
    if hash_phase_name and phase_timings is not None:
        hash_started, hash_start_ts = _phase_start()
        logical_hash = logical_content_sha256(parsed_rows, column_order)
        _record_phase(phase_timings, hash_phase_name, hash_started, hash_start_ts)
    else:
        logical_hash = logical_content_sha256(parsed_rows, column_order)
    identity = {
        "artifact_format": artifact_format or path.suffix.lower().lstrip("."),
        "compression": compression,
        "compression_codecs": compression_codecs,
        "resolved_artifact_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "logical_content_sha256": logical_hash,
        "schema_fingerprint": schema_fp,
        "stable_column_order": column_order,
        "row_count": row_count,
        "column_count": len(column_order),
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in parsed_rows if row.get("symbol")}),
        "decision_date_count": len({str(value)[:10] for value in decision_values}),
        "minimum_decision_timestamp": min((str(value) for value in decision_values), default=None),
        "maximum_decision_timestamp": max((str(value) for value in decision_values), default=None),
        "realized_target_count": len(realized),
        "unrealized_boundary_count": row_count - len(realized),
        "duplicate_symbol_decision_keys": _duplicate_symbol_decision_count(parsed_rows),
        "null_symbol_count": sum(1 for row in parsed_rows if not row.get("symbol")),
        "null_decision_timestamp_count": sum(1 for row in parsed_rows if not (row.get("decision_timestamp") or row.get("rebalance_date"))),
        "target_contract_version": target_contract_versions[0] if len(target_contract_versions) == 1 else None,
        "target_contract_versions": target_contract_versions,
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "source_dataset_hash_count": len(source_dataset_hashes),
        "source_dataset_hashes": source_dataset_hashes[:10],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_status": "complete",
    }
    if inspection_sample is not None:
        identity["inspection_sample"] = dict(inspection_sample)
    return identity


def schema_fingerprint(column_order: Sequence[str], schema: pa.Schema | None) -> str:
    payload = {
        "columns": list(column_order),
        "types": {
            name: str(schema.field(name).type) if schema is not None and name in schema.names else "csv_string"
            for name in column_order
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_content_sha256(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    payload = [
        {
            name: _canonical_json_value(row.get(name))
            for name in fieldnames
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value == "":
        return ""
    return value


def write_inspection_sample(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    _atomic_write_csv(path, rows, fieldnames=fieldnames)
    return {
        "path": str(path),
        "row_count": len(rows),
        "inspection_only": True,
        "not_model_input": True,
        "canonical": False,
        "sha256": file_sha256(path),
    }


def _remove_legacy_full_csv(path: Path, inspection_sample_path: Path | None) -> None:
    legacy_csv_path = path.with_suffix(".csv")
    if inspection_sample_path is not None and legacy_csv_path == inspection_sample_path:
        return
    if legacy_csv_path.exists():
        legacy_csv_path.unlink()


def _atomic_write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    compression: str,
    phase_timings: list[dict[str, Any]] | None = None,
    write_phase_name: str | None = None,
    validation_phase_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    normalized = [_normalize_row(row, fieldnames) for row in rows]
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table({name: pa.array([]) for name in fieldnames})
    if fieldnames:
        table = table.select([name for name in fieldnames if name in table.schema.names])
    if write_phase_name and phase_timings is not None:
        started, start_ts = _phase_start()
        pq.write_table(table, tmp, compression=compression)
        _record_phase(phase_timings, write_phase_name, started, start_ts)
    else:
        pq.write_table(table, tmp, compression=compression)
    if validation_phase_name and phase_timings is not None:
        started, start_ts = _phase_start()
        _validate_complete_parquet(tmp)
        _record_phase(phase_timings, validation_phase_name, started, start_ts)
    else:
        _validate_complete_parquet(tmp)
    os.replace(tmp, path)


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _normalize_row(row: Mapping[str, Any], fieldnames: Sequence[str]) -> dict[str, Any]:
    output = {}
    for name in fieldnames:
        value = row.get(name)
        if name in TIMESTAMP_COLUMNS and value not in (None, ""):
            output[name] = _parse_utc_datetime(value)
        else:
            output[name] = None if value == "" else value
    return output


def _parse_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_complete_parquet(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < 0:
        raise ValueError("Invalid Parquet row count")


def _duplicate_symbol_decision_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [
        (str(row.get("symbol", "")).upper(), str(row.get("decision_timestamp") or row.get("rebalance_date") or ""))
        for row in rows
        if row.get("symbol") and (row.get("decision_timestamp") or row.get("rebalance_date"))
    ]
    return len(keys) - len(set(keys))


def _phase_start() -> tuple[float, str]:
    return time.perf_counter(), datetime.now(timezone.utc).isoformat()


def _record_phase(
    timings: list[dict[str, Any]],
    phase_name: str,
    started: float,
    start_timestamp: str,
    *,
    requested_workers: int = 1,
    effective_workers: int = 1,
    task_count: int | None = None,
    execution_mode: str = "serial",
) -> None:
    timings.append(
        {
            "phase_name": phase_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": max(0.0, time.perf_counter() - started),
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "task_count": task_count,
            "execution_mode": execution_mode,
        }
    )
