from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CONTRACT_VERSION = "stock_artifact_target_provenance_repair_v1"
TARGET_PROVENANCE_V2 = "stock_level_target_provenance_v2"
PROVENANCE_COLUMN = "target_provenance_contract_version"
STATUS_COLUMN = "target_status"
ECONOMIC_KEY_COLUMNS = ("rebalance_date", "symbol")
APPROVED_BLANK_STATUSES = frozenset({"unrealized_boundary"})
REQUIRED_COLUMNS = {
    PROVENANCE_COLUMN,
    STATUS_COLUMN,
    *ECONOMIC_KEY_COLUMNS,
}


class RepairBlockedError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("target-provenance repair blocked: " + ", ".join(report["blockers"]))


def repair_stock_artifact_target_provenance_v2(
    *,
    input_path: Path,
    output_path: Path | None,
    report_root: Path,
    dry_run: bool,
    promote: bool,
    batch_rows: int,
    writer_factory: type[pq.ParquetWriter] = pq.ParquetWriter,
) -> dict[str, Any]:
    if batch_rows < 1:
        raise ValueError("batch_rows must be at least one")
    input_path = Path(input_path)
    report_root = Path(report_root)
    report_path = report_root / "target_provenance_repair_report.json"
    source_checksum = _file_sha256(input_path)
    source = _scan_projected(input_path, batch_rows=batch_rows)
    blockers = _source_blockers(source)
    source_invariants = (
        _invariant_fingerprint(input_path, batch_rows=batch_rows)
        if not blockers
        else None
    )
    report = _base_report(
        input_path=input_path,
        output_path=output_path,
        source_checksum=source_checksum,
        source=source,
        source_invariants=source_invariants,
        dry_run=dry_run,
        promote=promote,
        batch_rows=batch_rows,
        blockers=blockers,
    )
    if blockers:
        report["status"] = "BLOCKED"
        _write_json_atomic(report_path, report)
        raise RepairBlockedError(report)

    effective_dry_run = bool(dry_run or (output_path is None and not promote))
    if effective_dry_run:
        report.update(
            {
                "status": "DRY_RUN_COMPLETE",
                "promotion_status": "not_requested",
                "repaired_checksum": None,
                "distinct_values_after": source["distinct_provenance_values"],
                "approved_rows_repaired": 0,
                "row_count_match": True,
                "economic_key_match": True,
                "invariant_column_checksum_match": True,
            }
        )
        _write_json_atomic(report_path, report)
        return report

    repair_output = Path(output_path) if output_path is not None else input_path.with_name(
        f"{input_path.stem}.target_provenance_v2_repaired{input_path.suffix}"
    )
    if repair_output.resolve() == input_path.resolve():
        raise ValueError("output must be side-by-side; use --promote for canonical replacement")
    temporary = repair_output.with_name(
        f".{repair_output.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_repaired(
            input_path,
            temporary,
            batch_rows=batch_rows,
            writer_factory=writer_factory,
        )
        repaired = _scan_projected(temporary, batch_rows=batch_rows)
        repaired_invariants = _invariant_fingerprint(
            temporary, batch_rows=batch_rows
        )
        validation_blockers = _repaired_blockers(
            source,
            repaired,
            source_invariants=source_invariants or {},
            repaired_invariants=repaired_invariants,
        )
        if validation_blockers:
            report["blockers"] = validation_blockers
            report["status"] = "BLOCKED"
            _write_json_atomic(report_path, report)
            raise RepairBlockedError(report)
        repaired_checksum = _file_sha256(temporary)
        os.replace(temporary, repair_output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    promoted_at: str | None = None
    publication_path = repair_output
    if promote:
        os.replace(repair_output, input_path)
        publication_path = input_path
        promoted_at = datetime.now(timezone.utc).isoformat()

    report.update(
        {
            "status": "COMPLETE",
            "output_path": str(publication_path),
            "repaired_checksum": repaired_checksum,
            "total_rows": repaired["row_count"],
            "approved_rows_repaired": source["blank_or_null_rows"],
            "unapproved_blank_rows": 0,
            "distinct_values_after": repaired["distinct_provenance_values"],
            "target_status_population_after": repaired["target_status_population"],
            "row_count_match": source["row_count"] == repaired["row_count"],
            "economic_key_match": (
                source["economic_key_order_sha256"]
                == repaired["economic_key_order_sha256"]
            ),
            "invariant_column_checksum_match": (
                source_invariants["sha256"] == repaired_invariants["sha256"]
            ),
            "schema_match": source_invariants["schema"] == repaired_invariants["schema"],
            "column_order_match": (
                source_invariants["column_order"]
                == repaired_invariants["column_order"]
            ),
            "null_populations_outside_provenance_match": (
                source_invariants["null_counts"]
                == repaired_invariants["null_counts"]
            ),
            "promotion_status": "promoted" if promote else "side_by_side",
            "promoted_at": promoted_at,
            "blockers": [],
        }
    )
    _write_json_atomic(report_path, report)
    return report


def _scan_projected(path: Path, *, batch_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise ValueError(f"Parquet metadata is unreadable: {path}: {exc}") from exc
    missing = sorted(REQUIRED_COLUMNS - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"source artifact is missing required columns: {missing}")
    if not (
        pa.types.is_string(parquet.schema_arrow.field(PROVENANCE_COLUMN).type)
        or pa.types.is_large_string(parquet.schema_arrow.field(PROVENANCE_COLUMN).type)
    ):
        raise ValueError("target provenance column must be string-typed")

    provenance = Counter()
    statuses = Counter()
    blank_statuses = Counter()
    row_count = 0
    key_hasher = hashlib.sha256()
    duplicate_keys = 0
    sqlite_path: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        handle = tempfile.NamedTemporaryFile(
            prefix="target-provenance-keys-", suffix=".sqlite", delete=False
        )
        sqlite_path = Path(handle.name)
        handle.close()
        connection = sqlite3.connect(sqlite_path)
        connection.execute(
            "CREATE TABLE economic_keys (rebalance_date TEXT, symbol TEXT, "
            "PRIMARY KEY (rebalance_date, symbol)) WITHOUT ROWID"
        )
        for batch in parquet.iter_batches(
            batch_size=batch_rows,
            columns=[*ECONOMIC_KEY_COLUMNS, PROVENANCE_COLUMN, STATUS_COLUMN],
        ):
            dates = batch.column(0)
            symbols = batch.column(1)
            versions = batch.column(2)
            target_statuses = batch.column(3)
            for index in range(batch.num_rows):
                date_value = str(dates[index].as_py() or "")[:10]
                symbol_value = str(symbols[index].as_py() or "").upper()
                version = versions[index].as_py()
                status = str(target_statuses[index].as_py() or "")
                blank = version is None or not str(version).strip()
                normalized_version = "<blank_or_null>" if blank else str(version)
                provenance[normalized_version] += 1
                statuses[status] += 1
                if blank:
                    blank_statuses[status] += 1
                key_hasher.update(
                    f"{date_value}\x1f{symbol_value}\n".encode("utf-8")
                )
                try:
                    connection.execute(
                        "INSERT INTO economic_keys VALUES (?, ?)",
                        (date_value, symbol_value),
                    )
                except sqlite3.IntegrityError:
                    duplicate_keys += 1
                row_count += 1
            connection.commit()
    finally:
        if connection is not None:
            connection.close()
        if sqlite_path is not None and sqlite_path.exists():
            sqlite_path.unlink()
    return {
        "row_count": row_count,
        "metadata_row_count": parquet.metadata.num_rows,
        "column_order": list(parquet.schema_arrow.names),
        "schema": str(parquet.schema_arrow),
        "distinct_provenance_values": sorted(provenance),
        "provenance_population": dict(sorted(provenance.items())),
        "existing_v2_rows": provenance[TARGET_PROVENANCE_V2],
        "blank_or_null_rows": provenance["<blank_or_null>"],
        "target_status_population": dict(sorted(statuses.items())),
        "blank_status_population": dict(sorted(blank_statuses.items())),
        "approved_boundary_population": statuses["unrealized_boundary"],
        "unapproved_blank_rows": sum(
            count
            for status, count in blank_statuses.items()
            if status not in APPROVED_BLANK_STATUSES
        ),
        "duplicate_economic_keys": duplicate_keys,
        "economic_key_order_sha256": key_hasher.hexdigest(),
    }


def _source_blockers(scan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    nonblank = set(scan["distinct_provenance_values"]) - {"<blank_or_null>"}
    if nonblank - {TARGET_PROVENANCE_V2}:
        blockers.append("NON_V2_PROVENANCE_PRESENT")
    if scan["row_count"] != scan["metadata_row_count"]:
        blockers.append("ROW_COUNT_METADATA_MISMATCH")
    if scan["duplicate_economic_keys"]:
        blockers.append("DUPLICATE_ECONOMIC_KEYS")
    if scan["unapproved_blank_rows"]:
        blockers.append("UNAPPROVED_BLANK_TARGET_STATUS")
    blank_count = scan["blank_or_null_rows"]
    if blank_count and blank_count != scan["approved_boundary_population"]:
        blockers.append("BLANK_COUNT_DOES_NOT_MATCH_APPROVED_BOUNDARY_POPULATION")
    return blockers


def _repaired_blockers(
    source: dict[str, Any],
    repaired: dict[str, Any],
    *,
    source_invariants: dict[str, Any],
    repaired_invariants: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if repaired["distinct_provenance_values"] != [TARGET_PROVENANCE_V2]:
        blockers.append("REPAIRED_PROVENANCE_NOT_SOLE_V2")
    if source["row_count"] != repaired["row_count"]:
        blockers.append("ROW_COUNT_CHANGED")
    if source["economic_key_order_sha256"] != repaired["economic_key_order_sha256"]:
        blockers.append("ECONOMIC_KEY_ORDER_CHANGED")
    if source_invariants.get("sha256") != repaired_invariants.get("sha256"):
        blockers.append("NON_PROVENANCE_VALUE_CHANGED")
    if source_invariants.get("schema") != repaired_invariants.get("schema"):
        blockers.append("SCHEMA_CHANGED")
    if source_invariants.get("column_order") != repaired_invariants.get("column_order"):
        blockers.append("COLUMN_ORDER_CHANGED")
    if source_invariants.get("null_counts") != repaired_invariants.get("null_counts"):
        blockers.append("NON_PROVENANCE_NULL_POPULATION_CHANGED")
    return blockers


def _invariant_fingerprint(path: Path, *, batch_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = [
        name for name in parquet.schema_arrow.names if name != PROVENANCE_COLUMN
    ]
    hasher = hashlib.sha256()
    null_counts = {name: 0 for name in columns}
    row_count = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        for column_index, name in enumerate(columns):
            column = batch.column(column_index)
            null_counts[name] += column.null_count
            hasher.update(name.encode("utf-8") + b"\0")
            for scalar in column:
                hasher.update(_scalar_bytes(scalar.as_py()))
        row_count += batch.num_rows
    return {
        "sha256": hasher.hexdigest(),
        "row_count": row_count,
        "column_order": list(parquet.schema_arrow.names),
        "schema": str(parquet.schema_arrow),
        "null_counts": null_counts,
    }


def _scalar_bytes(value: Any) -> bytes:
    if value is None:
        return b"N\0"
    if isinstance(value, bytes):
        return b"B" + len(value).to_bytes(8, "big") + value
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=True
    ).encode("utf-8")
    return b"V" + len(encoded).to_bytes(8, "big") + encoded


def _write_repaired(
    input_path: Path,
    temporary: Path,
    *,
    batch_rows: int,
    writer_factory: type[pq.ParquetWriter],
) -> None:
    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    try:
        writer = writer_factory(
            temporary,
            parquet.schema_arrow,
            compression=_compression_codec(parquet),
        )
        provenance_index = parquet.schema_arrow.get_field_index(PROVENANCE_COLUMN)
        provenance_type = parquet.schema_arrow.field(PROVENANCE_COLUMN).type
        for batch in parquet.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches([batch])
            values = table[PROVENANCE_COLUMN]
            normalized = pc.fill_null(values, "")
            blank = pc.equal(pc.utf8_trim_whitespace(normalized), "")
            repaired = pc.if_else(
                blank,
                pa.scalar(TARGET_PROVENANCE_V2, type=provenance_type),
                values,
            )
            table = table.set_column(
                provenance_index, PROVENANCE_COLUMN, repaired
            )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _compression_codec(parquet: pq.ParquetFile) -> str:
    codecs = {
        parquet.metadata.row_group(row_group)
        .column(column)
        .compression.lower()
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    return next(iter(codecs)) if len(codecs) == 1 else "zstd"


def _base_report(
    *,
    input_path: Path,
    output_path: Path | None,
    source_checksum: str,
    source: dict[str, Any],
    source_invariants: dict[str, Any] | None,
    dry_run: bool,
    promote: bool,
    batch_rows: int,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path is not None else None,
        "source_checksum": source_checksum,
        "repaired_checksum": None,
        "total_rows": source["row_count"],
        "existing_v2_rows": source["existing_v2_rows"],
        "blank_or_null_rows_found": source["blank_or_null_rows"],
        "approved_rows_repaired": 0,
        "unapproved_blank_rows": source["unapproved_blank_rows"],
        "distinct_values_before": source["distinct_provenance_values"],
        "distinct_values_after": None,
        "target_status_population": source["target_status_population"],
        "blank_status_population": source["blank_status_population"],
        "approved_boundary_statuses": sorted(APPROVED_BLANK_STATUSES),
        "approved_boundary_population": source["approved_boundary_population"],
        "row_count_match": None,
        "economic_key_match": None,
        "source_economic_key_order_sha256": source[
            "economic_key_order_sha256"
        ],
        "source_invariant_column_sha256": (
            source_invariants["sha256"] if source_invariants else None
        ),
        "invariant_column_checksum_match": None,
        "promotion_requested": promote,
        "promotion_status": "pending" if promote else "not_requested",
        "dry_run": dry_run,
        "batch_rows": batch_rows,
        "bounded_arrow_batches": True,
        "blockers": blockers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair approved blank target-provenance boundary rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=65_536)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = repair_stock_artifact_target_provenance_v2(
            input_path=args.input,
            output_path=args.output,
            report_root=args.report_root,
            dry_run=args.dry_run,
            promote=args.promote,
            batch_rows=args.batch_rows,
        )
    except RepairBlockedError as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
