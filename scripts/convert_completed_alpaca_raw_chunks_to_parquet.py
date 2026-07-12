from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


RAW_ROOT = Path("data/raw/alpaca/stock_bars")
PARQUET_ROOT = Path("data/processed/alpaca/stock_bars_parquet")
PAYLOAD_FILES = ("normalized_rows.json", "provider_pages.json")
REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close")
SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("vwap", pa.float64()),
        ("provider", pa.string()),
        ("feed", pa.string()),
        ("collection_timestamp", pa.string()),
        ("requested_timeframe", pa.string()),
        ("native_timeframe", pa.string()),
        ("adjustment_mode", pa.string()),
        ("extended_hours", pa.bool_()),
        ("session_policy", pa.string()),
        ("session_type", pa.string()),
        ("raw_chunk_identifier", pa.string()),
        ("normalizer_version", pa.string()),
    ]
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert completed Alpaca raw chunk JSON payloads to validated Parquet."
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--parquet-root", type=Path, default=PARQUET_ROOT)
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--row-group-size", type=int, default=64_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of chunk conversions to run in parallel.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Write one progress line to stderr after each processed chunk.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write Parquet and delete validated JSON payloads.",
    )
    args = parser.parse_args()
    if args.max_chunks < 0:
        raise SystemExit("--max-chunks must be zero or positive")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run cannot be combined")

    candidates = find_candidates(args.raw_root, args.parquet_root)
    selected = candidates if args.max_chunks == 0 else candidates[: args.max_chunks]
    results = []
    for candidate in selected:
        if args.dry_run or not args.execute:
            results.append({"status": "dry_run", **candidate})
    if args.execute and not args.dry_run:
        results = run_conversions(selected, args.row_group_size, args.workers, args.progress)

    report = {
        "raw_root": str(args.raw_root),
        "parquet_root": str(args.parquet_root),
        "dry_run": bool(args.dry_run or not args.execute),
        "candidate_count": len(candidates),
        "processed_count": len(results),
        "results": results,
        "source_bytes_deleted": sum(int(row.get("source_bytes_deleted", 0) or 0) for row in results),
        "parquet_bytes": sum(int(row.get("parquet_bytes", 0) or 0) for row in results),
        "space_reclaimed": sum(int(row.get("space_reclaimed", 0) or 0) for row in results),
        "free_bytes_after": shutil.disk_usage(args.raw_root).free if args.raw_root.exists() else None,
    }
    print(json.dumps(report, indent=2, default=str))
    return 1 if any(row.get("status") == "failed" for row in results) else 0


def run_conversions(
    candidates: list[dict[str, Any]],
    row_group_size: int,
    workers: int,
    progress: bool = False,
) -> list[dict[str, Any]]:
    if workers == 1 or len(candidates) <= 1:
        results = []
        for index, candidate in enumerate(candidates, start=1):
            result = convert_one(candidate, row_group_size)
            results.append(result)
            print_progress(result, index, len(candidates), progress)
        return results

    results = []
    max_workers = min(workers, len(candidates))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_one, candidate, row_group_size) for candidate in candidates]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print_progress(result, index, len(candidates), progress)
    return results


def print_progress(result: Mapping[str, Any], index: int, total: int, enabled: bool) -> None:
    if not enabled:
        return
    source_path = result.get("source_path", "<unknown>")
    print(f"[{index}/{total}] {result.get('status')}: {source_path}", file=sys.stderr, flush=True)


def find_candidates(raw_root: Path, parquet_root: Path) -> list[dict[str, Any]]:
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(raw_root, onerror=lambda _error: None):
        chunk_dir = Path(dirpath)
        if any(part.endswith(".tmp") or part == ".tmp" for part in chunk_dir.parts):
            continue
        if "manifest.json" not in filenames or "normalized_rows.json" not in filenames:
            continue
        manifest_path = chunk_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("completion_state") != "completed":
            continue
        final_path = parquet_path(parquet_root, raw_root, chunk_dir)
        if already_converted(chunk_dir, final_path):
            continue
        source_bytes = payload_size(chunk_dir)
        candidates.append(
            {
                "source_path": str(chunk_dir),
                "manifest_path": str(manifest_path),
                "parquet_path": str(final_path),
                "manifest_row_count": int(manifest.get("row_count", 0) or 0),
                "source_bytes": source_bytes,
                "manifest": manifest,
            }
        )
    return sorted(candidates, key=lambda row: (int(row["source_bytes"]), row["source_path"]))


def convert_one(candidate: Mapping[str, Any], row_group_size: int) -> dict[str, Any]:
    chunk_dir = Path(str(candidate["source_path"]))
    final_path = Path(str(candidate["parquet_path"]))
    tmp_path = final_path.with_suffix(final_path.suffix + f".converter.{os.getpid()}.tmp")
    normalized_path = chunk_dir / "normalized_rows.json"
    source_bytes = int(candidate["source_bytes"])
    expected_rows = int(candidate["manifest_row_count"])
    started_at = utc_now()
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp_path.exists():
            tmp_path.unlink()
        written_rows = write_parquet(normalized_path, tmp_path, row_group_size)
        validation = validate_parquet(tmp_path, expected_rows)
        if written_rows != expected_rows:
            validation["errors"].append(f"source_read_row_count_mismatch:{written_rows}!={expected_rows}")
        if validation["errors"]:
            tmp_path.unlink(missing_ok=True)
            return failed(candidate, validation["errors"])
        os.replace(tmp_path, final_path)
        final_validation = validate_parquet(final_path, expected_rows)
        if final_validation["errors"]:
            return failed(candidate, final_validation["errors"])
        deleted = []
        deleted_bytes = 0
        for name in PAYLOAD_FILES:
            path = chunk_dir / name
            if path.exists():
                deleted_bytes += path.stat().st_size
                path.unlink()
                deleted.append(str(path))
        parquet_bytes = final_path.stat().st_size
        tombstone = {
            "parquet_path": str(final_path),
            "source_path": str(chunk_dir),
            "source_row_count": expected_rows,
            "parquet_row_count": final_validation["row_count"],
            "source_bytes": source_bytes,
            "source_bytes_deleted": deleted_bytes,
            "parquet_bytes": parquet_bytes,
            "validation_result": "passed",
            "validation_errors": [],
            "deleted_payload_files": deleted,
            "converted_at": started_at,
            "validated_at": utc_now(),
            "deletion_timestamp": utc_now(),
        }
        atomic_write_json(chunk_dir / "parquet_conversion.json", tombstone)
        return {
            "status": "converted",
            "source_path": str(chunk_dir),
            "parquet_path": str(final_path),
            "source_row_count": expected_rows,
            "parquet_row_count": final_validation["row_count"],
            "source_bytes": source_bytes,
            "source_bytes_deleted": deleted_bytes,
            "parquet_bytes": parquet_bytes,
            "space_reclaimed": deleted_bytes - parquet_bytes,
            "deleted_payload_files": deleted,
        }
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return failed(candidate, [f"{type(exc).__name__}: {exc}"])


def write_parquet(normalized_path: Path, tmp_path: Path, row_group_size: int) -> int:
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    rows = 0
    try:
        for raw in iter_json_array(normalized_path):
            batch.append(normalize_row(raw))
            rows += 1
            if len(batch) >= row_group_size:
                writer = write_batch(writer, tmp_path, batch, row_group_size)
                batch.clear()
        if batch or rows == 0:
            writer = write_batch(writer, tmp_path, batch, row_group_size)
    finally:
        if writer is not None:
            writer.close()
    return rows


def write_batch(
    writer: pq.ParquetWriter | None,
    tmp_path: Path,
    batch: list[dict[str, Any]],
    row_group_size: int,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(batch, schema=SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
    writer.write_table(table, row_group_size=row_group_size)
    return writer


def validate_parquet(path: Path, expected_rows: int) -> dict[str, Any]:
    errors = []
    parquet = pq.ParquetFile(path)
    row_count = parquet.metadata.num_rows
    if row_count != expected_rows:
        errors.append(f"row_count_mismatch:{row_count}!={expected_rows}")
    schema = parquet.schema_arrow
    for field in SCHEMA:
        if field.name not in schema.names:
            errors.append(f"missing_column:{field.name}")
        elif schema.field(field.name).type != field.type:
            errors.append(f"type_mismatch:{field.name}")
    table = parquet.read(columns=list(REQUIRED_COLUMNS))
    data = table.to_pydict()
    for column in REQUIRED_COLUMNS:
        if any(value is None for value in data[column]):
            errors.append(f"null_required_column:{column}")
    return {"row_count": row_count, "errors": errors}


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    eof = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not in_array:
                if buffer.startswith("["):
                    buffer = buffer[1:]
                    in_array = True
                elif eof and not buffer:
                    return
                elif eof:
                    raise ValueError("normalized_rows.json is not a top-level JSON array")
            while in_array:
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                try:
                    item, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(item, dict):
                    raise ValueError("normalized_rows.json item is not an object")
                yield item
                buffer = buffer[index:]
            if eof and not buffer.strip():
                return


def normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {name: row.get(name) for name in SCHEMA.names}
    output["timestamp"] = parse_datetime(output["timestamp"])
    return output


def parquet_path(parquet_root: Path, raw_root: Path, chunk_dir: Path) -> Path:
    relative = chunk_dir.relative_to(raw_root)
    return parquet_root / relative / "bars.parquet"


def already_converted(chunk_dir: Path, final_path: Path) -> bool:
    tombstone_path = chunk_dir / "parquet_conversion.json"
    if not tombstone_path.exists() or not final_path.exists():
        return False
    try:
        tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    converted = (
        tombstone.get("validation_result") == "passed"
        and str(tombstone.get("parquet_path")) == str(final_path)
    )
    try:
        return converted and int(tombstone.get("parquet_row_count", -1)) == pq.ParquetFile(final_path).metadata.num_rows
    except Exception:
        return False


def payload_size(chunk_dir: Path) -> int:
    return sum((chunk_dir / name).stat().st_size for name in PAYLOAD_FILES if (chunk_dir / name).exists())


def failed(candidate: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "status": "failed",
        "source_path": candidate["source_path"],
        "parquet_path": candidate["parquet_path"],
        "source_row_count": candidate["manifest_row_count"],
        "source_bytes": candidate["source_bytes"],
        "errors": errors,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
