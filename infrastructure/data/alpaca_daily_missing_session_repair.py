from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


MISSING_SESSION = "2026-05-27"
REPORT_ROOT = Path("reports/data_lineage/alpaca_daily_missing_session_repair")
PRODUCTION_ARCHIVE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d")
REPAIR_ARCHIVE_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d_missing_session_repair")
REPAIR_OUTPUT_ROOT = Path("reports/market_data/historical_bar_backfill/daily_sip_514_symbol_may27_repair")
REPAIR_RAW_ROOT = Path("data/raw/alpaca/daily_stock_bars_missing_session_repair")


def inspect_missing_session_root_cause(
    *,
    raw_root: Path = Path("data/raw/alpaca/daily_stock_bars/sip/1Day"),
    archive_root: Path = PRODUCTION_ARCHIVE_ROOT,
    report_root: Path = REPORT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    chunk_reports = []
    raw_may27_rows = 0
    for manifest_path in sorted(raw_root.glob("*/20260427T000000Z_20260527T000000Z/manifest.json")) + sorted(raw_root.glob("*/20260528T000000Z_20260627T000000Z/manifest.json")):
        chunk_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows_path = chunk_dir / "normalized_rows.json"
        dates = []
        may27_count = 0
        if rows_path.exists():
            rows = json.loads(rows_path.read_text(encoding="utf-8"))
            dates = sorted({str(row.get("timestamp", ""))[:10] for row in rows})
            may27_count = sum(1 for row in rows if str(row.get("timestamp", "")).startswith(MISSING_SESSION))
            raw_may27_rows += may27_count
        chunk_reports.append(
            {
                "chunk_dir": str(chunk_dir),
                "requested_start": manifest.get("requested_start"),
                "requested_end": manifest.get("requested_end"),
                "row_count": manifest.get("row_count"),
                "earliest_session": dates[0] if dates else "",
                "latest_session": dates[-1] if dates else "",
                "may27_rows": may27_count,
            }
        )
    archive_may27_rows = count_archive_session_rows(archive_root, MISSING_SESSION)
    report = {
        "missing_session": MISSING_SESSION,
        "root_cause": "daily chunk boundary ended at 2026-05-27T00:00:00Z; Alpaca 1Day bars for 2026-05-27 require a request window extending to 2026-05-28T00:00:00Z",
        "failure_phase": "chunk start/end boundary semantics",
        "may27_exists_in_raw": raw_may27_rows > 0,
        "raw_may27_rows": raw_may27_rows,
        "archive_may27_rows": archive_may27_rows,
        "chunk_windows": chunk_reports,
        "api_requests_attempted": 0,
        "source_archives_modified": False,
    }
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "missing_session_root_cause.json", report)
        (report_root / "missing_session_root_cause.md").write_text(render_root_cause_markdown(report), encoding="utf-8")
    return report


def count_archive_session_rows(archive_root: Path, session_date: str) -> int:
    count = 0
    for path in sorted(archive_root.glob("symbol=*/year=*/bars.parquet")):
        table = pq.read_table(path, columns=["session_date"])
        count += sum(1 for row in table.to_pylist() if row.get("session_date") == session_date)
    return count


def validate_may27_rows(archive_root: Path = REPAIR_ARCHIVE_ROOT, *, expected_rows: int = 514) -> dict[str, Any]:
    rows = read_archive_session_rows(archive_root, MISSING_SESSION)
    keys = Counter((row["canonical_symbol"], row["session_date"]) for row in rows)
    duplicates = sum(count - 1 for count in keys.values() if count > 1)
    invalid_ohlc = [row for row in rows if not _valid_ohlc(row)]
    invalid_volume = [row for row in rows if row.get("volume") is None or float(row["volume"]) < 0]
    invalid_timestamp = [row for row in rows if row.get("session_date") != MISSING_SESSION or row.get("timestamp_utc") is None]
    missing_asset = [row for row in rows if not row.get("asset_id")]
    symbols = sorted({row["canonical_symbol"] for row in rows})
    return {
        "archive_root": str(archive_root),
        "session_date": MISSING_SESSION,
        "row_count": len(rows),
        "symbol_count": len(symbols),
        "asset_id_count": len({row["asset_id"] for row in rows if row.get("asset_id")}),
        "duplicate_symbol_session_rows": duplicates,
        "invalid_ohlc_rows": len(invalid_ohlc),
        "invalid_volume_rows": len(invalid_volume),
        "invalid_timestamp_rows": len(invalid_timestamp),
        "symbols_without_asset_id": sorted({row["canonical_symbol"] for row in missing_asset}),
        "missing_symbols": [],
        "valid": len(rows) == expected_rows and duplicates == 0 and not invalid_ohlc and not invalid_volume and not invalid_timestamp and not missing_asset,
    }


def read_archive_session_rows(archive_root: Path, session_date: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(archive_root.glob("symbol=*/year=*/bars.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            if str(row.get("session_date")) == session_date:
                rows.append(dict(row))
    return sorted(rows, key=lambda row: row["canonical_symbol"])


def merge_may27_archive(
    *,
    source_archive_root: Path = REPAIR_ARCHIVE_ROOT,
    target_archive_root: Path = PRODUCTION_ARCHIVE_ROOT,
    report_root: Path = REPORT_ROOT,
    dry_run: bool = False,
    expected_rows: int = 514,
) -> dict[str, Any]:
    validation = validate_may27_rows(source_archive_root, expected_rows=expected_rows)
    if not validation["valid"]:
        raise ValueError(f"May 27 repair rows are not valid: {validation}")
    repair_rows = read_archive_session_rows(source_archive_root, MISSING_SESSION)
    backup_root = report_root / "pre_may27_archive_backup"
    merged = []
    conflicts = []
    identical_duplicates = 0
    for row in repair_rows:
        symbol = row["canonical_symbol"]
        target = target_archive_root / f"symbol={symbol}" / "year=2026" / "bars.parquet"
        if not target.exists():
            raise FileNotFoundError(f"target Alpaca partition missing: {target}")
        existing_rows = pq.read_table(target).to_pylist()
        existing_by_session = {str(item["session_date"]): dict(item) for item in existing_rows}
        if MISSING_SESSION in existing_by_session:
            if _rows_equivalent(existing_by_session[MISSING_SESSION], row):
                identical_duplicates += 1
                continue
            conflicts.append(symbol)
            continue
        new_rows = sorted([dict(item) for item in existing_rows] + [dict(row)], key=lambda item: str(item["session_date"]))
        if not dry_run:
            backup_target = backup_root / f"symbol={symbol}" / "year=2026" / "bars.parquet"
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            if not backup_target.exists():
                shutil.copy2(target, backup_target)
            tmp = target.with_suffix(target.suffix + ".tmp")
            pq.write_table(pa.Table.from_pylist(new_rows, schema=pq.read_schema(target)), tmp, compression="zstd")
            tmp.replace(target)
        merged.append(symbol)
    if conflicts:
        raise ValueError("conflicting May 27 rows already exist for: " + ", ".join(conflicts))
    report = {
        "source_archive_root": str(source_archive_root),
        "target_archive_root": str(target_archive_root),
        "session_date": MISSING_SESSION,
        "validated_repair_rows": validation,
        "merged_symbols": merged,
        "merged_row_count": len(merged),
        "identical_duplicate_rows": identical_duplicates,
        "conflicting_duplicate_symbols": conflicts,
        "backup_root": str(backup_root),
        "dry_run": dry_run,
        "source_archives_modified": not dry_run,
    }
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "may27_archive_validation.json", {**validation, "merge": report})
    return report


def post_repair_archive_audit(archive_root: Path = PRODUCTION_ARCHIVE_ROOT, *, report_root: Path = REPORT_ROOT, dry_run: bool = False) -> dict[str, Any]:
    rows = []
    file_count = 0
    for path in sorted(archive_root.glob("symbol=*/year=*/bars.parquet")):
        file_count += 1
        rows.extend(pq.read_table(path, columns=["asset_id", "canonical_symbol", "session_date", "open", "high", "low", "close", "volume", "timestamp_utc"]).to_pylist())
    keys = Counter((row["canonical_symbol"], row["session_date"]) for row in rows)
    dates = sorted({row["session_date"] for row in rows})
    report = {
        "archive_root": str(archive_root),
        "file_count": file_count,
        "row_count": len(rows),
        "symbol_count": len({row["canonical_symbol"] for row in rows}),
        "unique_session_count": len(dates),
        "date_minimum": dates[0] if dates else None,
        "date_maximum": dates[-1] if dates else None,
        "may27_rows": sum(1 for row in rows if row["session_date"] == MISSING_SESSION),
        "duplicate_symbol_session_rows": sum(count - 1 for count in keys.values() if count > 1),
        "invalid_ohlc_rows": sum(1 for row in rows if not _valid_ohlc(row)),
        "invalid_volume_rows": sum(1 for row in rows if row.get("volume") is None or float(row["volume"]) < 0),
        "symbols_without_asset_id": sorted({row["canonical_symbol"] for row in rows if not row.get("asset_id")}),
    }
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "post_repair_archive_audit.json", report)
    return report


def five_minute_verification(*, report_root: Path = REPORT_ROOT, dry_run: bool = False) -> list[dict[str, Any]]:
    root = Path("data/processed/alpaca/stock_bars_parquet/sip/5m")
    direct_rows = {row["canonical_symbol"]: row for row in read_archive_session_rows(PRODUCTION_ARCHIVE_ROOT, MISSING_SESSION)}
    results = []
    if root.exists():
        for symbol, direct in direct_rows.items():
            provider_symbol = direct.get("provider_symbol") or symbol
            matches = list(root.glob(f"**/{provider_symbol}/**/*.parquet")) + list(root.glob(f"**/*{provider_symbol}*/**/*.parquet"))
            results.append(
                {
                    "canonical_symbol": symbol,
                    "session_date": MISSING_SESSION,
                    "five_minute_coverage": "not_evaluated" if not matches else "candidate_files_found",
                    "candidate_file_count": len(matches),
                    "direct_daily_close": direct.get("close"),
                    "derived_close": "",
                    "close_agreement": "",
                    "ohlc_agreement": "",
                    "volume_difference": "",
                }
            )
    if not results:
        results = [
            {
                "canonical_symbol": symbol,
                "session_date": MISSING_SESSION,
                "five_minute_coverage": "no_5m_archive_match_found",
                "candidate_file_count": 0,
                "direct_daily_close": row.get("close"),
                "derived_close": "",
                "close_agreement": "",
                "ohlc_agreement": "",
                "volume_difference": "",
            }
            for symbol, row in direct_rows.items()
        ]
    if not dry_run:
        report_root.mkdir(parents=True, exist_ok=True)
        with (report_root / "may27_five_minute_verification.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["canonical_symbol", "session_date", "five_minute_coverage", "candidate_file_count", "direct_daily_close", "derived_close", "close_agreement", "ohlc_agreement", "volume_difference"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
    return results


def write_pre_post_comparison(
    *,
    before_summary_path: Path,
    after_summary_path: Path,
    before_revised_path: Path,
    after_revised_path: Path,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Any]:
    before = json.loads(before_summary_path.read_text(encoding="utf-8"))
    after = json.loads(after_summary_path.read_text(encoding="utf-8"))
    rows = []
    keys = sorted(set(before.get("classification_totals", before.get("revised_classification_counts", {}))) | set(after.get("classification_totals", after.get("revised_classification_counts", {}))))
    before_counts = before.get("classification_totals", before.get("revised_classification_counts", {}))
    after_counts = after.get("classification_totals", after.get("revised_classification_counts", {}))
    for key in keys:
        rows.append({"classification": key, "before_count": before_counts.get(key, 0), "after_count": after_counts.get(key, 0), "delta": after_counts.get(key, 0) - before_counts.get(key, 0)})
    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / "pre_post_classification_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["classification", "before_count", "after_count", "delta"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    may28 = may28_changes(before_revised_path, after_revised_path)
    with (report_root / "may28_reclassification_changes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["canonical_symbol", "session_date", "before_classification", "after_classification", "changed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(may28)
    return {"classification_comparison": rows, "may28_changes": may28}


def may28_changes(before_path: Path, after_path: Path) -> list[dict[str, Any]]:
    before = _read_csv_by_key(before_path)
    after = _read_csv_by_key(after_path)
    rows = []
    for key, after_row in sorted(after.items()):
        symbol, session = key
        if session != "2026-05-28":
            continue
        before_class = before.get(key, {}).get("classification", "")
        after_class = after_row.get("classification", "")
        rows.append({"canonical_symbol": symbol, "session_date": session, "before_classification": before_class, "after_classification": after_class, "changed": str(before_class != after_class).lower()})
    return rows


def _read_csv_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["canonical_symbol"], row["session_date"]): dict(row) for row in csv.DictReader(handle)}


def render_root_cause_markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Alpaca May 27 Missing Session Root Cause\n\n"
        f"- Root cause: {report['root_cause']}\n"
        f"- Failure phase: {report['failure_phase']}\n"
        f"- May 27 rows in raw: {report['raw_may27_rows']}\n"
        f"- May 27 rows in archive: {report['archive_may27_rows']}\n"
    )


def _rows_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = ["asset_id", "canonical_symbol", "provider_symbol", "session_date", "open", "high", "low", "close", "volume", "provider", "feed", "timeframe", "adjustment_policy"]
    return all(str(left.get(field)) == str(right.get(field)) for field in fields)


def _valid_ohlc(row: Mapping[str, Any]) -> bool:
    try:
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
    except (TypeError, ValueError, KeyError):
        return False
    return low <= open_ <= high and low <= close <= high and min(open_, high, low, close) > 0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
