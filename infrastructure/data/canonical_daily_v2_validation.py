from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "asset_id",
    "canonical_symbol",
    "session_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
    "model_open",
    "model_high",
    "model_low",
    "model_close",
    "source_provider",
    "source_feed",
    "source_adjustment",
    "source_path",
    "compatibility_tier",
    "selector_eligible",
    "eligibility_reason",
}


def validate_partitioned_dataset(root: Path, *, report_root: Path | None = None) -> dict[str, Any]:
    files = sorted(root.glob("symbol=*/year=*/bars.parquet"))
    asset_keys: Counter[tuple[str, str]] = Counter()
    symbol_keys: Counter[tuple[str, str]] = Counter()
    symbol_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    provider_transitions = 0
    bridge_rows = 0
    quarantine_rows = 0
    invalid_ohlc = 0
    invalid_timestamp = 0
    nonpositive_model = 0
    missing_lineage = 0
    tier_d_eligible = 0
    quarantined_eligible = 0
    unexpected_boundary_return = 0
    row_count = 0
    global_hash = hashlib.sha256()
    date_min: str | None = None
    date_max: str | None = None
    for path in files:
        table = pq.read_table(path)
        missing = sorted(REQUIRED_COLUMNS - set(table.schema.names))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        file_hash = _file_sha256(path)
        global_hash.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        global_hash.update(file_hash.encode("ascii"))
        for row in table.to_pylist():
            row_count += 1
            asset = str(row.get("asset_id") or "")
            symbol = str(row.get("canonical_symbol") or "")
            session = str(row.get("session_date") or "")
            asset_keys[(asset, session)] += 1
            symbol_keys[(symbol, session)] += 1
            symbol_counts[symbol] += 1
            session_counts[session] += 1
            tier = str(row.get("compatibility_tier") or "")
            tier_counts[tier] += 1
            eligibility_counts[str(bool(row.get("selector_eligible"))).lower()] += 1
            provider_transitions += int(bool(row.get("provider_transition_flag")))
            bridge_rows += int(float(row.get("price_bridge_factor") or 1.0) != 1.0)
            quarantine_rows += int(bool(row.get("quarantine_flag")))
            if not session:
                invalid_timestamp += 1
            else:
                date_min = session if date_min is None else min(date_min, session)
                date_max = session if date_max is None else max(date_max, session)
            try:
                raw_open = float(row["raw_open"])
                raw_high = float(row["raw_high"])
                raw_low = float(row["raw_low"])
                raw_close = float(row["raw_close"])
                model_close = float(row["model_close"])
            except (TypeError, ValueError, KeyError):
                invalid_ohlc += 1
                continue
            if not (raw_low <= raw_open <= raw_high and raw_low <= raw_close <= raw_high and min(raw_open, raw_high, raw_low, raw_close) > 0):
                invalid_ohlc += 1
            if model_close <= 0:
                nonpositive_model += 1
            if not row.get("source_path"):
                missing_lineage += 1
            if tier == "TIER_D_SYMBOL_QUARANTINE" and row.get("selector_eligible"):
                tier_d_eligible += 1
            if row.get("quarantine_flag") and row.get("selector_eligible"):
                quarantined_eligible += 1
            if row.get("provider_changed_since_previous_row") and row.get("return_valid") and float(row.get("price_bridge_factor") or 1.0) == 1.0:
                unexpected_boundary_return += 1
    duplicate_asset = sum(count - 1 for count in asset_keys.values() if count > 1)
    duplicate_symbol = sum(count - 1 for count in symbol_keys.values() if count > 1)
    validation = {
        "root": str(root),
        "partition_file_count": len(files),
        "row_count": row_count,
        "symbol_count": len([symbol for symbol in symbol_counts if symbol]),
        "date_min": date_min,
        "date_max": date_max,
        "duplicate_asset_session_rows": duplicate_asset,
        "duplicate_symbol_session_rows": duplicate_symbol,
        "invalid_ohlc_rows": invalid_ohlc,
        "invalid_timestamp_rows": invalid_timestamp,
        "nonpositive_model_price_rows": nonpositive_model,
        "missing_source_lineage_rows": missing_lineage,
        "tier_d_selector_eligible_rows": tier_d_eligible,
        "quarantined_selector_eligible_rows": quarantined_eligible,
        "unexpected_provider_boundary_return_rows": unexpected_boundary_return,
        "provider_transition_count": provider_transitions,
        "price_bridge_row_count": bridge_rows,
        "quarantine_row_count": quarantine_rows,
        "tier_row_counts": dict(sorted(tier_counts.items())),
        "selector_eligibility_row_counts": dict(sorted(eligibility_counts.items())),
        "dataset_logical_partition_hash": global_hash.hexdigest(),
        "valid": all(
            value == 0
            for value in [
                duplicate_asset,
                duplicate_symbol,
                invalid_ohlc,
                invalid_timestamp,
                nonpositive_model,
                missing_lineage,
                tier_d_eligible,
                quarantined_eligible,
                unexpected_boundary_return,
            ]
        ),
    }
    if report_root:
        report_root.mkdir(parents=True, exist_ok=True)
        _write_json(report_root / "validation.json", validation)
        (report_root / "validation.md").write_text(_markdown(validation), encoding="utf-8")
        _write_csv(report_root / "row_counts_by_symbol.csv", [{"canonical_symbol": k, "row_count": v} for k, v in sorted(symbol_counts.items())], ["canonical_symbol", "row_count"])
        _write_csv(report_root / "row_counts_by_session.csv", [{"session_date": k, "row_count": v} for k, v in sorted(session_counts.items())], ["session_date", "row_count"])
    return validation


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(validation: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Canonical Daily V2 Validation",
            "",
            f"- Valid: {validation['valid']}",
            f"- Rows: {validation['row_count']}",
            f"- Symbols: {validation['symbol_count']}",
            f"- Date range: {validation['date_min']} through {validation['date_max']}",
            f"- Duplicate asset/session rows: {validation['duplicate_asset_session_rows']}",
            f"- Duplicate symbol/session rows: {validation['duplicate_symbol_session_rows']}",
        ]
    )

