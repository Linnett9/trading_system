from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from infrastructure.data.canonical_daily_v2_validation import validate_partitioned_dataset


REGISTRY_PATH = Path("data/reference/assets/canonical_asset_registry.csv")
STOOQ_ROOT = Path("data/processed/stooq_parquet")
ALPACA_ROOT = Path("data/processed/alpaca/symbol_bars/sip/1d")
RESIDUAL_ROOT = Path("reports/data_lineage/provider_residual_resolution")
OUTPUT_ROOT = Path("data/processed/market_data/canonical_daily_v2")
DATASET_ROOT = OUTPUT_ROOT / "full"
REPORT_ROOT = Path("reports/data_lineage/canonical_daily_v2")
SCHEMA_VERSION = "canonical_daily_v2.partitioned.v1"


def build_full_canonical_daily_v2(
    *,
    workers: int = 12,
    output_root: Path = OUTPUT_ROOT,
    report_root: Path = REPORT_ROOT,
    force: bool = False,
    symbols: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    report_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "full"
    manifest_root = report_root / "partition_manifests"
    failed_path = report_root / "failed_partitions.json"
    profile = profile_build_inputs(symbols=symbols)
    _write_json(report_root / "performance_profile.json", profile)
    assets = _assets()
    tiers = _read_by_symbol(RESIDUAL_ROOT / "symbol_compatibility_tiers.csv")
    quarantines = _quarantines()
    selected_symbols = sorted(symbols or assets)
    plan = [{"symbol": symbol, "asset": assets[symbol]} for symbol in selected_symbols if symbol in assets]
    _write_json(
        report_root / "partition_plan.json",
        {
            "schema_version": "canonical_daily_v2.partition_plan.v1",
            "partition_unit": "symbol with year parquet outputs",
            "planned_partitions": len(plan),
            "requested_workers": workers,
            "effective_workers": min(max(1, workers), max(1, len(plan))),
            "dry_run": dry_run,
        },
    )
    if dry_run:
        return {"status": "DRY_RUN", "planned_partitions": len(plan), "source_archives_modified": False}
    manifest_root.mkdir(parents=True, exist_ok=True)
    completed_before = _completed_symbols(manifest_root)
    pending = [item for item in plan if force or item["symbol"] not in completed_before]
    failed: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    rows_written = 0
    effective_workers = min(max(1, workers), max(1, len(pending)))
    if pending:
        if effective_workers == 1:
            for item in pending:
                try:
                    result = _build_symbol_partition(item["symbol"], item["asset"], tiers.get(item["symbol"], {}), quarantines.get(item["symbol"], {}), data_root, manifest_root)
                    completed.append(result)
                    rows_written += int(result["row_count"])
                    _progress(report_root, len(plan), len(completed_before) + len(completed), len(failed), rows_written, started)
                except Exception as exc:  # pragma: no cover - exercised by retry tests with monkeypatching.
                    failed.append({"symbol": item["symbol"], "error": f"{type(exc).__name__}: {exc}"})
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(_build_symbol_partition, item["symbol"], item["asset"], tiers.get(item["symbol"], {}), quarantines.get(item["symbol"], {}), data_root, manifest_root): item["symbol"]
                    for item in pending
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        completed.append(result)
                        rows_written += int(result["row_count"])
                    except Exception as exc:
                        failed.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
                    _progress(report_root, len(plan), len(completed_before) + len(completed), len(failed), rows_written, started)
    _write_json(failed_path, {"failed_partitions": failed, "failed_partition_count": len(failed)})
    all_completed = _completed_symbols(manifest_root)
    source_reports = _aggregate_symbol_reports(manifest_root, selected_symbols)
    _write_csv(report_root / "source_selection_by_symbol.csv", source_reports["source_selection"], ["asset_id", "canonical_symbol", "stooq_rows_selected", "alpaca_rows_selected", "first_alpaca_session", "compatibility_tier"])
    _write_csv(report_root / "provider_transitions.csv", source_reports["provider_transitions"], ["asset_id", "canonical_symbol", "transition_date", "from_provider", "to_provider", "compatibility_tier", "price_bridge_factor"])
    _write_csv(report_root / "price_bridge_statistics.csv", source_reports["price_bridges"], ["asset_id", "canonical_symbol", "price_bridge_factor", "price_bridge_method", "calibration_start", "calibration_end"])
    _write_csv(report_root / "quarantined_rows.csv", source_reports["quarantined_rows"], ["asset_id", "canonical_symbol", "session_date", "quarantine_reason"])
    _write_csv(report_root / "eligibility_summary.csv", source_reports["eligibility"], ["asset_id", "canonical_symbol", "compatibility_tier", "row_count", "selector_eligible_rows", "quarantined_rows", "tier_d_rows"])
    validation = validate_partitioned_dataset(data_root, report_root=report_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE" if not failed and len(all_completed) == len(plan) and validation["valid"] else "INCOMPLETE",
        "dataset_root": str(data_root),
        "planned_partitions": len(plan),
        "completed_partitions": len(all_completed),
        "pending_partitions": max(0, len(plan) - len(all_completed)),
        "failed_partitions": len(failed),
        "requested_workers": workers,
        "effective_workers": effective_workers,
        "row_count": validation["row_count"],
        "symbol_count": validation["symbol_count"],
        "date_min": validation["date_min"],
        "date_max": validation["date_max"],
        "provider_transition_count": validation["provider_transition_count"],
        "price_bridge_row_count": validation["price_bridge_row_count"],
        "quarantine_row_count": validation["quarantine_row_count"],
        "dataset_logical_partition_hash": validation["dataset_logical_partition_hash"],
        "elapsed_seconds": time.perf_counter() - started,
        "source_archives_modified": False,
    }
    _write_json(report_root / "build_manifest.json", manifest)
    _progress(report_root, len(plan), len(all_completed), len(failed), validation["row_count"], started)
    return manifest


def retry_failed_partitions(*, workers: int = 12) -> dict[str, Any]:
    failed = _read_json(REPORT_ROOT / "failed_partitions.json").get("failed_partitions", [])
    symbols = [row["symbol"] for row in failed if row.get("symbol")]
    return build_full_canonical_daily_v2(workers=workers, symbols=symbols, force=True)


def profile_build_inputs(*, symbols: Sequence[str] | None = None) -> dict[str, Any]:
    assets = _assets()
    selected = sorted(symbols or assets)
    stooq_files = [p for symbol in selected for p in [_stooq_path(symbol)] if p.exists()]
    alpaca_files = [ALPACA_ROOT / f"symbol={symbol}" / "year=2026" / "bars.parquet" for symbol in selected]
    alpaca_files = [p for p in alpaca_files if p.exists()]
    stooq_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in stooq_files)
    alpaca_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in alpaca_files)
    return {
        "root_cause_previous_nine_hour_build": "readiness builder accumulated all symbols into one Python row list and attempted one monolithic Parquet publication; it also repeated expensive artifact/report setup before each smoke/full run and lacked completed-partition resume gates",
        "files_opened_estimate": len(stooq_files) + len(alpaca_files),
        "stooq_file_count": len(stooq_files),
        "alpaca_file_count": len(alpaca_files),
        "rows_scanned_estimate": stooq_rows + alpaca_rows,
        "stooq_rows_estimate": stooq_rows,
        "alpaca_rows_estimate": alpaca_rows,
        "repeated_source_scans_removed": True,
        "compatibility_tiers_preloaded_once": True,
        "quarantines_preloaded_once": True,
        "monolithic_table_removed": True,
        "approximate_memory_policy": "bounded by one symbol plus yearly output groups per worker",
    }


def _build_symbol_partition(
    symbol: str,
    asset: Mapping[str, str],
    tier_row: Mapping[str, str],
    symbol_quarantines: Mapping[str, str],
    data_root: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    tier = tier_row.get("compatibility_tier", "TIER_A_NATIVE_COMPATIBLE")
    stooq = _read_stooq_rows(symbol)
    alpaca = _read_alpaca_rows(symbol)
    first_alpaca = min((row["session_date"] for row in alpaca), default="")
    stooq_selected = [row for row in stooq if not first_alpaca or row["session_date"] < first_alpaca]
    bridge_factor = _bridge_factor(tier, stooq, alpaca)
    rows: list[dict[str, Any]] = []
    for row in stooq_selected:
        rows.append(_canonical_row(asset, row, tier, "stooq", False, "", 1.0, symbol_quarantines))
    for row in alpaca:
        rows.append(_canonical_row(asset, row, tier, "alpaca", row["session_date"] == first_alpaca, first_alpaca, bridge_factor, symbol_quarantines))
    rows.sort(key=lambda row: row["session_date"])
    _add_returns_and_volume(rows)
    by_year: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_year[row["session_date"][:4]].append(row)
    output_files = []
    for year, year_rows in sorted(by_year.items()):
        target = data_root / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(year_rows), tmp, compression="zstd")
        tmp.replace(target)
        output_files.append({"path": str(target), "row_count": len(year_rows), "sha256": _file_sha256(target)})
    manifest = {
        "symbol": symbol,
        "asset_id": asset["asset_id"],
        "status": "COMPLETE",
        "row_count": len(rows),
        "stooq_rows_selected": len(stooq_selected),
        "alpaca_rows_selected": len(alpaca),
        "first_alpaca_session": first_alpaca,
        "compatibility_tier": tier,
        "provider_transition_count": 1 if first_alpaca else 0,
        "price_bridge_factor": bridge_factor,
        "price_bridge_count": 1 if tier == "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE" and bridge_factor != 1.0 else 0,
        "quarantined_rows": sum(1 for row in rows if row["quarantine_flag"]),
        "selector_eligible_rows": sum(1 for row in rows if row["selector_eligible"]),
        "output_files": output_files,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_root / f"{_safe_symbol(symbol)}.json", manifest)
    return manifest


def _canonical_row(asset: Mapping[str, str], row: Mapping[str, Any], tier: str, provider: str, transition: bool, transition_id: str, bridge_factor: float, quarantines: Mapping[str, str]) -> dict[str, Any]:
    symbol = asset["canonical_symbol"]
    date = row["session_date"]
    factor = bridge_factor if provider == "alpaca" and tier == "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE" else 1.0
    quarantine_reason = quarantines.get(date, "")
    selector_eligible = not quarantine_reason and tier not in {"TIER_D_SYMBOL_QUARANTINE", "TIER_E_REVIEW_BLOCKED"}
    eligibility_reason = "eligible"
    if quarantine_reason:
        eligibility_reason = f"quarantined:{quarantine_reason}"
    elif tier == "TIER_D_SYMBOL_QUARANTINE":
        eligibility_reason = "tier_d_symbol_quarantine"
    elif tier == "TIER_E_REVIEW_BLOCKED":
        eligibility_reason = "tier_e_review_blocked"
    raw_open = float(row["open"])
    raw_high = float(row["high"])
    raw_low = float(row["low"])
    raw_close = float(row["close"])
    return {
        "asset_id": asset["asset_id"],
        "canonical_symbol": symbol,
        "session_date": date,
        "raw_open": raw_open,
        "raw_high": raw_high,
        "raw_low": raw_low,
        "raw_close": raw_close,
        "raw_volume": float(row.get("volume") or 0.0),
        "model_open": raw_open * factor,
        "model_high": raw_high * factor,
        "model_low": raw_low * factor,
        "model_close": raw_close * factor,
        "source_provider": provider,
        "source_feed": row.get("feed", "stooq_bulk" if provider == "stooq" else "sip"),
        "source_adjustment": row.get("adjustment_policy", "stooq_adjusted" if provider == "stooq" else "all"),
        "source_path": row.get("source_path", ""),
        "compatibility_tier": tier,
        "provider_transition_flag": transition,
        "provider_transition_id": transition_id,
        "price_bridge_factor": factor,
        "price_bridge_method": "median_overlap_ratio" if factor != 1.0 else "none",
        "price_bridge_calibration_start": "",
        "price_bridge_calibration_end": date if factor != 1.0 else "",
        "quarantine_flag": bool(quarantine_reason),
        "quarantine_reason": quarantine_reason,
        "selector_eligible": selector_eligible,
        "eligibility_reason": eligibility_reason,
        "previous_session_date": "",
        "session_gap_calendar_days": None,
        "session_gap_trading_sessions": None,
        "provider_changed_since_previous_row": False,
        "return_valid": False,
        "return_invalid_reason": "first_observed_session",
        "model_return": None,
        "provider_local_volume_percentile": None,
        "provider_local_volume_zscore": None,
        "provider_local_relative_volume": None,
        "provider_transition_volume_guard": transition,
    }


def _add_returns_and_volume(rows: list[dict[str, Any]]) -> None:
    provider_volumes: defaultdict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        volumes = provider_volumes[row["source_provider"]]
        if volumes:
            mean = statistics.mean(volumes)
            stdev = statistics.pstdev(volumes) or 1.0
            row["provider_local_volume_percentile"] = sum(1 for value in volumes if value <= row["raw_volume"]) / len(volumes)
            row["provider_local_volume_zscore"] = (row["raw_volume"] - mean) / stdev
            row["provider_local_relative_volume"] = row["raw_volume"] / mean if mean else None
        volumes.append(row["raw_volume"])
        if index == 0:
            continue
        previous = rows[index - 1]
        row["previous_session_date"] = previous["session_date"]
        row["session_gap_calendar_days"] = (datetime.fromisoformat(row["session_date"]) - datetime.fromisoformat(previous["session_date"])).days
        row["session_gap_trading_sessions"] = 1
        provider_changed = row["source_provider"] != previous["source_provider"]
        row["provider_changed_since_previous_row"] = provider_changed
        if row["quarantine_flag"] or previous["quarantine_flag"]:
            row["return_invalid_reason"] = "quarantined_current_or_previous_row"
        elif provider_changed and row["price_bridge_factor"] == 1.0:
            row["return_invalid_reason"] = "unbridged_provider_transition"
        elif not row["selector_eligible"] or not previous["selector_eligible"]:
            row["return_invalid_reason"] = "selector_ineligible_current_or_previous_row"
        else:
            row["return_valid"] = True
            row["return_invalid_reason"] = ""
            row["model_return"] = row["model_close"] / previous["model_close"] - 1.0


def _bridge_factor(tier: str, stooq: Sequence[Mapping[str, Any]], alpaca: Sequence[Mapping[str, Any]]) -> float:
    if tier != "TIER_B_COMPATIBLE_WITH_PRICE_BRIDGE":
        return 1.0
    stooq_by_date = {row["session_date"]: row for row in stooq}
    ratios = [float(stooq_by_date[row["session_date"]]["close"]) / float(row["close"]) for row in alpaca if row["session_date"] in stooq_by_date and float(row["close"]) > 0]
    return statistics.median(ratios) if ratios else 1.0


def _read_stooq_rows(symbol: str) -> list[dict[str, Any]]:
    path = _stooq_path(symbol)
    if not path.exists():
        return []
    table = pq.read_table(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return [
        {
            "session_date": row["timestamp"].date().isoformat() if hasattr(row["timestamp"], "date") else str(row["timestamp"])[:10],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "source_path": str(path),
        }
        for row in table.to_pylist()
        if row["close"] is not None and math.isfinite(float(row["close"]))
    ]


def _read_alpaca_rows(symbol: str) -> list[dict[str, Any]]:
    path = ALPACA_ROOT / f"symbol={symbol}" / "year=2026" / "bars.parquet"
    if not path.exists():
        return []
    table = pq.read_table(path)
    return [
        {
            "session_date": str(row["session_date"]),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "feed": row.get("feed", "sip"),
            "adjustment_policy": row.get("adjustment_policy", "all"),
            "source_path": str(path),
        }
        for row in table.to_pylist()
    ]


def _stooq_path(symbol: str) -> Path:
    mapped = symbol.replace("-", ".").upper()
    path = STOOQ_ROOT / f"{mapped}.parquet"
    return path if path.exists() else STOOQ_ROOT / f"{symbol.upper()}.parquet"


def _assets() -> dict[str, dict[str, str]]:
    rows = _read_csv(REGISTRY_PATH)
    assets = {}
    for row in rows:
        symbol = row.get("canonical_symbol") or row.get("symbol")
        if symbol:
            assets[symbol] = {"asset_id": row.get("asset_id", ""), "canonical_symbol": symbol}
    return assets


def _quarantines() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for row in _read_csv(RESIDUAL_ROOT / "quarantined_symbol_dates.csv"):
        result[row["canonical_symbol"]][row["session_date"]] = row.get("quarantine_reason", "quarantined")
    return result


def _read_by_symbol(path: Path) -> dict[str, dict[str, str]]:
    return {row["canonical_symbol"]: row for row in _read_csv(path) if row.get("canonical_symbol")}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _completed_symbols(manifest_root: Path) -> set[str]:
    completed = set()
    for path in manifest_root.glob("*.json"):
        payload = _read_json(path)
        if payload.get("status") == "COMPLETE":
            completed.add(str(payload.get("symbol")))
    return completed


def _aggregate_symbol_reports(manifest_root: Path, symbols: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    source_selection = []
    transitions = []
    bridges = []
    quarantined = []
    eligibility = []
    for symbol in symbols:
        payload = _read_json(manifest_root / f"{_safe_symbol(symbol)}.json")
        if not payload:
            continue
        source_selection.append(
            {
                "asset_id": payload["asset_id"],
                "canonical_symbol": symbol,
                "stooq_rows_selected": payload["stooq_rows_selected"],
                "alpaca_rows_selected": payload["alpaca_rows_selected"],
                "first_alpaca_session": payload["first_alpaca_session"],
                "compatibility_tier": payload["compatibility_tier"],
            }
        )
        if payload.get("first_alpaca_session"):
            transitions.append(
                {
                    "asset_id": payload["asset_id"],
                    "canonical_symbol": symbol,
                    "transition_date": payload["first_alpaca_session"],
                    "from_provider": "stooq",
                    "to_provider": "alpaca",
                    "compatibility_tier": payload["compatibility_tier"],
                    "price_bridge_factor": payload["price_bridge_factor"],
                }
            )
        if payload.get("price_bridge_count"):
            bridges.append(
                {
                    "asset_id": payload["asset_id"],
                    "canonical_symbol": symbol,
                    "price_bridge_factor": payload["price_bridge_factor"],
                    "price_bridge_method": "median_overlap_ratio",
                    "calibration_start": "",
                    "calibration_end": payload["first_alpaca_session"],
                }
            )
        eligibility.append(
            {
                "asset_id": payload["asset_id"],
                "canonical_symbol": symbol,
                "compatibility_tier": payload["compatibility_tier"],
                "row_count": payload["row_count"],
                "selector_eligible_rows": payload["selector_eligible_rows"],
                "quarantined_rows": payload["quarantined_rows"],
                "tier_d_rows": payload["row_count"] if payload["compatibility_tier"] == "TIER_D_SYMBOL_QUARANTINE" else 0,
            }
        )
    quarantine_rows = _read_csv(RESIDUAL_ROOT / "quarantined_symbol_dates.csv")
    wanted = set(symbols)
    for row in quarantine_rows:
        if row["canonical_symbol"] in wanted:
            quarantined.append(row)
    return {"source_selection": source_selection, "provider_transitions": transitions, "price_bridges": bridges, "quarantined_rows": quarantined, "eligibility": eligibility}


def _progress(report_root: Path, planned: int, completed: int, failed: int, rows_written: int, started: float) -> None:
    elapsed = max(0.001, time.perf_counter() - started)
    payload = {
        "planned_partitions": planned,
        "completed_partitions": completed,
        "pending_partitions": max(0, planned - completed - failed),
        "failed_partitions": failed,
        "rows_written": rows_written,
        "elapsed_seconds": elapsed,
        "partitions_per_minute": completed / elapsed * 60.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(report_root / "progress_manifest.json", payload)
    print(
        f"[canonical-v2] completed={completed}/{planned} failed={failed} rows={rows_written} "
        f"elapsed={elapsed:.1f}s ppm={payload['partitions_per_minute']:.2f}",
        flush=True,
    )


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(".", "_")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

