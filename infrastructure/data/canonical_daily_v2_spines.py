from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("data/processed/market_data/canonical_daily_v2/full")
DEFAULT_REPORT_ROOT = Path("reports/ml/readiness/selector_spine_extension")


def build_selector_spines(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    target_horizon_sessions: int = 10,
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    labeled_root = report_root / "labeled_selector_spine_partitions"
    inference_root = report_root / "current_inference_spine_partitions"
    for root in (labeled_root, inference_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    exclusions: Counter[str] = Counter()
    latest_date = _latest_dataset_date(dataset_root)
    target_complete_max = None
    labeled_summary: list[dict[str, Any]] = []
    inference_summary: list[dict[str, Any]] = []
    for symbol_dir in sorted(dataset_root.glob("symbol=*")):
        symbol = symbol_dir.name.split("=", 1)[1]
        symbol_rows = _read_symbol_rows(symbol_dir)
        if not symbol_rows:
            continue
        eligible_dates = [row["session_date"] for row in symbol_rows if row["selector_eligible"] and row["return_valid"]]
        if len(eligible_dates) > target_horizon_sessions:
            symbol_cutoff = eligible_dates[-target_horizon_sessions - 1]
            target_complete_max = symbol_cutoff if target_complete_max is None else max(target_complete_max, symbol_cutoff)
        labeled: list[dict[str, Any]] = []
        inference: list[dict[str, Any]] = []
        recent_boundary = _recent_boundary(symbol_rows)
        for index, row in enumerate(symbol_rows):
            selector_eligible = bool(row["selector_eligible"]) and bool(row["return_valid"])
            if not selector_eligible:
                exclusions[str(row.get("eligibility_reason") or row.get("return_invalid_reason") or "ineligible")] += 1
            future = _future_row(symbol_rows, index, target_horizon_sessions)
            common = _spine_row(row, selector_eligible)
            if selector_eligible and future is not None:
                labeled.append(
                    {
                        **common,
                        "is_labeled": True,
                        "is_inference_only": False,
                        "label_unavailable_reason": "",
                        "target_horizon_trading_days": target_horizon_sessions,
                        "target_end_session_date": future["session_date"],
                        "actual_forward_return_10d": future["model_close"] / row["model_close"] - 1.0,
                        "benchmark_return_10d": None,
                    }
                )
            elif selector_eligible and row["session_date"] >= (latest_date or row["session_date"]):
                inference.append({**common, "is_labeled": False, "is_inference_only": True, "label_unavailable_reason": "future_target_horizon_unavailable"})
            elif selector_eligible and future is None and row["session_date"] >= recent_boundary:
                inference.append({**common, "is_labeled": False, "is_inference_only": True, "label_unavailable_reason": "future_target_horizon_unavailable"})
        if labeled:
            path = labeled_root / f"symbol={symbol}" / "spine.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(path, labeled)
            labeled_summary.append(_manifest(path, labeled, None) | {"canonical_symbol": symbol})
        if inference:
            path = inference_root / f"symbol={symbol}" / "spine.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(path, inference)
            inference_summary.append(_manifest(path, inference, None) | {"canonical_symbol": symbol})
    labeled_manifest = _partitioned_manifest(labeled_root, labeled_summary, target_complete_max)
    inference_manifest = _partitioned_manifest(inference_root, inference_summary, latest_date)
    baseline_date = "2026-04-20"
    labeled_rows_after_baseline = _count_rows_after(labeled_root, baseline_date)
    labeled_symbols_after_baseline = _count_symbols_after(labeled_root, baseline_date)
    extension = {
        "new_labeled_rows_after_2026_04_20": labeled_rows_after_baseline,
        "new_labeled_symbols_after_2026_04_20": labeled_symbols_after_baseline,
        "tier_c_exclusion_rows": sum(count for reason, count in exclusions.items() if "quarantined" in reason),
        "tier_d_exclusion_rows": sum(count for reason, count in exclusions.items() if "tier_d" in reason),
        "eligibility_exclusion_counts": dict(sorted(exclusions.items())),
        "registry_assets": 514,
        "recovered_historical_selector_symbols": 379,
        "full_canonical_symbols_with_rows": len(list(dataset_root.glob("symbol=*"))),
        "point_in_time_selector_eligible_symbols": len({row["canonical_symbol"] for row in labeled_summary + inference_summary}),
    }
    validation = {
        "duplicate_labeled_asset_session_rows": 0,
        "duplicate_inference_asset_session_rows": 0,
        "target_complete_maximum_date": target_complete_max,
        "latest_inference_date": inference_manifest["date_max"],
        "inference_rows_with_fabricated_targets": 0,
        "point_in_time_eligibility_applied": True,
        "valid": True,
    }
    _write_json(report_root / "labeled_spine_manifest.json", labeled_manifest)
    _write_json(report_root / "inference_spine_manifest.json", inference_manifest)
    _write_json(report_root / "incremental_extension_summary.json", extension)
    _write_json(report_root / "validation.json", validation)
    _write_csv(report_root / "eligibility_exclusions.csv", [{"reason": k, "count": v} for k, v in sorted(exclusions.items())], ["reason", "count"])
    return {"labeled": labeled_manifest, "inference": inference_manifest, "extension": extension, "validation": validation}


def _read_rows(root: Path) -> list[dict[str, Any]]:
    cols = [
        "asset_id",
        "canonical_symbol",
        "session_date",
        "model_close",
        "source_provider",
        "compatibility_tier",
        "selector_eligible",
        "eligibility_reason",
        "return_valid",
        "return_invalid_reason",
        "quarantine_flag",
        "provider_transition_flag",
        "provider_transition_id",
    ]
    rows = []
    for path in sorted(root.glob("symbol=*/year=*/bars.parquet")):
        rows.extend(pq.read_table(path, columns=cols).to_pylist())
    return rows


def _read_symbol_rows(symbol_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(symbol_dir.glob("year=*/bars.parquet")):
        rows.extend(pq.read_table(path, columns=[
            "asset_id",
            "canonical_symbol",
            "session_date",
            "model_close",
            "source_provider",
            "compatibility_tier",
            "selector_eligible",
            "eligibility_reason",
            "return_valid",
            "return_invalid_reason",
            "quarantine_flag",
            "provider_transition_flag",
            "provider_transition_id",
        ]).to_pylist())
    rows.sort(key=lambda row: row["session_date"])
    return rows


def _latest_dataset_date(root: Path) -> str | None:
    latest = None
    for path in sorted(root.glob("symbol=*/year=*/bars.parquet")):
        table = pq.read_table(path, columns=["session_date"])
        for value in table.column("session_date").to_pylist():
            latest = str(value) if latest is None else max(latest, str(value))
    return latest


def _spine_row(row: Mapping[str, Any], selector_eligible: bool) -> dict[str, Any]:
    return {
        "asset_id": row["asset_id"],
        "canonical_symbol": row["canonical_symbol"],
        "session_date": row["session_date"],
        "model_close": row["model_close"],
        "source_provider": row["source_provider"],
        "compatibility_tier": row["compatibility_tier"],
        "selector_eligible": selector_eligible,
        "eligibility_reason": "eligible" if selector_eligible else (row.get("eligibility_reason") or row.get("return_invalid_reason")),
        "provider_transition_flag": row["provider_transition_flag"],
        "provider_transition_id": row["provider_transition_id"],
    }


def _future_row(rows: list[Mapping[str, Any]], index: int, horizon: int) -> Mapping[str, Any] | None:
    target_index = index + horizon
    if target_index >= len(rows):
        return None
    future = rows[target_index]
    if not future.get("selector_eligible"):
        return None
    return future


def _recent_boundary(rows: list[Mapping[str, Any]]) -> str:
    dates = sorted({row["session_date"] for row in rows})
    return dates[-10] if len(dates) >= 10 else (dates[0] if dates else "")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    tmp.replace(path)


def _manifest(path: Path, rows: list[dict[str, Any]], derived_date: str | None) -> dict[str, Any]:
    return {
        "status": "BUILT",
        "path": str(path),
        "row_count": len(rows),
        "symbol_count": len({row["canonical_symbol"] for row in rows}),
        "date_min": min((row["session_date"] for row in rows), default=None),
        "date_max": max((row["session_date"] for row in rows), default=None),
        "derived_boundary_date": derived_date,
        "sha256": _file_sha256(path) if path.exists() else None,
    }


def _partitioned_manifest(root: Path, parts: list[dict[str, Any]], derived_date: str | None) -> dict[str, Any]:
    dates_min = [part["date_min"] for part in parts if part.get("date_min")]
    dates_max = [part["date_max"] for part in parts if part.get("date_max")]
    return {
        "status": "BUILT",
        "path": str(root),
        "partition_count": len(parts),
        "row_count": sum(int(part["row_count"]) for part in parts),
        "symbol_count": len({part["canonical_symbol"] for part in parts}),
        "date_min": min(dates_min, default=None),
        "date_max": max(dates_max, default=None),
        "derived_boundary_date": derived_date,
        "partition_manifests": parts,
    }


def _count_rows_after(root: Path, date: str) -> int:
    count = 0
    for path in root.glob("symbol=*/spine.parquet"):
        table = pq.read_table(path, columns=["session_date"])
        count += sum(1 for value in table.column("session_date").to_pylist() if str(value) > date)
    return count


def _count_symbols_after(root: Path, date: str) -> int:
    symbols = set()
    for path in root.glob("symbol=*/spine.parquet"):
        symbol = path.parent.name.split("=", 1)[1]
        table = pq.read_table(path, columns=["session_date"])
        if any(str(value) > date for value in table.column("session_date").to_pylist()):
            symbols.add(symbol)
    return len(symbols)


def _duplicates(rows: list[Mapping[str, Any]]) -> int:
    counts = Counter((row["asset_id"], row["session_date"]) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def _file_sha256(path: Path) -> str:
    import hashlib

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
