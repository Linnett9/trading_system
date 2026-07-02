from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _load_raw_price_ranges(data_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(data_dir.glob("*.parquet")):
        summary = _parquet_date_range(path)
        if summary:
            output.append(summary)
    return output
def _load_adjusted_price_ranges(data_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == "manifest.json":
            continue
        summary = _csv_date_range(path, date_columns=("date", "Date", "timestamp"))
        if summary:
            summary["symbol"] = path.stem.upper()
            output.append(summary)
    return output
def _prediction_artifact_range(path: Path) -> dict[str, Any]:
    summary = _csv_date_range(
        path,
        date_columns=("rebalance_date", "prediction_date", "date", "feature_date"),
    )
    if not summary:
        return {
            "path": str(path),
            "available": False,
            "row_count": 0,
            "unique_rebalance_dates": 0,
        }
    summary["name"] = path.parent.name
    return summary
def _parquet_date_range(path: Path) -> dict[str, Any] | None:
    try:
        import pandas as pd
    except ImportError:
        return {
            "path": str(path),
            "symbol": path.stem.upper(),
            "available": False,
            "row_count": 0,
            "error": "pandas_not_available_for_parquet_scan",
        }
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:
        return {
            "path": str(path),
            "symbol": path.stem.upper(),
            "available": False,
            "row_count": 0,
            "error": str(exc),
        }
    if frame.empty:
        return None
    dates = [str(value)[:10] for value in frame["timestamp"].dropna().tolist()]
    if not dates:
        return None
    return {
        "path": str(path),
        "symbol": path.stem.upper(),
        "available": True,
        "row_count": len(dates),
        "earliest_date": min(dates),
        "latest_date": max(dates),
    }
def _csv_date_range(path: Path, *, date_columns: tuple[str, ...]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row_count = 0
    dates = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            value = next((row.get(name) for name in date_columns if row.get(name)), None)
            if value:
                dates.append(str(value)[:10])
    if not dates:
        return {
            "path": str(path),
            "available": True,
            "row_count": row_count,
            "unique_rebalance_dates": 0,
        }
    return {
        "path": str(path),
        "available": True,
        "row_count": row_count,
        "earliest_date": min(dates),
        "latest_date": max(dates),
        "unique_rebalance_dates": len(set(dates)),
    }
def _aggregate_ranges(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    available = [row for row in rows if row.get("available") and row.get("earliest_date")]
    return {
        "name": name,
        "available": bool(available),
        "item_count": len(rows),
        "available_item_count": len(available),
        "earliest_date": min((row["earliest_date"] for row in available), default=None),
        "latest_date": max((row["latest_date"] for row in available), default=None),
        "latest_common_start_date": max(
            (row["earliest_date"] for row in available),
            default=None,
        ),
        "earliest_common_end_date": min(
            (row["latest_date"] for row in available),
            default=None,
        ),
        "row_count": sum(int(row.get("row_count") or 0) for row in rows),
        "unique_rebalance_dates": max(
            (int(row.get("unique_rebalance_dates") or 0) for row in rows),
            default=0,
        ),
        "items": rows,
    }
def _range_summary(row: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": bool(row.get("available")),
        "path": row.get("path"),
        "row_count": int(row.get("row_count") or 0),
        "earliest_date": row.get("earliest_date"),
        "latest_date": row.get("latest_date"),
        "unique_rebalance_dates": int(row.get("unique_rebalance_dates") or 0),
    }
