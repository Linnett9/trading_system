from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.compute as pc
import pyarrow.parquet as pq


METADATA_COLUMNS = ("timestamp", "symbol", "timeframe")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class FeatureBankSlice:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    metadata_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]


def default_feature_bank_path(timeframe: str) -> Path:
    canonical = canonical_timeframe(timeframe)
    return Path("cache/ml/features") / f"stock_features_{canonical}.parquet"


def load_feature_bank_slice(
    timeframe: str,
    *,
    path: str | Path | None = None,
    symbols: Sequence[str] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    columns: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> FeatureBankSlice:
    canonical = canonical_timeframe(timeframe)
    feature_path = Path(path) if path is not None else default_feature_bank_path(canonical)
    parquet = pq.ParquetFile(feature_path)
    schema_names = tuple(parquet.schema_arrow.names)
    requested_columns = _requested_columns(schema_names, columns)
    filters = _filters(canonical, symbols=symbols, start=start, end=end)
    table = pq.read_table(feature_path, columns=list(requested_columns), filters=filters or None)
    if max_rows is not None:
        table = table.slice(0, max(0, int(max_rows)))
    if table.num_rows:
        order = pc.sort_indices(table, sort_keys=[("timestamp", "ascending"), ("symbol", "ascending")])
        table = pc.take(table, order)
    rows = table.to_pylist()
    duplicate_count = _duplicate_key_count(rows)
    if duplicate_count:
        raise ValueError(f"duplicate feature-bank keys detected: {duplicate_count}")
    metadata_columns = tuple(name for name in METADATA_COLUMNS if name in table.column_names)
    label_columns = tuple(name for name in table.column_names if _is_label_column(name))
    feature_columns = tuple(
        name
        for name in table.column_names
        if name not in metadata_columns and name not in label_columns
    )
    metadata = _metadata(
        feature_path,
        parquet,
        rows,
        canonical,
        schema_names=schema_names,
        selected_columns=table.column_names,
        symbols=symbols,
        start=start,
        end=end,
    )
    return FeatureBankSlice(
        rows=rows,
        metadata=metadata,
        metadata_columns=metadata_columns,
        feature_columns=feature_columns,
        label_columns=label_columns,
    )


def canonical_timeframe(timeframe: str) -> str:
    value = str(timeframe)
    if value == "daily":
        return "1Day"
    if value == "hourly":
        return "1h"
    return value


def _requested_columns(schema_names: Sequence[str], columns: Sequence[str] | None) -> tuple[str, ...]:
    if columns is None:
        return tuple(schema_names)
    requested = []
    for name in (*METADATA_COLUMNS, *PRICE_COLUMNS, *columns):
        if name in schema_names and name not in requested:
            requested.append(name)
    return tuple(requested)


def _filters(
    timeframe: str,
    *,
    symbols: Sequence[str] | None,
    start: str | datetime | None,
    end: str | datetime | None,
) -> list[tuple[str, str, Any]]:
    filters: list[tuple[str, str, Any]] = [("timeframe", "==", timeframe)]
    if symbols:
        filters.append(("symbol", "in", [str(symbol).upper() for symbol in symbols]))
    start_ts = _timestamp(start)
    end_ts = _timestamp(end)
    if start_ts is not None:
        filters.append(("timestamp", ">=", start_ts))
    if end_ts is not None:
        filters.append(("timestamp", "<=", end_ts))
    return filters


def _metadata(
    path: Path,
    parquet: pq.ParquetFile,
    rows: Sequence[Mapping[str, Any]],
    timeframe: str,
    *,
    schema_names: Sequence[str],
    selected_columns: Sequence[str],
    symbols: Sequence[str] | None,
    start: str | datetime | None,
    end: str | datetime | None,
) -> dict[str, Any]:
    timestamps = [row["timestamp"] for row in rows if row.get("timestamp") is not None]
    selected_symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    return {
        "path": str(path),
        "timeframe": timeframe,
        "source_row_count": parquet.metadata.num_rows,
        "source_column_count": parquet.metadata.num_columns,
        "source_row_group_count": parquet.metadata.num_row_groups,
        "schema_columns": list(schema_names),
        "selected_columns": list(selected_columns),
        "row_count": len(rows),
        "symbol_count": len(selected_symbols),
        "symbols": selected_symbols,
        "timestamp_min": min(timestamps).isoformat() if timestamps else None,
        "timestamp_max": max(timestamps).isoformat() if timestamps else None,
        "requested_symbols": [str(symbol).upper() for symbol in symbols or []],
        "requested_start": _iso_or_none(start),
        "requested_end": _iso_or_none(end),
        "duplicate_key_count": 0,
    }


def _duplicate_key_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [
        (row.get("timestamp"), str(row.get("symbol", "")).upper())
        for row in rows
        if row.get("timestamp") is not None and row.get("symbol")
    ]
    return len(keys) - len(set(keys))


def _is_label_column(name: str) -> bool:
    return (
        name.startswith("target_")
        or name.startswith("actual_")
        or name.startswith("forward_return")
    )


def _timestamp(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _iso_or_none(value: str | datetime | None) -> str | None:
    parsed = _timestamp(value)
    return parsed.isoformat() if parsed is not None else None
