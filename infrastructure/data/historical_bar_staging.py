from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from infrastructure.data.market_sessions import expected_rth_timestamps, is_rth_timestamp


STAGING_COLUMNS = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "provider",
    "feed",
    "collection_timestamp",
    "requested_timeframe",
    "native_timeframe",
    "adjustment_mode",
    "extended_hours",
    "session_policy",
    "session_type",
    "raw_chunk_identifier",
    "normalizer_version",
)


def validate_normalized_bars(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    duplicate_count = _duplicate_count(rows)
    invalid_ohlc = 0
    invalid_volume = 0
    invalid_timestamp = 0
    invalid_numeric = 0
    by_symbol: dict[str, list[datetime]] = defaultdict(list)
    for row in rows:
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, datetime):
            invalid_timestamp += 1
            continue
        if timestamp.tzinfo is None:
            invalid_timestamp += 1
        by_symbol[str(row.get("symbol", "")).upper()].append(timestamp)
        values = [_float(row.get(field)) for field in ("open", "high", "low", "close")]
        if any(value is None for value in values):
            invalid_numeric += 1
            continue
        open_, high, low, close = values
        if high < max(open_, close, low) or low > min(open_, close, high):
            invalid_ohlc += 1
        volume = _float(row.get("volume"))
        if volume is None or volume < 0:
            invalid_volume += 1
    unsorted_symbols = [
        symbol for symbol, timestamps in by_symbol.items() if timestamps != sorted(timestamps)
    ]
    return {
        "row_count": len(rows),
        "duplicate_key_count": duplicate_count,
        "invalid_ohlc_count": invalid_ohlc,
        "invalid_volume_count": invalid_volume,
        "invalid_timestamp_count": invalid_timestamp,
        "invalid_numeric_price_count": invalid_numeric,
        "unsorted_symbol_count": len(unsorted_symbols),
        "valid": not any([duplicate_count, invalid_ohlc, invalid_volume, invalid_timestamp, invalid_numeric, unsorted_symbols]),
        "timestamp_storage": "UTC timezone-aware datetimes",
        "timestamp_semantics": "bar interval start as returned by Alpaca",
        "duplicate_policy": "reject conflicting duplicates; identical duplicates may be dropped by consolidation",
    }


def write_staging_parquet(rows: Sequence[Mapping[str, Any]], path: str | Path) -> dict[str, Any]:
    serial = [_staging_row(row) for row in rows]
    table = pa.Table.from_pylist(serial, schema=_schema())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return {"path": str(path), "row_count": len(serial), "validation": validate_normalized_bars(rows)}


def consolidate_staging_chunks(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_conflicting_duplicates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
    conflicts = []
    duplicate_count = 0
    for raw in sorted(rows, key=lambda row: (str(row.get("symbol", "")), row.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc), str(row.get("raw_chunk_identifier", "")))):
        row = dict(raw)
        key = (str(row["symbol"]).upper(), row["timestamp"])
        if key in by_key:
            duplicate_count += 1
            existing = by_key[key]
            if any(_float(existing.get(field)) != _float(row.get(field)) for field in ("open", "high", "low", "close", "volume")):
                conflicts.append({"symbol": key[0], "timestamp": key[1].isoformat()})
                if not allow_conflicting_duplicates:
                    continue
            existing["source_chunk_ids"] = sorted(set(existing.get("source_chunk_ids", [existing.get("raw_chunk_identifier")]) + [row.get("raw_chunk_identifier")]))
            continue
        row["source_chunk_ids"] = [row.get("raw_chunk_identifier")]
        by_key[key] = row
    if conflicts and not allow_conflicting_duplicates:
        raise ValueError(f"conflicting duplicate bars detected: {len(conflicts)}")
    output = sorted(by_key.values(), key=lambda row: (row["symbol"], row["timestamp"]))
    return output, {
        "input_row_count": len(rows),
        "output_row_count": len(output),
        "duplicate_key_count": duplicate_count,
        "conflicting_duplicate_count": len(conflicts),
        "provenance_retained": True,
    }


def coverage_gap_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
    provider: str,
    feed: str,
    holidays: set[str] | None = None,
    early_closes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_rth_row(row):
            by_symbol[str(row.get("symbol", "")).upper()].append(row)
    reports = []
    for symbol, symbol_rows in sorted(by_symbol.items()):
        ordered = sorted(symbol_rows, key=lambda row: row["timestamp"])
        timestamps = [row["timestamp"] for row in ordered]
        gaps = [
            (right - left)
            for left, right in zip(timestamps, timestamps[1:])
            if _is_intraday_gap(left, right, timeframe)
        ]
        expected = _expected_rth_timestamps(
            requested_start,
            requested_end,
            timeframe,
            holidays=holidays,
            early_closes=early_closes,
        )
        actual = set(timestamps)
        missing = sorted(ts for ts in expected if ts not in actual)
        expected_by_session = _dates(expected)
        observed_by_session = _dates(timestamps)
        missing_by_session = _dates(missing)
        fully_missing = sorted(expected_by_session - observed_by_session)
        incomplete = sorted((missing_by_session & observed_by_session) - set(fully_missing))
        reports.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "requested_start": requested_start.isoformat(),
                "requested_end": requested_end.isoformat(),
                "observed_start": min(timestamps).isoformat() if timestamps else None,
                "observed_end": max(timestamps).isoformat() if timestamps else None,
                "row_count": len(ordered),
                "expected_session_count": len(expected_by_session),
                "observed_session_count": len(observed_by_session),
                "session_count": len(observed_by_session),
                "duplicate_count": _duplicate_count(ordered),
                "missing_expected_rth_bars": len(missing),
                "fully_missing_session_count": len(fully_missing),
                "incomplete_session_count": len(incomplete),
                "missing_session_count": len(fully_missing),
                "intraday_gap_count": len(gaps),
                "longest_gap_seconds": max((gap.total_seconds() for gap in gaps), default=0.0),
                "empty_windows": 1 if not ordered else 0,
                "provider": provider,
                "feed": feed,
                "research_view": "rth_only",
                "no_forward_fill": True,
                "no_synthetic_bars": True,
                "no_stale_close_carry": True,
                "structural_validity_status": "valid" if validate_normalized_bars(ordered)["valid"] else "invalid",
                "completeness_status": "complete" if not missing and ordered else "empty" if not ordered else "incomplete",
            }
        )
    return reports


def aggregate_5m_to_1h(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, datetime], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        ts = row["timestamp"]
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        buckets[(str(row["symbol"]).upper(), bucket)].append(row)
    output = []
    for (symbol, bucket), bucket_rows in sorted(buckets.items()):
        ordered = sorted(bucket_rows, key=lambda row: row["timestamp"])
        output.append(
            {
                "symbol": symbol,
                "timestamp": bucket,
                "open": float(ordered[0]["open"]),
                "high": max(float(row["high"]) for row in ordered),
                "low": min(float(row["low"]) for row in ordered),
                "close": float(ordered[-1]["close"]),
                "volume": sum(float(row["volume"]) for row in ordered),
                "partial_input_bar_count": len(ordered),
            }
        )
    return output


def compare_aggregated_1h(aggregated: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from infrastructure.data.historical_bar_overlap import audit_historical_bar_overlap

    report = audit_historical_bar_overlap(
        aggregated,
        existing,
        left_provider="local_5m_aggregation",
        right_provider="existing_1h",
    )
    report["recommendation"] = (
        "derive_1h_locally_from_validated_5m"
        if report["relative_close_difference_count"] == 0 and report["missing_keys_on_left_count"] == 0
        else "collect_or_validate_1h_independently_before_use"
    )
    return report


def _staging_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in STAGING_COLUMNS}


def _schema() -> pa.Schema:
    return pa.schema(
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


def _duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [(str(row.get("symbol", "")).upper(), row.get("timestamp")) for row in rows]
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_intraday_gap(left: datetime, right: datetime, timeframe: str) -> bool:
    if left.date() != right.date() or left.weekday() >= 5 or right.weekday() >= 5:
        return False
    expected = timedelta(minutes=5 if timeframe in {"5m", "5Min"} else 60)
    return right - left > expected * 1.5


def _is_rth_row(row: Mapping[str, Any]) -> bool:
    if row.get("session_type") == "rth":
        return True
    if row.get("session_type") in {"pre_market", "after_hours"}:
        return False
    timestamp = row.get("timestamp")
    if not isinstance(timestamp, datetime):
        return False
    return is_rth_timestamp(timestamp)


def _expected_rth_timestamps(
    start: datetime,
    end: datetime,
    timeframe: str,
    *,
    holidays: set[str] | None = None,
    early_closes: Mapping[str, str] | None = None,
) -> list[datetime]:
    if timeframe not in {"5m", "5Min"}:
        return []
    step = timedelta(minutes=5)
    if holidays or early_closes:
        return _expected_rth_timestamps_override(start, end, step=step, holidays=holidays or set(), early_closes=early_closes or {})
    return expected_rth_timestamps(start, end, step=step)


def _expected_rth_timestamps_override(
    start: datetime,
    end: datetime,
    *,
    step: timedelta,
    holidays: set[str],
    early_closes: Mapping[str, str],
) -> list[datetime]:
    from datetime import date, time
    from infrastructure.data.market_sessions import EASTERN, RTH_OPEN, RTH_CLOSE

    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    day = start_utc.astimezone(EASTERN).date()
    last = end_utc.astimezone(EASTERN).date()
    output = []
    while day <= last:
        day_text = day.isoformat()
        if day.weekday() < 5 and day_text not in holidays:
            close_text = early_closes.get(day_text)
            close = RTH_CLOSE
            if close_text:
                hour, minute = [int(part) for part in close_text.split(":", maxsplit=1)]
                close = time(hour, minute)
            current = datetime.combine(day, RTH_OPEN, tzinfo=EASTERN)
            session_end = datetime.combine(day, close, tzinfo=EASTERN)
            while current < session_end:
                current_utc = current.astimezone(timezone.utc)
                if start_utc <= current_utc <= end_utc:
                    output.append(current_utc)
                current += step
        day = date.fromordinal(day.toordinal() + 1)
    return output


def _dates(timestamps: Sequence[datetime]) -> set[str]:
    return {timestamp.date().isoformat() for timestamp in timestamps}
