from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from infrastructure.data.historical_bar_staging import validate_normalized_bars
from infrastructure.data.market_sessions import (
    EASTERN,
    RTH_OPEN,
    expected_rth_timestamps,
    is_trading_session,
    rth_close_for_date,
)


def diagnose_staging_session_gaps(
    *,
    staging_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    requested_start: datetime,
    requested_end: datetime,
) -> dict[str, Any]:
    rows = _read_staging_rows(Path(staging_path))
    manifest = _read_json(Path(manifest_path))
    chunks = list(manifest.get("chunks", {}).values())
    expected_by_date = _expected_sessions_by_date(requested_start, requested_end)
    all_expected_dates = sorted(expected_by_date)
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row["symbol"]).upper()].append(row)

    fully_missing = []
    incomplete = []
    for symbol in sorted(by_symbol):
        symbol_rows = sorted(by_symbol[symbol], key=lambda row: row["timestamp"])
        all_local_dates = {
            _local_date(row["timestamp"])
            for row in symbol_rows
        }
        rth_rows = [
            row for row in symbol_rows if row.get("session_type") == "rth"
        ]
        rth_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for row in rth_rows:
            rth_by_date[_local_date(row["timestamp"])].append(row)
        min_date = min(all_local_dates) if all_local_dates else None
        max_date = max(all_local_dates) if all_local_dates else None
        for session_date in all_expected_dates:
            expected = expected_by_date[session_date]
            observed_rows = sorted(rth_by_date.get(session_date, []), key=lambda row: row["timestamp"])
            observed = {row["timestamp"] for row in observed_rows}
            missing = [timestamp for timestamp in expected if timestamp not in observed]
            if not missing:
                continue
            has_before = any(local_date < session_date for local_date in all_local_dates)
            has_after = any(local_date > session_date for local_date in all_local_dates)
            active = bool(min_date and max_date and min_date <= session_date <= max_date)
            chunk = _chunk_for_symbol_date(chunks, symbol, session_date)
            if not observed_rows:
                fully_missing.append(
                    {
                        "symbol": symbol,
                        "local_exchange_date": session_date.isoformat(),
                        "expected_session_open": _local_session_timestamp(session_date, RTH_OPEN).isoformat(),
                        "expected_session_close": _local_session_timestamp(session_date, rth_close_for_date(session_date)).isoformat(),
                        "exchange_should_have_been_open": is_trading_session(session_date),
                        "symbol_had_data_before_date": has_before,
                        "symbol_had_data_after_date": has_after,
                        "symbol_plausibly_active_from_local_history": active,
                        "collection_chunk_id": chunk.get("chunk_id") if chunk else None,
                        "collection_chunk_status": chunk.get("status") if chunk else None,
                        "classification": _classify_fully_missing(
                            session_date=session_date,
                            active=active,
                            chunk=chunk,
                        ),
                    }
                )
                continue
            incomplete.append(
                {
                    "symbol": symbol,
                    "local_exchange_date": session_date.isoformat(),
                    "expected_bar_count": len(expected),
                    "observed_bar_count": len(observed_rows),
                    "missing_timestamps": [timestamp.isoformat() for timestamp in missing],
                    "bars_immediately_before_and_after_gap": [
                        {
                            "missing_timestamp": timestamp.isoformat(),
                            "before": _summarize_row(_previous_row(observed_rows, timestamp)),
                            "after": _summarize_row(_next_row(observed_rows, timestamp)),
                        }
                        for timestamp in missing
                    ],
                    "exchange_should_have_been_open": is_trading_session(session_date),
                    "collection_chunk_id": chunk.get("chunk_id") if chunk else None,
                    "collection_chunk_status": chunk.get("status") if chunk else None,
                    "classification": _classify_incomplete(chunk=chunk),
                }
            )

    structural = validate_normalized_bars(rows)
    duplicate_count = _duplicate_count(rows)
    classification_counts = Counter(item["classification"] for item in fully_missing)
    report = {
        "staging_path": str(Path(staging_path)),
        "manifest_path": str(Path(manifest_path)),
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "calendar_source": "infrastructure.data.market_sessions",
        "calendar_coverage": "NYSE-like regular sessions for 2016-2026, America/New_York timezone",
        "all_session_row_count": len(rows),
        "rth_row_count": sum(1 for row in rows if row.get("session_type") == "rth"),
        "outside_rth_row_count": sum(1 for row in rows if row.get("session_type") != "rth"),
        "duplicate_count": duplicate_count,
        "structural_validity": structural["valid"],
        "normal_full_day_rth_5m_bar_count": 78,
        "early_close_rth_5m_bar_count": 42,
        "fully_missing_session_count": len(fully_missing),
        "fully_missing_classification_counts": dict(sorted(classification_counts.items())),
        "incomplete_session_count": len(incomplete),
        "missing_rth_bars_after_excluding_invalid_expectations": (
            sum(78 if _session_bar_count(item["local_exchange_date"]) == 78 else 42 for item in fully_missing if item["classification"] not in {"calendar_false_positive", "symbol_lifecycle_gap"})
            + sum(item["expected_bar_count"] - item["observed_bar_count"] for item in incomplete)
        ),
        "fully_missing_sessions": fully_missing,
        "incomplete_sessions": incomplete,
        "no_forward_fill": True,
        "no_synthetic_5m_bars": True,
        "no_stale_close_carry": True,
    }
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _read_staging_rows(path: Path) -> list[dict[str, Any]]:
    columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session_type",
        "raw_chunk_identifier",
    ]
    return pq.read_table(path, columns=columns).to_pylist()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_sessions_by_date(start: datetime, end: datetime) -> dict[date, list[datetime]]:
    sessions: dict[date, list[datetime]] = defaultdict(list)
    for timestamp in expected_rth_timestamps(start, end, step=timedelta(minutes=5)):
        sessions[_local_date(timestamp)].append(timestamp)
    return dict(sessions)


def _local_date(timestamp: datetime) -> date:
    return timestamp.astimezone(EASTERN).date()


def _local_session_timestamp(session_date: date, session_time: Any) -> datetime:
    return datetime.combine(session_date, session_time, tzinfo=EASTERN).astimezone(timezone.utc)


def _chunk_for_symbol_date(chunks: Sequence[Mapping[str, Any]], symbol: str, session_date: date) -> Mapping[str, Any] | None:
    session_start = datetime.combine(session_date, RTH_OPEN, tzinfo=EASTERN).astimezone(timezone.utc)
    matches = []
    for chunk in chunks:
        symbols = {str(value).upper() for value in chunk.get("symbols", [])}
        if symbol not in symbols:
            continue
        start = datetime.fromisoformat(str(chunk["start"]))
        end = datetime.fromisoformat(str(chunk["end"]))
        if start <= session_start <= end:
            matches.append(chunk)
    completed = [chunk for chunk in matches if chunk.get("status") in {"completed", "skipped_completed"}]
    return completed[0] if completed else matches[0] if matches else None


def _classify_fully_missing(*, session_date: date, active: bool, chunk: Mapping[str, Any] | None) -> str:
    if not is_trading_session(session_date):
        return "calendar_false_positive"
    if not chunk or chunk.get("status") not in {"completed", "skipped_completed"}:
        return "symbol_lifecycle_gap" if not active else "collection_gap"
    if not active:
        return "collection_gap"
    return "provider_empty_session"


def _classify_incomplete(*, chunk: Mapping[str, Any] | None) -> str:
    if not chunk or chunk.get("status") not in {"completed", "skipped_completed"}:
        return "collection_gap"
    return "provider_or_no_trade_interval"


def _previous_row(rows: Sequence[Mapping[str, Any]], timestamp: datetime) -> Mapping[str, Any] | None:
    previous = [row for row in rows if row["timestamp"] < timestamp]
    return previous[-1] if previous else None


def _next_row(rows: Sequence[Mapping[str, Any]], timestamp: datetime) -> Mapping[str, Any] | None:
    for row in rows:
        if row["timestamp"] > timestamp:
            return row
    return None


def _summarize_row(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "timestamp": row["timestamp"].isoformat(),
        "open": row.get("open"),
        "high": row.get("high"),
        "low": row.get("low"),
        "close": row.get("close"),
        "volume": row.get("volume"),
        "session_type": row.get("session_type"),
        "raw_chunk_identifier": row.get("raw_chunk_identifier"),
    }


def _duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    counts = Counter((str(row["symbol"]).upper(), row["timestamp"]) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def _session_bar_count(date_text: str) -> int:
    close = rth_close_for_date(date.fromisoformat(date_text))
    if close is None:
        return 0
    if close.hour == 13:
        return 42
    return 78
