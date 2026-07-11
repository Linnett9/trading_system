from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Mapping, Sequence


def audit_historical_bar_overlap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_provider: str,
    right_provider: str,
    relative_close_tolerance: float = 0.0005,
) -> dict[str, Any]:
    left = {_key(row): row for row in left_rows if _key(row) is not None}
    right = {_key(row): row for row in right_rows if _key(row) is not None}
    left_keys = set(left)
    right_keys = set(right)
    matched = sorted(left_keys & right_keys)
    left_only = left_keys - right_keys
    right_only = right_keys - left_keys
    ohlc_differences = []
    relative_close_differences = []
    volume_differences = []
    session_alignment_differences = 0
    adjustment_differences = 0
    for key in matched:
        left_row = left[key]
        right_row = right[key]
        ohlc_delta = {
            field: _number(left_row.get(field)) - _number(right_row.get(field))
            for field in ("open", "high", "low", "close")
        }
        if any(abs(value) > 0.0 for value in ohlc_delta.values()):
            ohlc_differences.append({"key": _serial_key(key), **ohlc_delta})
        right_close = _number(right_row.get("close"))
        relative_close = (
            (_number(left_row.get("close")) - right_close) / right_close
            if right_close
            else 0.0
        )
        if abs(relative_close) > relative_close_tolerance:
            relative_close_differences.append({"key": _serial_key(key), "relative_close_difference": relative_close})
        volume_delta = _number(left_row.get("volume")) - _number(right_row.get("volume"))
        if abs(volume_delta) > 0.0:
            volume_differences.append({"key": _serial_key(key), "volume_difference": volume_delta})
        if left_row.get("session_policy") != right_row.get("session_policy"):
            session_alignment_differences += 1
        if left_row.get("adjustment_mode") != right_row.get("adjustment_mode"):
            adjustment_differences += 1
    return {
        "left_provider": left_provider,
        "right_provider": right_provider,
        "canonical_source_selected": False,
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "left_duplicate_key_count": _duplicate_count(left_rows),
        "right_duplicate_key_count": _duplicate_count(right_rows),
        "matched_key_count": len(matched),
        "timestamp_match_rate": len(matched) / len(left_keys | right_keys) if left_keys or right_keys else 0.0,
        "missing_keys_on_left_count": len(right_only),
        "missing_keys_on_right_count": len(left_only),
        "ohlc_difference_count": len(ohlc_differences),
        "relative_close_difference_count": len(relative_close_differences),
        "volume_difference_count": len(volume_differences),
        "session_alignment_difference_count": session_alignment_differences,
        "adjustment_difference_count": adjustment_differences,
        "sample_ohlc_differences": ohlc_differences[:10],
        "sample_relative_close_differences": relative_close_differences[:10],
        "sample_volume_differences": volume_differences[:10],
    }


def _key(row: Mapping[str, Any]) -> tuple[str, datetime] | None:
    symbol = str(row.get("symbol", "")).upper()
    timestamp = row.get("timestamp")
    if not symbol or not isinstance(timestamp, datetime):
        return None
    return symbol, timestamp


def _duplicate_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [_key(row) for row in rows if _key(row) is not None]
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _serial_key(key: tuple[str, datetime]) -> dict[str, str]:
    return {"symbol": key[0], "timestamp": key[1].isoformat()}
