from __future__ import annotations

from typing import Any

from core.research.ml.audits.data_adjustment_validation_price_rows import (
    _normalized_price_rows,
    _split_like_factor,
)
from core.research.ml.audits.data_adjustment_validation_types import RESEARCH_METADATA
from core.research.ml.audits.data_adjustment_validation_utils import _number


def detect_split_like_jumps(
    symbol: str,
    rows: list[dict[str, Any]],
    *,
    suspicious_daily_return_abs: float = 0.50,
    impossible_daily_return_abs: float = 4.0,
    split_ratio_tolerance: float = 0.08,
) -> list[dict[str, Any]]:
    normalized = _normalized_price_rows(rows)
    events: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in normalized:
        close = _number(row.get("close"))
        if close is None or close <= 0.0:
            events.append({
                "symbol": symbol.upper(),
                "date": row.get("date"),
                "previous_date": previous.get("date") if previous else None,
                "previous_close": previous.get("close") if previous else None,
                "close": close,
                "daily_return": None,
                "price_ratio": None,
                "event_type": "impossible_ohlcv",
                "split_like_factor": None,
                "severity": "impossible",
                **RESEARCH_METADATA,
            })
            previous = row if close is not None and close > 0.0 else previous
            continue
        if previous is None:
            previous = row
            continue
        previous_close = _number(previous.get("close"))
        if previous_close is None or previous_close <= 0.0:
            previous = row
            continue
        ratio = close / previous_close
        daily_return = ratio - 1.0
        split_factor = _split_like_factor(ratio, split_ratio_tolerance)
        is_suspicious = abs(daily_return) >= suspicious_daily_return_abs
        is_impossible = abs(daily_return) >= impossible_daily_return_abs
        if split_factor is not None or is_suspicious or is_impossible:
            events.append({
                "symbol": symbol.upper(),
                "date": row.get("date"),
                "previous_date": previous.get("date"),
                "previous_close": previous_close,
                "close": close,
                "daily_return": daily_return,
                "price_ratio": ratio,
                "event_type": (
                    "impossible_daily_jump"
                    if is_impossible
                    else "split_like_jump"
                    if split_factor is not None
                    else "suspicious_daily_jump"
                ),
                "split_like_factor": split_factor,
                "severity": (
                    "impossible"
                    if is_impossible
                    else "suspicious"
                ),
                **RESEARCH_METADATA,
            })
        previous = row
    return events
