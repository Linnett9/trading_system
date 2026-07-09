from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence

from core.research.framework.ranking import finite_number


def add_forward_return_targets(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon_bars: int,
    price_column: str = "close",
    allow_cross_session_horizon: bool = True,
    expected_bar_seconds: int | None = None,
    allow_missing_intermediate_bars: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least one")
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        symbol = str(row.get("symbol", "")).upper()
        if symbol and row.get("timestamp") is not None:
            by_symbol[symbol].append(row)

    output: list[dict[str, Any]] = []
    target_column = target_column_name(horizon_bars)
    maturity_column = label_maturity_column_name(horizon_bars)
    gap_column = session_gap_column_name(horizon_bars)
    matured_count = 0
    gap_count = 0
    cross_session_target_count = 0
    dropped_cross_session_target_count = 0
    missing_intermediate_gap_count = 0
    dropped_missing_intermediate_target_count = 0
    for symbol, symbol_rows in sorted(by_symbol.items()):
        ordered = sorted(symbol_rows, key=lambda item: item["timestamp"])
        for index, row in enumerate(ordered):
            enriched = dict(row)
            target_index = index + horizon_bars
            if target_index < len(ordered):
                window = ordered[index:target_index + 1]
                maturity = ordered[target_index]["timestamp"]
                enriched[maturity_column] = maturity
                gap = _session_gap_count(window)
                enriched[gap_column] = gap
                gap_count += gap
                missing_gaps = _missing_intermediate_gap_count(
                    window,
                    expected_bar_seconds=expected_bar_seconds,
                )
                missing_intermediate_gap_count += missing_gaps
                if gap:
                    cross_session_target_count += 1
                if gap and not allow_cross_session_horizon:
                    enriched[target_column] = None
                    dropped_cross_session_target_count += 1
                elif missing_gaps and not allow_missing_intermediate_bars:
                    enriched[target_column] = None
                    dropped_missing_intermediate_target_count += 1
                else:
                    start = finite_number(row.get(price_column))
                    end = finite_number(ordered[target_index].get(price_column))
                    if start is not None and end is not None and start > 0.0:
                        enriched[target_column] = end / start - 1.0
                        matured_count += 1
                    else:
                        enriched[target_column] = None
            else:
                enriched[target_column] = None
                enriched[maturity_column] = None
                enriched[gap_column] = None
            output.append(enriched)
    output.sort(key=lambda item: (item["timestamp"], str(item["symbol"])))
    return output, {
        "target_column": target_column,
        "label_maturity_column": maturity_column,
        "session_gap_column": gap_column,
        "horizon_bars": horizon_bars,
        "input_row_count": len(rows),
        "target_row_count": matured_count,
        "immature_row_count": len(rows) - matured_count,
        "session_gap_count": gap_count,
        "cross_session_target_count": cross_session_target_count,
        "dropped_cross_session_target_count": dropped_cross_session_target_count,
        "missing_intermediate_gap_count": missing_intermediate_gap_count,
        "dropped_missing_intermediate_target_count": dropped_missing_intermediate_target_count,
        "allow_cross_session_horizon": allow_cross_session_horizon,
        "expected_bar_seconds": expected_bar_seconds,
        "allow_missing_intermediate_bars": allow_missing_intermediate_bars,
        "label_price_column": price_column,
    }


def target_column_name(horizon_bars: int) -> str:
    return f"target_forward_return_{int(horizon_bars)}b"


def label_maturity_column_name(horizon_bars: int) -> str:
    return f"label_maturity_timestamp_{int(horizon_bars)}b"


def session_gap_column_name(horizon_bars: int) -> str:
    return f"session_gap_count_{int(horizon_bars)}b"


def label_is_mature(row: Mapping[str, Any], *, horizon_bars: int, fit_cutoff: datetime) -> bool:
    maturity = row.get(label_maturity_column_name(horizon_bars))
    return isinstance(maturity, datetime) and maturity <= fit_cutoff


def _session_gap_count(rows: Sequence[Mapping[str, Any]]) -> int:
    dates = [
        value.date()
        for row in rows
        if isinstance((value := row.get("timestamp")), datetime)
    ]
    return sum(1 for left, right in zip(dates, dates[1:]) if right != left)


def _missing_intermediate_gap_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_bar_seconds: int | None,
) -> int:
    if expected_bar_seconds is None or expected_bar_seconds <= 0:
        return 0
    timestamps = [
        value
        for row in rows
        if isinstance((value := row.get("timestamp")), datetime)
    ]
    threshold = expected_bar_seconds * 1.5
    return sum(
        1
        for left, right in zip(timestamps, timestamps[1:])
        if (right - left).total_seconds() > threshold
    )
