from __future__ import annotations

import math
from typing import Any

from core.research.ml.audits.adjusted_data_comparison import _number


def _close(
    closes_by_symbol: dict[str, dict[str, float]],
    symbol: str,
    day: str,
) -> float | None:
    if not symbol or not day:
        return None
    return _number(closes_by_symbol.get(symbol.upper(), {}).get(day))


def _period_return(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return end / start - 1.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _expected_adjusted_return(
    raw_return: float | None,
    ratio_start: float | None,
    ratio_end: float | None,
) -> float | None:
    ratio_change = _ratio(ratio_end, ratio_start)
    if raw_return is None or ratio_change is None:
        return None
    return (1.0 + raw_return) * ratio_change - 1.0


def _mismatch(
    first: float | None,
    second: float | None,
    *,
    tolerance: float,
) -> bool:
    if first is None and second is None:
        return False
    if first is None or second is None:
        return True
    return abs(first - second) > tolerance


def _count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(row.get(field)) for row in rows)


def _max_abs(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        abs(float(value))
        for value in (row.get(field) for row in rows)
        if _number(value) is not None
    ]
    return max(values, default=None)


def _top_rows(
    rows: list[dict[str, Any]],
    field: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = [
        row for row in rows
        if _number(row.get(field)) is not None
    ]
    ranked.sort(key=lambda row: abs(float(row[field])), reverse=True)
    return ranked[:limit]


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
