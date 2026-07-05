from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


def _date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _date_string(value: Any) -> str | None:
    parsed = _date(value)
    return parsed.date().isoformat() if parsed is not None else None


def _first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return row[name]
    return None


def _first_number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(row.get(name))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.6f}"
