from __future__ import annotations

import math
from typing import Any


def _descending(value: Any) -> float:
    number = _number(value)
    return math.inf if number is None else -number


def _ascending(value: Any) -> float:
    number = _number(value)
    return math.inf if number is None else number


def _drawdown_magnitude(value: Any) -> float | None:
    number = _number(value)
    return abs(number) if number is not None else None


def _first_number(payload: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(payload.get(name))
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

def _format(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.4f}"
