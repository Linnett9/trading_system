from __future__ import annotations

import math
from statistics import mean
from typing import Any


def _flat_numbers(value: Any) -> list[float]:
    if value is None:
        return []
    candidate = value.tolist() if hasattr(value, "tolist") else value
    while isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], list):
        candidate = candidate[0]
    if not isinstance(candidate, (list, tuple)):
        candidate = [candidate]
    return [float(item) for item in candidate]
def _first_number(value: Any) -> float | None:
    numbers = _flat_numbers(value)
    return numbers[0] if numbers else None
def _average(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else None
def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)
