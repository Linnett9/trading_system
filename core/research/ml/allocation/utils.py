from __future__ import annotations

import json
import math
from typing import Any


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Allocation inputs must be finite")
    return result


def _is_probability(value: Any) -> bool:
    try:
        result = _finite_float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= result <= 1.0


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _json_safe_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, value in row.items():
        output[name] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    return output
