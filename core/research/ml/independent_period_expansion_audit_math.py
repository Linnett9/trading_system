from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def _compound(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    for value in returns:
        equity *= 1.0 + float(value)
    return equity - 1.0
def _max_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max_drawdown
def _top_positive_share(returns: list[float], top_n: int) -> float | None:
    positives = sorted((value for value in returns if value > 0.0), reverse=True)
    total = sum(positives)
    if total <= 0.0:
        return None
    return sum(positives[:top_n]) / total
def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None
