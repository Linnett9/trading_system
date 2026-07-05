from __future__ import annotations

from typing import Any

from core.research.ml.audits.data_adjustment_validation_types import (
    COMMON_SPLIT_FACTORS,
    RESEARCH_METADATA,
)
from core.research.ml.audits.data_adjustment_validation_utils import (
    _date_string,
    _first_number,
    _first_present,
    _number,
)


def _normalized_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        close = _first_number(row, "close", "Close", "<CLOSE>")
        normalized.append({
            "date": _date_string(_first_present(row, "timestamp", "date", "Date", "<DATE>")),
            "close": close,
            "raw_close": _first_number(
                row,
                "raw_close",
                "RawClose",
                "raw_Close",
                "unadjusted_close",
                "UnadjustedClose",
            ),
            "adjusted_close": _first_number(
                row,
                "adjusted_close",
                "adj_close",
                "Adj Close",
                "AdjClose",
                "adjusted",
            ),
            "open": _first_number(row, "open", "Open", "<OPEN>"),
            "high": _first_number(row, "high", "High", "<HIGH>"),
            "low": _first_number(row, "low", "Low", "<LOW>"),
            "volume": _first_number(row, "volume", "Volume", "<VOL>"),
            **row,
        })
    return sorted(
        [row for row in normalized if row.get("date")],
        key=lambda row: str(row["date"]),
    )
def _split_like_factor(ratio: float, tolerance: float) -> float | None:
    if ratio <= 0.0:
        return None
    for factor in COMMON_SPLIT_FACTORS:
        inverse = 1.0 / factor
        if abs(ratio - factor) / factor <= tolerance:
            return factor
        if abs(ratio - inverse) / inverse <= tolerance:
            return factor
    return None
