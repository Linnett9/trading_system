from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdjustedPricePoint:
    symbol: str
    timestamp: datetime
    adjusted_close: float
    source: str
    raw_close: float | None = None


class LocalAdjustedPriceCsvDataFeed:
    """Local adjusted-close adapter for research audits only."""

    def __init__(
        self,
        data_dir: str = "data/reference/adjusted_prices",
        *,
        combined_path: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.combined_path = Path(combined_path) if combined_path else None

    def get_adjusted_prices(self, symbol: str) -> list[AdjustedPricePoint]:
        normalized = symbol.upper()
        path = self._symbol_path(normalized)
        if path is None:
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                point for point in self._parse_rows(normalized, handle, str(path))
                if point.symbol == normalized
            ]
        return sorted(rows, key=lambda point: point.timestamp)

    def _symbol_path(self, symbol: str) -> Path | None:
        if self.combined_path and self.combined_path.exists():
            return self.combined_path
        candidates = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol.lower()}.csv",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _parse_rows(
        self,
        default_symbol: str,
        handle,
        source: str,
    ) -> list[AdjustedPricePoint]:
        reader = csv.DictReader(handle)
        output = []
        for row in reader:
            symbol = str(
                _first_present(row, "symbol", "Symbol") or default_symbol
            ).upper()
            date_value = _first_present(row, "date", "Date", "timestamp")
            adjusted_close = _first_number(
                row,
                "adjusted_close",
                "adj_close",
                "Adj Close",
                "AdjClose",
                "close",
                "Close",
            )
            if date_value is None or adjusted_close is None or adjusted_close <= 0.0:
                continue
            output.append(
                AdjustedPricePoint(
                    symbol=symbol,
                    timestamp=_parse_date(date_value),
                    adjusted_close=adjusted_close,
                    raw_close=_first_number(row, "raw_close", "Raw Close", "raw"),
                    source=source,
                )
            )
        return output


def _first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) not in {None, ""}:
            return row[name]
    return None


def _first_number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _parse_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value)[:10])
