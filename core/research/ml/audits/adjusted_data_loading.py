from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from infrastructure.data.adjusted_price_csv_data_feed import (
    AdjustedPricePoint,
    LocalAdjustedPriceCsvDataFeed,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_raw_stooq_rows_by_symbol(
    data_dir: Path,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        symbol: _load_raw_stooq_rows(data_dir / f"{symbol}.parquet")
        for symbol in symbols
    }


def _load_raw_stooq_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Adjusted data comparison requires pyarrow to read raw Stooq parquet"
        ) from exc
    table = pq.read_table(path, columns=["timestamp", "close"])
    columns = table.to_pydict()
    return [
        {"date": _date_string(timestamp), "close": close}
        for timestamp, close in zip(columns["timestamp"], columns["close"])
    ]


def _load_adjusted_rows_by_symbol(
    config: dict[str, Any],
    symbols: list[str],
) -> dict[str, list[AdjustedPricePoint]]:
    feed = LocalAdjustedPriceCsvDataFeed(
        str(config["adjusted_data_dir"]),
        combined_path=config.get("adjusted_combined_path"),
    )
    return {symbol: feed.get_adjusted_prices(symbol) for symbol in symbols}


def _raw_close_by_date(rows: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for row in rows:
        day = _date_string(row.get("timestamp") or row.get("date"))
        close = _number(row.get("close") or row.get("Close"))
        if day and close is not None and close > 0:
            output[day] = close
    return output


def _adjusted_close_by_date(
    rows: list[AdjustedPricePoint | dict[str, Any]],
) -> dict[str, float]:
    output = {}
    for row in rows:
        if isinstance(row, AdjustedPricePoint):
            output[row.timestamp.date().isoformat()] = row.adjusted_close
            continue
        day = _date_string(row.get("timestamp") or row.get("date"))
        close = _number(
            row.get("adjusted_close")
            or row.get("adj_close")
            or row.get("Adj Close")
            or row.get("close")
        )
        if day and close is not None and close > 0:
            output[day] = close
    return output


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
    return parsed.date().isoformat() if parsed else None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
