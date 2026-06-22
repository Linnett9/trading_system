from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from typing import Callable
from urllib.request import urlopen

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class StooqDataFeed(IDataFeed):
    """Daily US-equity history adapter for research and backtesting only."""

    def __init__(
        self,
        opener: Callable = urlopen,
        timeout_seconds: float = 30.0,
    ):
        self._opener = opener
        self._timeout_seconds = timeout_seconds

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if timeframe != "1Day":
            raise ValueError("StooqDataFeed supports only the 1Day timeframe")

        url = self._url(symbol, start, end)
        with self._opener(url, timeout=self._timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        if not payload.startswith("Date,Open,High,Low,Close"):
            raise RuntimeError(
                "Stooq did not return daily CSV data; its endpoint may require "
                "interactive browser verification."
            )
        rows = csv.DictReader(StringIO(payload))
        candles = []
        for row in rows:
            timestamp = _parse_date(row.get("Date", ""))
            if timestamp is None:
                continue
            try:
                candles.append(Candle(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0) or 0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(candles, key=lambda candle: candle.timestamp)

    def _url(self, symbol: str, start: datetime, end: datetime) -> str:
        stooq_symbol = f"{symbol.lower()}.us"
        return (
            "https://stooq.com/q/d/l/"
            f"?s={stooq_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
        )


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
