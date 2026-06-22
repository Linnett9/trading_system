import csv
from datetime import datetime
from pathlib import Path

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class CsvDataFeed(IDataFeed):
    """Local CSV data feed for pipeline tests.

    Expected file pattern: <data_dir>/<symbol>.csv with columns:
    timestamp, open, high, low, close, volume.
    """

    def __init__(self, data_dir: str = "cache/data"):
        self.data_dir = Path(data_dir)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        return [
            candle
            for candle in self._load_symbol(symbol)
            if start <= candle.timestamp <= end
        ]

    def get_latest_candles(
        self,
        symbols: list[str],
        lookback: int,
        timeframe: str = "1Day",
    ) -> dict[str, list[Candle]]:
        return {
            symbol: self._load_symbol(symbol)[-lookback:]
            for symbol in symbols
        }

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return {
            symbol: candles[-1].close
            for symbol in symbols
            if (candles := self._load_symbol(symbol))
        }

    def get_historical_candles(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> dict[str, list[Candle]]:
        return {
            symbol: self.get_historical_bars(symbol, timeframe, start, end)
            for symbol in symbols
        }

    def _load_symbol(self, symbol: str) -> list[Candle]:
        path = self.data_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV data not found for {symbol}: {path}")

        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            candles = [
                Candle(
                    symbol=symbol,
                    timestamp=parse_timestamp(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0) or 0),
                )
                for row in rows
            ]

        return sorted(candles, key=lambda candle: candle.timestamp)


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d")
