from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class StooqCsvDataFeed(IDataFeed):
    """Local Stooq daily CSV adapter for long-history research only."""

    def __init__(self, data_dir: str = "data/raw/stooq"):
        self.data_dir = Path(data_dir)
        self._last_request_metadata: dict[str, dict] = {}

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if timeframe != "1Day":
            raise ValueError("StooqCsvDataFeed supports only the 1Day timeframe")
        path = self.data_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Stooq CSV data not found for {symbol}: {path}"
            )
        candles = [
            candle for candle in self._load(path, symbol)
            if start <= candle.timestamp <= end
        ]
        self._last_request_metadata[symbol] = {
            "source": "stooq_csv",
            "source_file": str(path),
            "page_count": 0,
        }
        return candles

    def get_last_request_metadata(self, symbol: str) -> dict:
        return dict(self._last_request_metadata.get(symbol, {}))

    def _load(self, path: Path, symbol: str) -> list[Candle]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            candles = [
                Candle(
                    symbol=symbol,
                    timestamp=datetime.strptime(row["Date"], "%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0) or 0),
                )
                for row in rows
                if row.get("Close") not in {None, ""}
            ]
        return sorted(candles, key=lambda candle: candle.timestamp)
