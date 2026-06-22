from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class StooqParquetDataFeed(IDataFeed):
    """Local processed Stooq bulk data adapter for research only."""

    def __init__(self, data_dir: str = "data/processed/stooq_parquet"):
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
            raise ValueError("StooqParquetDataFeed supports only the 1Day timeframe")
        path = self.data_dir / f"{symbol.upper()}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Stooq Parquet data not found for {symbol}: {path}. "
                "Run --mode import-stooq-bulk first."
            )
        candles = [
            candle for candle in self._load(path, symbol.upper())
            if start <= candle.timestamp <= end
        ]
        self._last_request_metadata[symbol] = {
            "source": "stooq_parquet",
            "source_file": str(path),
            "page_count": 0,
        }
        return candles

    def get_last_request_metadata(self, symbol: str) -> dict:
        return dict(self._last_request_metadata.get(symbol, {}))

    def _load(self, path: Path, symbol: str) -> list[Candle]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Stooq Parquet research requires pyarrow. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        table = pq.read_table(path, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
        ])
        columns = table.to_pydict()
        return sorted([
            Candle(
                symbol=symbol,
                timestamp=value,
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(close_price),
                volume=float(volume),
            )
            for value, open_price, high_price, low_price, close_price, volume in zip(
                columns["timestamp"],
                columns["open"],
                columns["high"],
                columns["low"],
                columns["close"],
                columns["volume"],
            )
        ], key=lambda candle: candle.timestamp)
