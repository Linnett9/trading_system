from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from core.entities.candle import Candle


class IDataFeed(ABC):

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[Candle]:
        pass

    def get_latest_candles(
        self,
        symbols: list[str],
        lookback: int,
        timeframe: str = "1Day",
    ) -> dict[str, list[Candle]]:
        raise NotImplementedError

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

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        candles_by_symbol = self.get_latest_candles(symbols, lookback=1)
        return {
            symbol: candles[-1].close
            for symbol, candles in candles_by_symbol.items()
            if candles
        }
