# core/services/market_data_service.py

from collections import deque
from typing import List, Optional

from core.entities.candle import Candle
from core.indicators.sma import sma
from core.indicators.ema import ema
from core.indicators.rsi import rsi
from core.indicators.atr import atr


class MarketDataService:
    """
    Maintains a rolling window of candles for a single
    symbol/timeframe combination.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        max_bars: int = 1000
    ):
        self._symbol = symbol
        self._timeframe = timeframe
        self._candles = deque(maxlen=max_bars)

    # -------------------------------------------------
    # METADATA
    # -------------------------------------------------
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def candle_count(self) -> int:
        return len(self._candles)

    # -------------------------------------------------
    # CORE DATA ACCESS (FIXED)
    # -------------------------------------------------
    @property
    def candles(self) -> List[Candle]:
        """
        Return all candles as a list.
        (FIX: now a property, not a method)
        """
        return list(self._candles)

    # -------------------------------------------------
    # DATA INGESTION
    # -------------------------------------------------
    def add_candle(self, candle: Candle) -> None:

        if candle.symbol != self._symbol:
            raise ValueError(
                f"Expected symbol {self._symbol}, "
                f"received {candle.symbol}"
            )

        self._candles.append(candle)

    def add_candles(self, candles: List[Candle]) -> None:

        for candle in candles:
            self.add_candle(candle)

    # -------------------------------------------------
    # TIME ACCESSORS
    # -------------------------------------------------
    def latest(self) -> Optional[Candle]:

        if not self._candles:
            return None

        return self._candles[-1]

    def oldest(self) -> Optional[Candle]:

        if not self._candles:
            return None

        return self._candles[0]

    def last_n_candles(self, n: int) -> List[Candle]:

        return list(self._candles)[-n:]

    # -------------------------------------------------
    # OHLCV SERIES
    # -------------------------------------------------
    def opens(self) -> List[float]:
        return [c.open for c in self._candles]

    def highs(self) -> List[float]:
        return [c.high for c in self._candles]

    def lows(self) -> List[float]:
        return [c.low for c in self._candles]

    def closes(self) -> List[float]:
        return [c.close for c in self._candles]

    def volumes(self) -> List[float]:
        return [c.volume for c in self._candles]

    # -------------------------------------------------
    # PRICE HELPERS
    # -------------------------------------------------
    def latest_price(self) -> Optional[float]:

        latest = self.latest()

        if latest is None:
            return None

        return latest.close

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    def has_minimum_bars(self, required_bars: int) -> bool:
        return len(self._candles) >= required_bars

    # -------------------------------------------------
    # UTILITIES
    # -------------------------------------------------
    def clear(self) -> None:
        self._candles.clear()

    def summary(self) -> dict:

        latest = self.latest()

        return {
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "bars": len(self._candles),
            "latest_price": latest.close if latest else None
        }