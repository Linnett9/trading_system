import math

from core.indicators.atr import atr
from core.indicators.ema import ema
from core.indicators.rsi import rsi
from core.indicators.sma import sma
from core.services.market_data_service import (
    MarketDataService
)


class IndicatorService:

    def __init__(
        self,
        market_data: MarketDataService
    ):
        self.market_data = market_data

    def sma(self, period):

        return sma(
            self.market_data.closes(),
            period
        )

    def previous_sma(self, period):
        closes = self.market_data.closes()

        if len(closes) <= period:
            return None

        return sma(closes[:-1], period)

    def ema(self, period):

        return ema(
            self.market_data.closes(),
            period
        )

    def rsi(self, period=14):

        return rsi(
            self.market_data.closes(),
            period
        )

    def atr(self, period=14):

        return atr(
            self.market_data.candles,
            period
        )

    def volatility(self, period=20):

        closes = self.market_data.closes()

        if len(closes) < period + 1:
            return None

        recent = closes[-(period + 1):]
        returns = []

        for index in range(1, len(recent)):
            previous = recent[index - 1]
            current = recent[index]

            if previous == 0:
                return None

            returns.append((current - previous) / previous)

        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)

        return math.sqrt(variance)

    def highest_high(self, period=20, exclude_latest=False):

        highs = self.market_data.highs()
        if exclude_latest:
            highs = highs[:-1]

        if len(highs) < period:
            return None

        return max(highs[-period:])

    def lowest_low(self, period=20, exclude_latest=False):

        lows = self.market_data.lows()
        if exclude_latest:
            lows = lows[:-1]

        if len(lows) < period:
            return None

        return min(lows[-period:])
