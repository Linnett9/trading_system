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

    def adx(self, period=14):
        candles = self.market_data.candles

        if len(candles) < period + 1:
            return None

        directional_values = []

        for index in range(1, len(candles)):
            current = candles[index]
            previous = candles[index - 1]

            up_move = current.high - previous.high
            down_move = previous.low - current.low

            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0

            true_range = max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )

            if true_range == 0:
                directional_values.append(0)
                continue

            plus_di = 100 * (plus_dm / true_range)
            minus_di = 100 * (minus_dm / true_range)
            denominator = plus_di + minus_di

            if denominator == 0:
                directional_values.append(0)
            else:
                directional_values.append(
                    100 * abs(plus_di - minus_di) / denominator
                )

        if len(directional_values) < period:
            return None

        return sum(directional_values[-period:]) / period

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

    def bollinger_bands(self, period=20, std_multiplier=2.0):
        closes = self.market_data.closes()

        if len(closes) < period:
            return None, None, None

        recent = closes[-period:]
        middle = sum(recent) / period
        variance = sum((value - middle) ** 2 for value in recent) / period
        std = math.sqrt(variance)

        return (
            middle,
            middle + std * std_multiplier,
            middle - std * std_multiplier,
        )

    def bollinger_bandwidth(self, period=20, std_multiplier=2.0):
        middle, upper, lower = self.bollinger_bands(period, std_multiplier)

        if middle is None or middle == 0:
            return None

        return (upper - lower) / middle

    def volume_sma(self, period=20):
        volumes = self.market_data.volumes()

        if len(volumes) < period:
            return None

        return sum(volumes[-period:]) / period

    def relative_volume(self, period=20):
        latest = self.market_data.latest()
        volume_average = self.volume_sma(period)

        if latest is None or volume_average in (None, 0):
            return None

        return latest.volume / volume_average

    def volatility_percentile(self, period=20, lookback=100):
        closes = self.market_data.closes()

        if len(closes) < period + lookback:
            return None

        volatilities = []
        start = len(closes) - lookback

        for end in range(start, len(closes) + 1):
            window = closes[end - period - 1:end]
            if len(window) < period + 1:
                continue

            returns = []
            for index in range(1, len(window)):
                previous = window[index - 1]
                current = window[index]
                if previous == 0:
                    continue
                returns.append((current - previous) / previous)

            if not returns:
                continue

            mean = sum(returns) / len(returns)
            variance = (
                sum((value - mean) ** 2 for value in returns)
                / len(returns)
            )
            volatilities.append(math.sqrt(variance))

        if not volatilities:
            return None

        current_volatility = volatilities[-1]
        below_or_equal = sum(
            1 for value in volatilities
            if value <= current_volatility
        )

        return below_or_equal / len(volatilities)

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
