from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy
from strategies.filters import (
    BullMarketFilter,
    TrendStrengthFilter,
    VolatilityFilter,
    VolumeConfirmationFilter,
)


class TrendPullbackStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        fast_period: int = 20,
        trend_fast_period: int = 50,
        trend_slow_period: int = 200,
        pullback_tolerance: float = 0.02,
        exit_extension: float = 0.08,
        use_regime_filter: bool = True,
        min_adx: float | None = None,
        min_relative_volume: float | None = None,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = trend_slow_period
        self.trend_fast_period = trend_fast_period
        self.trend_slow_period = trend_slow_period
        self.pullback_tolerance = pullback_tolerance
        self.exit_extension = exit_extension
        self.use_regime_filter = use_regime_filter
        self.min_adx = min_adx
        self.min_relative_volume = min_relative_volume
        self.bull_filter = BullMarketFilter()
        self.trend_filter = TrendStrengthFilter(min_adx=min_adx)
        self.volatility_filter = VolatilityFilter()
        self.volume_filter = (
            VolumeConfirmationFilter(min_relative_volume)
            if min_relative_volume is not None
            else None
        )

    def generate_signal(self, context) -> Signal:
        if (
            context.close is None
            or context.ema_fast is None
            or context.ema_slow is None
            or context.sma_20 is None
        ):
            return self._hold(context, "Trend pullback indicators unavailable")

        if context.current_position == "LONG":
            if context.close >= context.sma_20 * (1 + self.exit_extension):
                return Signal(
                    symbol=self.symbol,
                    action="SELL",
                    timestamp=context.timestamp,
                    reason="Trend pullback extension exit",
                )

            if context.ema_fast < context.ema_slow:
                return Signal(
                    symbol=self.symbol,
                    action="SELL",
                    timestamp=context.timestamp,
                    reason="Trend pullback trend exit",
                )

            return self._hold(context, "Trend pullback position held")

        if context.ema_fast <= context.ema_slow:
            return self._hold(context, "Trend filter not bullish")

        if self.use_regime_filter and not self._regime_passes(context):
            return self._hold(context, "Market regime filter blocked entry")

        near_pullback_average = (
            context.close <= context.sma_20 * (1 + self.pullback_tolerance)
        )

        if near_pullback_average:
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason="Bullish trend pullback entry",
            )

        return self._hold(context, "No pullback entry")

    def _regime_passes(self, context) -> bool:
        return (
            self.bull_filter.passes(context)
            and self.trend_filter.passes(context)
            and self.volatility_filter.passes(context)
            and (
                self.volume_filter is None
                or self.volume_filter.passes(context)
            )
        )

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
