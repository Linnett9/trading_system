from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class EMARSIPullbackStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        fast_period: int = 20,
        slow_period: int = 200,
        rsi_pullback: float = 45,
        rsi_recover: float = 55,
        use_regime_filter: bool = True,
        min_relative_volume: float | None = None,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_pullback = rsi_pullback
        self.rsi_recover = rsi_recover
        self.use_regime_filter = use_regime_filter
        self.min_relative_volume = min_relative_volume

    def generate_signal(self, context) -> Signal:
        if (
            context.close is None
            or context.ema_fast is None
            or context.ema_slow is None
            or context.rsi is None
        ):
            return self._hold(context, "EMA RSI pullback indicators unavailable")

        if context.current_position == "LONG":
            if context.rsi >= self.rsi_recover:
                return Signal(
                    symbol=self.symbol,
                    action="SELL",
                    timestamp=context.timestamp,
                    reason="EMA RSI pullback recovery exit",
                )

            if context.ema_fast < context.ema_slow:
                return Signal(
                    symbol=self.symbol,
                    action="SELL",
                    timestamp=context.timestamp,
                    reason="EMA RSI pullback trend exit",
                )

            return self._hold(context, "EMA RSI pullback position held")

        if context.ema_fast <= context.ema_slow:
            return self._hold(context, "EMA trend not bullish")

        if self.use_regime_filter and context.market_regime == "bear":
            return self._hold(context, "Bear regime blocked pullback")

        if (
            self.min_relative_volume is not None
            and context.relative_volume is not None
            and context.relative_volume < self.min_relative_volume
        ):
            return self._hold(context, "Volume confirmation blocked")

        if context.rsi <= self.rsi_pullback:
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason="EMA trend RSI pullback entry",
            )

        return self._hold(context, "No EMA RSI pullback signal")

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
