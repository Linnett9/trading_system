from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class EMARSIFilterStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        fast_period: int = 20,
        slow_period: int = 50,
        rsi_entry: float = 50,
        rsi_exit: float = 45,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit

    def generate_signal(self, context) -> Signal:
        if context.rsi is None:
            return self._hold(context, "RSI unavailable")

        bullish = context.ema_fast > context.ema_slow
        bearish_or_weak = (
            context.ema_fast < context.ema_slow
            or context.rsi < self.rsi_exit
        )

        if (
            context.current_position is None
            and bullish
            and context.rsi >= self.rsi_entry
        ):
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason=(
                    f"EMA trend with RSI confirmation: "
                    f"RSI {context.rsi:.2f}"
                ),
            )

        if context.current_position == "LONG" and bearish_or_weak:
            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=context.timestamp,
                reason=(
                    f"EMA/RSI exit: RSI {context.rsi:.2f}, "
                    f"EMA fast {context.ema_fast:.2f}, "
                    f"EMA slow {context.ema_slow:.2f}"
                ),
            )

        return self._hold(context, "No EMA/RSI signal")

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
