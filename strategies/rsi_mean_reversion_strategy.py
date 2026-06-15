from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class RSIMeanReversionStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        period: int = 14,
        oversold: float = 35,
        exit_level: float = 50,
        overbought: float = 70,
        require_sideways_regime: bool = False,
    ):
        self.symbol = symbol
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.overbought = overbought
        self.require_sideways_regime = require_sideways_regime

    def generate_signal(self, context) -> Signal:
        if context.rsi is None:
            return self._hold(context, "RSI unavailable")

        if (
            self.require_sideways_regime
            and context.market_regime != "sideways"
        ):
            return self._hold(context, "RSI sideways regime filter blocked")

        if (
            context.current_position is None
            and context.rsi <= self.oversold
        ):
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason=f"RSI mean reversion entry: {context.rsi:.2f}",
            )

        if (
            context.current_position == "LONG"
            and (
                context.rsi >= self.exit_level
                or context.rsi >= self.overbought
            )
        ):
            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=context.timestamp,
                reason=f"RSI mean reversion exit: {context.rsi:.2f}",
            )

        return self._hold(context, "No RSI mean reversion signal")

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
