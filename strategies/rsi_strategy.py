from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class RSIStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
    ):
        self.symbol = symbol
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, context) -> Signal:

        if context.rsi is None:
            return Signal(
                symbol=self.symbol,
                action="HOLD",
                timestamp=context.timestamp,
                reason="RSI unavailable",
            )

        if context.rsi <= self.oversold:
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason=f"RSI oversold: {context.rsi:.2f}",
            )

        if context.rsi >= self.overbought:
            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=context.timestamp,
                reason=f"RSI overbought: {context.rsi:.2f}",
            )

        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=f"RSI neutral: {context.rsi:.2f}",
        )
