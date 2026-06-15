from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class BuyAndHoldStrategy(IStrategy):

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.has_entered = False
        self.use_engine_exits = False

    def generate_signal(self, context) -> Signal:
        if not self.has_entered and context.current_position is None:
            self.has_entered = True
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason="Buy-and-hold initial entry",
            )

        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason="Buy-and-hold remains invested",
        )
