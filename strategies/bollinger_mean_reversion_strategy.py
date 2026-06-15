from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class BollingerMeanReversionStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        rsi_entry: float = 40,
        rsi_exit: float = 55,
        require_sideways_regime: bool = False,
        min_bandwidth: float | None = None,
        max_bandwidth: float | None = None,
    ):
        self.symbol = symbol
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.require_sideways_regime = require_sideways_regime
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

    def generate_signal(self, context) -> Signal:
        if (
            context.close is None
            or context.bollinger_lower is None
            or context.bollinger_middle is None
            or context.rsi is None
        ):
            return self._hold(context, "Bollinger indicators unavailable")

        if (
            self.require_sideways_regime
            and context.market_regime != "sideways"
        ):
            return self._hold(context, "Sideways regime filter blocked")

        if (
            self.min_bandwidth is not None
            and context.bollinger_bandwidth is not None
            and context.bollinger_bandwidth < self.min_bandwidth
        ):
            return self._hold(context, "Bollinger bandwidth too low")

        if (
            self.max_bandwidth is not None
            and context.bollinger_bandwidth is not None
            and context.bollinger_bandwidth > self.max_bandwidth
        ):
            return self._hold(context, "Bollinger bandwidth too high")

        if context.current_position == "LONG":
            if (
                context.close >= context.bollinger_middle
                or context.rsi >= self.rsi_exit
            ):
                return Signal(
                    symbol=self.symbol,
                    action="SELL",
                    timestamp=context.timestamp,
                    reason="Bollinger mean reversion exit",
                )

            return self._hold(context, "Bollinger position held")

        if (
            context.close <= context.bollinger_lower
            and context.rsi <= self.rsi_entry
        ):
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason="Bollinger lower-band mean reversion entry",
            )

        return self._hold(context, "No Bollinger mean reversion signal")

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
