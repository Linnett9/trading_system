from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class DonchianBreakoutStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        lookback_period: int = 20,
        use_volatility_filter: bool = False,
        use_volume_filter: bool = False,
        min_relative_volume: float = 1.0,
        allowed_volatility_regimes=None,
    ):
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.use_volatility_filter = use_volatility_filter
        self.use_volume_filter = use_volume_filter
        self.min_relative_volume = min_relative_volume
        self.allowed_volatility_regimes = (
            allowed_volatility_regimes or {"normal", "high"}
        )

    def generate_signal(self, context) -> Signal:
        if context.close is None:
            return self._hold(context, "Close unavailable")

        if context.recent_high is None or context.recent_low is None:
            return self._hold(context, "Donchian channel unavailable")

        if (
            self.use_volatility_filter
            and context.volatility_regime not in self.allowed_volatility_regimes
        ):
            return self._hold(context, "Volatility regime filter blocked")

        if (
            self.use_volume_filter
            and context.relative_volume is not None
            and context.relative_volume < self.min_relative_volume
        ):
            return self._hold(context, "Volume confirmation blocked")

        if (
            context.current_position is None
            and context.close >= context.recent_high
        ):
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                reason=(
                    f"Donchian breakout: close {context.close:.2f} >= "
                    f"high {context.recent_high:.2f}"
                ),
            )

        if (
            context.current_position == "LONG"
            and context.close <= context.recent_low
        ):
            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=context.timestamp,
                reason=(
                    f"Donchian breakdown: close {context.close:.2f} <= "
                    f"low {context.recent_low:.2f}"
                ),
            )

        return self._hold(context, "No Donchian breakout")

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
