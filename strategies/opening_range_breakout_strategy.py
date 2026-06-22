from __future__ import annotations

from core.entities.signal import Signal
from core.entities.strategy_context import StrategyContext
from core.interfaces.strategy import IStrategy


class OpeningRangeBreakoutStrategy(IStrategy):
    """Long-only intraday research strategy; not connected to execution."""

    def __init__(
        self,
        symbol: str,
        min_relative_volume: float = 1.5,
        require_vwap_confirmation: bool = True,
    ) -> None:
        self.symbol = symbol
        self.min_relative_volume = min_relative_volume
        self.require_vwap_confirmation = require_vwap_confirmation

    def generate_signal(self, context: StrategyContext) -> Signal:
        missing = self._missing_fields(context)
        if missing:
            return self._hold(context, f"Opening-range inputs unavailable: {', '.join(missing)}")

        if context.current_position == "LONG":
            if context.close <= context.opening_range_low:
                return self._signal(context, "SELL", "Opening-range breakdown exit")
            if self.require_vwap_confirmation and context.close < context.vwap:
                return self._signal(context, "SELL", "VWAP loss exit")
            return self._hold(context, "Opening-range position held")

        if context.market_regime == "bear":
            return self._hold(context, "Bear market regime blocks entry")
        if context.relative_volume < self.min_relative_volume:
            return self._hold(context, "Relative-volume confirmation blocked entry")
        if context.close <= context.opening_range_high:
            return self._hold(context, "Price has not broken the opening range")
        if self.require_vwap_confirmation and context.close < context.vwap:
            return self._hold(context, "VWAP confirmation blocked entry")

        return self._signal(context, "BUY", "Opening-range breakout with volume confirmation")

    def _missing_fields(self, context: StrategyContext) -> list[str]:
        required = {
            "close": context.close,
            "opening_range_high": context.opening_range_high,
            "opening_range_low": context.opening_range_low,
            "relative_volume": context.relative_volume,
        }
        if self.require_vwap_confirmation:
            required["vwap"] = context.vwap
        return [name for name, value in required.items() if value is None]

    def _signal(self, context: StrategyContext, action: str, reason: str) -> Signal:
        return Signal(
            symbol=self.symbol,
            action=action,
            timestamp=context.timestamp,
            confidence=0.75,
            reason=reason,
        )

    def _hold(self, context: StrategyContext, reason: str) -> Signal:
        return self._signal(context, "HOLD", reason)
