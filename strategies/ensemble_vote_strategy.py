from core.entities.signal import Signal
from core.interfaces.strategy import IStrategy


class EnsembleVoteStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        min_buy_votes: int = 2,
        min_sell_votes: int = 1,
        rsi_entry: float = 50,
        rsi_exit: float = 45,
        rsi_pullback: float = 45,
        pullback_tolerance: float = 0.02,
        use_regime_filter: bool = True,
        use_breakout_vote: bool = True,
    ):
        self.symbol = symbol
        self.min_buy_votes = min_buy_votes
        self.min_sell_votes = min_sell_votes
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.rsi_pullback = rsi_pullback
        self.pullback_tolerance = pullback_tolerance
        self.use_regime_filter = use_regime_filter
        self.use_breakout_vote = use_breakout_vote

    def generate_signal(self, context) -> Signal:
        if (
            context.close is None
            or context.ema_fast is None
            or context.ema_slow is None
            or context.rsi is None
        ):
            return self._hold(context, "Ensemble indicators unavailable")

        buy_reasons = self._buy_reasons(context)
        sell_reasons = self._sell_reasons(context)

        if (
            context.current_position is None
            and len(buy_reasons) >= self.min_buy_votes
        ):
            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=context.timestamp,
                confidence=len(buy_reasons) / 4,
                reason="Ensemble buy vote: " + ", ".join(buy_reasons),
            )

        if (
            context.current_position == "LONG"
            and len(sell_reasons) >= self.min_sell_votes
        ):
            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=context.timestamp,
                confidence=len(sell_reasons) / 3,
                reason="Ensemble sell vote: " + ", ".join(sell_reasons),
            )

        return self._hold(
            context,
            f"Ensemble votes buy={len(buy_reasons)} sell={len(sell_reasons)}",
        )

    def _buy_reasons(self, context) -> list[str]:
        reasons = []
        bullish_trend = context.ema_fast > context.ema_slow

        if bullish_trend:
            reasons.append("trend")

        if self.use_regime_filter and context.market_regime == "bear":
            return []

        if context.rsi >= self.rsi_entry and bullish_trend:
            reasons.append("momentum")

        if (
            context.sma_20 is not None
            and bullish_trend
            and context.close <= context.sma_20 * (1 + self.pullback_tolerance)
            and context.rsi <= self.rsi_pullback
        ):
            reasons.append("pullback")

        if (
            self.use_breakout_vote
            and context.recent_high is not None
            and context.close >= context.recent_high
        ):
            reasons.append("breakout")

        return reasons

    def _sell_reasons(self, context) -> list[str]:
        reasons = []

        if context.ema_fast < context.ema_slow:
            reasons.append("trend_exit")

        if context.rsi < self.rsi_exit:
            reasons.append("rsi_exit")

        if (
            context.recent_low is not None
            and context.close <= context.recent_low
        ):
            reasons.append("breakdown")

        return reasons

    def _hold(self, context, reason):
        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=context.timestamp,
            reason=reason,
        )
