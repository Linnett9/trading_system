# strategies/ema_crossover_strategy.py

from core.interfaces.strategy import IStrategy
from core.entities.signal import Signal


class EMACrossoverStrategy(IStrategy):

    def __init__(
        self,
        symbol: str,
        fast_period: int = 50,
        slow_period: int = 200,
    ):
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._last_state = None  # "above" | "below" | "flat" | None

    def generate_signal(self, context) -> Signal:

        ema_fast = context.ema_fast
        ema_slow = context.ema_slow
        timestamp = context.timestamp

        # -----------------------------
        # 1. Determine current EMA state
        # -----------------------------
        if ema_fast > ema_slow:
            current_state = "above"
        elif ema_fast < ema_slow:
            current_state = "below"
        else:
            current_state = "flat"

        # -----------------------------
        # 2. First valid bar after warmup
        # -----------------------------
        if self._last_state is None:
            self._last_state = current_state

            if current_state == "above":
                return Signal(
                    symbol=self.symbol,
                    action="BUY",
                    timestamp=timestamp,
                    confidence=1.0,
                    reason=(
                        f"Initial bullish EMA state: "
                        f"EMA{self.fast_period} ({ema_fast:.2f}) > "
                        f"EMA{self.slow_period} ({ema_slow:.2f})"
                    )
                )

            return Signal(
                symbol=self.symbol,
                action="HOLD",
                timestamp=timestamp,
                confidence=1.0,
                reason="Initial state is not bullish"
            )

        # -----------------------------
        # 3. Bullish crossover
        # below/flat -> above
        # -----------------------------
        if (
            self._last_state in ["below", "flat"]
            and current_state == "above"
        ):
            self._last_state = current_state

            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=timestamp,
                confidence=1.0,
                reason=(
                    f"EMA crossover UP: "
                    f"EMA{self.fast_period} ({ema_fast:.2f}) > "
                    f"EMA{self.slow_period} ({ema_slow:.2f})"
                )
            )

        # -----------------------------
        # 3b. Bullish trend re-entry
        # flat -> above without a fresh crossover
        # -----------------------------
        if (
            current_state == "above"
            and getattr(context, "current_position", None) is None
        ):
            self._last_state = current_state

            return Signal(
                symbol=self.symbol,
                action="BUY",
                timestamp=timestamp,
                confidence=1.0,
                reason=(
                    f"Bullish EMA trend re-entry: "
                    f"EMA{self.fast_period} ({ema_fast:.2f}) > "
                    f"EMA{self.slow_period} ({ema_slow:.2f})"
                )
            )

        # -----------------------------
        # 4. Bearish crossover
        # above/flat -> below
        # -----------------------------
        if (
            self._last_state in ["above", "flat"]
            and current_state == "below"
        ):
            self._last_state = current_state

            return Signal(
                symbol=self.symbol,
                action="SELL",
                timestamp=timestamp,
                confidence=1.0,
                reason=(
                    f"EMA crossover DOWN: "
                    f"EMA{self.fast_period} ({ema_fast:.2f}) < "
                    f"EMA{self.slow_period} ({ema_slow:.2f})"
                )
            )

        # -----------------------------
        # 5. No crossover
        # -----------------------------
        self._last_state = current_state

        return Signal(
            symbol=self.symbol,
            action="HOLD",
            timestamp=timestamp,
            confidence=1.0,
            reason="No EMA crossover event"
        )
