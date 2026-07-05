


class BacktestEngineControlsMixin:
    def _is_warming_up(self, candle) -> bool:

        if self.market_data.candle_count < self.warmup_bars:

            if self.debug:
                print(
                    f"{candle.timestamp} | "
                    f"WARMUP ({self.market_data.candle_count})"
                )

            return True

        return False
    def _should_skip_signal(self, signal, candle) -> bool:

        current_position = self.trade_manager.get_position(
            self.symbol
        )

        if signal.action == "HOLD":

            if self.debug:
                print(f"{candle.timestamp} | SKIP HOLD")

            return True

        if current_position == "LONG" and signal.action == "BUY":
            self.signal_diagnostics.duplicate_buy_skips += 1

            if self.debug:
                print(f"{candle.timestamp} | SKIP BUY — already LONG")

            return True

        if current_position is None and signal.action == "SELL":
            self.signal_diagnostics.flat_sell_skips += 1

            if self.debug:
                print(f"{candle.timestamp} | SKIP SELL — no LONG position")

            return True

        risk_valid = self.risk_manager.validate(signal)

        if self.debug:
            print(
                f"{candle.timestamp} | "
                f"POSITION={current_position} "
                f"RISK_VALID={risk_valid}"
            )

        if not risk_valid:
            self.signal_diagnostics.risk_blocked_signals += 1

            if self.debug:
                print(f"{candle.timestamp} | BLOCKED BY RISK")

            return True

        return False
    def _should_stop_early(self) -> bool:
        if (
            self.early_stop_max_drawdown is not None
            and self.portfolio_engine.max_drawdown
            > self.early_stop_max_drawdown
        ):
            return True

        if (
            self.early_stop_equity_floor is not None
            and self.portfolio_engine.current_equity
            < self.early_stop_equity_floor
        ):
            return True

        return False
    def _debug_signal(
        self,
        candle,
        ema_fast,
        ema_slow,
        signal,
    ):

        if not self.debug:
            return

        print(
            f"{candle.timestamp} | "
            f"EMA_FAST={ema_fast:.2f} "
            f"EMA_SLOW={ema_slow:.2f} "
            f"SIGNAL={signal.action} "
            f"CONF={signal.confidence}"
        )
