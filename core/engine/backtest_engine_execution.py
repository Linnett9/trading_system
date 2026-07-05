


class BacktestEngineExecutionMixin:
    def _check_exit_rules(self, candle) -> bool:

        trade = self.trade_manager.get_open_trade(self.symbol)

        if trade is None or trade.side != "LONG":
            return False

        if trade.stop_loss is not None and candle.low <= trade.stop_loss:
            self.signal_diagnostics.stop_loss_exits += 1
            signal = self._exit_signal(
                candle=candle,
                reason=(
                    f"ATR stop loss hit: "
                    f"price <= {trade.stop_loss:.2f}"
                ),
            )
            self._execute_exit_signal(
                signal=signal,
                exit_price=trade.stop_loss,
            )
            return True

        if trade.trailing_stop is not None and candle.low <= trade.trailing_stop:
            self.signal_diagnostics.stop_loss_exits += 1
            signal = self._exit_signal(
                candle=candle,
                reason=(
                    f"ATR trailing stop hit: "
                    f"price <= {trade.trailing_stop:.2f}"
                ),
            )
            self._execute_exit_signal(
                signal=signal,
                exit_price=trade.trailing_stop,
            )
            return True

        if trade.take_profit is not None and candle.high >= trade.take_profit:
            self.signal_diagnostics.take_profit_exits += 1
            signal = self._exit_signal(
                candle=candle,
                reason=(
                    f"ATR take profit hit: "
                    f"price >= {trade.take_profit:.2f}"
                ),
            )
            self._execute_exit_signal(
                signal=signal,
                exit_price=trade.take_profit,
            )
            return True

        return False
    def _exit_signal(self, candle, reason):

        from core.entities.signal import Signal

        return Signal(
            symbol=self.symbol,
            action="SELL",
            timestamp=candle.timestamp,
            confidence=1.0,
            reason=reason,
        )
    def _execute_exit_signal(self, signal, exit_price):

        trade = self.trade_manager.get_open_trade(self.symbol)

        if trade is None:
            return None

        return self.execution_engine.execute(
            signal=signal,
            size=trade.quantity,
            market_price=exit_price,
        )
    def _close_open_trades_at_end(self, candle):
        if candle is None:
            return

        open_trades = list(self.trade_manager.open_trades.values())

        for trade in open_trades:
            signal = self._exit_signal(
                candle=candle,
                reason="End of backtest liquidation",
            )
            self._execute_exit_signal(
                signal=signal,
                exit_price=candle.close,
            )

        if open_trades:
            self._update_portfolio(candle)
    def _execute_signal(self, signal, candle, risk_context):

        account_equity = self.portfolio_engine.current_equity

        size = self.risk_manager.position_size(
            signal,
            account_equity,
            candle.close,
            risk_context=risk_context,
        )

        trade = self.execution_engine.execute(
            signal=signal,
            size=size,
            market_price=candle.close,
        )

        if signal.action == "BUY":
            self._set_exit_levels(trade, risk_context)

        if self.debug:
            print(
                f"{candle.timestamp} | "
                f"EXECUTED {signal.action} "
                f"SIZE={size}"
            )
    def _set_exit_levels(self, trade, risk_context):

        if trade is None:
            return

        if trade.side != "LONG":
            return

        if risk_context is None or risk_context.atr is None:
            return

        if risk_context.atr <= 0:
            return

        trade.stop_loss = (
            trade.entry_price
            - risk_context.atr * self.atr_stop_multiplier
        )
        trade.highest_price = trade.entry_price

        if self.trailing_atr_multiplier is not None:
            trade.trailing_stop = (
                trade.entry_price
                - risk_context.atr * self.trailing_atr_multiplier
            )

        if self.atr_take_profit_multiplier is None:
            trade.take_profit = None
            return

        trade.take_profit = (
            trade.entry_price
            + risk_context.atr * self.atr_take_profit_multiplier
        )
    def _update_trailing_stop(self, candle, atr):
        if self.trailing_atr_multiplier is None or atr is None or atr <= 0:
            return

        trade = self.trade_manager.get_open_trade(self.symbol)

        if trade is None or trade.side != "LONG":
            return

        highest_price = max(trade.highest_price or trade.entry_price, candle.high)
        trailing_stop = highest_price - atr * self.trailing_atr_multiplier

        trade.highest_price = highest_price
        trade.trailing_stop = max(
            trade.trailing_stop or trailing_stop,
            trailing_stop,
        )
    def _update_portfolio(self, candle):

        equity = self.portfolio_engine.update(
            trade_manager=self.trade_manager,
            latest_price=candle.close,
            timestamp=candle.timestamp,
        )

        if self.debug:
            print(f"{candle.timestamp} | EQUITY={equity:.2f}")

        return equity
