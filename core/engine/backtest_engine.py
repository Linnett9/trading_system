# core/engine/backtest_engine.py

from core.services.indicator_service import IndicatorService
from core.entities.backtest_result import BacktestResult
from core.entities.signal_diagnostics import SignalDiagnostics
from core.entities.strategy_context import StrategyContext
from core.entities.risk_context import RiskContext
from core.research.capital_utilization_analyzer import (
    CapitalUtilizationAnalyzer,
)
from core.research.market_regime_analyzer import MarketRegimeAnalyzer
from core.research.trade_analyzer import TradeAnalyzer
from core.services.market_data_service import MarketDataService


class BacktestEngine:

    def __init__(
        self,
        data_feed,
        strategy,
        risk_manager,
        execution_engine,
        trade_manager,
        portfolio_engine,
        symbol: str,
        timeframe: str,
        account_equity: float = 10_000,
        warmup_bars: int = 200,
        atr_stop_multiplier: float = 2.0,
        atr_take_profit_multiplier: float = 3.0,
        trailing_atr_multiplier: float | None = None,
        close_open_trades_at_end: bool = False,
        early_stop_max_drawdown: float | None = None,
        early_stop_equity_floor: float | None = None,
        debug: bool = False,
    ):
        self.data_feed = data_feed
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.trade_manager = trade_manager
        self.portfolio_engine = portfolio_engine

        self.symbol = symbol
        self.timeframe = timeframe
        self.account_equity = account_equity
        self.warmup_bars = warmup_bars
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.close_open_trades_at_end = close_open_trades_at_end
        self.early_stop_max_drawdown = early_stop_max_drawdown
        self.early_stop_equity_floor = early_stop_equity_floor
        self.debug = debug
        self._last_result = None
        self.signal_diagnostics = SignalDiagnostics()

        self.market_data = MarketDataService(
            symbol=symbol,
            timeframe=timeframe,
        )

    def run(self, candles):
        last_candle = None

        for candle in candles:
            last_candle = candle

            self.market_data.add_candle(candle)

            if self._is_warming_up(candle):
                continue

            indicators = IndicatorService(self.market_data)

            fast_period = getattr(self.strategy, "fast_period", 50)
            slow_period = getattr(self.strategy, "slow_period", 200)
            channel_period = getattr(self.strategy, "lookback_period", 20)

            ema_fast = indicators.ema(fast_period)
            ema_slow = indicators.ema(slow_period)
            sma_20 = indicators.sma(20)
            sma_50 = indicators.sma(50)
            sma_200 = indicators.sma(200)
            previous_sma_200 = indicators.previous_sma(200)
            atr = indicators.atr(14)
            volatility = indicators.volatility(20)
            volatility_average = indicators.volatility(60)
            rsi = indicators.rsi(14)
            recent_high = indicators.highest_high(
                channel_period,
                exclude_latest=True,
            )
            recent_low = indicators.lowest_low(
                channel_period,
                exclude_latest=True,
            )

            if ema_fast is None or ema_slow is None:
                continue

            current_position = self.trade_manager.get_position(self.symbol)
            regime = MarketRegimeAnalyzer().classify(
                close=candle.close,
                sma_200=sma_200,
                previous_sma_200=previous_sma_200,
                volatility=volatility,
                volatility_average=volatility_average,
            )

            context = StrategyContext(
                symbol=self.symbol,
                timestamp=candle.timestamp,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                atr=atr,
                volatility=volatility,
                rsi=rsi,
                current_position=current_position,
                close=candle.close,
                recent_high=recent_high,
                recent_low=recent_low,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                previous_sma_200=previous_sma_200,
                volatility_average=volatility_average,
                market_regime=regime.market_regime,
                volatility_regime=regime.volatility_regime,
            )

            risk_context = RiskContext(
                atr=atr,
                volatility=volatility,
            )

            self._update_trailing_stop(candle, atr)

            if (
                getattr(self.strategy, "use_engine_exits", True)
                and self._check_exit_rules(candle)
            ):
                self._update_portfolio(candle)
                if self._should_stop_early():
                    break
                continue

            signal = self.strategy.generate_signal(context)
            self.signal_diagnostics.record_signal(signal.action)

            self._debug_signal(
                candle=candle,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                signal=signal,
            )

            if self._should_skip_signal(signal, candle):
                self._update_portfolio(candle)
                if self._should_stop_early():
                    break
                continue

            self._execute_signal(signal, candle, risk_context)

            self._update_portfolio(candle)
            if self._should_stop_early():
                break

        if self.close_open_trades_at_end:
            self._close_open_trades_at_end(last_candle)
        self._last_result = self._build_result()
        return self._last_result

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

    def summary(self):

        if self._last_result is not None:
            return self._last_result.to_dict()

        if hasattr(self.portfolio_engine, "summary"):
            stats = self.portfolio_engine.summary(self.trade_manager)
            return self._result_from_summary(stats).to_dict()

        return {
            "open_trades": len(self.trade_manager.open_trades),
            "closed_trades": len(self.trade_manager.closed_trades),
        }

    def save_report(
        self,
        result=None,
        report_dir: str = "reports/backtests",
    ):

        backtest_result = result or self._last_result

        if backtest_result is None:
            backtest_result = self._build_result()

        return backtest_result.save_json(
            symbol=self.symbol,
            timeframe=self.timeframe,
            report_dir=report_dir,
        )

    def _build_result(self):

        if hasattr(self.portfolio_engine, "summary"):
            stats = self.portfolio_engine.summary(self.trade_manager)
            return self._result_from_summary(stats)

        return BacktestResult(
            starting_equity=self.account_equity,
            final_equity=self.account_equity,
            total_return=0,
            max_drawdown=0,
            sharpe=0,
            closed_trades=len(self.trade_manager.closed_trades),
            open_trades=len(self.trade_manager.open_trades),
            equity_curve=[],
            profit_factor=0,
            trade_analysis=TradeAnalyzer().analyze([]),
            capital_utilization=CapitalUtilizationAnalyzer().analyze(
                [],
                [],
                self.account_equity,
            ),
            signal_diagnostics=self.signal_diagnostics,
        )

    def _result_from_summary(self, stats):
        equity_curve = list(self.portfolio_engine.equity_curve)
        period_start = equity_curve[0].timestamp if equity_curve else None
        period_end = equity_curve[-1].timestamp if equity_curve else None
        trade_analysis = TradeAnalyzer().analyze(
            self.trade_manager.closed_trades,
            period_start=period_start,
            period_end=period_end,
        )
        all_trades = [
            *self.trade_manager.closed_trades,
            *self.trade_manager.open_trades.values(),
        ]
        capital_utilization = CapitalUtilizationAnalyzer().analyze(
            trades=all_trades,
            equity_curve=equity_curve,
            starting_equity=stats["starting_cash"],
        )

        return BacktestResult(
            starting_equity=stats["starting_cash"],
            final_equity=stats["final_equity"],
            total_return=stats["total_return"],
            max_drawdown=stats["max_drawdown"],
            sharpe=stats["sharpe_ratio"],
            closed_trades=len(self.trade_manager.closed_trades),
            open_trades=len(self.trade_manager.open_trades),
            equity_curve=equity_curve,
            profit_factor=stats.get("profit_factor", 0),
            trade_analysis=trade_analysis,
            capital_utilization=capital_utilization,
            signal_diagnostics=self.signal_diagnostics,
        )
