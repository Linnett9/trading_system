from core.services.indicator_service import IndicatorService
from core.entities.strategy_context import StrategyContext
from core.entities.risk_context import RiskContext
from core.research.market_regime_analyzer import MarketRegimeAnalyzer


class BacktestEngineLoopMixin:
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
            volatility_percentile = indicators.volatility_percentile(20, 100)
            rsi = indicators.rsi(14)
            adx = indicators.adx(14)
            relative_volume = indicators.relative_volume(20)
            bollinger_middle, bollinger_upper, bollinger_lower = (
                indicators.bollinger_bands(20, 2.0)
            )
            bollinger_bandwidth = indicators.bollinger_bandwidth(20, 2.0)
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
                volatility_percentile=volatility_percentile,
                rsi=rsi,
                adx=adx,
                relative_volume=relative_volume,
                current_position=current_position,
                close=candle.close,
                recent_high=recent_high,
                recent_low=recent_low,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                previous_sma_200=previous_sma_200,
                volatility_average=volatility_average,
                bollinger_middle=bollinger_middle,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                bollinger_bandwidth=bollinger_bandwidth,
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
