# core/engine/trading_engine.py

from core.entities.strategy_context import StrategyContext
from core.entities.risk_context import RiskContext
from core.services.market_data_service import MarketDataService
from core.services.indicator_service import IndicatorService


class TradingEngine:

    def __init__(
        self,
        data_feed,
        strategy,
        risk_manager,
        execution_engine,
        symbol: str,
        timeframe: str,
        account_equity: float = 10000
    ):
        self.data_feed = data_feed
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine

        self.symbol = symbol
        self.timeframe = timeframe
        self.account_equity = account_equity

        self.market_data = MarketDataService(symbol, timeframe)

    def load_data(self, start, end):
        candles = self.data_feed.get_historical_bars(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start=start,
            end=end
        )
        self.market_data.add_candles(candles)

    def run_once(self):

        indicators = IndicatorService(self.market_data)

        latest = self.market_data.latest()
        fast_period = getattr(self.strategy, "fast_period", 50)
        slow_period = getattr(self.strategy, "slow_period", 200)

        context = StrategyContext(
            symbol=self.symbol,
            timestamp=latest.timestamp,
            ema_fast=indicators.ema(fast_period),
            ema_slow=indicators.ema(slow_period),
            atr=indicators.atr(14),
            volatility=indicators.volatility(20),
            rsi=indicators.rsi(14),
            current_position=None,
            close=latest.close,
            recent_high=indicators.highest_high(20),
            recent_low=indicators.lowest_low(20),
        )

        risk_context = RiskContext(
            atr=context.atr,
            volatility=context.volatility,
        )

        signal = self.strategy.generate_signal(context)

        if not self.risk_manager.validate(signal):
            print("❌ Risk blocked signal")
            return None

        size = self.risk_manager.position_size(
            signal,
            self.account_equity,
            latest.close,
            risk_context=risk_context,
        )

        return self.execution_engine.execute(signal, size, latest.close)
