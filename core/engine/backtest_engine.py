# core/engine/backtest_engine.py

from core.entities.signal_diagnostics import SignalDiagnostics
from core.engine.backtest_engine_controls import BacktestEngineControlsMixin
from core.engine.backtest_engine_execution import BacktestEngineExecutionMixin
from core.engine.backtest_engine_loop import BacktestEngineLoopMixin
from core.engine.backtest_engine_results import BacktestEngineResultsMixin
from core.services.market_data_service import MarketDataService


class BacktestEngine(
    BacktestEngineLoopMixin,
    BacktestEngineExecutionMixin,
    BacktestEngineControlsMixin,
    BacktestEngineResultsMixin,
):

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
