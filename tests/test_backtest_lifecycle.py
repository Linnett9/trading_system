from core.engine.backtest_engine import BacktestEngine
from core.entities.candle import Candle
from core.risk.simple_risk_manager import SimpleRiskManager
from core.services.portfolio_engine import PortfolioEngine
from core.services.trade_manager import TradeManager

from tests.test_backtest_accounting import (
    MarketPriceExecutionEngine,
    RecordingRiskManager,
    SequenceStrategy,
    make_candles,
)


def make_engine(actions, candles_count=201):
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(actions),
        risk_manager=RecordingRiskManager(),
        execution_engine=MarketPriceExecutionEngine(trade_manager),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
        atr_stop_multiplier=100,
        atr_take_profit_multiplier=100,
    )

    return engine, trade_manager, portfolio, make_candles(candles_count)


def replace_candle(candles, index, **overrides):
    candle = candles[index]
    values = {
        "symbol": candle.symbol,
        "timestamp": candle.timestamp,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }
    values.update(overrides)
    candles[index] = Candle(**values)


def test_buy_opens_trade():
    engine, trade_manager, _, candles = make_engine(["BUY"], 200)

    result = engine.run(candles)

    assert len(trade_manager.open_trades) == 1
    assert result.open_trades == 1


def test_sell_closes_trade():
    engine, trade_manager, _, candles = make_engine(["BUY", "SELL"], 201)

    result = engine.run(candles)

    assert len(trade_manager.closed_trades) == 1
    assert result.closed_trades == 1
    assert result.open_trades == 0


def test_atr_stop_closes_trade():
    engine, trade_manager, _, candles = make_engine(["BUY"], 201)
    engine.atr_stop_multiplier = 2
    replace_candle(candles, 200, low=290)

    result = engine.run(candles)

    assert result.closed_trades == 1
    assert trade_manager.closed_trades[0].exit_reason.startswith(
        "ATR stop loss hit"
    )


def test_take_profit_closes_trade():
    engine, trade_manager, _, candles = make_engine(["BUY"], 201)
    engine.atr_take_profit_multiplier = 3
    replace_candle(candles, 200, high=310)

    result = engine.run(candles)

    assert result.closed_trades == 1
    assert trade_manager.closed_trades[0].exit_reason.startswith(
        "ATR take profit hit"
    )


def test_trailing_atr_stop_closes_trade():
    engine, trade_manager, _, candles = make_engine(["BUY", "HOLD"], 201)
    engine.atr_stop_multiplier = 100
    engine.atr_take_profit_multiplier = None
    engine.trailing_atr_multiplier = 1
    replace_candle(candles, 200, high=320, low=305, close=306)

    result = engine.run(candles)

    assert result.closed_trades == 1
    assert trade_manager.closed_trades[0].exit_reason.startswith(
        "ATR trailing stop hit"
    )


def test_duplicate_buy_is_blocked():
    engine, trade_manager, _, candles = make_engine(["BUY", "BUY"], 201)

    result = engine.run(candles)

    assert len(trade_manager.open_trades) == 1
    assert result.closed_trades == 0
    assert result.open_trades == 1


def test_flat_sell_is_ignored():
    engine, trade_manager, _, candles = make_engine(["SELL"], 200)

    result = engine.run(candles)

    assert len(trade_manager.open_trades) == 0
    assert len(trade_manager.closed_trades) == 0
    assert result.closed_trades == 0
    assert result.open_trades == 0


def test_portfolio_equity_updates_with_open_trade():
    engine, _, portfolio, candles = make_engine(["BUY", "HOLD"], 201)

    result = engine.run(candles)

    assert portfolio.equity_curve[-1].equity == 10_001
    assert result.final_equity == 10_001


def test_engine_accepts_plain_simple_risk_manager():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(["BUY"]),
        risk_manager=SimpleRiskManager(),
        execution_engine=MarketPriceExecutionEngine(trade_manager),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
    )

    result = engine.run(make_candles(200))

    assert result.open_trades == 1
