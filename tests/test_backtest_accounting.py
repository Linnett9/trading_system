from datetime import datetime, timedelta

from core.engine.backtest_engine import BacktestEngine
from core.entities.backtest_result import BacktestResult
from core.entities.candle import Candle
from core.entities.signal import Signal
from core.entities.risk_context import RiskContext
from core.execution.simple_execution_model import SimpleExecutionModel
from core.risk.simple_risk_manager import SimpleRiskManager
from core.services.portfolio_engine import PortfolioEngine
from core.services.trade_manager import TradeManager


class HoldStrategy:
    def generate_signal(self, context):
        return Signal(
            symbol=context.symbol,
            action="HOLD",
            timestamp=context.timestamp,
        )


class NoopExecutionEngine:
    def execute(self, signal, size, market_price):
        raise AssertionError("HOLD signals should not execute")


class SequenceStrategy:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def generate_signal(self, context):
        action = self.actions[self.index]
        self.index += 1

        return Signal(
            symbol=context.symbol,
            action=action,
            timestamp=context.timestamp,
        )


class RecordingRiskManager(SimpleRiskManager):
    def __init__(self):
        super().__init__()
        self.equity_inputs = []
        self.risk_contexts = []

    def position_size(
        self,
        signal,
        account_equity,
        market_price,
        risk_context=None,
    ):
        self.equity_inputs.append(account_equity)
        self.risk_contexts.append(risk_context)
        return 1


class MarketPriceExecutionEngine:
    def __init__(self, trade_manager):
        self.trade_manager = trade_manager

    def execute(self, signal, size, market_price):
        from core.entities.fill import Fill

        quantity = size
        if signal.action == "SELL":
            quantity = -size

        fill = Fill(
            symbol=signal.symbol,
            quantity=quantity,
            price=market_price,
            timestamp=signal.timestamp,
        )

        if signal.action == "BUY":
            return self.trade_manager.open_trade(fill)

        if signal.action == "SELL":
            return self.trade_manager.close_trade(
                fill,
                exit_reason=signal.reason,
            )

        return None


def make_candles(count):
    start = datetime(2024, 1, 1)

    return [
        Candle(
            symbol="AAPL",
            timestamp=start + timedelta(days=i),
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100 + i,
            volume=1_000,
        )
        for i in range(count)
    ]


def test_backtest_updates_portfolio_on_hold_bars():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=HoldStrategy(),
        risk_manager=SimpleRiskManager(),
        execution_engine=NoopExecutionEngine(),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
    )

    engine.run(make_candles(205))

    assert len(portfolio.equity_curve) == 6
    assert portfolio.summary()["num_returns_samples"] == 5


def test_simple_execution_model_can_be_seeded():
    signal = Signal(
        symbol="AAPL",
        action="BUY",
        timestamp=datetime(2024, 1, 1),
    )

    first = SimpleExecutionModel(seed=42).create_fill_price(signal, 100)
    second = SimpleExecutionModel(seed=42).create_fill_price(signal, 100)

    assert first == second


def test_backtest_sizes_from_current_equity_and_passes_risk_context():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)
    risk_manager = RecordingRiskManager()

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(["BUY", "HOLD", "SELL", "BUY"]),
        risk_manager=risk_manager,
        execution_engine=MarketPriceExecutionEngine(trade_manager),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
    )

    engine.run(make_candles(203))

    assert risk_manager.equity_inputs == [10_000, 10_001, 10_002]
    assert all(
        isinstance(context, RiskContext)
        for context in risk_manager.risk_contexts
    )
    assert all(
        context.atr is not None and context.volatility is not None
        for context in risk_manager.risk_contexts
    )


def test_backtest_closes_long_when_atr_stop_is_hit():
    candles = make_candles(201)
    candles[200] = Candle(
        symbol=candles[200].symbol,
        timestamp=candles[200].timestamp,
        open=candles[200].open,
        high=candles[200].high,
        low=290,
        close=candles[200].close,
        volume=candles[200].volume,
    )

    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(["BUY"]),
        risk_manager=RecordingRiskManager(),
        execution_engine=MarketPriceExecutionEngine(trade_manager),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
        atr_stop_multiplier=2,
        atr_take_profit_multiplier=3,
    )

    engine.run(candles)

    assert len(trade_manager.closed_trades) == 1
    closed_trade = trade_manager.closed_trades[0]

    assert closed_trade.exit_reason.startswith("ATR stop loss hit")
    assert closed_trade.exit_price == closed_trade.stop_loss


def test_portfolio_summary_includes_trade_quality_metrics():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(["BUY", "HOLD", "SELL"]),
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

    engine.run(make_candles(202))

    stats = portfolio.summary(trade_manager)

    assert stats["closed_trades"] == 1
    assert stats["win_rate"] == 1
    assert stats["average_win"] == 2
    assert stats["expectancy"] == 2
    assert stats["max_consecutive_losses"] == 0


def test_large_sample_backtest_tracks_many_bars_and_trades():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    valid_bars = 1_000 - 199
    actions = []

    while len(actions) < valid_bars:
        actions.extend(["BUY"] + ["HOLD"] * 8 + ["SELL"])

    engine = BacktestEngine(
        data_feed=None,
        strategy=SequenceStrategy(actions[:valid_bars]),
        risk_manager=RecordingRiskManager(),
        execution_engine=MarketPriceExecutionEngine(trade_manager),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
        atr_stop_multiplier=1_000,
        atr_take_profit_multiplier=1_000,
    )

    result = engine.run(make_candles(1_000))

    assert len(portfolio.equity_curve) == valid_bars
    assert len(trade_manager.closed_trades) == 80
    assert len(trade_manager.open_trades) == 1
    assert result.closed_trades == 80
    assert result.final_equity > result.starting_equity


def test_backtest_run_returns_result_object():
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=HoldStrategy(),
        risk_manager=SimpleRiskManager(),
        execution_engine=NoopExecutionEngine(),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
    )

    result = engine.run(make_candles(205))

    assert isinstance(result, BacktestResult)
    assert result.starting_equity == 10_000
    assert result.final_equity == 10_000
    assert result.closed_trades == 0
    assert result.open_trades == 0
    assert len(result.equity_curve) == 6
    assert result.trade_analysis.total_trades == 0


def test_backtest_result_can_be_saved_to_report_file(tmp_path):
    trade_manager = TradeManager()
    portfolio = PortfolioEngine(starting_cash=10_000)

    engine = BacktestEngine(
        data_feed=None,
        strategy=HoldStrategy(),
        risk_manager=SimpleRiskManager(),
        execution_engine=NoopExecutionEngine(),
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        warmup_bars=200,
    )

    result = engine.run(make_candles(205))
    path = result.save_json(
        symbol="AAPL",
        timeframe="1Day",
        report_dir=str(tmp_path),
    )

    assert path.exists()
    assert path.name.endswith("_AAPL_1Day.json")
    text = path.read_text(encoding="utf-8")
    assert '"symbol": "AAPL"' in text
    assert '"trade_analysis"' in text
