from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed

from strategies.ema_crossover_strategy import EMACrossoverStrategy
from core.risk.simple_risk_manager import SimpleRiskManager

from core.services.trade_manager import TradeManager
from core.services.portfolio_engine import PortfolioEngine

from core.engine.execution_engine import ExecutionEngine
from core.engine.backtest_engine import BacktestEngine

from infrastructure.broker.paper_broker import PaperBroker

from core.execution.simple_execution_model import SimpleExecutionModel


def main():

    print("\n🚀 STARTING BACKTEST ENGINE\n")

    # -------------------------
    # 1. CONFIG + DATA
    # -------------------------
    config = load_config()

    feed = AlpacaDataFeed(
        config["alpaca"]["api_key"],
        config["alpaca"]["secret_key"]
    )

    candles = feed.get_historical_bars(
        symbol="AAPL",
        timeframe="1Day",
        start=datetime.utcnow() - timedelta(days=365),
        end=datetime.utcnow()
    )

    print(f"📊 Loaded {len(candles)} candles\n")

    # -------------------------
    # 2. CORE COMPONENTS
    # -------------------------
    strategy = EMACrossoverStrategy("AAPL")

    risk = SimpleRiskManager()

    trade_manager = TradeManager()

    portfolio = PortfolioEngine(
        starting_cash=10_000
    )

    broker = PaperBroker(None)

    execution_model = SimpleExecutionModel(
        spread_bps=2.0,
        slippage_bps=1.0
    )

    execution = ExecutionEngine(
        broker=broker,
        trade_manager=trade_manager,
        execution_model=execution_model
    )

    # -------------------------
    # 3. ENGINE
    # -------------------------
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        risk_manager=risk,
        execution_engine=execution,
        trade_manager=trade_manager,
        portfolio_engine=portfolio,
        symbol="AAPL",
        timeframe="1Day",
        debug=True
    )

    # -------------------------
    # 4. RUN BACKTEST
    # -------------------------
    engine.run(candles)

    # -------------------------
    # 5. FINAL REPORT
    # -------------------------
    print("\n==============================")
    print("📊 BACKTEST COMPLETE REPORT")
    print("==============================")

    print(f"🔥 Closed Trades: {len(trade_manager.closed_trades)}")
    print(f"📂 Open Trades: {len(trade_manager.open_trades)}")

    if trade_manager.closed_trades:

        total_pnl = sum(
            trade.pnl for trade in trade_manager.closed_trades
        )

        wins = sum(
            1 for trade in trade_manager.closed_trades
            if trade.pnl > 0
        )

        losses = sum(
            1 for trade in trade_manager.closed_trades
            if trade.pnl <= 0
        )

        print(f"💰 Total PnL: {total_pnl:.2f}")
        print(f"📈 Wins: {wins}")
        print(f"📉 Losses: {losses}")

    if portfolio.equity_curve:

        final_equity = portfolio.equity_curve[-1].equity

        print(f"📊 Final Equity: {final_equity:.2f}")

        try:
            stats = portfolio.summary()

            print("\n📈 PERFORMANCE")
            print(f"Return: {stats['total_return'] * 100:.2f}%")
            print(f"Max DD: {stats['max_drawdown'] * 100:.2f}%")
            print(f"Sharpe: {stats['sharpe_ratio']:.2f}")

        except Exception as exc:
            print(f"Performance summary unavailable: {exc}")

    print("\n✅ Done\n")


if __name__ == "__main__":
    main()
