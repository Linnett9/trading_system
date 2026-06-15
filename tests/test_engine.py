from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed
from infrastructure.broker.paper_broker import PaperBroker

from strategies.ema_crossover_strategy import EMACrossoverStrategy
from core.risk.simple_risk_manager import SimpleRiskManager
from core.engine.execution_engine import ExecutionEngine
from core.engine.trading_engine import TradingEngine


def main():

    config = load_config()

    feed = AlpacaDataFeed(
        api_key=config["alpaca"]["api_key"],
        secret_key=config["alpaca"]["secret_key"]
    )

    strategy = EMACrossoverStrategy(symbol="AAPL")

    risk_manager = SimpleRiskManager()

    # broker depends on price feed (market data)
    temp_market_data = None  # will be injected later cleanly

    # engine creates market data internally → so we pass broker later
    engine = TradingEngine(
        data_feed=feed,
        strategy=strategy,
        risk_manager=risk_manager,
        execution_engine=None,  # TEMP (fix below)
        symbol="AAPL",
        timeframe="1Day",
        account_equity=10000
    )

    engine.load_data(
        start=datetime.utcnow() - timedelta(days=365),
        end=datetime.utcnow()
    )

    # FIX: inject broker after market data exists
    broker = PaperBroker(engine.market_data)

    execution_engine = ExecutionEngine(broker)

    engine.execution_engine = execution_engine

    engine.run_once()


if __name__ == "__main__":
    main()
