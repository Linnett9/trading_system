# tests/test_strategy.py

from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed
from core.services.market_data_service import MarketDataService
from core.services.indicator_service import IndicatorService
from core.entities.strategy_context import StrategyContext
from strategies.ema_crossover_strategy import EMACrossoverStrategy


def main():

    config = load_config()

    feed = AlpacaDataFeed(
        api_key=config["alpaca"]["api_key"],
        secret_key=config["alpaca"]["secret_key"]
    )

    candles = feed.get_historical_bars(
        symbol="AAPL",
        timeframe="1Day",
        start=datetime.utcnow() - timedelta(days=365),
        end=datetime.utcnow()
    )

    market_data = MarketDataService("AAPL", "1Day")
    market_data.add_candles(candles)

    indicators = IndicatorService(market_data)

    latest = market_data.latest()

    context = StrategyContext(
        symbol="AAPL",
        timestamp=latest.timestamp,
        ema_fast=indicators.ema(50),
        ema_slow=indicators.ema(200)
    )

    strategy = EMACrossoverStrategy(symbol="AAPL")

    signal = strategy.generate_signal(context)

    print("\n=== STRATEGY SIGNAL ===")
    print(f"Symbol: {signal.symbol}")
    print(f"Action: {signal.action}")
    print(f"Timestamp: {signal.timestamp}")
    print(f"Confidence: {signal.confidence}")
    print(f"Reason: {signal.reason}")


if __name__ == "__main__":
    main()
