from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed
from core.services.market_data_service import MarketDataService


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

    market_data = MarketDataService(
        symbol="AAPL",
        timeframe="1Day"
    )

    market_data.add_candles(candles)

    print("Total candles:", market_data.candle_count)
    print("Latest price:", market_data.latest_price())

    print("\nLatest candle:")
    print(market_data.latest())


if __name__ == "__main__":
    main()
