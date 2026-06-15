from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed


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

    print(f"Retrieved {len(candles)} candles\n")

    print("First candle:")
    print(candles[0])

    print("\nLast candle:")
    print(candles[-1])


if __name__ == "__main__":
    main()
