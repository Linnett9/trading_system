from datetime import datetime, timedelta

import pytest

pytest.importorskip("alpaca")

from config.config_loader import load_config
from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed
from core.services.market_data_service import MarketDataService
from core.services.indicator_service import IndicatorService


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

    indicators = IndicatorService(market_data)

    print("EMA 50:", indicators.ema(50))
    print("EMA 200:", indicators.ema(200))
    print("RSI 14:", indicators.rsi())
    print("ATR 14:", indicators.atr())


if __name__ == "__main__":
    main()
