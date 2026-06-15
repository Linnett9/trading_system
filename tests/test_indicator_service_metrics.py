from datetime import datetime, timedelta

from core.entities.candle import Candle
from core.services.indicator_service import IndicatorService
from core.services.market_data_service import MarketDataService


def make_candles(count, symbol="AAPL"):
    start = datetime(2024, 1, 1)
    candles = []

    for index in range(count):
        close = 100 + index
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=start + timedelta(days=index),
                open=close - 0.5,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1_000 + index * 10,
            )
        )

    return candles


def make_indicator_service(count=140):
    market_data = MarketDataService("AAPL", "1Day")
    market_data.add_candles(make_candles(count))
    return IndicatorService(market_data)


def test_adx_returns_trend_strength_value():
    indicators = make_indicator_service()

    assert indicators.adx(14) is not None
    assert indicators.adx(14) >= 0


def test_bollinger_bandwidth_returns_positive_value():
    indicators = make_indicator_service()

    assert indicators.bollinger_bandwidth(20, 2.0) > 0


def test_relative_volume_uses_volume_average():
    indicators = make_indicator_service()

    assert indicators.relative_volume(20) > 1


def test_volatility_percentile_is_between_zero_and_one():
    indicators = make_indicator_service()
    percentile = indicators.volatility_percentile(20, 100)

    assert percentile is not None
    assert 0 <= percentile <= 1
