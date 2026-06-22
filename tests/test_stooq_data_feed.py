from datetime import datetime

import pytest

from infrastructure.data.stooq_data_feed import StooqDataFeed


class _Response:
    def __init__(self, payload: str):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload.encode("utf-8")


def test_stooq_data_feed_parses_daily_candles_and_uses_us_symbol_suffix():
    requested_urls = []

    def opener(url, timeout):
        requested_urls.append((url, timeout))
        return _Response(
            "Date,Open,High,Low,Close,Volume\n"
            "2015-01-02,10,12,9,11,100\n"
            "2015-01-05,11,13,10,12,200\n"
        )

    candles = StooqDataFeed(opener=opener).get_historical_bars(
        "AAPL",
        "1Day",
        datetime(2015, 1, 1),
        datetime(2015, 1, 31),
    )

    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert "s=aapl.us" in requested_urls[0][0]
    assert "d1=20150101" in requested_urls[0][0]


def test_stooq_data_feed_rejects_intraday_timeframes():
    with pytest.raises(ValueError, match="1Day"):
        StooqDataFeed().get_historical_bars(
            "AAPL",
            "1Hour",
            datetime(2015, 1, 1),
            datetime(2015, 1, 31),
        )


def test_stooq_data_feed_rejects_browser_verification_pages():
    def opener(url, timeout):
        return _Response("<html>browser verification</html>")

    with pytest.raises(RuntimeError, match="verification"):
        StooqDataFeed(opener=opener).get_historical_bars(
            "AAPL",
            "1Day",
            datetime(2015, 1, 1),
            datetime(2015, 1, 31),
        )
