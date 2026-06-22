from datetime import datetime, timedelta

from application.services.market_data_loader import data_quality_report
from core.entities.candle import Candle


def candle(symbol="AAPL", close=100, timestamp=None):
    timestamp = timestamp or datetime(2026, 6, 1)
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000,
    )


def test_data_quality_report_flags_missing_and_bad_prices():
    report = data_quality_report(
        {
            "AAPL": [],
            "MSFT": [candle("MSFT", close=0)],
        },
        min_lookback_bars=1,
    )

    assert report["issues_by_symbol"]["AAPL"][0]["reason"] == "missing_candles"
    assert any(
        issue["reason"] == "zero_or_negative_prices"
        for issue in report["issues_by_symbol"]["MSFT"]
    )


def test_data_quality_report_flags_duplicate_dates_and_large_latest_gap():
    start = datetime(2026, 6, 1)
    report = data_quality_report(
        {
            "AAPL": [
                candle(timestamp=start, close=100),
                candle(timestamp=start, close=100),
                candle(timestamp=start + timedelta(days=1), close=200),
            ],
        },
        min_lookback_bars=1,
        max_latest_gap_percent=0.40,
    )
    reasons = {
        issue["reason"]
        for issue in report["issues_by_symbol"]["AAPL"]
    }

    assert "duplicate_candle_dates" in reasons
    assert "large_latest_price_gap" in reasons
