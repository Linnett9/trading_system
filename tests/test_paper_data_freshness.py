from datetime import datetime, timedelta
from types import SimpleNamespace

from main import latest_data_freshness


def test_latest_data_freshness_flags_stale_data():
    old_timestamp = datetime.utcnow() - timedelta(days=10)
    candles_by_symbol = {
        "AAPL": [SimpleNamespace(timestamp=old_timestamp)],
    }

    freshness = latest_data_freshness(candles_by_symbol, max_age_days=3)

    assert freshness["age_days"] >= 10
    assert freshness["is_stale"] is True


def test_latest_data_freshness_accepts_recent_data():
    recent_timestamp = datetime.utcnow() - timedelta(days=1)
    candles_by_symbol = {
        "AAPL": [SimpleNamespace(timestamp=recent_timestamp)],
    }

    freshness = latest_data_freshness(candles_by_symbol, max_age_days=3)

    assert freshness["age_days"] <= 1
    assert freshness["is_stale"] is False
