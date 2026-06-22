from datetime import datetime
import json
from urllib.parse import parse_qs, urlparse

from infrastructure.alpaca.alpaca_data_feed import AlpacaDataFeed


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_alpaca_feed_requests_all_pages_and_records_metadata():
    requests = []

    def opener(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        requests.append(query)
        if "page_token" not in query:
            return _Response({
                "bars": {"AAPL": [{"t": "2016-01-04T00:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}]},
                "next_page_token": "page-two",
            })
        return _Response({
            "bars": {"AAPL": [{"t": "2016-01-05T00:00:00Z", "o": 11, "h": 13, "l": 10, "c": 12, "v": 200}]},
            "next_page_token": None,
        })

    feed = AlpacaDataFeed(
        "key",
        "secret",
        historical_bar_limit=10_000,
        opener=opener,
    )
    candles = feed.get_historical_bars(
        "AAPL",
        "1Day",
        datetime(2016, 1, 1),
        datetime(2026, 1, 1),
    )

    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert requests[0]["limit"] == ["10000"]
    assert requests[0]["start"] == ["2016-01-01T00:00:00Z"]
    assert requests[1]["page_token"] == ["page-two"]
    assert feed.get_last_request_metadata("AAPL") == {
        "page_count": 2,
        "next_page_token_handled": True,
    }
