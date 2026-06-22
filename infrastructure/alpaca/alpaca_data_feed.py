from __future__ import annotations

from datetime import datetime
import json
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class AlpacaDataFeed(IDataFeed):
    """Alpaca historical market-data adapter with explicit page handling."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        data_feed: str = "iex",
        adjustment: str = "all",
        historical_bar_limit: int = 10_000,
        opener: Callable = urlopen,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_feed = data_feed
        self.adjustment = adjustment
        self.historical_bar_limit = historical_bar_limit
        self._opener = opener
        self._last_request_metadata: dict[str, dict] = {}

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": self.data_feed,
            "adjustment": self.adjustment,
            "limit": self.historical_bar_limit,
            "sort": "asc",
        }
        bars = []
        page_count = 0
        next_page_token = None
        while True:
            page_params = dict(params)
            if next_page_token:
                page_params["page_token"] = next_page_token
            payload = self._request_page(page_params)
            bars.extend(payload.get("bars", {}).get(symbol, []))
            page_count += 1
            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break

        candles = [self._to_candle(symbol, bar) for bar in bars]
        candles.sort(key=lambda candle: candle.timestamp)
        self._last_request_metadata[symbol] = {
            "page_count": page_count,
            "next_page_token_handled": page_count > 1,
        }
        return candles

    def get_last_request_metadata(self, symbol: str) -> dict:
        return dict(self._last_request_metadata.get(symbol, {}))

    def _request_page(self, params: dict) -> dict:
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params)
        request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Alpaca historical bars request failed ({error.code}): {details}"
            ) from error

    def _to_candle(self, symbol: str, bar: dict) -> Candle:
        return Candle(
            symbol=symbol,
            timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
            open=float(bar["o"]),
            high=float(bar["h"]),
            low=float(bar["l"]),
            close=float(bar["c"]),
            volume=float(bar["v"]),
        )
