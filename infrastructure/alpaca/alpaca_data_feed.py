from datetime import datetime
from typing import List
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class AlpacaDataFeed(IDataFeed):

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        data_feed: str = "iex",
        adjustment: str = "all",
    ):
        self.client = StockHistoricalDataClient(
            api_key,
            secret_key,
        )
        self.data_feed = data_feed
        self.adjustment = adjustment

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[Candle]:

        tf_map = {
            "1Min": TimeFrame.Minute,
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day
        }
        feed_map = {
            "iex": DataFeed.IEX,
            "sip": getattr(DataFeed, "SIP", DataFeed.IEX),
            "otc": getattr(DataFeed, "OTC", DataFeed.IEX),
        }

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map[timeframe],
            start=start,
            end=end,
            feed=feed_map.get(self.data_feed.lower(), DataFeed.IEX),
            adjustment=self._adjustment(),
        )

        response = self.client.get_stock_bars(request)

        candles = []

        for bar in response[symbol]:
            candles.append(
                Candle(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume)
                )
            )

        return candles

    def _adjustment(self):
        value = (self.adjustment or "raw").upper()
        return getattr(Adjustment, value, Adjustment.RAW)
