from datetime import datetime
from typing import List
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.entities.candle import Candle
from core.interfaces.data_feed import IDataFeed


class AlpacaDataFeed(IDataFeed):

    def __init__(
        self,
        api_key: str,
        secret_key: str
    ):
        self.client = StockHistoricalDataClient(
            api_key,
            secret_key,
        )

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

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map[timeframe],
            start=start,
            end=end,
            feed=DataFeed.IEX
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