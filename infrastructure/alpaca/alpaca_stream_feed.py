# infrastructure/alpaca/alpaca_stream_feed.py

from alpaca.data.live import StockDataStream


class AlpacaStreamFeed:

    def __init__(self, api_key: str, secret_key: str, symbol: str):
        self.stream = StockDataStream(api_key, secret_key)
        self.symbol = symbol

    def set_handler(self, callback):
        """
        callback(candle) will be triggered on every new bar
        """

        async def handle_bar(bar):

            candle = {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }

            callback(candle)

        self.stream.subscribe_bars(handle_bar, self.symbol)

    def run(self):
        self.stream.run()