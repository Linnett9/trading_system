from datetime import timedelta

from core.research import portfolio_utils
from core.research.walk_forward import normalize_datetime


class MultiStrategyDataMixin:

    def _slice_candles(self, candles_by_symbol, start_at=None, end_at=None):
        if start_at is None and end_at is None:
            return candles_by_symbol

        warmup_start = (
            normalize_datetime(start_at) - timedelta(days=self.warmup_days)
            if start_at is not None
            else None
        )
        normalized_end = normalize_datetime(end_at) if end_at is not None else None

        return {
            symbol: [
                candle
                for candle in candles
                if (
                    warmup_start is None
                    or normalize_datetime(candle.timestamp) >= warmup_start
                )
                and (
                    normalized_end is None
                    or normalize_datetime(candle.timestamp) <= normalized_end
                )
            ]
            for symbol, candles in candles_by_symbol.items()
        }

    def _prices_by_symbol(self, candles_by_symbol):
        return portfolio_utils.prices_by_symbol(candles_by_symbol)

    def _benchmark_return(self, prices_by_symbol, timestamps):
        return portfolio_utils.benchmark_return(
            prices_by_symbol,
            timestamps,
            self.benchmark_symbol,
        )

    def _equal_weight_return(self, prices_by_symbol, timestamps):
        return portfolio_utils.equal_weight_return(
            prices_by_symbol,
            timestamps,
        )
