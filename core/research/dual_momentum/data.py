from core.research import portfolio_utils
from core.research.walk_forward import normalize_datetime


class DualMomentumDataMixin:

    def _prices_by_symbol(self, candles_by_symbol):
        return portfolio_utils.prices_by_symbol(candles_by_symbol)

    def _common_timestamps(self, prices_by_symbol, start_at=None, end_at=None):
        max_lookback = max(
            [
                self.regime_sma_period,
                self.asset_sma_period if self.use_asset_trend_filter else 0,
            ]
            + self.momentum_periods
        )

        timestamps = portfolio_utils.common_timestamps(
            prices_by_symbol,
            max_lookback,
        )

        if start_at is not None:
            normalized_start = normalize_datetime(start_at)
            timestamps = [
                timestamp
                for timestamp in timestamps
                if normalize_datetime(timestamp) >= normalized_start
            ]

        if end_at is not None:
            normalized_end = normalize_datetime(end_at)
            timestamps = [
                timestamp
                for timestamp in timestamps
                if normalize_datetime(timestamp) <= normalized_end
            ]

        return timestamps

    def _prices_at(self, prices_by_symbol, timestamp):
        return portfolio_utils.prices_at(prices_by_symbol, timestamp)

    def _timestamp_index(self, timestamps, timestamp):
        return portfolio_utils.timestamp_index(timestamps, timestamp)

    def _should_rebalance(self, timestamp, last_rebalance_key):
        return portfolio_utils.should_rebalance(
            timestamp,
            last_rebalance_key,
            self.rebalance_frequency,
        )

    def _rebalance_key(self, timestamp):
        return portfolio_utils.rebalance_key(
            timestamp,
            self.rebalance_frequency,
        )
