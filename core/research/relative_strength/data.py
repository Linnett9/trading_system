from core.research import portfolio_utils


class RelativeStrengthDataMixin:

    def _prices_by_symbol(self, candles_by_symbol):
        return portfolio_utils.prices_by_symbol(candles_by_symbol)

    def _common_timestamps(self, prices_by_symbol):
        max_lookback = max([self.sma_period] + self.momentum_periods)
        return portfolio_utils.common_timestamps(prices_by_symbol, max_lookback)

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
