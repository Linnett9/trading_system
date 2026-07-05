


class DualMomentumRankingFiltersMixin:
    def _has_short_term_weakness(self, prices, timestamps, index):
        if not getattr(self, "avoid_short_term_weakness", False):
            return False

        period = getattr(self, "short_term_momentum_period", 21)
        floor = getattr(self, "short_term_momentum_floor", -0.02)

        if period <= 0:
            return False

        if index is None or index < period:
            return False

        current = prices[timestamps[index]]
        previous = prices[timestamps[index - period]]

        if previous <= 0:
            return False

        short_term_momentum = (current / previous) - 1

        return short_term_momentum < floor
    def _passes_leadership_filter(
        self,
        symbol,
        prices,
        timestamps,
        index,
        prices_by_symbol,
        timestamp,
    ):
        if not self.leadership_filter_enabled:
            return True

        if symbol == self.leadership_symbol:
            return True

        benchmark_prices = prices_by_symbol.get(self.leadership_symbol)

        if not benchmark_prices:
            return False

        benchmark_timestamps = sorted(benchmark_prices)
        benchmark_index = self._timestamp_index(benchmark_timestamps, timestamp)

        if benchmark_index is None:
            return False

        asset_score = self._momentum_score(
            prices,
            timestamps,
            index,
            self.leadership_momentum_periods,
        )
        benchmark_score = self._momentum_score(
            benchmark_prices,
            benchmark_timestamps,
            benchmark_index,
            self.leadership_momentum_periods,
        )

        if asset_score is None or benchmark_score is None:
            return False

        return asset_score > benchmark_score
    def _passes_relative_strength_filter(
        self,
        symbol,
        prices,
        timestamps,
        index,
        prices_by_symbol,
        timestamp,
    ):
        if not getattr(self, "relative_strength_filter_enabled", False):
            return True

        benchmark_symbol = getattr(
            self,
            "relative_strength_filter_symbol",
            "SPY",
        )

        if symbol == benchmark_symbol:
            return True

        benchmark_prices = prices_by_symbol.get(benchmark_symbol)
        if not benchmark_prices:
            return False

        period = getattr(self, "relative_strength_filter_period", 63)
        benchmark_timestamps = sorted(benchmark_prices)
        benchmark_index = self._timestamp_index(benchmark_timestamps, timestamp)

        if benchmark_index is None:
            return False

        asset_return = self._period_return(prices, timestamps, index, period)
        benchmark_return = self._period_return(
            benchmark_prices,
            benchmark_timestamps,
            benchmark_index,
            period,
        )

        if asset_return is None or benchmark_return is None:
            return False

        min_excess = getattr(self, "relative_strength_filter_min_excess", 0)
        return asset_return - benchmark_return >= min_excess
    def _passes_quality_filter(self, prices, timestamps, index):
        if not self.quality_filter_enabled:
            return True

        required_index = max(
            self.quality_momentum_period,
            self.quality_sma_period,
        )

        if self.quality_require_momentum_improving:
            required_index = max(
                required_index,
                self.quality_momentum_period * 2,
            )

        if index < required_index:
            return False

        close = prices[timestamps[index]]
        previous = prices[timestamps[index - self.quality_momentum_period]]

        if previous <= 0 or (close / previous) - 1 <= 0:
            return False

        sma_start = index - self.quality_sma_period + 1
        sma = sum(
            prices[timestamps[position]]
            for position in range(sma_start, index + 1)
        ) / self.quality_sma_period

        if close <= sma:
            return False

        if not self.quality_require_momentum_improving:
            return True

        older = prices[timestamps[index - self.quality_momentum_period * 2]]

        if older <= 0:
            return False

        recent_momentum = (close / previous) - 1
        prior_momentum = (previous / older) - 1

        return recent_momentum > prior_momentum
