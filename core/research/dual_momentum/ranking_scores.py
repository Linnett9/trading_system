import math


class DualMomentumRankingScoresMixin:
    def _rank_score(
        self,
        symbol,
        prices,
        timestamps,
        index,
        periods,
        prices_by_symbol,
        timestamp,
    ):
        if self.ranking_score_mode == "enhanced":
            return self._enhanced_rank_score(
                symbol,
                prices,
                timestamps,
                index,
                prices_by_symbol,
                timestamp,
            )

        score = self._momentum_score(prices, timestamps, index, periods)
        return self._apply_short_term_weakness_penalty(
            score,
            prices,
            timestamps,
            index,
        )
    def _enhanced_rank_score(
        self,
        symbol,
        prices,
        timestamps,
        index,
        prices_by_symbol,
        timestamp,
    ):
        momentum_score = self._weighted_momentum_score(
            prices,
            timestamps,
            index,
            self.enhanced_momentum_periods,
            self.enhanced_momentum_weights,
        )

        if momentum_score is None:
            return None

        relative_score = 0
        benchmark_prices = prices_by_symbol.get(self.relative_strength_symbol)

        if benchmark_prices and symbol != self.relative_strength_symbol:
            benchmark_timestamps = sorted(benchmark_prices)
            benchmark_index = self._timestamp_index(
                benchmark_timestamps,
                timestamp,
            )

            benchmark_score = (
                self._weighted_momentum_score(
                    benchmark_prices,
                    benchmark_timestamps,
                    benchmark_index,
                    self.relative_strength_periods,
                    [1 / len(self.relative_strength_periods)]
                    * len(self.relative_strength_periods),
                )
                if benchmark_index is not None
                else None
            )

            asset_relative_score = self._weighted_momentum_score(
                prices,
                timestamps,
                index,
                self.relative_strength_periods,
                [1 / len(self.relative_strength_periods)]
                * len(self.relative_strength_periods),
            )

            if benchmark_score is not None and asset_relative_score is not None:
                relative_score = asset_relative_score - benchmark_score

        volatility = self._realized_volatility(
            prices,
            timestamps,
            index,
            self.ranking_volatility_lookback,
        )
        annualized_volatility = volatility * math.sqrt(252)

        score = (
            momentum_score
            + self.relative_strength_weight * relative_score
            - self.volatility_penalty_weight * annualized_volatility
        )

        return self._apply_short_term_weakness_penalty(
            score,
            prices,
            timestamps,
            index,
        )
    def _apply_short_term_weakness_penalty(
        self,
        score,
        prices,
        timestamps,
        index,
    ):
        if score is None:
            return None

        if not getattr(self, "short_term_weakness_penalty_enabled", False):
            return score

        period = getattr(self, "short_term_weakness_penalty_period", 21)
        floor = getattr(self, "short_term_weakness_penalty_floor", -0.02)
        weight = getattr(self, "short_term_weakness_penalty_weight", 1.0)
        momentum = self._period_return(prices, timestamps, index, period)

        if momentum is None or momentum >= floor:
            return score

        return score - weight * abs(momentum - floor)
