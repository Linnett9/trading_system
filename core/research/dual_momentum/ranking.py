import math


class DualMomentumRankingMixin:

    def _select_symbols(self, ranked):
        eligible = [
            (symbol, score)
            for symbol, score in ranked
            if score >= self.min_selection_score
        ]

        if self.selection_mode == "all_positive":
            selected = [symbol for symbol, _ in eligible]

            if self.max_selected_assets is not None:
                return selected[:self.max_selected_assets]

            return selected

        selected = [symbol for symbol, _ in eligible[:self.top_n]]

        if self.max_selected_assets is not None:
            return selected[:self.max_selected_assets]

        return selected

    def _rank_symbols(
        self,
        timestamp,
        prices_by_symbol,
        allowed_symbols=None,
        momentum_periods=None,
        skip_regime_symbol=True,
        blocked_symbols=None,
        apply_quality_filter=True,
        apply_leadership_filter=True,
        apply_relative_strength_filter=True,
        apply_short_term_weakness_filter=True,
    ):
        ranked = []
        periods = momentum_periods or self.momentum_periods
        blocked_symbols = blocked_symbols or set()

        for symbol, prices in prices_by_symbol.items():
            if skip_regime_symbol and symbol == self.regime_symbol:
                continue

            if symbol in blocked_symbols:
                continue

            if allowed_symbols is not None and symbol not in allowed_symbols:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)

            if index is None:
                continue

            if (
                self.use_asset_trend_filter
                and not self._above_sma(prices, timestamps, index)
            ):
                continue

            if (
                apply_short_term_weakness_filter
                and self._has_short_term_weakness(prices, timestamps, index)
            ):
                continue

            if (
                apply_quality_filter
                and not self._passes_quality_filter(prices, timestamps, index)
            ):
                continue

            if (
                apply_leadership_filter
                and not self._passes_leadership_filter(
                    symbol,
                    prices,
                    timestamps,
                    index,
                    prices_by_symbol,
                    timestamp,
                )
            ):
                continue

            if (
                apply_relative_strength_filter
                and not self._passes_relative_strength_filter(
                    symbol,
                    prices,
                    timestamps,
                    index,
                    prices_by_symbol,
                    timestamp,
                )
            ):
                continue

            score = self._rank_score(
                symbol=symbol,
                prices=prices,
                timestamps=timestamps,
                index=index,
                periods=periods,
                prices_by_symbol=prices_by_symbol,
                timestamp=timestamp,
            )

            if score is not None and score > 0:
                ranked.append((symbol, score))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

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

    def _apply_rank_hysteresis(self, selected, ranked, positions):
        max_replacements = getattr(
            self,
            "max_rebalance_replacements",
            None,
        )
        max_rank_override = getattr(
            self,
            "rank_hysteresis_max_rank",
            None,
        )
        replacement_score_gap = max(
            0,
            getattr(self, "replacement_score_gap", 0) or 0,
        )

        if (
            not getattr(self, "rank_hysteresis_enabled", False)
            and max_replacements is None
            and max_rank_override is None
        ):
            return selected

        if not positions:
            return selected

        margin = max(0, getattr(self, "rank_hysteresis_margin", 0))
        max_rank = (
            max_rank_override
            if max_rank_override is not None
            else self.top_n + margin
        )
        selected_set = set(selected)
        ranked_symbols = [symbol for symbol, _ in ranked]
        ranked_scores = dict(ranked)
        kept = []

        for symbol in positions:
            if symbol not in ranked_symbols:
                continue

            rank = ranked_symbols.index(symbol) + 1
            if (
                symbol in selected_set
                or (
                    getattr(self, "rank_hysteresis_enabled", False)
                    and rank <= max_rank
                )
            ):
                kept.append((rank, symbol))

        kept = [
            symbol
            for _, symbol in sorted(kept, key=lambda item: item[0])
        ]
        target_count = (
            self.max_selected_assets
            if self.max_selected_assets is not None
            else len(selected)
        )
        kept = kept[:target_count]
        replacement_slots = max(0, target_count - len(kept))

        if max_replacements is not None:
            replacement_slots = min(
                replacement_slots,
                max(0, max_replacements),
            )

        displaced = [
            symbol
            for symbol in positions
            if symbol in ranked_scores and symbol not in kept
        ]
        displaced = sorted(
            displaced,
            key=lambda symbol: ranked_symbols.index(symbol),
        )
        replacement_candidates = [
            symbol
            for symbol in selected
            if symbol not in kept
        ]
        replacements = []
        available_displaced = list(displaced)

        for candidate in replacement_candidates:
            if len(replacements) >= replacement_slots:
                break

            incumbent = (
                available_displaced[0]
                if available_displaced
                else None
            )

            if not self._passes_replacement_score_gap(
                candidate,
                [incumbent] if incumbent else [],
                ranked_scores,
                replacement_score_gap,
            ):
                available_displaced.pop(0)
                replacements.append(incumbent)
                continue

            if incumbent:
                available_displaced.pop(0)

            replacements.append(candidate)

        result = kept + replacements

        if not result:
            return selected

        return result[:target_count]

    def _passes_replacement_score_gap(
        self,
        candidate,
        displaced,
        ranked_scores,
        replacement_score_gap,
    ):
        if replacement_score_gap <= 0 or not displaced:
            return True

        candidate_score = ranked_scores.get(candidate)
        displaced_score = ranked_scores.get(displaced[0])

        if candidate_score is None or displaced_score is None:
            return True

        return (candidate_score - displaced_score) >= replacement_score_gap

    def _weighted_momentum_score(
        self,
        prices,
        timestamps,
        index,
        periods,
        weights,
    ):
        if not periods:
            return None

        if len(weights) != len(periods):
            weights = [1 / len(periods)] * len(periods)

        weighted_score = 0
        total_weight = 0

        for period, weight in zip(periods, weights):
            if index is None or index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]

            if previous <= 0:
                return None

            weighted_score += weight * ((current / previous) - 1)
            total_weight += weight

        return weighted_score / total_weight if total_weight else None

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

    def _period_return(self, prices, timestamps, index, period):
        if period <= 0:
            return None

        if index is None or index < period:
            return None

        current = prices[timestamps[index]]
        previous = prices[timestamps[index - period]]

        if previous <= 0:
            return None

        return (current / previous) - 1

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

    def _momentum_score(self, prices, timestamps, index, periods=None):
        scores = []
        periods = periods or self.momentum_periods

        for period in periods:
            if index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]

            if previous <= 0:
                return None

            scores.append((current / previous) - 1)

        return sum(scores) / len(scores) if scores else None
