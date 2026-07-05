


class DualMomentumRankingSelectionMixin:
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
