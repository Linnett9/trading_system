import math


class DualMomentumWeightingMixin:

    def _fallback_symbols(
        self,
        selected,
        timestamp,
        prices_by_symbol,
        risk_asset_mode,
    ):
        if (
            not risk_asset_mode
            or self.fallback_allocation <= 0
            or len(selected) >= self.fallback_min_risk_assets
            or not self.fallback_symbols
        ):
            return []

        ranked = self._rank_symbols(
            timestamp=timestamp,
            prices_by_symbol=prices_by_symbol,
            allowed_symbols=set(self.fallback_symbols),
            momentum_periods=self.fallback_momentum_periods,
            skip_regime_symbol=False,
            apply_quality_filter=False,
            apply_leadership_filter=False,
            apply_relative_strength_filter=False,
        )

        selected_set = set(selected)

        return [
            symbol
            for symbol, _ in ranked
            if symbol not in selected_set
        ][: max(0, self.fallback_min_risk_assets - len(selected))]

    def _apply_benchmark_sleeve_weights(
        self,
        target_weights,
        timestamp,
        prices_by_symbol,
        risk_asset_mode,
    ):
        if (
            not risk_asset_mode
            or self.benchmark_sleeve_allocation <= 0
            or not self.benchmark_sleeve_symbols
        ):
            return target_weights

        if not self._benchmark_participation_active(
            target_weights,
            timestamp,
            prices_by_symbol,
        ):
            return target_weights

        ranked = self._rank_symbols(
            timestamp=timestamp,
            prices_by_symbol=prices_by_symbol,
            allowed_symbols=set(self.benchmark_sleeve_symbols),
            momentum_periods=self.benchmark_sleeve_momentum_periods,
            skip_regime_symbol=False,
            apply_quality_filter=False,
            apply_leadership_filter=False,
            apply_relative_strength_filter=False,
        )

        sleeve_symbols = [
            symbol
            for symbol, _ in ranked[:self.benchmark_sleeve_top_n]
        ]

        if not sleeve_symbols:
            return target_weights

        allocation = min(max(self.benchmark_sleeve_allocation, 0), 1)

        scaled_weights = {
            symbol: weight * (1 - allocation)
            for symbol, weight in target_weights.items()
        }

        sleeve_weight = allocation / len(sleeve_symbols)

        for symbol in sleeve_symbols:
            scaled_weights[symbol] = (
                scaled_weights.get(symbol, 0) + sleeve_weight
            )

        return scaled_weights

    def _benchmark_participation_active(
        self,
        target_weights,
        timestamp,
        prices_by_symbol,
    ):
        if not getattr(self, "benchmark_participation_filter_enabled", False):
            return True

        period = getattr(self, "benchmark_participation_period", 63)
        min_return = getattr(
            self,
            "benchmark_participation_min_return",
            0.03,
        )
        max_selected_excess = getattr(
            self,
            "benchmark_participation_max_selected_excess",
            0,
        )

        benchmark_returns = [
            self._symbol_momentum(symbol, timestamp, prices_by_symbol, period)
            for symbol in self.benchmark_sleeve_symbols
        ]
        benchmark_returns = [
            value for value in benchmark_returns if value is not None
        ]

        if not benchmark_returns:
            return False

        benchmark_return = max(benchmark_returns)

        if benchmark_return < min_return:
            return False

        selected_returns = [
            self._symbol_momentum(symbol, timestamp, prices_by_symbol, period)
            for symbol in target_weights
            if symbol not in self.benchmark_sleeve_symbols
        ]
        selected_returns = [
            value for value in selected_returns if value is not None
        ]

        if not selected_returns:
            return True

        selected_average = sum(selected_returns) / len(selected_returns)

        return (
            selected_average - benchmark_return
        ) <= max_selected_excess

    def _apply_fallback_weights(self, target_weights, fallback_symbols):
        if not fallback_symbols or self.fallback_allocation <= 0:
            return target_weights

        fallback_allocation = min(max(self.fallback_allocation, 0), 1)

        scaled_weights = {
            symbol: weight * (1 - fallback_allocation)
            for symbol, weight in target_weights.items()
        }

        fallback_weight = fallback_allocation / len(fallback_symbols)

        for symbol in fallback_symbols:
            scaled_weights[symbol] = (
                scaled_weights.get(symbol, 0) + fallback_weight
            )

        return scaled_weights

    def _target_weights(self, selected, timestamp, prices_by_symbol):
        if not selected:
            return {}

        if self.weighting == "inverse_volatility":
            weights = self._inverse_volatility_weights(
                selected,
                timestamp,
                prices_by_symbol,
            )
        else:
            weights = {
                symbol: 1 / len(selected)
                for symbol in selected
            }

        return self._apply_sector_caps(self._cap_weights(weights))

    def _inverse_volatility_weights(
        self,
        selected,
        timestamp,
        prices_by_symbol,
    ):
        inverse_volatilities = {}

        for symbol in selected:
            prices = prices_by_symbol.get(symbol)

            if not prices:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)

            if index is None:
                continue

            volatility = self._realized_volatility(
                prices,
                timestamps,
                index,
                self.weight_volatility_lookback,
            )

            inverse_volatilities[symbol] = (
                1 / volatility
                if volatility > 0
                else 1
            )

        total = sum(inverse_volatilities.values())

        if total <= 0:
            return {
                symbol: 1 / len(selected)
                for symbol in selected
            }

        return {
            symbol: inverse_volatilities.get(symbol, 0) / total
            for symbol in selected
        }

    def _cap_weights(self, weights):
        if self.max_position_weight is None:
            return weights

        remaining_symbols = set(weights)
        remaining_weight = 1.0
        capped = {}

        while remaining_symbols and remaining_weight > 0:
            raw_total = sum(weights[symbol] for symbol in remaining_symbols)

            if raw_total <= 0:
                break

            capped_this_round = False

            for symbol in sorted(remaining_symbols):
                proposed_weight = (
                    remaining_weight
                    * weights[symbol]
                    / raw_total
                )

                if proposed_weight > self.max_position_weight:
                    capped[symbol] = self.max_position_weight
                    remaining_weight -= self.max_position_weight
                    remaining_symbols.remove(symbol)
                    capped_this_round = True

            if not capped_this_round:
                for symbol in sorted(remaining_symbols):
                    capped[symbol] = (
                        remaining_weight
                        * weights[symbol]
                        / raw_total
                    )

                break

        return capped

    def _apply_sector_caps(self, weights):
        max_sector_weight = getattr(self, "max_sector_weight", None)

        if max_sector_weight is None or max_sector_weight <= 0:
            return weights

        sector_map = getattr(self, "sector_map", {}) or {}

        if not sector_map:
            return weights

        capped = dict(weights)

        for _ in range(10):
            sector_totals = {}

            for symbol, weight in capped.items():
                sector = sector_map.get(symbol)

                if sector is None:
                    continue

                sector_totals[sector] = sector_totals.get(sector, 0) + weight

            excess_by_sector = {
                sector: total - max_sector_weight
                for sector, total in sector_totals.items()
                if total > max_sector_weight
            }
            total_excess = sum(excess_by_sector.values())

            if total_excess <= 1e-12:
                break

            for sector, excess in excess_by_sector.items():
                sector_symbols = [
                    symbol
                    for symbol in capped
                    if sector_map.get(symbol) == sector
                ]
                sector_total = sum(
                    capped[symbol] for symbol in sector_symbols
                )

                if sector_total <= 0:
                    continue

                scale = max(0, (sector_total - excess) / sector_total)

                for symbol in sector_symbols:
                    capped[symbol] *= scale

            sector_totals = {}

            for symbol, weight in capped.items():
                sector = sector_map.get(symbol)

                if sector is None:
                    continue

                sector_totals[sector] = sector_totals.get(sector, 0) + weight

            eligible_symbols = [
                symbol
                for symbol in capped
                if (
                    sector_map.get(symbol) is None
                    or sector_totals.get(sector_map.get(symbol), 0)
                    < max_sector_weight - 1e-12
                )
            ]
            eligible_total = sum(capped[symbol] for symbol in eligible_symbols)

            if eligible_total <= 0:
                break

            for symbol in eligible_symbols:
                capped[symbol] += total_excess * (
                    capped[symbol] / eligible_total
                )

        total = sum(capped.values())

        if total <= 0:
            return weights

        return {
            symbol: weight / total
            for symbol, weight in capped.items()
        }

    def _target_exposure_for_rebalance(self, returns):
        if self.target_volatility is None:
            return self.target_exposure

        if len(returns) < self.volatility_lookback:
            return self.target_exposure

        recent = returns[-self.volatility_lookback:]
        mean = sum(recent) / len(recent)

        variance = (
            sum((value - mean) ** 2 for value in recent)
            / len(recent)
        )

        annualized_volatility = math.sqrt(variance) * math.sqrt(252)

        if annualized_volatility <= 0:
            return self.target_exposure

        volatility_scalar = self.target_volatility / annualized_volatility

        return min(
            self.target_exposure,
            self.target_exposure * volatility_scalar,
        )

    def _realized_volatility(self, prices, timestamps, index, lookback):
        if index < 1:
            return 0

        start = max(1, index - lookback + 1)
        returns = []

        for position in range(start, index + 1):
            previous = prices[timestamps[position - 1]]
            current = prices[timestamps[position]]

            if previous:
                returns.append((current / previous) - 1)

        if not returns:
            return 0

        mean = sum(returns) / len(returns)

        variance = (
            sum((value - mean) ** 2 for value in returns)
            / len(returns)
        )

        return math.sqrt(variance)
