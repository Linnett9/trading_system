class DualMomentumRegimeMixin:

    def _risk_on(self, timestamp, prices_by_symbol):
        symbols = getattr(self, "regime_confirmation_symbols", None)
        mode = getattr(self, "regime_confirmation_mode", "primary")

        if not symbols or mode == "primary":
            return self._symbol_above_sma(
                self.regime_symbol,
                timestamp,
                prices_by_symbol,
                self.regime_sma_period,
            )

        checks = [
            self._symbol_above_sma(
                symbol,
                timestamp,
                prices_by_symbol,
                self.regime_sma_period,
            )
            for symbol in symbols
        ]

        if not checks:
            return False

        if mode == "all":
            return all(checks)

        if mode == "any":
            return any(checks)

        return checks[0]

    def _fast_reentry_signal(self, timestamp, prices_by_symbol):
        signal_symbols = self.fast_reentry_symbols or [self.regime_symbol]

        return (
            any(
                self._symbol_above_sma(
                    symbol,
                    timestamp,
                    prices_by_symbol,
                    self.fast_reentry_sma_period,
                )
                for symbol in signal_symbols
            )
            or any(
                self._symbol_momentum_positive(
                    symbol,
                    timestamp,
                    prices_by_symbol,
                    self.fast_reentry_momentum_period,
                )
                for symbol in signal_symbols
            )
            or self._breadth_passes_threshold(
                timestamp,
                prices_by_symbol,
                self.fast_reentry_breadth_percent,
            )
        )

    def _chop_filter_active(self, timestamp, prices_by_symbol):
        if not self.chop_filter_enabled:
            return False

        momentum = self._regime_momentum(
            timestamp,
            prices_by_symbol,
            self.chop_lookback,
        )

        if momentum is None:
            return False

        return momentum < self.min_chop_momentum

    def _regime_above_sma(self, timestamp, prices_by_symbol, period):
        return self._symbol_above_sma(
            self.regime_symbol,
            timestamp,
            prices_by_symbol,
            period,
        )

    def _symbol_above_sma(self, symbol, timestamp, prices_by_symbol, period):
        prices = prices_by_symbol.get(symbol)

        if not prices:
            return False

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)

        if index is None or index < period:
            return False

        close = prices[timestamps[index]]

        sma = sum(
            prices[timestamps[position]]
            for position in range(index - period + 1, index + 1)
        ) / period

        return close > sma

    def _regime_momentum_positive(self, timestamp, prices_by_symbol, period):
        return self._symbol_momentum_positive(
            self.regime_symbol,
            timestamp,
            prices_by_symbol,
            period,
        )

    def _symbol_momentum_positive(
        self,
        symbol,
        timestamp,
        prices_by_symbol,
        period,
    ):
        momentum = self._symbol_momentum(
            symbol,
            timestamp,
            prices_by_symbol,
            period,
        )

        return momentum is not None and momentum > 0

    def _regime_momentum(self, timestamp, prices_by_symbol, period):
        return self._symbol_momentum(
            self.regime_symbol,
            timestamp,
            prices_by_symbol,
            period,
        )

    def _symbol_momentum(self, symbol, timestamp, prices_by_symbol, period):
        prices = prices_by_symbol.get(symbol)

        if not prices:
            return None

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)

        if index is None or index < period:
            return None

        previous = prices[timestamps[index - period]]
        current = prices[timestamps[index]]

        return ((current / previous) - 1) if previous > 0 else None

    def _above_sma(self, prices, timestamps, index):
        if index < self.asset_sma_period:
            return False

        close = prices[timestamps[index]]

        sma = sum(
            prices[timestamps[position]]
            for position in range(
                index - self.asset_sma_period + 1,
                index + 1,
            )
        ) / self.asset_sma_period

        return close > sma

    def _breadth_passes(self, timestamp, prices_by_symbol):
        if self.min_breadth_percent <= 0:
            return True

        return self._breadth_passes_threshold(
            timestamp,
            prices_by_symbol,
            self.min_breadth_percent,
        )

    def _breadth_passes_threshold(
        self,
        timestamp,
        prices_by_symbol,
        threshold,
    ):
        if threshold <= 0:
            return True

        checked = 0
        passing = 0

        for symbol, prices in prices_by_symbol.items():
            if symbol == self.regime_symbol:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)

            if index is None or index < self.asset_sma_period:
                continue

            checked += 1

            if self._above_sma(prices, timestamps, index):
                passing += 1

        if checked == 0:
            return False

        return (passing / checked) >= threshold

    def _breadth_ratio(self, timestamp, prices_by_symbol):
        checked = 0
        passing = 0

        for symbol, prices in prices_by_symbol.items():
            if symbol == self.regime_symbol:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)

            if index is None or index < self.asset_sma_period:
                continue

            checked += 1

            if self._above_sma(prices, timestamps, index):
                passing += 1

        return passing / checked if checked else 0

    def _breadth_exposure_multiplier(self, timestamp, prices_by_symbol):
        if not getattr(self, "breadth_scaled_exposure_enabled", False):
            return 1.0

        breadth = self._breadth_ratio(timestamp, prices_by_symbol)
        tiers = getattr(self, "breadth_exposure_tiers", []) or []

        for threshold, multiplier in sorted(tiers, reverse=True):
            if breadth >= threshold:
                return multiplier

        return getattr(self, "breadth_exposure_floor", 0)

    def _drawdown_recovery_exposure_cap(self, current_drawdown):
        if not getattr(self, "drawdown_recovery_scaling_enabled", False):
            return 1.0

        caps = getattr(self, "drawdown_recovery_exposure_caps", []) or []

        for threshold, cap in sorted(caps, reverse=True):
            if current_drawdown >= threshold:
                return cap

        return 1.0

    def _volatility_shock_multiplier(self, timestamp, prices_by_symbol):
        if not getattr(self, "volatility_shock_filter_enabled", False):
            return 1.0

        symbol = getattr(
            self,
            "volatility_shock_symbol",
            self.regime_symbol,
        )
        prices = prices_by_symbol.get(symbol)

        if not prices:
            return 1.0

        timestamps = sorted(prices)
        index = self._timestamp_index(timestamps, timestamp)
        short_lookback = getattr(self, "volatility_shock_short_lookback", 21)
        long_lookback = getattr(self, "volatility_shock_long_lookback", 126)

        short_volatility = self._realized_volatility(
            prices,
            timestamps,
            index,
            short_lookback,
        )
        long_volatility = self._realized_volatility(
            prices,
            timestamps,
            index,
            long_lookback,
        )

        if long_volatility <= 0:
            return 1.0

        threshold = getattr(self, "volatility_shock_ratio_threshold", 2.0)

        if short_volatility / long_volatility <= threshold:
            return 1.0

        return getattr(self, "volatility_shock_exposure_multiplier", 0.50)

    def _scale_regime_exposure(
        self,
        regime_exposure,
        timestamp,
        prices_by_symbol,
        current_drawdown,
    ):
        scaled = regime_exposure
        scaled *= self._breadth_exposure_multiplier(
            timestamp,
            prices_by_symbol,
        )
        scaled *= self._volatility_shock_multiplier(
            timestamp,
            prices_by_symbol,
        )
        scaled = min(
            scaled,
            self._drawdown_recovery_exposure_cap(current_drawdown),
        )
        return max(0, min(1, scaled))

    def _drawdown_guard_active(
        self,
        current_drawdown,
        guard_rebalances_remaining,
    ):
        if self.max_drawdown_guard is None:
            return False

        return (
            current_drawdown >= self.max_drawdown_guard
            or guard_rebalances_remaining > 0
        )
