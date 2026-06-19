class DualMomentumExecutionMixin:

    def _rebalance(
        self,
        positions,
        entry_values,
        selected,
        target_weights,
        prices,
        cash,
        equity,
        target_exposure,
    ):
        pnls = []
        sold = 0
        bought = 0
        traded_value = 0
        total_cost = 0
        cooldown_symbols = []
        selected_symbols = set(selected)

        for symbol in list(positions):
            if symbol in selected_symbols:
                continue

            value = positions[symbol] * prices[symbol]
            cost = self._transaction_cost(value)
            cash += value - cost
            entry_value = entry_values.get(symbol, value)
            pnl = value - entry_value - cost
            pnls.append(pnl)

            if self._should_cooldown(pnl, entry_value):
                cooldown_symbols.append(symbol)

            traded_value += value
            total_cost += cost
            sold += 1
            del positions[symbol]
            entry_values.pop(symbol, None)

        if not selected:
            return (
                cash,
                pnls,
                sold,
                bought,
                traded_value,
                total_cost,
                cooldown_symbols,
            )

        for symbol in selected:
            if symbol not in prices or prices[symbol] <= 0:
                continue

            target_value = equity * target_exposure * target_weights.get(
                symbol,
                0,
            )

            if target_value <= 0:
                continue

            current_value = positions.get(symbol, 0) * prices[symbol]
            difference = target_value - current_value
            drift_band = max(0, getattr(self, "rebalance_drift_band", 0))

            if drift_band > 0:
                target_portfolio_weight = (
                    target_exposure * target_weights.get(symbol, 0)
                )
                current_portfolio_weight = (
                    current_value / equity
                    if equity > 0
                    else 0
                )

                if (
                    abs(current_portfolio_weight - target_portfolio_weight)
                    <= drift_band
                ):
                    continue

            min_trade_value = (
                equity
                * max(0, getattr(self, "rebalance_min_trade_weight", 0))
            )

            if abs(difference) < min_trade_value:
                continue

            if difference > 0:
                value = min(difference, cash)
                cost = self._transaction_cost(value)
                investable_value = max(0, value - cost)

                positions[symbol] = (
                    positions.get(symbol, 0)
                    + investable_value / prices[symbol]
                )
                entry_values[symbol] = entry_values.get(symbol, 0) + value
                cash -= value
                traded_value += investable_value
                total_cost += cost
                bought += 1
                continue

            if difference < 0 and symbol in positions:
                sell_value = min(abs(difference), current_value)
                quantity_to_sell = sell_value / prices[symbol]
                cost = self._transaction_cost(sell_value)
                original_entry = entry_values.get(symbol, current_value)
                entry_reduction = (
                    original_entry
                    * (sell_value / current_value)
                    if current_value
                    else 0
                )

                cash += sell_value - cost
                positions[symbol] -= quantity_to_sell
                entry_values[symbol] = original_entry - entry_reduction
                traded_value += sell_value
                total_cost += cost

                if positions[symbol] <= 1e-12:
                    pnl = sell_value - entry_reduction - cost
                    pnls.append(pnl)

                    if self._should_cooldown(pnl, entry_reduction):
                        cooldown_symbols.append(symbol)

                    del positions[symbol]
                    entry_values.pop(symbol, None)
                    sold += 1

        return (
            cash,
            pnls,
            sold,
            bought,
            traded_value,
            total_cost,
            cooldown_symbols,
        )

    def _apply_decay_exits(
        self,
        positions,
        entry_values,
        prices,
        timestamp,
        prices_by_symbol,
        cash,
    ):
        pnls = []
        sold = 0
        traded_value = 0
        total_cost = 0
        cooldown_symbols = []

        exit_symbols = self._decay_exit_symbols(
            positions,
            timestamp,
            prices_by_symbol,
        )

        for symbol in exit_symbols:
            if symbol not in positions or symbol not in prices:
                continue

            value = positions[symbol] * prices[symbol]
            cost = self._transaction_cost(value)
            cash += value - cost
            entry_value = entry_values.get(symbol, value)
            pnl = value - entry_value - cost
            pnls.append(pnl)

            if self._should_cooldown(pnl, entry_value):
                cooldown_symbols.append(symbol)

            traded_value += value
            total_cost += cost
            sold += 1
            del positions[symbol]
            entry_values.pop(symbol, None)

        return cash, pnls, sold, traded_value, total_cost, cooldown_symbols

    def _decay_exit_symbols(self, positions, timestamp, prices_by_symbol):
        if not positions:
            return []

        required_period = (
            self.decay_momentum_period
            if self.decay_exit_enabled
            else max(self.momentum_periods)
        )
        ranked = self._rank_symbols(
            timestamp=timestamp,
            prices_by_symbol=prices_by_symbol,
            momentum_periods=[required_period],
            apply_relative_strength_filter=False,
        )

        ranked_symbols = [symbol for symbol, _ in ranked]
        top_ranked = {
            symbol
            for symbol, _ in (
                ranked[:self.rank_drop_exit_top_n]
                if self.rank_drop_exit_top_n
                else []
            )
        }

        exit_symbols = []

        for symbol in positions:
            prices = prices_by_symbol.get(symbol)

            if not prices:
                continue

            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)

            if index is None or index < required_period:
                continue

            if self.decay_exit_enabled:
                score = self._momentum_score(
                    prices,
                    timestamps,
                    index,
                    [self.decay_momentum_period],
                )

                if score is not None and score <= 0:
                    exit_symbols.append(symbol)
                    continue

            if self.rank_drop_exit_top_n and symbol not in top_ranked:
                exit_symbols.append(symbol)
                continue

            if (
                self.rank_deterioration_exit_enabled
                and self._rank_deteriorated(symbol, ranked_symbols)
            ):
                exit_symbols.append(symbol)

        return exit_symbols

    def _rank_deteriorated(self, symbol, ranked_symbols):
        threshold = getattr(self, "rank_deterioration_exit_rank", None)

        if threshold is None:
            return False

        if symbol not in ranked_symbols:
            return True

        return ranked_symbols.index(symbol) + 1 > threshold

    def _should_cooldown(self, pnl, entry_value):
        if not self.cooldown_enabled or entry_value <= 0:
            return False

        return (pnl / entry_value) <= self.cooldown_loss_threshold

    def _tick_cooldowns(self, cooldowns):
        if not self.cooldown_enabled:
            cooldowns.clear()
            return

        for symbol in list(cooldowns):
            cooldowns[symbol] -= 1

            if cooldowns[symbol] <= 0:
                del cooldowns[symbol]

    def _apply_cooldowns(self, cooldowns, symbols):
        if not self.cooldown_enabled:
            return

        for symbol in symbols:
            cooldowns[symbol] = max(1, self.cooldown_periods)

    def _transaction_cost(self, trade_value):
        return trade_value * (self.effective_transaction_cost_bps / 10_000)
