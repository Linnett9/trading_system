class RelativeStrengthExecutionMixin:

    def _sell_unselected(
        self,
        positions,
        entry_values,
        selected,
        prices,
        cash,
    ):
        pnls = []
        sold = 0
        traded_value = 0
        total_cost = 0

        for symbol in list(positions):
            if symbol in selected:
                continue

            value = positions[symbol] * prices[symbol]
            cost = self._transaction_cost(value)
            cash += value - cost
            pnls.append(value - entry_values.get(symbol, value) - cost)
            traded_value += value
            total_cost += cost
            del positions[symbol]
            entry_values.pop(symbol, None)
            sold += 1

        return cash, pnls, sold, traded_value, total_cost

    def _buy_selected(
        self,
        positions,
        entry_values,
        selected,
        prices,
        cash,
        equity,
    ):
        if not selected:
            return cash, 0, 0, 0

        target_value = equity * self.target_exposure / len(selected)
        bought = 0
        traded_value = 0
        total_cost = 0

        for symbol in selected:
            if symbol in positions:
                continue

            value = min(target_value, cash)
            if value <= 0 or prices[symbol] <= 0:
                continue

            cost = self._transaction_cost(value)
            investable_value = max(0, value - cost)
            positions[symbol] = investable_value / prices[symbol]
            entry_values[symbol] = value
            cash -= value
            traded_value += investable_value
            total_cost += cost
            bought += 1

        return cash, bought, traded_value, total_cost

    def _transaction_cost(self, trade_value):
        return trade_value * (self.transaction_cost_bps / 10_000)
