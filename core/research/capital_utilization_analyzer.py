from core.entities.capital_utilization import CapitalUtilization


class CapitalUtilizationAnalyzer:

    def analyze(
        self,
        trades,
        equity_curve,
        starting_equity: float,
    ) -> CapitalUtilization:
        trades = list(trades)
        equity_curve = list(equity_curve)

        position_values = [
            trade.entry_price * trade.quantity
            for trade in trades
            if trade.quantity and trade.entry_price
        ]
        average_position_value = (
            sum(position_values) / len(position_values)
            if position_values
            else 0
        )

        exposures = [
            self._exposure_at(point, trades, starting_equity)
            for point in equity_curve
        ]
        average_exposure = (
            sum(exposures) / len(exposures)
            if exposures
            else 0
        )
        max_exposure = max(exposures) if exposures else 0

        return CapitalUtilization(
            average_position_value=average_position_value,
            average_exposure_percent=average_exposure,
            max_exposure_percent=max_exposure,
            average_cash_percent=max(0, 1 - average_exposure),
            average_leverage=average_exposure,
        )

    def _exposure_at(self, equity_point, trades, starting_equity):
        equity = equity_point.equity or starting_equity

        if equity <= 0:
            return 0

        notional = sum(
            trade.entry_price * trade.quantity
            for trade in trades
            if self._is_active_at(trade, equity_point.timestamp)
        )

        return notional / equity

    def _is_active_at(self, trade, timestamp) -> bool:
        if trade.entry_time is None:
            return False

        if timestamp < trade.entry_time:
            return False

        if trade.exit_time is None:
            return True

        return timestamp <= trade.exit_time
