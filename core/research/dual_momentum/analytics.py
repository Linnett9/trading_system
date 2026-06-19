from core.research import portfolio_utils


class DualMomentumAnalyticsMixin:

    def _equity(self, cash, positions, prices):
        return portfolio_utils.equity(cash, positions, prices)

    def _position_value(self, positions, prices):
        return portfolio_utils.position_value(positions, prices)

    def _profit_factor(self, pnls):
        return portfolio_utils.profit_factor(pnls)

    def _trade_analysis(self, pnls, exposure_values):
        return portfolio_utils.trade_analysis(pnls, exposure_values)

    def _capital_utilization(self, exposure_values, position_values):
        return portfolio_utils.capital_utilization(
            exposure_values,
            position_values,
        )

    def _benchmark_return(self, prices_by_symbol, timestamps):
        return portfolio_utils.benchmark_return(
            prices_by_symbol,
            timestamps,
            self.benchmark_symbol,
            excluded_equal_weight_symbols={self.regime_symbol},
        )

    def _equal_weight_benchmark(self, prices_by_symbol, timestamps):
        return portfolio_utils.equal_weight_return(
            prices_by_symbol,
            timestamps,
            excluded_symbols={self.regime_symbol},
        )

    def _period_returns(self, equity_curve, period):
        if not equity_curve:
            return {}

        grouped = {}

        for point in equity_curve:
            if period == "annual":
                key = point.timestamp.year
            else:
                key = point.timestamp.strftime("%Y-%m")

            grouped.setdefault(key, []).append(point.equity)

        return {
            key: (values[-1] / values[0]) - 1 if values[0] else 0
            for key, values in grouped.items()
        }

    def _rolling_12_month_returns(self, equity_curve):
        if len(equity_curve) < 252:
            return {}

        returns = {}

        for index in range(252, len(equity_curve)):
            start = equity_curve[index - 252]
            end = equity_curve[index]

            if start.equity:
                returns[end.timestamp.strftime("%Y-%m-%d")] = (
                    end.equity / start.equity
                ) - 1

        return returns

    def _elapsed_days(self, equity_curve):
        if len(equity_curve) < 2:
            return 0

        return (equity_curve[-1].timestamp - equity_curve[0].timestamp).days

    def _drawdown_statistics(self, equity_curve):
        peak = None
        max_dd = 0
        current_drawdown = 0
        drawdowns = []
        longest_days = 0
        current_start = None

        for point in equity_curve:
            if peak is None or point.equity >= peak:
                peak = point.equity
                current_start = None
                current_drawdown = 0
                continue

            current_drawdown = (peak - point.equity) / peak if peak else 0
            drawdowns.append(current_drawdown)
            max_dd = max(max_dd, current_drawdown)

            if current_start is None:
                current_start = point.timestamp

            longest_days = max(
                longest_days,
                (point.timestamp - current_start).days,
            )

        return {
            "max_drawdown": max_dd,
            "average_drawdown": (
                sum(drawdowns) / len(drawdowns)
                if drawdowns
                else 0
            ),
            "current_drawdown": current_drawdown,
            "longest_drawdown_days": longest_days,
        }
