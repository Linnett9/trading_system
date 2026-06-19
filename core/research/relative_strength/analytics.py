from core.research import portfolio_utils


class RelativeStrengthAnalyticsMixin:

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
        )

    def _equal_weight_benchmark(self, prices_by_symbol, timestamps):
        return portfolio_utils.equal_weight_return(
            prices_by_symbol,
            timestamps,
        )
