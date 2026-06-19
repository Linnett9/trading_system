from core.entities.backtest_result import BacktestResult
from core.entities.capital_utilization import CapitalUtilization
from core.entities.signal_diagnostics import SignalDiagnostics
from core.entities.trade_analysis import TradeAnalysis
from core.research.performance_metrics import (
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.services.portfolio_engine import EquityPoint


class MultiStrategyAggregationMixin:

    def _combine_results(self, sleeve_results):
        if not sleeve_results:
            return BacktestResult(
                starting_equity=self.starting_equity,
                final_equity=self.starting_equity,
                total_return=0,
                max_drawdown=0,
                sharpe=0,
                closed_trades=0,
                open_trades=0,
                equity_curve=[],
            )

        common_timestamps = set(
            point.timestamp
            for point in sleeve_results[0].result.equity_curve
        )
        for sleeve in sleeve_results[1:]:
            common_timestamps &= set(
                point.timestamp for point in sleeve.result.equity_curve
            )

        timestamps = sorted(common_timestamps)
        equity_by_sleeve = {
            sleeve.name: {
                point.timestamp: point.equity
                for point in sleeve.result.equity_curve
            }
            for sleeve in sleeve_results
        }
        equity_curve = [
            EquityPoint(
                timestamp=timestamp,
                equity=sum(
                    equity_by_sleeve[sleeve.name][timestamp]
                    for sleeve in sleeve_results
                ),
            )
            for timestamp in timestamps
        ]
        returns = []
        for index in range(1, len(equity_curve)):
            previous = equity_curve[index - 1].equity
            current = equity_curve[index].equity
            returns.append((current / previous) - 1 if previous else 0)

        final_equity = (
            equity_curve[-1].equity
            if equity_curve
            else self.starting_equity
        )
        exposure = self._weighted_metric(
            sleeve_results,
            "capital_utilization",
            "average_exposure_percent",
        )
        time_in_market = self._weighted_metric(
            sleeve_results,
            "trade_analysis",
            "time_in_market_percent",
        )

        return BacktestResult(
            starting_equity=self.starting_equity,
            final_equity=final_equity,
            total_return=total_return(self.starting_equity, final_equity),
            max_drawdown=max_drawdown([point.equity for point in equity_curve]),
            sharpe=sharpe_ratio(returns),
            closed_trades=sum(
                sleeve.result.closed_trades for sleeve in sleeve_results
            ),
            open_trades=sum(
                sleeve.result.open_trades for sleeve in sleeve_results
            ),
            equity_curve=equity_curve,
            profit_factor=self._average_profit_factor(sleeve_results),
            trade_analysis=TradeAnalysis(
                total_trades=sum(
                    sleeve.result.closed_trades for sleeve in sleeve_results
                ),
                profit_factor=self._average_profit_factor(sleeve_results),
                time_in_market_percent=time_in_market,
            ),
            capital_utilization=CapitalUtilization(
                average_exposure_percent=exposure,
                max_exposure_percent=max(
                    (
                        sleeve.result.capital_utilization.max_exposure_percent
                        * sleeve.weight
                        for sleeve in sleeve_results
                    ),
                    default=0,
                ),
                average_cash_percent=1 - exposure,
                average_leverage=exposure,
            ),
            signal_diagnostics=SignalDiagnostics(
                buy_signals=sum(
                    sleeve.result.signal_diagnostics.buy_signals
                    for sleeve in sleeve_results
                ),
                sell_signals=sum(
                    sleeve.result.signal_diagnostics.sell_signals
                    for sleeve in sleeve_results
                ),
                hold_signals=sum(
                    sleeve.result.signal_diagnostics.hold_signals
                    for sleeve in sleeve_results
                ),
            ),
        )

    def _weighted_metric(self, sleeve_results, parent_name, field_name):
        return sum(
            getattr(getattr(sleeve.result, parent_name), field_name)
            * sleeve.weight
            for sleeve in sleeve_results
        )

    def _average_profit_factor(self, sleeve_results):
        values = [
            sleeve.result.profit_factor
            for sleeve in sleeve_results
            if sleeve.result.profit_factor > 0
        ]
        return sum(values) / len(values) if values else 0
