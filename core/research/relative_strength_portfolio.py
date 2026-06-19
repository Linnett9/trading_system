from core.entities.backtest_result import BacktestResult
from core.entities.signal_diagnostics import SignalDiagnostics
from core.research.performance_metrics import (
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from core.research.relative_strength.analytics import (
    RelativeStrengthAnalyticsMixin,
)
from core.research.relative_strength.data import RelativeStrengthDataMixin
from core.research.relative_strength.execution import (
    RelativeStrengthExecutionMixin,
)
from core.research.relative_strength.models import (
    RelativeStrengthPortfolioResult,
    RelativeStrengthSelection,
)
from core.research.relative_strength.ranking import RelativeStrengthRankingMixin
from core.services.portfolio_engine import EquityPoint


class RelativeStrengthPortfolioBacktester(
    RelativeStrengthAnalyticsMixin,
    RelativeStrengthDataMixin,
    RelativeStrengthExecutionMixin,
    RelativeStrengthRankingMixin,
):

    def __init__(
        self,
        starting_equity: float = 500,
        top_n: int = 2,
        momentum_periods: list[int] | None = None,
        sma_period: int = 200,
        rebalance_frequency: str = "monthly",
        target_exposure: float = 1.0,
        benchmark_symbol: str = "SPY",
        transaction_cost_bps: float = 0,
    ):
        self.starting_equity = starting_equity
        self.top_n = top_n
        self.momentum_periods = momentum_periods or [63, 126]
        self.sma_period = sma_period
        self.rebalance_frequency = rebalance_frequency
        self.target_exposure = target_exposure
        self.benchmark_symbol = benchmark_symbol
        self.transaction_cost_bps = transaction_cost_bps

    def run(self, candles_by_symbol: dict[str, list]) -> RelativeStrengthPortfolioResult:
        prices_by_symbol = self._prices_by_symbol(candles_by_symbol)
        timestamps = self._common_timestamps(prices_by_symbol)
        cash = self.starting_equity
        positions: dict[str, float] = {}
        equity_curve = []
        returns = []
        selections = []
        trade_pnls = []
        entry_values: dict[str, float] = {}
        exposure_values = []
        position_values = []
        last_rebalance_key = None
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0
        turnover_value = 0
        estimated_cost = 0

        for timestamp in timestamps:
            prices = {
                symbol: prices_by_symbol[symbol][timestamp]
                for symbol in prices_by_symbol
            }
            equity = cash + sum(
                quantity * prices[symbol]
                for symbol, quantity in positions.items()
                if symbol in prices
            )

            if self._should_rebalance(timestamp, last_rebalance_key):
                last_rebalance_key = self._rebalance_key(timestamp)
                ranked = self._rank_symbols(
                    timestamp=timestamp,
                    prices_by_symbol=prices_by_symbol,
                )
                selected = [symbol for symbol, _ in ranked[:self.top_n]]
                scores = dict(ranked)
                selections.append(
                    RelativeStrengthSelection(
                        timestamp=timestamp,
                        symbols=selected,
                        scores=scores,
                    )
                )
                cash, closed_pnls, sold, sold_value, sell_cost = (
                    self._sell_unselected(
                        positions=positions,
                        entry_values=entry_values,
                        selected=selected,
                        prices=prices,
                        cash=cash,
                    )
                )
                turnover_value += sold_value
                estimated_cost += sell_cost
                trade_pnls.extend(closed_pnls)
                sell_signals += sold
                cash, bought, bought_value, buy_cost = self._buy_selected(
                    positions=positions,
                    entry_values=entry_values,
                    selected=selected,
                    prices=prices,
                    cash=cash,
                    equity=equity,
                )
                turnover_value += bought_value
                estimated_cost += buy_cost
                buy_signals += bought
            else:
                hold_signals += 1

            equity = cash + sum(
                quantity * prices[symbol]
                for symbol, quantity in positions.items()
                if symbol in prices
            )
            equity_curve.append(EquityPoint(timestamp=timestamp, equity=equity))

            if len(equity_curve) > 1:
                previous = equity_curve[-2].equity
                returns.append((equity - previous) / previous if previous else 0)

            exposure = sum(
                quantity * prices[symbol]
                for symbol, quantity in positions.items()
                if symbol in prices
            )
            exposure_values.append(exposure / equity if equity else 0)
            position_values.append(
                exposure / len(positions)
                if positions
                else 0
            )

        final_prices = {
            symbol: prices_by_symbol[symbol][timestamps[-1]]
            for symbol in prices_by_symbol
        } if timestamps else {}
        final_equity = (
            cash + sum(
                quantity * final_prices[symbol]
                for symbol, quantity in positions.items()
                if symbol in final_prices
            )
            if timestamps
            else self.starting_equity
        )
        result = BacktestResult(
            starting_equity=self.starting_equity,
            final_equity=final_equity,
            total_return=total_return(self.starting_equity, final_equity),
            max_drawdown=max_drawdown([point.equity for point in equity_curve]),
            sharpe=sharpe_ratio(returns),
            closed_trades=len(trade_pnls),
            open_trades=len(positions),
            equity_curve=equity_curve,
            profit_factor=self._profit_factor(trade_pnls),
            trade_analysis=self._trade_analysis(trade_pnls, exposure_values),
            capital_utilization=self._capital_utilization(
                exposure_values,
                position_values,
            ),
            signal_diagnostics=SignalDiagnostics(
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                hold_signals=hold_signals,
            ),
        )
        benchmark_return = self._benchmark_return(prices_by_symbol, timestamps)
        equal_weight_return = self._equal_weight_benchmark(
            prices_by_symbol,
            timestamps,
        )

        return RelativeStrengthPortfolioResult(
            result=result,
            selections=selections,
            benchmark_return=benchmark_return,
            excess_return=result.total_return - benchmark_return,
            equal_weight_return=equal_weight_return,
            excess_vs_equal_weight=result.total_return - equal_weight_return,
            turnover_percent=(
                turnover_value / self.starting_equity
                if self.starting_equity
                else 0
            ),
            rebalance_count=len(selections),
            estimated_cost=estimated_cost,
            config={
                "top_n": self.top_n,
                "momentum_periods": self.momentum_periods,
                "sma_period": self.sma_period,
                "rebalance_frequency": self.rebalance_frequency,
                "target_exposure": self.target_exposure,
                "benchmark_symbol": self.benchmark_symbol,
                "transaction_cost_bps": self.transaction_cost_bps,
            },
        )
