from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

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


@dataclass(frozen=True)
class RelativeStrengthSelection:
    timestamp: object
    symbols: list[str]
    scores: dict[str, float]

    def to_dict(self) -> dict:
        timestamp = self.timestamp
        return {
            "timestamp": (
                timestamp.isoformat()
                if hasattr(timestamp, "isoformat")
                else str(timestamp)
            ),
            "symbols": self.symbols,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class RelativeStrengthPortfolioResult:
    result: BacktestResult
    selections: list[RelativeStrengthSelection]
    benchmark_return: float
    excess_return: float
    config: dict
    equal_weight_return: float = 0
    excess_vs_equal_weight: float = 0
    turnover_percent: float = 0
    rebalance_count: int = 0
    estimated_cost: float = 0

    def save_json(
        self,
        report_dir: str = "reports/summary",
        filename: str = "relative_strength_portfolio.json",
    ) -> Path:
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        payload = {
            "result": self.result.to_dict(),
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "equal_weight_return": self.equal_weight_return,
            "excess_vs_equal_weight": self.excess_vs_equal_weight,
            "turnover_percent": self.turnover_percent,
            "rebalance_count": self.rebalance_count,
            "estimated_cost": self.estimated_cost,
            "config": self.config,
            "selections": [
                selection.to_dict()
                for selection in self.selections
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


class RelativeStrengthPortfolioBacktester:

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
        open_trades = len(positions)
        result = BacktestResult(
            starting_equity=self.starting_equity,
            final_equity=final_equity,
            total_return=total_return(self.starting_equity, final_equity),
            max_drawdown=max_drawdown([point.equity for point in equity_curve]),
            sharpe=sharpe_ratio(returns),
            closed_trades=len(trade_pnls),
            open_trades=open_trades,
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

    def _prices_by_symbol(self, candles_by_symbol):
        return {
            symbol: {
                candle.timestamp: candle.close
                for candle in candles
            }
            for symbol, candles in candles_by_symbol.items()
            if candles
        }

    def _common_timestamps(self, prices_by_symbol):
        if not prices_by_symbol:
            return []

        common = set.intersection(
            *[
                set(prices.keys())
                for prices in prices_by_symbol.values()
            ]
        )
        max_lookback = max([self.sma_period] + self.momentum_periods)
        return sorted(common)[max_lookback:]

    def _rank_symbols(self, timestamp, prices_by_symbol):
        ranked = []

        for symbol, prices in prices_by_symbol.items():
            timestamps = sorted(prices)
            index = self._timestamp_index(timestamps, timestamp)
            if index is None:
                continue

            if not self._above_sma_filter(prices, timestamps, index):
                continue

            score = self._momentum_score(prices, timestamps, index)
            if score is not None:
                ranked.append((symbol, score))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _timestamp_index(self, timestamps, timestamp):
        try:
            return timestamps.index(timestamp)
        except ValueError:
            return None

    def _above_sma_filter(self, prices, timestamps, index):
        if index < self.sma_period:
            return False

        close = prices[timestamps[index]]
        sma = sum(
            prices[timestamps[position]]
            for position in range(index - self.sma_period + 1, index + 1)
        ) / self.sma_period
        previous_sma = sum(
            prices[timestamps[position]]
            for position in range(index - self.sma_period, index)
        ) / self.sma_period

        return close > sma and sma >= previous_sma

    def _momentum_score(self, prices, timestamps, index):
        scores = []

        for period in self.momentum_periods:
            if index < period:
                return None

            current = prices[timestamps[index]]
            previous = prices[timestamps[index - period]]
            if previous <= 0:
                return None

            scores.append((current / previous) - 1)

        return sum(scores) / len(scores) if scores else None

    def _should_rebalance(self, timestamp, last_rebalance_key):
        return self._rebalance_key(timestamp) != last_rebalance_key

    def _rebalance_key(self, timestamp):
        if self.rebalance_frequency == "weekly":
            calendar = timestamp.isocalendar()
            return calendar.year, calendar.week

        return timestamp.year, timestamp.month

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

    def _profit_factor(self, pnls):
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl <= 0))
        return gross_profit / gross_loss if gross_loss else 0

    def _trade_analysis(self, pnls, exposure_values):
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        return TradeAnalysis(
            total_trades=len(pnls),
            win_rate=len(wins) / len(pnls) if pnls else 0,
            average_win=gross_profit / len(wins) if wins else 0,
            average_loss=sum(losses) / len(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
            expectancy=sum(pnls) / len(pnls) if pnls else 0,
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss
                else 0
            ),
            time_in_market_percent=(
                sum(1 for exposure in exposure_values if exposure > 0)
                / len(exposure_values)
                if exposure_values
                else 0
            ),
        )

    def _capital_utilization(self, exposure_values, position_values):
        average_exposure = (
            sum(exposure_values) / len(exposure_values)
            if exposure_values
            else 0
        )
        return CapitalUtilization(
            average_position_value=(
                sum(position_values) / len(position_values)
                if position_values
                else 0
            ),
            average_exposure_percent=average_exposure,
            max_exposure_percent=max(exposure_values) if exposure_values else 0,
            average_cash_percent=1 - average_exposure,
            average_leverage=average_exposure,
        )

    def _benchmark_return(self, prices_by_symbol, timestamps):
        if not timestamps:
            return 0

        prices = prices_by_symbol.get(self.benchmark_symbol)
        if not prices:
            return self._equal_weight_benchmark(prices_by_symbol, timestamps)

        start = prices[timestamps[0]]
        end = prices[timestamps[-1]]
        return (end / start) - 1 if start else 0

    def _equal_weight_benchmark(self, prices_by_symbol, timestamps):
        returns = []

        for prices in prices_by_symbol.values():
            start = prices.get(timestamps[0])
            end = prices.get(timestamps[-1])
            if start:
                returns.append((end / start) - 1)

        return sum(returns) / len(returns) if returns else 0
