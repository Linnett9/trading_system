from dataclasses import dataclass

from core.entities.walk_forward_result import WalkForwardResult
from core.research.performance_metrics import composite_score, qualified_score


@dataclass(frozen=True)
class StrategyComparisonResult:
    strategy_name: str
    symbol: str
    walk_forward_result: WalkForwardResult

    @property
    def average_excess_return(self) -> float:
        return self.walk_forward_result.average_excess_return

    @property
    def average_test_sharpe(self) -> float:
        return self.walk_forward_result.average_test_sharpe

    @property
    def closed_trades(self) -> int:
        return sum(
            fold.test_result.closed_trades
            for fold in self.walk_forward_result.folds
        )

    @property
    def average_max_drawdown(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.max_drawdown
            for fold in folds
        ) / len(folds)

    @property
    def average_profit_factor(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.profit_factor
            for fold in folds
        ) / len(folds)

    @property
    def average_time_in_market_percent(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.trade_analysis.time_in_market_percent
            for fold in folds
        ) / len(folds)

    @property
    def average_trade_duration_days(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.trade_analysis.average_trade_duration_days
            for fold in folds
        ) / len(folds)

    @property
    def average_exposure_percent(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.capital_utilization.average_exposure_percent
            for fold in folds
        ) / len(folds)

    @property
    def average_cash_percent(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 1

        return sum(
            fold.test_result.capital_utilization.average_cash_percent
            for fold in folds
        ) / len(folds)

    @property
    def average_position_value(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.test_result.capital_utilization.average_position_value
            for fold in folds
        ) / len(folds)

    @property
    def average_benchmark_sharpe(self) -> float:
        return self.walk_forward_result.average_benchmark_sharpe

    @property
    def average_benchmark_max_drawdown(self) -> float:
        return self.walk_forward_result.average_benchmark_max_drawdown

    @property
    def average_excess_return_per_unit_risk(self) -> float:
        folds = self.walk_forward_result.folds

        if not folds:
            return 0

        return sum(
            fold.excess_return_per_unit_risk
            for fold in folds
        ) / len(folds)

    @property
    def passed_folds(self) -> int:
        return sum(
            1 for fold in self.walk_forward_result.folds
            if fold.passed
        )

    @property
    def composite_score(self) -> float:
        return composite_score(
            excess_return=self.average_excess_return,
            sharpe=self.average_test_sharpe,
            max_drawdown_value=self.average_max_drawdown,
            profit_factor_value=self.average_profit_factor,
            closed_trades=self.closed_trades,
            target_trades=20,
            passed_folds=self.passed_folds,
            total_folds=len(self.walk_forward_result.folds),
        )

    @property
    def qualified_score(self) -> float:
        return qualified_score(
            excess_return=self.average_excess_return,
            sharpe=self.average_test_sharpe,
            max_drawdown_value=self.average_max_drawdown,
            profit_factor_value=self.average_profit_factor,
            closed_trades=self.closed_trades,
            target_trades=20,
            passed_folds=self.passed_folds,
            total_folds=len(self.walk_forward_result.folds),
            is_benchmark_strategy=self.strategy_name == "buy_and_hold",
        )

    @property
    def buy_signals(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.buy_signals
            for fold in self.walk_forward_result.folds
        )

    @property
    def sell_signals(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.sell_signals
            for fold in self.walk_forward_result.folds
        )

    @property
    def hold_signals(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.hold_signals
            for fold in self.walk_forward_result.folds
        )

    @property
    def duplicate_buy_skips(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.duplicate_buy_skips
            for fold in self.walk_forward_result.folds
        )

    @property
    def flat_sell_skips(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.flat_sell_skips
            for fold in self.walk_forward_result.folds
        )

    @property
    def risk_blocked_signals(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.risk_blocked_signals
            for fold in self.walk_forward_result.folds
        )

    @property
    def stop_loss_exits(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.stop_loss_exits
            for fold in self.walk_forward_result.folds
        )

    @property
    def take_profit_exits(self) -> int:
        return sum(
            fold.test_result.signal_diagnostics.take_profit_exits
            for fold in self.walk_forward_result.folds
        )
