from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json

from core.entities.benchmark_result import BenchmarkResult
from core.entities.backtest_result import BacktestResult
from core.entities.optimization_result import OptimizationResult


@dataclass(frozen=True)
class WalkForwardFoldResult:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_training_result: OptimizationResult
    test_result: BacktestResult
    benchmark_return: float
    benchmark: BenchmarkResult
    excess_return: float
    excess_return_per_unit_risk: float
    passed: bool
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "best_params": self.best_training_result.parameters,
            "train_metric_name": self.best_training_result.metric_name,
            "train_metric_value": self.best_training_result.metric_value,
            "train_return": self.best_training_result.result.total_return,
            "train_sharpe": self.best_training_result.result.sharpe,
            "train_max_drawdown": self.best_training_result.result.max_drawdown,
            "train_closed_trades": self.best_training_result.result.closed_trades,
            "train_trade_analysis": (
                self.best_training_result.result.trade_analysis.to_dict()
            ),
            "test_return": self.test_result.total_return,
            "test_sharpe": self.test_result.sharpe,
            "test_max_drawdown": self.test_result.max_drawdown,
            "test_closed_trades": self.test_result.closed_trades,
            "test_trade_analysis": self.test_result.trade_analysis.to_dict(),
            "test_capital_utilization": (
                self.test_result.capital_utilization.to_dict()
            ),
            "test_signal_diagnostics": (
                self.test_result.signal_diagnostics.to_dict()
            ),
            "benchmark_return": self.benchmark_return,
            "benchmark": self.benchmark.to_dict(),
            "benchmark_sharpe": self.benchmark.sharpe,
            "benchmark_max_drawdown": self.benchmark.max_drawdown,
            "excess_return": self.excess_return,
            "excess_return_per_unit_risk": (
                self.excess_return_per_unit_risk
            ),
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class WalkForwardResult:
    symbol: str
    timeframe: str
    folds: list[WalkForwardFoldResult]

    @property
    def average_test_return(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.test_result.total_return
            for fold in self.folds
        ) / len(self.folds)

    @property
    def average_test_sharpe(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.test_result.sharpe
            for fold in self.folds
        ) / len(self.folds)

    @property
    def average_benchmark_return(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.benchmark_return
            for fold in self.folds
        ) / len(self.folds)

    @property
    def average_benchmark_sharpe(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.benchmark.sharpe
            for fold in self.folds
        ) / len(self.folds)

    @property
    def average_benchmark_max_drawdown(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.benchmark.max_drawdown
            for fold in self.folds
        ) / len(self.folds)

    @property
    def average_excess_return(self) -> float:
        if not self.folds:
            return 0

        return sum(
            fold.excess_return
            for fold in self.folds
        ) / len(self.folds)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "folds": [fold.to_dict() for fold in self.folds],
            "average_test_return": self.average_test_return,
            "average_test_sharpe": self.average_test_sharpe,
            "average_benchmark_return": self.average_benchmark_return,
            "average_benchmark_sharpe": self.average_benchmark_sharpe,
            "average_benchmark_max_drawdown": (
                self.average_benchmark_max_drawdown
            ),
            "average_excess_return": self.average_excess_return,
        }

    def save_json(
        self,
        report_dir: str = "reports/walk_forward",
        run_date: date | None = None,
    ) -> Path:
        report_date = run_date or date.today()
        directory = Path(report_dir)
        directory.mkdir(parents=True, exist_ok=True)

        path = (
            directory
            / f"{report_date.isoformat()}_{self.symbol}_{self.timeframe}.json"
        )
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

        return path
