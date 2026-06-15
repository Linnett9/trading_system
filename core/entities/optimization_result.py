from dataclasses import dataclass

from core.entities.backtest_result import BacktestResult


@dataclass(frozen=True)
class OptimizationResult:
    parameters: dict
    metric_name: str
    metric_value: float
    result: BacktestResult
