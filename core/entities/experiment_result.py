from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentResult:
    symbol: str
    source: str
    return_value: float
    walk_forward_sharpe: float
    max_drawdown: float
    closed_trades: int
    profit_factor: float
    time_in_market_percent: float = 0
    average_trade_duration_days: float = 0
    average_exposure_percent: float = 0
    average_cash_percent: float = 1
    average_position_value: float = 0
    benchmark_sharpe: float = 0
    benchmark_max_drawdown: float = 0
    excess_return_per_unit_risk: float = 0
    passed_folds: int = 0
    total_folds: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "return": self.return_value,
            "walk_forward_sharpe": self.walk_forward_sharpe,
            "max_drawdown": self.max_drawdown,
            "closed_trades": self.closed_trades,
            "profit_factor": self.profit_factor,
            "time_in_market_percent": self.time_in_market_percent,
            "average_trade_duration_days": self.average_trade_duration_days,
            "average_exposure_percent": self.average_exposure_percent,
            "average_cash_percent": self.average_cash_percent,
            "average_position_value": self.average_position_value,
            "benchmark_sharpe": self.benchmark_sharpe,
            "benchmark_max_drawdown": self.benchmark_max_drawdown,
            "excess_return_per_unit_risk": (
                self.excess_return_per_unit_risk
            ),
            "passed_folds": self.passed_folds,
            "total_folds": self.total_folds,
        }
