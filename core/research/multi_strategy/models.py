from dataclasses import dataclass
from pathlib import Path
import json

from core.entities.backtest_result import BacktestResult


@dataclass(frozen=True)
class MultiStrategySleeveResult:
    name: str
    weight: float
    result: BacktestResult

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class MultiStrategyPortfolioResult:
    result: BacktestResult
    sleeves: list[MultiStrategySleeveResult]
    benchmark_return: float
    excess_return: float
    equal_weight_return: float
    excess_vs_equal_weight: float
    diagnostics: dict
    config: dict

    def save_json(
        self,
        report_dir: str = "reports/summary",
        filename: str = "multi_strategy_portfolio.json",
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
            "diagnostics": self.diagnostics,
            "config": self.config,
            "sleeves": [sleeve.to_dict() for sleeve in self.sleeves],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
