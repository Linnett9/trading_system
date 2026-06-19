from dataclasses import dataclass
from pathlib import Path
import json

from core.entities.backtest_result import BacktestResult


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
