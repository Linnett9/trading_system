from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    total_return: float = 0
    max_drawdown: float = 0
    sharpe: float = 0

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
        }
