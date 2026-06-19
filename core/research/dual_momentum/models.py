from dataclasses import dataclass, field
from pathlib import Path
import json

from core.entities.backtest_result import BacktestResult


@dataclass(frozen=True)
class DualMomentumSelection:
    timestamp: object
    symbols: list[str]
    scores: dict[str, float]
    risk_on: bool
    regime_label: str = "risk-off"
    regime_exposure: float = 0
    exposure_target: float = 0
    fallback_symbols: list[str] | None = None
    breadth_passes: bool = False
    fast_reentry: bool = False
    drawdown_guard_active: bool = False
    target_weights: dict[str, float] | None = None
    chop_filter_active: bool = False
    cooldown_symbols: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbols": self.symbols,
            "scores": self.scores,
            "risk_on": self.risk_on,
            "regime_label": self.regime_label,
            "regime_exposure": self.regime_exposure,
            "exposure_target": self.exposure_target,
            "fallback_symbols": self.fallback_symbols or [],
            "breadth_passes": self.breadth_passes,
            "fast_reentry": self.fast_reentry,
            "drawdown_guard_active": self.drawdown_guard_active,
            "target_weights": self.target_weights or {},
            "chop_filter_active": self.chop_filter_active,
            "cooldown_symbols": self.cooldown_symbols or [],
        }


@dataclass(frozen=True)
class DualMomentumResult:
    result: BacktestResult
    selections: list[DualMomentumSelection]
    benchmark_return: float
    excess_return: float
    equal_weight_return: float
    excess_vs_equal_weight: float
    turnover_percent: float
    annualized_turnover_percent: float
    turnover_per_rebalance_percent: float
    rebalance_count: int
    estimated_cost: float
    cost_drag_percent: float
    cagr: float
    calmar: float
    annual_returns: dict[int, float]
    monthly_returns: dict[str, float]
    rolling_12_month_returns: dict[str, float]
    drawdown_statistics: dict
    config: dict
    walk_forward_filter_reasons: list[str] = field(default_factory=list)
    walk_forward_filter_passed: bool = True
    walk_forward_selector_mode: str = ""
    walk_forward_filter_fallback: bool = False

    def save_json(
        self,
        report_dir: str = "reports/summary",
        filename: str = "dual_momentum_portfolio.json",
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
            "annualized_turnover_percent": (
                self.annualized_turnover_percent
            ),
            "turnover_per_rebalance_percent": (
                self.turnover_per_rebalance_percent
            ),
            "rebalance_count": self.rebalance_count,
            "estimated_cost": self.estimated_cost,
            "cost_drag_percent": self.cost_drag_percent,
            "cagr": self.cagr,
            "calmar": self.calmar,
            "annual_returns": self.annual_returns,
            "monthly_returns": self.monthly_returns,
            "rolling_12_month_returns": self.rolling_12_month_returns,
            "drawdown_statistics": self.drawdown_statistics,
            "config": self.config,
            "selections": [
                selection.to_dict()
                for selection in self.selections
            ],
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return path