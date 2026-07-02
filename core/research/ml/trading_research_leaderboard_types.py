from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}

RANKING_BASIS = [
    "canonical_rows_before_period_grid_only_rows",
    "canonical_continuous_return_desc_when_available",
    "diagnostic_period_grid_return_desc",
    "max_drawdown_asc",
    "sharpe_desc",
    "sortino_desc",
    "calmar_desc",
    "turnover_asc",
    "estimated_transaction_costs_asc",
]

NOTICE = "Research only. Trading impact: none. Production validated: false."

@dataclass(frozen=True)
class TradingResearchLeaderboardPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
