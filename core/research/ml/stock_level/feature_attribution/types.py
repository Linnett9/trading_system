from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NOTICE = "Research only. Trading impact: none. Production validated: false."
METRIC_NAMES = (
    "mean_spearman_ic",
    "top_minus_bottom_spread",
    "top_decile_return",
    "bottom_decile_return",
    "top_decile_hit_rate",
    "risk_adjusted_spread",
    "spread_sharpe",
)
@dataclass(frozen=True)
class StockLevelFeatureAttributionPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
