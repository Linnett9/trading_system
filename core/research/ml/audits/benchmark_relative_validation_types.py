from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COST_STRESS_BPS = (5, 10, 25, 50, 100)
RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}


@dataclass(frozen=True)
class BenchmarkRelativeValidationPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
    promotion_readiness_path: Path
