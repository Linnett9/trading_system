from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
TARGET_PERIODS = (36, 60)
@dataclass(frozen=True)
class HistoricalCoverageAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
