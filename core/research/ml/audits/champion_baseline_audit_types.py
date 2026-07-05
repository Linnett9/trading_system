from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}


@dataclass(frozen=True)
class ChampionBaselineAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
