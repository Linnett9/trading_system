from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}
NOTICE = "Research only. Trading impact: none. Production validated: false."
DEFAULT_INSPECT_SYMBOLS = ("AMSC", "AXTI", "LEU", "LUMN", "MRVL", "MU")
COMMON_SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
REPORT_CANDIDATES = (
    "exact_champion_replay",
    "selected_bayesian_optimizer_diagnostic_policy",
    "spy_buy_and_hold",
    "qqq_buy_and_hold",
    "equal_weight_selected_universe",
)


@dataclass(frozen=True)
class DataAdjustmentAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class CleanDataReplayPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class IndependentPeriodValidationPaths:
    json_path: Path
    markdown_path: Path
