from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_METADATA = {
    "research_only": True,
    "trading_impact": "none",
    "production_validated": False,
}

AUDITED_CANDIDATES = (
    "champion_baseline",
    "always_full_exposure",
    "binary_exposure_overlay",
    "return_only_allocation",
    "risk_adjusted_allocation",
    "meta_ensemble_allocation",
    "best_grid_search_diagnostic_policy",
    "selected_bayesian_optimizer_diagnostic_policy",
)

CAP_SCENARIOS = {
    "cap_-50pct_+50pct": (-0.50, 0.50),
    "cap_-25pct_+25pct": (-0.25, 0.25),
    "cap_-10pct_+10pct": (-0.10, 0.10),
}

COST_SENSITIVITY_BPS = (5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class ReturnMechanicsAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
