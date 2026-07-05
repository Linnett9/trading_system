from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPORT_CANDIDATES = (
    "exact_champion_replay",
    "selected_bayesian_optimizer_diagnostic_policy",
)

@dataclass(frozen=True)
class AdjustedReplayAlignmentAuditPaths:
    csv_path: Path
    json_path: Path
    markdown_path: Path
