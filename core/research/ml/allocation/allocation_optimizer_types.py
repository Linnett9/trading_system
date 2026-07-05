from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


OPTIMIZER_NOTICE = (
    "Research-only optimizer. Parameters selected on out-of-fold data and evaluated "
    "once on frozen holdout; not production-valid."
)
OBJECTIVE_MODES = {
    "diagnostic_period_grid_return",
    "canonical_non_overlap_return",
    "anomaly_adjusted_canonical_return",
    "robustness_adjusted_canonical_score",
}


class CandidateSampler(Protocol):
    method: str
    sampler_requested: str
    sampler_used: str
    optuna_available: bool
    fallback_reason: str | None

    def sample(self, count: int) -> list[dict[str, float | str]]:
        ...

    def suggest(self, trial_number: int) -> dict[str, float | str]:
        ...

    def observe(
        self,
        candidate: dict[str, float | str],
        objective_value: float | None,
    ) -> None:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OptimizerPaths:
    candidates_csv: Path
    results_json: Path
    report_markdown: Path
    selected_exposure_path_csv: Path
    selected_exposure_path_json: Path
