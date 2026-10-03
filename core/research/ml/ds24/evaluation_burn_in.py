from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd


EVALUATION_BURN_IN_ID: Final = "DS24_EVALUATION_BURN_IN_2016_2017_V1"
DELL_EVALUATION_START: Final = pd.Timestamp("2018-01-02T14:35:00Z")


@dataclass(frozen=True)
class EvaluationBurnIn:
    """Explicit form of the Dell tournament's historical 2016-2017 burn-in."""

    contract_id: str = EVALUATION_BURN_IN_ID
    evaluation_start: pd.Timestamp = DELL_EVALUATION_START
    boundary_kind: Literal["decision_timestamp"] = "decision_timestamp"
    returns_before_boundary: Literal["excluded"] = "excluded"
    rank_ic_before_boundary: Literal["excluded"] = "excluded"
    predictions_before_boundary: Literal["persisted"] = "persisted"

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.evaluation_start)
        if start.tzinfo is None:
            raise ValueError("evaluation_start must be timezone-aware")
        object.__setattr__(self, "evaluation_start", start.tz_convert("UTC"))

    def is_evaluation_eligible(self, decision_timestamp: str | pd.Timestamp) -> bool:
        decision = pd.Timestamp(decision_timestamp)
        if decision.tzinfo is None:
            raise ValueError("decision_timestamp must be timezone-aware")
        return bool(decision.tz_convert("UTC") >= self.evaluation_start)

    def mark_predictions(self, rows: pd.DataFrame) -> pd.DataFrame:
        if "decision_timestamp" not in rows:
            raise ValueError("decision_timestamp is required")
        marked = rows.copy()
        timestamps = pd.to_datetime(marked["decision_timestamp"], utc=True, errors="raise")
        marked["evaluation_eligible"] = timestamps >= self.evaluation_start
        return marked

    def metric_population(
        self, rows: pd.DataFrame, *, metric: Literal["returns", "rank_ic"]
    ) -> pd.DataFrame:
        if metric not in {"returns", "rank_ic"}:
            raise ValueError(f"Unsupported metric population: {metric}")
        marked = self.mark_predictions(rows)
        return marked.loc[marked["evaluation_eligible"]].copy()


EVALUATION_BURN_IN: Final = EvaluationBurnIn()
