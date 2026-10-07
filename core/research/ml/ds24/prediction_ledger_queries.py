from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from core.research.ml.ds24.prediction_ledger_contract import PREDICTION_KEY


@dataclass(frozen=True)
class LedgerQuery:
    """Pure selection contract for read-only prediction-ledger research."""

    families: Sequence[str] | None = None
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    top_n: int | None = None
    bottom_n: int | None = None
    minimum_rank_percentile: float | None = None
    minimum_score: float | None = None

    def __post_init__(self) -> None:
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.bottom_n is not None and self.bottom_n <= 0:
            raise ValueError("bottom_n must be positive")
        if (
            self.minimum_rank_percentile is not None
            and not 0.0 < self.minimum_rank_percentile <= 1.0
        ):
            raise ValueError("minimum_rank_percentile must be in (0, 1]")


def apply_ledger_query(rows: pd.DataFrame, query: LedgerQuery) -> pd.DataFrame:
    """Apply a query without filesystem or outcome-authority dependencies."""

    selected = rows.copy()
    selected["decision_timestamp"] = pd.to_datetime(
        selected["decision_timestamp"], utc=True, errors="raise"
    )
    if query.families:
        selected = selected[selected["family"].isin(query.families)]
    if query.start is not None:
        selected = selected[
            selected["decision_timestamp"] >= _utc_timestamp(query.start, label="start")
        ]
    if query.end is not None:
        selected = selected[
            selected["decision_timestamp"] <= _utc_timestamp(query.end, label="end")
        ]
    if query.top_n is not None:
        selected = selected[selected["cross_section_rank"] <= query.top_n]
    if query.bottom_n is not None:
        selected = selected[
            selected["cross_section_rank"]
            > selected["eligible_universe_size"] - query.bottom_n
        ]
    if query.minimum_rank_percentile is not None:
        selected = selected[
            selected["rank_percentile"] >= query.minimum_rank_percentile
        ]
    if query.minimum_score is not None:
        selected = selected[selected["raw_prediction"] >= query.minimum_score]
    return selected.sort_values(list(PREDICTION_KEY), kind="mergesort").reset_index(drop=True)


def rank_changes(rows: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    if periods <= 0:
        raise ValueError("periods must be positive")
    work = rows.sort_values(
        ["family", "asset_id", "decision_timestamp"], kind="mergesort"
    ).copy()
    work["rank_change"] = work.groupby(["family", "asset_id"])[
        "cross_section_rank"
    ].diff(periods)
    return work


def score_spreads(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["family", "decision_timestamp"], sort=True)[
        "raw_prediction"
    ]
    return (grouped.max() - grouped.min()).rename("score_spread").reset_index()


def rebalance_candidates(
    rows: pd.DataFrame, *, minimum_rank_change: float
) -> pd.DataFrame:
    if minimum_rank_change < 0:
        raise ValueError("minimum_rank_change cannot be negative")
    changed = rank_changes(rows)
    return changed[changed["rank_change"].abs() >= minimum_rank_change].copy()


def join_certified_outcomes(
    rows: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    authority_id: str,
    holding_period: str,
    outcome_column: str,
) -> pd.DataFrame:
    """Join outcomes explicitly after ledger load; outcomes never enter storage."""

    if not authority_id.strip() or not holding_period.strip():
        raise ValueError("Certified authority identity and holding period are required")
    required = {"decision_timestamp", "asset_id", outcome_column}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"Certified outcome authority is missing columns: {missing}")
    return rows.merge(
        outcomes.loc[:, list(required)],
        on=["decision_timestamp", "asset_id"],
        how="left",
        validate="one_to_one",
    ).assign(outcome_authority_id=authority_id, holding_period=holding_period)


def _utc_timestamp(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")
