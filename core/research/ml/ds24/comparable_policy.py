from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.research.ml.ds24.canonical_prequential_engine import stable_hash


POLICY_ID = "DS24_CANONICAL_5M_COMPARABLE_NONCORE_POLICY_V1"
PARENT_POLICY_ID = "P0_C0_INCUMBENT"
PARENT_POLICY_HASH = "46426e0b4ff9e8948b089feb2d5192be0002a7e4f9a09479fa0b262d56344d21"
TRAINING_SESSIONS = 20
REFIT_SCORE_SESSION_CADENCE = 5


@dataclass(frozen=True)
class RefitPackageSpec:
    ordinal: int
    refit_T: pd.Timestamp
    training_session_dates: list[str]
    score_session_dates: list[str]
    policy_hash: str


def policy_payload() -> dict:
    return {
        "policy_id": POLICY_ID,
        "parent_policy_id": PARENT_POLICY_ID,
        "parent_policy_hash": PARENT_POLICY_HASH,
        "training_sessions": TRAINING_SESSIONS,
        "refit_score_session_cadence": REFIT_SCORE_SESSION_CADENCE,
        "training_rule": "decision_timestamp < refit_T and target_available_timestamp <= refit_T",
        "score_rule": "score every eligible five-minute T in the next five score sessions using the frozen fitted model",
        "shards_are_loading_units_only": True,
        "no_midcycle_outcome_performance_volatility_or_turnover_refit": True,
        "no_locked_holdout_metric_consumption": True,
        "no_paper_or_live_orders": True,
    }


def policy_hash() -> str:
    return stable_hash(policy_payload())


def session_dates_from_spine(spine: Iterable[pd.Timestamp]) -> list[str]:
    dates = sorted({pd.Timestamp(ts).date().isoformat() for ts in spine})
    return dates


def build_refit_schedule(spine: Iterable[pd.Timestamp], *, max_refits: int | None = None) -> list[RefitPackageSpec]:
    sessions = session_dates_from_spine(spine)
    out: list[RefitPackageSpec] = []
    digest = policy_hash()
    ordinal = 0
    for start in range(TRAINING_SESSIONS, len(sessions), REFIT_SCORE_SESSION_CADENCE):
        score_sessions = sessions[start : start + REFIT_SCORE_SESSION_CADENCE]
        if not score_sessions:
            break
        training_sessions = sessions[start - TRAINING_SESSIONS : start]
        refit_t = min(ts for ts in spine if pd.Timestamp(ts).date().isoformat() == score_sessions[0])
        out.append(
            RefitPackageSpec(
                ordinal=ordinal,
                refit_T=pd.Timestamp(refit_t).tz_convert("UTC"),
                training_session_dates=training_sessions,
                score_session_dates=score_sessions,
                policy_hash=digest,
            )
        )
        ordinal += 1
        if max_refits is not None and len(out) >= max_refits:
            break
    return out
