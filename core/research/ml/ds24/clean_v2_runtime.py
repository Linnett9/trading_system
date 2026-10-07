from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from core.research.ml.ds24.clean_v2_contracts import stable_hash
from core.research.ml.ds24.clean_v2_resources import RESOURCE_POLICY
from core.research.ml.ds24.comparable_policy import (
    REFIT_SCORE_SESSION_CADENCE,
    build_refit_schedule,
)


RUN_ID = "DS24_CLEAN_V2_TOURNAMENT_R1_20260926"
REFIT_POLICY_ID = "REFIT_EVERY_5_TRADING_SESSIONS_V1"
QUALIFIER_YEARS = (2017, 2019, 2020, 2022, 2024)
# Post-crash Dell recovery is deliberately serialized.  This is an execution
# control only; it does not alter any model, sample, refit, feature, or target
# authority.
MAX_DELL_MODEL_WORKERS = RESOURCE_POLICY.maximum_dell_model_workers
PAPER_ORDERS_ALLOWED = False
LIVE_ORDERS_ALLOWED = False


class CleanRuntimeError(ValueError):
    """Raised when a clean tournament runtime invariant is violated."""


def clean_refit_schedule(
    spine: Iterable[pd.Timestamp], *, max_refits: int | None = None
) -> list[Any]:
    if REFIT_SCORE_SESSION_CADENCE != 5:
        raise CleanRuntimeError("Five-session canonical policy was changed")
    schedule = build_refit_schedule(spine, max_refits=max_refits)
    if any(len(spec.score_session_dates) > 5 for spec in schedule):
        raise CleanRuntimeError("Refit schedule contains more than five score sessions")
    return schedule


def validate_ownership(ownership: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    for host, families in ownership.items():
        for family in families:
            previous = seen.setdefault(str(family), str(host))
            if previous != host:
                raise CleanRuntimeError(
                    f"Family {family} is assigned to both {previous} and {host}"
                )


def metric_key(
    *, family: str, decision_timestamp: str | pd.Timestamp, metric: str, cost_bps: int
) -> str:
    timestamp = pd.Timestamp(decision_timestamp)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return stable_hash(
        {
            "run_id": RUN_ID,
            "family": family,
            "decision_timestamp": timestamp.isoformat(),
            "metric": metric,
            "cost_bps": int(cost_bps),
        }
    )


@dataclass
class FamilyResumeState:
    family: str
    owner_host: str
    feature_authority_hash: str
    target_contract_hash: str
    model_config_hash: str
    refit_policy_hash: str
    latest_completed_refit: str | None = None
    latest_scored_decision: str | None = None
    metrics_cursor: int = 0
    terminal_state: str = "PENDING"
    completed_refits: list[str] = field(default_factory=list)
    committed_metric_keys: list[str] = field(default_factory=list)

    def record_refit(self, refit_timestamp: str) -> bool:
        value = pd.Timestamp(refit_timestamp).isoformat()
        if value in self.completed_refits:
            return False
        self.completed_refits.append(value)
        self.completed_refits.sort()
        self.latest_completed_refit = value
        return True

    def record_metric(self, key: str, decision_timestamp: str) -> bool:
        if key in self.committed_metric_keys:
            return False
        self.committed_metric_keys.append(key)
        self.metrics_cursor += 1
        self.latest_scored_decision = pd.Timestamp(decision_timestamp).isoformat()
        return True

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "run_id": RUN_ID,
                "refit_policy_id": REFIT_POLICY_ID,
                "paper_orders": 0,
                "live_orders": 0,
            }
        )
        return result


def assert_compact_output_path(path: Path) -> None:
    forbidden = {"predictions", "full_predictions", "prediction_matrix", "dense_scores"}
    lowered = {part.lower() for part in path.parts}
    conflict = sorted(forbidden & lowered)
    if conflict:
        raise CleanRuntimeError(f"Permanent full-prediction output is forbidden: {conflict}")


def validate_target_use(
    *,
    predictor_names: Iterable[str],
    training_rows: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    fit_timestamp: str | pd.Timestamp,
) -> None:
    predictors = {str(name) for name in predictor_names}
    target_like = {name for name in predictors if "target" in name.lower() or "forward_return" in name.lower()}
    if target_like:
        raise CleanRuntimeError(f"Target-like inference predictors are forbidden: {sorted(target_like)}")
    required = {"decision_timestamp", "target_available_timestamp", "target_is_trainable"}
    missing = sorted(required - set(training_rows.columns))
    if missing:
        raise CleanRuntimeError(f"Training rows are missing target maturity fields: {missing}")
    fit_t = pd.Timestamp(fit_timestamp)
    fit_t = fit_t.tz_localize("UTC") if fit_t.tzinfo is None else fit_t.tz_convert("UTC")
    decisions = pd.to_datetime(training_rows["decision_timestamp"], utc=True)
    maturities = pd.to_datetime(training_rows["target_available_timestamp"], utc=True)
    admitted = training_rows["target_is_trainable"].fillna(False).astype(bool)
    if not admitted.all() or (decisions >= fit_t).any() or (maturities > fit_t).any():
        raise CleanRuntimeError("Training population violates chronology or maturity")
    # Score rows intentionally need no target columns or maturity at score time.
    if "decision_timestamp" not in scoring_rows:
        raise CleanRuntimeError("Scoring rows are missing decision_timestamp")
