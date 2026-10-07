"""Exact rolling-window panel reuse for DS24 package candidates.

This module owns only deterministic in-memory composition.  Authority-keyed
storage, data reads, admission, and publication remain outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class RollingPanelReuseError(ValueError):
    """Raised when incremental panel inputs cannot prove exact compatibility."""


@dataclass(frozen=True)
class RollingPanelPlan:
    previous_session_dates: tuple[str, ...]
    next_session_dates: tuple[str, ...]
    retained_session_dates: tuple[str, ...]
    expired_session_dates: tuple[str, ...]
    appended_session_dates: tuple[str, ...]

    @classmethod
    def derive(
        cls,
        previous_session_dates: Iterable[str],
        next_session_dates: Iterable[str],
    ) -> "RollingPanelPlan":
        previous = tuple(sorted({str(value) for value in previous_session_dates}))
        following = tuple(sorted({str(value) for value in next_session_dates}))
        if not previous or not following:
            raise RollingPanelReuseError("Both rolling windows must be non-empty")
        previous_set = set(previous)
        following_set = set(following)
        retained = tuple(value for value in following if value in previous_set)
        expired = tuple(value for value in previous if value not in following_set)
        appended = tuple(value for value in following if value not in previous_set)
        if not retained:
            raise RollingPanelReuseError("Rolling windows have no reusable overlap")
        if not appended:
            raise RollingPanelReuseError("Next rolling window has no appended sessions")
        return cls(
            previous_session_dates=previous,
            next_session_dates=following,
            retained_session_dates=retained,
            expired_session_dates=expired,
            appended_session_dates=appended,
        )


def compose_incremental_panel(
    previous_panel: pd.DataFrame,
    appended_panel: pd.DataFrame,
    *,
    plan: RollingPanelPlan,
) -> pd.DataFrame:
    """Compose the next panel while preserving canonical full-build ordering."""

    required = {"asset_id", "decision_timestamp", "session_date"}
    for label, frame in (
        ("previous_panel", previous_panel),
        ("appended_panel", appended_panel),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RollingPanelReuseError(f"{label} missing columns: {missing}")
    if list(previous_panel.columns) != list(appended_panel.columns):
        raise RollingPanelReuseError("Incremental panel column order mismatch")
    previous_dtypes = previous_panel.dtypes.astype(str).to_dict()
    appended_dtypes = appended_panel.dtypes.astype(str).to_dict()
    if previous_dtypes != appended_dtypes:
        raise RollingPanelReuseError("Incremental panel dtype mismatch")

    previous_dates = set(previous_panel["session_date"].astype(str).unique())
    unexpected_previous = previous_dates - set(plan.previous_session_dates)
    if unexpected_previous:
        raise RollingPanelReuseError(
            f"Previous panel contains unexpected sessions: {sorted(unexpected_previous)}"
        )
    appended_dates = set(appended_panel["session_date"].astype(str).unique())
    expected_appended = set(plan.appended_session_dates)
    if appended_dates != expected_appended:
        raise RollingPanelReuseError(
            "Appended panel sessions mismatch: "
            f"expected={sorted(expected_appended)} actual={sorted(appended_dates)}"
        )

    retained = previous_panel[
        previous_panel["session_date"].astype(str).isin(plan.retained_session_dates)
    ]
    candidate = pd.concat([retained, appended_panel], ignore_index=True)
    candidate["decision_timestamp"] = pd.to_datetime(
        candidate["decision_timestamp"], utc=True, errors="raise"
    )
    candidate = candidate.sort_values(
        ["decision_timestamp", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)
    if candidate.duplicated(["asset_id", "decision_timestamp"]).any():
        raise RollingPanelReuseError("Incremental panel contains duplicate row keys")
    actual_dates = set(candidate["session_date"].astype(str).unique())
    expected_dates = set(plan.next_session_dates)
    if actual_dates != expected_dates:
        raise RollingPanelReuseError(
            "Incremental panel final sessions mismatch: "
            f"expected={sorted(expected_dates)} actual={sorted(actual_dates)}"
        )
    if candidate.dtypes.astype(str).to_dict() != previous_dtypes:
        raise RollingPanelReuseError("Incremental composition changed canonical dtypes")
    return candidate
