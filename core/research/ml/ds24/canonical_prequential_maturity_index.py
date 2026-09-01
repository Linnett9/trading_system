from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.research.ml.ds24.canonical_prequential_engine import PartitionRow


@dataclass(frozen=True)
class MaturityEvent:
    asset_id: str
    decision_timestamp: pd.Timestamp
    target_available_timestamp: pd.Timestamp
    partition_identity: str
    row_key: str


def target_events_for_partition(root: Path, row: PartitionRow) -> pd.DataFrame:
    target = pd.read_parquet(
        root / row.target_partition,
        columns=["asset_id", "decision_timestamp", "target_available_timestamp", "target_is_trainable"],
    )
    target["decision_timestamp"] = pd.to_datetime(target["decision_timestamp"], utc=True)
    target["target_available_timestamp"] = pd.to_datetime(target["target_available_timestamp"], utc=True)
    target = target[target["target_is_trainable"].astype(bool)].copy()
    target["partition_identity"] = f"{row.asset_id}|{row.year}"
    target["row_key"] = (
        target["asset_id"].astype(str)
        + "|"
        + target["decision_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return target[
        ["asset_id", "decision_timestamp", "target_available_timestamp", "partition_identity", "row_key"]
    ].sort_values(["target_available_timestamp", "decision_timestamp", "asset_id", "row_key"]).reset_index(drop=True)


def admission_batch(
    events: pd.DataFrame,
    *,
    previous_cursor: pd.Timestamp | None,
    current_t: pd.Timestamp,
) -> pd.DataFrame:
    current_t = pd.Timestamp(current_t).tz_convert("UTC")
    mask = events["target_available_timestamp"] <= current_t
    if previous_cursor is not None:
        previous_cursor = pd.Timestamp(previous_cursor).tz_convert("UTC")
        mask &= events["target_available_timestamp"] > previous_cursor
    mask &= events["decision_timestamp"] < current_t
    return events[mask].copy().sort_values(["target_available_timestamp", "decision_timestamp", "asset_id", "row_key"]).reset_index(drop=True)


def full_eligible_keys(events: pd.DataFrame, current_t: pd.Timestamp) -> set[str]:
    current_t = pd.Timestamp(current_t).tz_convert("UTC")
    eligible = events[
        (events["target_available_timestamp"] <= current_t)
        & (events["decision_timestamp"] < current_t)
    ]
    return set(eligible["row_key"].astype(str))


def incremental_eligible_keys(events: pd.DataFrame, timeline: Iterable[pd.Timestamp]) -> set[str]:
    admitted: set[str] = set()
    cursor: pd.Timestamp | None = None
    for timestamp in timeline:
        batch = admission_batch(events, previous_cursor=cursor, current_t=timestamp)
        admitted.update(batch["row_key"].astype(str))
        cursor = pd.Timestamp(timestamp).tz_convert("UTC")
    return admitted


def equivalence_check(events: pd.DataFrame, timeline: Iterable[pd.Timestamp]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    admitted: set[str] = set()
    cursor: pd.Timestamp | None = None
    for timestamp in timeline:
        timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
        batch = admission_batch(events, previous_cursor=cursor, current_t=timestamp)
        duplicate_keys = sorted(set(batch["row_key"].astype(str)) & admitted)
        admitted.update(batch["row_key"].astype(str))
        full = full_eligible_keys(events, timestamp)
        rows.append(
            {
                "decision_timestamp": timestamp.isoformat(),
                "incremental_count": len(admitted),
                "full_count": len(full),
                "key_sets_equal": admitted == full,
                "duplicate_keys": len(duplicate_keys),
                "latest_maturity": batch["target_available_timestamp"].max().isoformat() if not batch.empty else None,
            }
        )
        cursor = timestamp
    return rows
