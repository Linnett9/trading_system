from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.research.ml.ds24.canonical_prequential_engine import CanonicalPrequentialEngine, PartitionRow, stable_hash


@dataclass(frozen=True)
class PreparedPanel:
    decision_date: str
    rows: int
    assets: int
    timestamps: int
    content_hash: str
    frame: pd.DataFrame


class CanonicalPreparedPanelStream:
    """Read-only bounded cache over canonical DS24 session panels."""

    def __init__(self, *, engine: CanonicalPrequentialEngine, partitions: Iterable[PartitionRow], cache_budget_bytes: int = 0) -> None:
        self.engine = engine
        self.partitions = list(partitions)
        self.cache_budget_bytes = max(0, int(cache_budget_bytes))
        self._cache: OrderedDict[str, PreparedPanel] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    @property
    def cache_bytes(self) -> int:
        return sum(int(item.frame.memory_usage(deep=True).sum()) for item in self._cache.values())

    def session_panel(self, decision_date: str) -> PreparedPanel:
        if decision_date in self._cache:
            self.cache_hits += 1
            panel = self._cache.pop(decision_date)
            self._cache[decision_date] = panel
            return panel
        self.cache_misses += 1
        rows = [row for row in self.partitions if row.year == int(decision_date[:4])]
        frame = self.engine.assemble_partitions(rows=rows, decision_dates={decision_date})
        predictors = self.engine.predictor_manifest.predictors
        digest = hashlib.sha256()
        if not frame.empty:
            digest.update(pd.util.hash_pandas_object(frame[["asset_id", "decision_timestamp", *predictors]], index=False).values.tobytes())
        content_hash = stable_hash({"decision_date": decision_date, "rows": len(frame), "digest": digest.hexdigest()})
        panel = PreparedPanel(
            decision_date=decision_date,
            rows=int(len(frame)),
            assets=int(frame["asset_id"].nunique()) if not frame.empty else 0,
            timestamps=int(frame["decision_timestamp"].nunique()) if not frame.empty else 0,
            content_hash=content_hash,
            frame=frame,
        )
        self._cache[decision_date] = panel
        self._evict()
        return panel

    def _evict(self) -> None:
        if self.cache_budget_bytes <= 0:
            self._cache.clear()
            return
        while self._cache and self.cache_bytes > self.cache_budget_bytes:
            self._cache.popitem(last=False)
            self.evictions += 1
