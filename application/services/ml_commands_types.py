from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLResearchBatchItem:
    config_path: Path
    output_dir: Path


@dataclass(frozen=True)
class MLResearchBatchResult:
    config_path: str
    output_dir: str
    success: bool
    metrics_path: str | None = None
    prediction_artifacts_path: str | None = None
    error: str | None = None
