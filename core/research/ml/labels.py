from __future__ import annotations

# Compatibility re-export for older internal imports. The canonical
# implementation lives in core.research.ml.features.labels.
from core.research.ml.features.labels import (
    ChampionSuccessLabelBuilder,
    DrawdownRiskLabelBuilder,
    MLLabelBuildResult,
    RiskRegimeLabelBuilder,
    ShouldReduceExposureLabelBuilder,
    write_label_rows,
)

__all__ = [
    "ChampionSuccessLabelBuilder",
    "DrawdownRiskLabelBuilder",
    "MLLabelBuildResult",
    "RiskRegimeLabelBuilder",
    "ShouldReduceExposureLabelBuilder",
    "write_label_rows",
]
