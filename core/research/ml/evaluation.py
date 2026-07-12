from __future__ import annotations

# Compatibility re-export for older internal imports. The canonical
# implementation lives in core.research.ml.metrics.evaluation.
from core.research.ml.metrics.evaluation import classification_metrics

__all__ = ["classification_metrics"]
