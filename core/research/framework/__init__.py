"""Reusable, broker-agnostic building blocks for research pipelines."""

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.parallel import ParallelTaskExecutor
from core.research.framework.ranking import CrossSectionalRankingEvaluator
from core.research.framework.registry import (
    BenchmarkRegistry,
    FeatureRegistry,
    ModelRegistry,
    ReportRegistry,
    ValidationRegistry,
)
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.framework.walk_forward import (
    ExpandingWindowSpec,
    ExpandingWindowSplitter,
    WalkForwardFold,
)

__all__ = [
    "BenchmarkRegistry",
    "CrossSectionalRankingEvaluator",
    "ExpandingWindowSpec",
    "ExpandingWindowSplitter",
    "FeatureRegistry",
    "ModelRegistry",
    "ParallelTaskExecutor",
    "ReportRegistry",
    "ResearchArtifactWriter",
    "StockLevelResearchConfig",
    "ValidationRegistry",
    "WalkForwardFold",
]
