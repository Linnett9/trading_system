from __future__ import annotations

from dataclasses import dataclass


DEFAULT_REGRESSION_TARGETS = [
    "forward_return_5d",
    "forward_return_10d",
    "future_volatility",
    "future_drawdown",
    "max_adverse_excursion",
    "max_favourable_excursion",
]

LEAKAGE_FEATURE_PREFIXES = (
    "actual_",
    "forward_",
    "future_",
    "max_adverse_",
    "max_favourable_",
)
LEAKAGE_FEATURE_NAMES = {
    "research_label",
    "should_reduce_exposure",
    *DEFAULT_REGRESSION_TARGETS,
}


@dataclass(frozen=True)
class MultiTaskTransformerTrainingSummary:
    trained: bool
    sequence_count: int
    feature_count: int
    positive_rate: float
    regression_targets: list[str]
    missing_target_counts: dict[str, int]
