from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NewsTransformerConfig:
    lookback_window_days: int = 63
    label_horizon_days: int = 10
    enabled: bool = False
    training_enabled: bool = False
    inference_enabled: bool = False
    used_in_strategy: bool = False
    used_in_replay: bool = False


@dataclass(frozen=True)
class NewsSequenceExample:
    symbol: str
    decision_timestamp: str
    availability_timestamp: str | None = None
    publication_timestamp: str | None = None
    lookback_window_days: int = 63
    candidate_id: str | None = None
    strategy_variant: str | None = None
    price_model_score: float | None = None
    news_numeric_features: Mapping[str, float] = field(default_factory=dict)
    event_category: str | None = None
    headline_text: str | None = None
    article_text: str | None = None
    source: str | None = None
    provider: str | None = None
    duplicate_group_id: str | None = None
    label_name: str | None = None
    label_horizon_days: int | None = None
    label_value: float | int | str | None = None
    split_name: str | None = None


@dataclass(frozen=True)
class NewsSequenceWindow:
    symbol: str
    decision_timestamp: str
    lookback_window_days: int
    examples: tuple[NewsSequenceExample, ...] = ()


@dataclass(frozen=True)
class NewsTransformerFeatureSchema:
    required_fields: tuple[str, ...] = (
        "symbol",
        "decision_timestamp",
        "availability_timestamp",
        "publication_timestamp",
        "headline_text",
        "article_text",
        "duplicate_group_id",
        "split_name",
    )
    numeric_feature_field: str = "news_numeric_features"
    point_in_time_timestamp_field: str = "availability_timestamp"


@dataclass(frozen=True)
class NewsTransformerLabelSchema:
    label_name: str = "future_return_or_risk_label"
    label_horizon_days: int = 10
    required_fields: tuple[str, ...] = ("label_name", "label_horizon_days", "label_value")
    labels_generated_after_decision_timestamp: bool = True


@dataclass(frozen=True)
class NewsTransformerReadinessReport:
    status: str
    transformer_readiness: str
    bert_readiness: str
    finbert_readiness: str
    enabled: bool
    training_enabled: bool
    inference_enabled: bool
    used_in_strategy: bool
    used_in_replay: bool
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewsTransformerTrainingPlan:
    status: str = "NOT_READY"
    plan_only: bool = True
    training_enabled: bool = False
    inference_enabled: bool = False
    model_family: str = "disabled_news_transformer_scaffold"
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewsTransformerModelSpec:
    status: str = "NOT_READY"
    model_family: str = "news_transformer"
    requires_optional_torch: bool = True
    requires_external_weights: bool = False
    enabled: bool = False
    training_enabled: bool = False
    inference_enabled: bool = False
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
