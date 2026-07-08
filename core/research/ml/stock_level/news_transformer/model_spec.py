from __future__ import annotations

from .contracts import NewsTransformerModelSpec


def disabled_news_transformer_model_spec() -> NewsTransformerModelSpec:
    return NewsTransformerModelSpec(
        status="NOT_READY",
        model_family="news_transformer",
        requires_optional_torch=True,
        requires_external_weights=False,
        enabled=False,
        training_enabled=False,
        inference_enabled=False,
        hyperparameters={"status": "placeholder_only"},
    )
