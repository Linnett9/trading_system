from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.research.ml.news_analysis_transformer_model import (
    NewsAnalysisTransformerMLModel,
)
from core.research.ml.news_sentiment import (
    NewsEvent,
    aggregate_news_sentiment_features,
)


def _rows(count: int = 32) -> list[dict[str, float]]:
    return [
        {
            "return_21d": index / 100.0,
            "volatility_21d": (index % 5) / 10.0,
            "news_sentiment_5d_mean": (-1.0 if index % 4 == 0 else 0.25),
            "news_sentiment_5d_count": float(index % 3),
            "forward_return_5d": 99.0,
            "future_drawdown": 99.0,
        }
        for index in range(count)
    ]


def _labels(count: int = 32) -> list[int]:
    return [1 if index % 6 in {0, 1} else 0 for index in range(count)]


def test_news_sentiment_aggregation_excludes_future_and_duplicates():
    feature_timestamp = datetime(2024, 1, 10, 16, tzinfo=timezone.utc)
    events = [
        NewsEvent(
            event_id="1",
            symbol="AAPL",
            headline="AAPL beats profit expectations",
            published_at=datetime(2024, 1, 9, 12, tzinfo=timezone.utc),
            first_seen_at=datetime(2024, 1, 9, 12, 5, tzinfo=timezone.utc),
            retrieved_at=datetime(2024, 1, 9, 12, 10, tzinfo=timezone.utc),
            source_name="tier1",
            reliability_tier=1,
        ),
        NewsEvent(
            event_id="2",
            symbol="AAPL",
            headline="AAPL beats profit expectations",
            published_at=datetime(2024, 1, 9, 12, tzinfo=timezone.utc),
            first_seen_at=datetime(2024, 1, 9, 12, 6, tzinfo=timezone.utc),
            retrieved_at=datetime(2024, 1, 9, 12, 10, tzinfo=timezone.utc),
            source_name="duplicate",
            reliability_tier=1,
        ),
        NewsEvent(
            event_id="3",
            symbol="AAPL",
            headline="AAPL warning after close",
            published_at=datetime(2024, 1, 11, 12, tzinfo=timezone.utc),
            first_seen_at=datetime(2024, 1, 11, 12, tzinfo=timezone.utc),
            retrieved_at=datetime(2024, 1, 11, 12, tzinfo=timezone.utc),
            source_name="future",
            reliability_tier=1,
        ),
    ]

    features, audit = aggregate_news_sentiment_features(
        events,
        feature_timestamp,
        "AAPL",
        lookback_days=[5],
    )

    assert features["news_sentiment_5d_count"] == 1.0
    assert features["news_sentiment_5d_mean"] > 0.0
    assert audit.deduplicated_event_count == 1
    assert audit.excluded_future_event_count == 1


def test_news_analysis_transformer_outputs_components_and_round_trips(tmp_path: Path):
    pytest.importorskip("torch")
    model = NewsAnalysisTransformerMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=8,
        random_seed=7,
    )

    model.fit(_rows(), _labels())
    components = model.predict_news_components(_rows(12))

    assert len(components) == 12
    assert all(0.0 <= row["news_probability_should_reduce_exposure"] <= 1.0 for row in components)
    assert "forward_return_5d" not in model.feature_names
    assert "future_drawdown" not in model.feature_names

    model_path = tmp_path / "news_transformer.pt"
    model.save(model_path)
    loaded = NewsAnalysisTransformerMLModel.load(model_path)
    assert len(loaded.predict_news_components(_rows(12))) == 12
