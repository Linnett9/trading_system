from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.temporal_fusion_transformer_model import (
    TemporalFusionTransformerMLModel,
)


def _rows(count: int = 32) -> list[dict[str, float]]:
    return [
        {
            "return_21d": index / 100.0,
            "volatility_21d": (index % 5) / 10.0,
            "day_of_week": float(index % 5),
            "month": 1.0,
            "is_month_end": 1.0 if index % 21 == 0 else 0.0,
            "days_until_next_rebalance": float(21 - (index % 21)),
            "forward_return_5d": 99.0,
            "future_drawdown": 99.0,
        }
        for index in range(count)
    ]


def _labels(count: int = 32) -> list[int]:
    return [1 if index % 7 in {0, 1} else 0 for index in range(count)]


def test_temporal_fusion_transformer_outputs_auxiliary_predictions(tmp_path: Path):
    pytest.importorskip("torch")
    model = TemporalFusionTransformerMLModel(
        sequence_length=4,
        hidden_size=8,
        attention_heads=2,
        num_layers=1,
        epochs=1,
        batch_size=8,
        random_seed=7,
        known_future_features=["day_of_week", "month", "is_month_end", "days_until_next_rebalance"],
    )

    model.fit(_rows(), _labels())
    outputs = model.predict_tft_outputs(_rows(12))

    assert len(outputs) == 12
    assert all(0.0 <= row["probability_should_reduce_exposure"] <= 1.0 for row in outputs)
    assert all("predicted_forward_return_5d" in row for row in outputs)
    assert all("tft_gating_saturation" in row for row in outputs)
    assert "forward_return_5d" not in model.observed_feature_names
    assert "future_drawdown" not in model.observed_feature_names
    assert "day_of_week" in model.known_feature_names

    model_path = tmp_path / "tft.pt"
    model.save(model_path)
    loaded = TemporalFusionTransformerMLModel.load(model_path)
    assert len(loaded.predict_tft_outputs(_rows(12))) == 12
