from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.market_context_encoder_model import MarketContextEncoderMLModel


def _rows(count: int = 32) -> list[dict[str, float]]:
    return [
        {
            "spy_return_21d": index / 100.0,
            "qqq_return_21d": index / 120.0,
            "realized_volatility_21d": (index % 5) / 10.0,
            "breadth_above_sma_200": 0.7 if index % 3 else 0.4,
            "future_drawdown": 99.0,
            "forward_return_5d": 99.0,
        }
        for index in range(count)
    ]


def _labels(count: int = 32) -> list[int]:
    return [1 if index % 5 in {0, 1} else 0 for index in range(count)]


def _metadata(count: int = 32) -> list[dict[str, str]]:
    return [
        {"variant_id": "monthly_equal" if index % 2 else "weekly_inverse_vol"}
        for index in range(count)
    ]


def test_market_context_encoder_outputs_bounded_context(tmp_path: Path):
    pytest.importorskip("torch")
    model = MarketContextEncoderMLModel(
        sequence_length=4,
        hidden_size=8,
        epochs=1,
        batch_size=8,
        random_seed=7,
    )

    model.set_sequence_context(metadata=_metadata(), feature_dates=[])
    model.fit(_rows(), _labels())
    context = model.predict_context(_rows(12))

    assert len(context) == 12
    assert all(0.0 <= row["market_regime_probability_risk_off"] <= 1.0 for row in context)
    assert all(0.25 <= row["risk_multiplier"] <= 1.25 for row in context)
    assert "future_drawdown" not in model.feature_names
    assert "forward_return_5d" not in model.feature_names

    model_path = tmp_path / "market_context.pt"
    model.save(model_path)
    loaded = MarketContextEncoderMLModel.load(model_path)
    assert len(loaded.predict_context(_rows(12))) == 12
