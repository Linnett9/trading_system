from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.models.multitask_transformer_model import (
    MultiTaskTransformerSequenceMLModel,
    _make_multitask_transformer_module,
)


def _rows(count: int = 32) -> list[dict[str, float]]:
    return [
        {
            "return_1": index / 100.0,
            "volatility_10": (index % 5) / 10.0,
            "forward_return_5d": 99.0,
            "future_drawdown": 99.0,
            "actual_forward_return_5d": 99.0,
        }
        for index in range(count)
    ]


def _labels(count: int = 32) -> list[int]:
    return [1 if index % 4 in {0, 1} else 0 for index in range(count)]


def _regression_targets(count: int = 32) -> dict[str, list[float | None]]:
    return {
        "forward_return_5d": [index / 1000.0 for index in range(count)],
        "future_drawdown": [
            None if index % 7 == 0 else -(index % 5) / 100.0
            for index in range(count)
        ],
    }


def test_multitask_transformer_module_forward_pass_returns_all_heads():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    module = _make_multitask_transformer_module()(
        feature_count=3,
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        regression_head_count=2,
    )

    classification_logits, regression_outputs = module(torch.ones(5, 4, 3))

    assert tuple(classification_logits.shape) == (5,)
    assert tuple(regression_outputs.shape) == (5, 2)


def test_multitask_transformer_trains_predicts_and_round_trips(tmp_path: Path):
    pytest.importorskip("torch")
    rows = _rows()
    labels = _labels()
    model = MultiTaskTransformerSequenceMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=8,
        random_seed=7,
        regression_targets=["forward_return_5d", "future_drawdown"],
    )

    model.fit_multitask(rows, labels, _regression_targets())
    predictions = model.predict_multitask(rows)
    probabilities = model.predict_proba(rows)

    assert len(predictions) == len(rows)
    assert len(probabilities) == len(rows)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert all("predicted_forward_return_5d" in row for row in predictions)
    assert all("predicted_future_drawdown" in row for row in predictions)
    assert model.training_summary.missing_target_counts["future_drawdown"] > 0
    assert "forward_return_5d" not in model.feature_names
    assert "future_drawdown" not in model.feature_names
    assert "actual_forward_return_5d" not in model.feature_names

    model_path = tmp_path / "multitask_transformer.pt"
    model.save(model_path)
    loaded = MultiTaskTransformerSequenceMLModel.load(model_path)

    loaded_predictions = loaded.predict_multitask(rows)
    assert len(loaded_predictions) == len(rows)
    assert all(
        "predicted_forward_return_5d" in row
        for row in loaded_predictions
    )


def test_multitask_transformer_single_task_fit_remains_runner_compatible():
    pytest.importorskip("torch")
    model = MultiTaskTransformerSequenceMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=8,
        random_seed=7,
        regression_targets=["forward_return_5d"],
    )

    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(12))

    assert len(probabilities) == 12
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_multitask_transformer_sequence_context_stays_inside_groups():
    pytest.importorskip("torch")
    rows = _rows(8)
    labels = _labels(8)
    metadata = [
        {"variant_id": "variant_a"},
        {"variant_id": "variant_b"},
        {"variant_id": "variant_a"},
        {"variant_id": "variant_b"},
        {"variant_id": "variant_a"},
        {"variant_id": "variant_b"},
        {"variant_id": "variant_a"},
        {"variant_id": "variant_b"},
    ]
    model = MultiTaskTransformerSequenceMLModel(
        sequence_length=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=4,
        random_seed=7,
        regression_targets=["forward_return_5d"],
    )

    model.set_sequence_context(metadata=metadata, feature_dates=[])
    model.fit_multitask(rows, labels, {"forward_return_5d": [0.01] * 8})

    assert model.training_summary.sequence_count == 4
