from datetime import datetime, timedelta

import pytest

from core.research.ml.online_learning import (
    FrozenLogisticModel,
    IncrementalLogisticModel,
    OnlineObservation,
    PeriodicRefitLogisticModel,
    PrequentialEvaluator,
    WarmStartNeuralModel,
)


def _observations(count=40, horizon_minutes=15):
    start = datetime(2025, 1, 2, 9, 30)
    return [
        OnlineObservation(
            observation_id=str(index),
            observed_at=start + timedelta(minutes=5 * index),
            label_available_at=start + timedelta(
                minutes=5 * index + horizon_minutes
            ),
            features={"momentum": float(index % 7), "volatility": float(index % 5)},
            label=int(index % 4 >= 2),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    "model",
    [
        IncrementalLogisticModel(),
        FrozenLogisticModel(minimum_training_samples=4),
        PeriodicRefitLogisticModel(refit_every=4, minimum_training_samples=4),
    ],
)
def test_prequential_models_never_train_on_unmatured_labels(model):
    result = PrequentialEvaluator(model).run(_observations())

    assert result["temporal_leakage_check_passed"] is True
    assert result["scored_prediction_count"] == 37
    assert result["pending_label_count"] == 3
    assert result["update_count"] > 0


def test_prediction_is_recorded_before_its_label_can_update_model():
    result = PrequentialEvaluator(IncrementalLogisticModel(update_batch_size=1)).run(
        _observations(6, horizon_minutes=10)
    )

    assert result["predictions"][0]["trained_through"] is None
    assert result["predictions"][1]["trained_through"] is None
    assert result["predictions"][2]["trained_through"] <= result["predictions"][2]["predicted_at"]


def test_warm_start_neural_retains_weights_and_uses_replay():
    pytest.importorskip("torch")
    model = WarmStartNeuralModel(
        hidden_size=4, replay_size=16, replay_batch_size=4, gradient_steps=1
    )

    result = PrequentialEvaluator(model).run(_observations(12, horizon_minutes=10))

    assert result["temporal_leakage_check_passed"] is True
    assert model.model is not None
    assert len(model.replay) > 0
