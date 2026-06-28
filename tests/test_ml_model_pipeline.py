from __future__ import annotations

from core.research.ml.config import MLExperimentConfig
from core.research.ml.datasets import MLDataset
from core.research.ml.pipelines.model_pipeline import MLModelPipeline
from core.research.ml.validation import ChronologicalSplit


def test_model_pipeline_builds_noop_model_and_predicts_probabilities():
    pipeline = MLModelPipeline(
        {"ml": {"model_type": "noop", "decision_threshold": 0.6}},
        MLExperimentConfig.from_config(
            {"ml": {"model_type": "noop", "decision_threshold": 0.6}}
        ),
    )
    dataset = _dataset([{"x": 1.0}, {"x": 2.0}], [0, 1])
    model = pipeline.build_model()

    pipeline.fit(model, dataset)
    prediction = pipeline.predict(model, dataset)

    assert prediction.probabilities == [0.5, 0.5]
    assert prediction.auxiliary_predictions == [{}, {}]
    assert pipeline.predictions_from_probabilities(prediction.probabilities) == [0, 0]


def test_model_pipeline_uses_multitask_fit_and_predict_contract():
    config = {
        "ml": {
            "model_type": "noop",
            "multitask_enabled": True,
            "multitask_regression_targets": ["future_drawdown"],
        }
    }
    pipeline = MLModelPipeline(config, MLExperimentConfig.from_config(config))
    dataset = _dataset(
        [{"x": 1.0}, {"x": 2.0}],
        [0, 1],
        auxiliary_targets=[
            {"future_drawdown": -0.1},
            {"future_drawdown": None},
        ],
    )
    model = _MultitaskModel()

    pipeline.fit(model, dataset)
    prediction = pipeline.predict(model, dataset)

    assert model.fit_targets == {"future_drawdown": [-0.1, None]}
    assert prediction.probabilities == [0.7, 0.2]
    assert prediction.auxiliary_predictions == [
        {"predicted_future_drawdown": -0.05},
        {"predicted_future_drawdown": -0.02},
    ]


def test_model_pipeline_uses_component_predictions_and_auxiliary_mapping():
    pipeline = MLModelPipeline(
        {"ml": {"model_type": "noop"}},
        MLExperimentConfig.from_config({"ml": {"model_type": "noop"}}),
    )
    dataset = _dataset([{"x": 1.0}, {"x": 2.0}], [0, 1])

    prediction = pipeline.predict(_ComponentModel(), dataset)

    assert prediction.probabilities == [0.8, 0.3]
    assert prediction.auxiliary_predictions == [
        {
            "predicted_trend_score": 0.2,
            "predicted_context_risk_multiplier": 0.9,
        },
        {"predicted_rank_score": 0.4},
    ]


def test_model_pipeline_prediction_context_concatenates_split():
    pipeline = MLModelPipeline(
        {"ml": {"model_type": "noop"}},
        MLExperimentConfig.from_config({"ml": {"model_type": "noop"}}),
    )
    train = _dataset([{"x": 1.0}], [0], feature_dates=["2024-01-01"])
    test = _dataset([{"x": 2.0}], [1], feature_dates=["2024-01-02"])

    context = pipeline.prediction_context(
        ChronologicalSplit(
            train=train,
            test=test,
            test_start_date="2024-01-02",
            purged_train_samples=0,
        )
    )

    assert context.features == [{"x": 1.0}, {"x": 2.0}]
    assert context.labels == [0, 1]
    assert context.feature_dates == ["2024-01-01", "2024-01-02"]


def _dataset(
    features: list[dict[str, float]],
    labels: list[int],
    *,
    feature_dates: list[str] | None = None,
    auxiliary_targets: list[dict[str, float | None]] | None = None,
) -> MLDataset:
    feature_dates = feature_dates or [
        f"2024-01-{index + 1:02d}"
        for index in range(len(features))
    ]
    return MLDataset(
        features=features,
        labels=labels,
        feature_dates=feature_dates,
        label_start_dates=feature_dates,
        label_end_dates=feature_dates,
        feature_ids=feature_dates,
        metadata=[{} for _ in features],
        auxiliary_targets=auxiliary_targets or [{} for _ in features],
    )


class _MultitaskModel:
    def __init__(self) -> None:
        self.fit_targets = None

    def set_sequence_context(self, **kwargs) -> None:
        self.context = kwargs

    def fit_multitask(self, features, labels, auxiliary_targets) -> None:
        self.fit_targets = auxiliary_targets

    def predict_multitask(self, features):
        return [
            {
                "probability_should_reduce_exposure": 0.7,
                "predicted_future_drawdown": -0.05,
            },
            {
                "probability_should_reduce_exposure": 0.2,
                "predicted_future_drawdown": -0.02,
            },
        ]


class _ComponentModel:
    def set_sequence_context(self, **kwargs) -> None:
        self.context = kwargs

    def predict_components(self, features):
        return [
            {
                "probability_should_reduce_exposure": 0.8,
                "trend_score": 0.2,
                "risk_multiplier": 0.9,
            },
            {
                "probability_should_reduce_exposure": 0.3,
                "rank_score": 0.4,
            },
        ]
