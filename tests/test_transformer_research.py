from __future__ import annotations

from pathlib import Path

import pytest

from core.research.ml.datasets import MLDataset
from core.research.ml.itransformer_model import (
    ITransformerSequenceMLModel,
    _build_itransformer_module,
)
from core.research.ml.momentum_transformer_model import (
    MomentumTransformerSequenceMLModel,
    _build_momentum_transformer_module,
)
from core.research.ml.sequence_dataset import build_sequence_dataset
from core.research.ml.transformer_model import TransformerSequenceMLModel
from core.research.ml.models import build_ml_model


def _dataset() -> MLDataset:
    features = [
        {"return_1m": index / 100.0, "volatility": (index % 3) / 10.0}
        for index in range(12)
    ]
    labels = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    feature_dates = [f"2024-01-{index + 1:02d}" for index in range(12)]
    label_start_dates = [f"2024-02-{index + 1:02d}" for index in range(12)]
    label_end_dates = [f"2024-03-{index + 1:02d}" for index in range(12)]
    return MLDataset(features, labels, feature_dates, label_start_dates, label_end_dates)


def test_build_sequence_dataset_aligns_label_with_last_row():
    dataset = _dataset()

    sequence_dataset = build_sequence_dataset(dataset, sequence_length=4)

    assert sequence_dataset.sample_count == 9
    assert sequence_dataset.sequence_length == 4
    assert sequence_dataset.feature_dates[0] == dataset.feature_dates[3]
    assert sequence_dataset.labels[0] == dataset.labels[3]
    assert sequence_dataset.label_start_dates[0] == dataset.label_start_dates[3]
    assert sequence_dataset.feature_names == ["return_1m", "volatility"]


def test_build_sequence_dataset_does_not_cross_variant_groups():
    dataset = MLDataset(
        features=[
            {"return_1m": index / 100.0, "volatility": 0.1}
            for index in range(6)
        ],
        labels=[0, 1, 0, 1, 0, 1],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(6)],
        label_start_dates=[f"2024-02-{index + 1:02d}" for index in range(6)],
        label_end_dates=[f"2024-03-{index + 1:02d}" for index in range(6)],
        metadata=[
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
        ],
    )

    sequence_dataset = build_sequence_dataset(dataset, sequence_length=3)

    assert sequence_dataset.sample_count == 2
    assert sequence_dataset.feature_dates == ["2024-01-05", "2024-01-06"]
    assert sequence_dataset.sequence_group_ids == ["variant_a", "variant_b"]


def test_build_sequence_dataset_groups_symbol_rows_by_symbol():
    dataset = MLDataset(
        features=[
            {"return_1m": index / 100.0, "volatility": 0.1}
            for index in range(6)
        ],
        labels=[0, 1, 0, 1, 0, 1],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(6)],
        label_start_dates=[f"2024-02-{index + 1:02d}" for index in range(6)],
        label_end_dates=[f"2024-03-{index + 1:02d}" for index in range(6)],
        metadata=[
            {"symbol": "AAPL", "variant_id": "shared"},
            {"symbol": "MSFT", "variant_id": "shared"},
            {"symbol": "AAPL", "variant_id": "shared"},
            {"symbol": "MSFT", "variant_id": "shared"},
            {"symbol": "AAPL", "variant_id": "shared"},
            {"symbol": "MSFT", "variant_id": "shared"},
        ],
    )

    sequence_dataset = build_sequence_dataset(dataset, sequence_length=3)

    assert sequence_dataset.sample_count == 2
    assert sequence_dataset.sequence_group_ids == ["AAPL", "MSFT"]


def test_transformer_sequence_context_keeps_training_windows_inside_variants():
    pytest.importorskip("torch")
    dataset = MLDataset(
        features=[
            {"return_1m": index / 100.0, "volatility": (index % 3) / 10.0}
            for index in range(8)
        ],
        labels=[0, 1, 0, 1, 0, 1, 1, 0],
        feature_dates=[f"2024-01-{index + 1:02d}" for index in range(8)],
        label_start_dates=[f"2024-02-{index + 1:02d}" for index in range(8)],
        label_end_dates=[f"2024-03-{index + 1:02d}" for index in range(8)],
        metadata=[
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
            {"variant_id": "variant_a"},
            {"variant_id": "variant_b"},
        ],
    )
    model = TransformerSequenceMLModel(
        sequence_length=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=4,
        random_seed=7,
    )

    model.set_sequence_context(metadata=dataset.metadata, feature_dates=dataset.feature_dates)
    model.fit(dataset.features, dataset.labels)

    assert model.training_summary.sequence_count == 4


def test_transformer_model_returns_one_probability_per_input_row(tmp_path: Path):
    pytest.importorskip("torch")
    dataset = _dataset()
    model = TransformerSequenceMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=4,
        random_seed=7,
    )

    model.fit(dataset.features, dataset.labels)
    probabilities = model.predict_proba(dataset.features)

    assert len(probabilities) == dataset.sample_count
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    model_path = tmp_path / "transformer.pt"
    model.save(model_path)
    loaded = TransformerSequenceMLModel.load(model_path)
    assert len(loaded.predict_proba(dataset.features)) == dataset.sample_count


def test_build_ml_model_can_create_transformer():
    pytest.importorskip("torch")

    model = build_ml_model(
        "transformer",
        random_seed=11,
        model_config={
            "sequence_length": 4,
            "transformer_d_model": 8,
            "transformer_heads": 2,
            "transformer_layers": 1,
            "transformer_epochs": 1,
        },
    )

    assert isinstance(model, TransformerSequenceMLModel)
    assert model.sequence_length == 4
    assert model.d_model == 8


def test_itransformer_module_forward_pass_returns_batch_logits():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    module = _build_itransformer_module(
        torch=torch,
        nn=nn,
        sequence_length=4,
        feature_count=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )

    logits = module(torch.ones(5, 4, 3))

    assert tuple(logits.shape) == (5,)


def test_itransformer_model_returns_one_probability_per_input_row(tmp_path: Path):
    pytest.importorskip("torch")
    dataset = _dataset()
    model = ITransformerSequenceMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=4,
        random_seed=7,
    )

    model.fit(dataset.features, dataset.labels)
    probabilities = model.predict_proba(dataset.features)

    assert len(probabilities) == dataset.sample_count
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    model_path = tmp_path / "itransformer.pt"
    model.save(model_path)
    loaded = ITransformerSequenceMLModel.load(model_path)
    assert len(loaded.predict_proba(dataset.features)) == dataset.sample_count


def test_momentum_transformer_module_forward_pass_returns_heads():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    module = _build_momentum_transformer_module(
        torch=torch,
        nn=nn,
        sequence_length=4,
        feature_count=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )

    logits, trend_logits, regime_logits = module(torch.ones(5, 4, 3))

    assert tuple(logits.shape) == (5,)
    assert tuple(trend_logits.shape) == (5,)
    assert tuple(regime_logits.shape) == (5,)


def test_momentum_transformer_model_returns_probabilities_and_components(tmp_path: Path):
    pytest.importorskip("torch")
    dataset = _dataset()
    model = MomentumTransformerSequenceMLModel(
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        epochs=1,
        batch_size=4,
        random_seed=7,
    )

    model.fit(dataset.features, dataset.labels)
    probabilities = model.predict_proba(dataset.features)
    components = model.predict_components(dataset.features)

    assert len(probabilities) == dataset.sample_count
    assert len(components) == dataset.sample_count
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert all(-1.0 <= row["trend_score"] <= 1.0 for row in components)
    assert all(0.0 <= row["regime_score"] <= 1.0 for row in components)
    assert all(0.25 <= row["size_multiplier"] <= 1.25 for row in components)

    model_path = tmp_path / "momentum_transformer.pt"
    model.save(model_path)
    loaded = MomentumTransformerSequenceMLModel.load(model_path)
    assert len(loaded.predict_proba(dataset.features)) == dataset.sample_count
