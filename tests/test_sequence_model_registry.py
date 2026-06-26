from __future__ import annotations

import pytest

from core.research.ml.models import build_ml_model


def _rows(count: int = 160) -> list[dict[str, float]]:
    return [
        {
            "return_1": i / 100.0,
            "volatility_10": (i % 11) / 10.0,
            "breadth": 1.0 if i % 3 else 0.0,
        }
        for i in range(count)
    ]


def _labels(count: int = 160) -> list[int]:
    return [1 if i % 7 in {0, 1, 2} else 0 for i in range(count)]


def test_build_dlinear_model_from_registry():
    pytest.importorskip("torch")
    model = build_ml_model(
        "dlinear",
        random_seed=7,
        model_config={"sequence_length": 16, "dlinear_epochs": 2, "dlinear_batch_size": 16},
    )
    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(40))
    assert len(probabilities) == 40
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_dlinear_sequence_length_uses_dlinear_specific_config():
    pytest.importorskip("torch")
    model = build_ml_model(
        "dlinear",
        random_seed=7,
        model_config={"sequence_length": 63, "dlinear_sequence_length": 16},
    )

    assert model.sequence_length == 16


def test_build_patchtst_model_from_registry():
    pytest.importorskip("torch")
    model = build_ml_model(
        "patchtst",
        random_seed=7,
        model_config={
            "sequence_length": 16,
            "patchtst_patch_length": 4,
            "patchtst_patch_stride": 2,
            "patchtst_d_model": 16,
            "patchtst_heads": 4,
            "patchtst_layers": 1,
            "patchtst_feedforward": 32,
            "patchtst_epochs": 2,
            "patchtst_batch_size": 16,
        },
    )
    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(40))
    assert len(probabilities) == 40
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_patchtst_sequence_length_uses_patchtst_specific_config():
    pytest.importorskip("torch")
    model = build_ml_model(
        "patchtst",
        random_seed=7,
        model_config={
            "sequence_length": 63,
            "patchtst_sequence_length": 16,
            "patchtst_patch_length": 4,
            "patchtst_patch_stride": 2,
            "patchtst_d_model": 16,
            "patchtst_heads": 4,
        },
    )

    assert model.sequence_length == 16


def test_build_itransformer_model_from_registry():
    pytest.importorskip("torch")
    model = build_ml_model(
        "itransformer",
        random_seed=7,
        model_config={
            "sequence_length": 16,
            "itransformer_d_model": 16,
            "itransformer_heads": 4,
            "itransformer_layers": 1,
            "itransformer_feedforward": 32,
            "itransformer_epochs": 2,
            "itransformer_batch_size": 16,
        },
    )
    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(40))
    assert len(probabilities) == 40
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_itransformer_sequence_length_uses_itransformer_specific_config():
    pytest.importorskip("torch")
    model = build_ml_model(
        "itransformer",
        random_seed=7,
        model_config={
            "sequence_length": 63,
            "itransformer_sequence_length": 16,
            "itransformer_d_model": 16,
            "itransformer_heads": 4,
        },
    )

    assert model.sequence_length == 16


def test_build_momentum_transformer_model_from_registry():
    pytest.importorskip("torch")
    model = build_ml_model(
        "momentum_transformer",
        random_seed=7,
        model_config={
            "sequence_length": 16,
            "momentum_transformer_d_model": 16,
            "momentum_transformer_heads": 4,
            "momentum_transformer_layers": 1,
            "momentum_transformer_feedforward": 32,
            "momentum_transformer_epochs": 2,
            "momentum_transformer_batch_size": 16,
        },
    )
    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(40))
    assert len(probabilities) == 40
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_momentum_transformer_sequence_length_uses_specific_config():
    pytest.importorskip("torch")
    model = build_ml_model(
        "momentum_transformer",
        random_seed=7,
        model_config={
            "sequence_length": 63,
            "momentum_transformer_sequence_length": 16,
            "momentum_transformer_d_model": 16,
            "momentum_transformer_heads": 4,
        },
    )

    assert model.sequence_length == 16


def test_build_multitask_transformer_model_from_registry():
    pytest.importorskip("torch")
    model = build_ml_model(
        "multitask_transformer",
        random_seed=7,
        model_config={
            "sequence_length": 16,
            "multitask_transformer_d_model": 16,
            "multitask_transformer_heads": 4,
            "multitask_transformer_layers": 1,
            "multitask_transformer_feedforward": 32,
            "multitask_transformer_epochs": 2,
            "multitask_transformer_batch_size": 16,
            "multitask_regression_targets": ["forward_return_5d", "future_drawdown"],
        },
    )
    model.fit(_rows(), _labels())
    probabilities = model.predict_proba(_rows(40))
    assert len(probabilities) == 40
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_multitask_transformer_sequence_length_uses_specific_config():
    pytest.importorskip("torch")
    model = build_ml_model(
        "multitask_transformer",
        random_seed=7,
        model_config={
            "sequence_length": 63,
            "multitask_transformer_sequence_length": 16,
            "multitask_transformer_d_model": 16,
            "multitask_transformer_heads": 4,
        },
    )

    assert model.sequence_length == 16
