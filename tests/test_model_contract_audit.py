from __future__ import annotations

import json

import pytest

from core.research.ml.artifact_schema import ARTIFACT_SCHEMA_VERSION
from core.research.ml.model_contract_audit import (
    COMPLETE_V1,
    MODEL_CONTRACT_SPECS,
    build_model_contract_audit,
    write_model_contract_audit,
)
from core.research.ml.models import build_ml_model


def test_model_contract_audit_reports_every_phase2_model(tmp_path):
    markdown_path, json_path = write_model_contract_audit(tmp_path)

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    registry_keys = {row["registry_key"] for row in rows}

    assert markdown_path.exists()
    assert json_path.exists()
    assert registry_keys == {spec.registry_key for spec in MODEL_CONTRACT_SPECS}
    assert all(row["artifact_support"] for row in rows)
    assert all(row["status"] == COMPLETE_V1 for row in rows)
    assert "Momentum Transformer" in markdown_path.read_text(encoding="utf-8")


def test_model_contract_audit_records_known_design_todos():
    rows = {
        row.registry_key: row
        for row in build_model_contract_audit()
    }

    assert "predicted_rank_score" in rows["itransformer"].todos[0]
    assert rows["meta_ensemble"].save_load_support
    assert rows["meta_ensemble"].todos
    assert rows["temporal_fusion_transformer"].expected_predicted_outputs == (
        "predicted_forward_return_5d",
        "predicted_forward_return_10d",
        "predicted_future_volatility",
        "predicted_future_drawdown",
    )


@pytest.mark.parametrize(
    ("model_type", "model_config"),
    [
        ("logistic_regression", {}),
        ("random_forest", {}),
        ("gradient_boosting", {}),
        (
            "dlinear",
            {
                "sequence_length": 8,
                "dlinear_epochs": 1,
                "dlinear_batch_size": 8,
            },
        ),
        (
            "patchtst",
            {
                "sequence_length": 8,
                "patchtst_sequence_length": 8,
                "patchtst_patch_length": 4,
                "patchtst_patch_stride": 2,
                "patchtst_d_model": 8,
                "patchtst_heads": 2,
                "patchtst_layers": 1,
                "patchtst_feedforward": 16,
                "patchtst_epochs": 1,
                "patchtst_batch_size": 8,
            },
        ),
        (
            "transformer",
            {
                "sequence_length": 8,
                "transformer_d_model": 8,
                "transformer_heads": 2,
                "transformer_layers": 1,
                "transformer_feedforward": 16,
                "transformer_epochs": 1,
                "transformer_batch_size": 8,
            },
        ),
        (
            "itransformer",
            {
                "sequence_length": 8,
                "itransformer_sequence_length": 8,
                "itransformer_d_model": 8,
                "itransformer_heads": 2,
                "itransformer_layers": 1,
                "itransformer_feedforward": 16,
                "itransformer_epochs": 1,
                "itransformer_batch_size": 8,
            },
        ),
        (
            "momentum_transformer",
            {
                "sequence_length": 8,
                "momentum_transformer_sequence_length": 8,
                "momentum_transformer_d_model": 8,
                "momentum_transformer_heads": 2,
                "momentum_transformer_layers": 1,
                "momentum_transformer_feedforward": 16,
                "momentum_transformer_epochs": 1,
                "momentum_transformer_batch_size": 8,
            },
        ),
        (
            "multitask_transformer",
            {
                "sequence_length": 8,
                "multitask_transformer_sequence_length": 8,
                "multitask_transformer_d_model": 8,
                "multitask_transformer_heads": 2,
                "multitask_transformer_layers": 1,
                "multitask_transformer_feedforward": 16,
                "multitask_transformer_epochs": 1,
                "multitask_transformer_batch_size": 8,
                "multitask_regression_targets": ["forward_return_5d"],
            },
        ),
        (
            "market_context_encoder",
            {
                "sequence_length": 8,
                "market_context_sequence_length": 8,
                "market_context_hidden_size": 8,
                "market_context_epochs": 1,
                "market_context_batch_size": 8,
            },
        ),
        (
            "news_analysis_transformer",
            {
                "sequence_length": 8,
                "news_transformer_sequence_length": 8,
                "news_transformer_d_model": 8,
                "news_transformer_heads": 2,
                "news_transformer_layers": 1,
                "news_transformer_feedforward": 16,
                "news_transformer_epochs": 1,
                "news_transformer_batch_size": 8,
            },
        ),
        (
            "temporal_fusion_transformer",
            {
                "sequence_length": 8,
                "tft_encoder_length": 8,
                "tft_hidden_size": 8,
                "tft_attention_heads": 2,
                "tft_epochs": 1,
                "tft_batch_size": 8,
            },
        ),
    ],
)
def test_registered_model_tiny_fit_predict_save_load_contract(
    tmp_path,
    model_type,
    model_config,
):
    if model_type not in {"logistic_regression", "random_forest", "gradient_boosting"}:
        pytest.importorskip("torch")

    model = build_ml_model(
        model_type,
        random_seed=7,
        model_config=model_config,
    )
    rows = _rows(72)
    labels = _labels(72)

    model.fit(rows, labels)
    probabilities = model.predict_proba(_rows(16))
    predictions = model.predict(_rows(16))

    assert len(probabilities) == 16
    assert len(predictions) == 16
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)

    model_path = tmp_path / f"{model_type}.model"
    model.save(model_path)
    loaded = type(model).load(model_path)
    loaded_probabilities = loaded.predict_proba(_rows(16))

    assert model_path.exists()
    assert len(loaded_probabilities) == 16
    assert all(0.0 <= probability <= 1.0 for probability in loaded_probabilities)


def test_artifact_schema_version_constant_is_v1():
    assert ARTIFACT_SCHEMA_VERSION == "ml_prediction_artifact_v1"


def _rows(count: int) -> list[dict[str, float]]:
    return [
        {
            "return_1": index / 100.0,
            "return_5": (index % 5) / 10.0,
            "volatility_10": (index % 11) / 10.0,
            "breadth": 1.0 if index % 3 else 0.0,
            "day_of_week": float(index % 5),
            "month": float((index % 12) + 1),
        }
        for index in range(count)
    ]


def _labels(count: int) -> list[int]:
    return [1 if index % 7 in {0, 1, 2} else 0 for index in range(count)]
