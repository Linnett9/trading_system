from __future__ import annotations

from core.research.ml.models import build_ml_model
from core.research.ml.models import dlinear_model
from core.research.ml.models import itransformer_model
from core.research.ml.models import (
    market_context_encoder_model,
)
from core.research.ml.models import (
    momentum_transformer_model,
)
from core.research.ml.models import (
    multitask_transformer_model,
)
from core.research.ml.models import (
    news_analysis_transformer_model,
)
from core.research.ml.models import patchtst_model
from core.research.ml.models import (
    temporal_fusion_transformer_model,
)
from core.research.ml.models import transformer_model


def test_model_implementation_modules_import_from_models_package():
    assert transformer_model.TransformerSequenceMLModel
    assert patchtst_model.PatchTSTSequenceMLModel
    assert dlinear_model.DLinearSequenceMLModel
    assert itransformer_model.ITransformerSequenceMLModel
    assert momentum_transformer_model.MomentumTransformerSequenceMLModel
    assert multitask_transformer_model.MultiTaskTransformerSequenceMLModel
    assert market_context_encoder_model.MarketContextEncoderMLModel
    assert news_analysis_transformer_model.NewsAnalysisTransformerMLModel
    assert temporal_fusion_transformer_model.TemporalFusionTransformerMLModel


def test_models_package_preserves_registry_import_path():
    model = build_ml_model("noop")

    assert model.model_type == "noop"


def test_logistic_regression_scales_features_and_reports_convergence():
    model = build_ml_model(
        "logistic_regression",
        model_config={
            "logistic_scale_features": True,
            "logistic_max_iter": 5000,
            "logistic_l2_penalty": 1.0,
        },
    )
    rows = [
        {"small": index / 100.0, "large": float(index * 10000)}
        for index in range(40)
    ]
    labels = [int(index >= 20) for index in range(40)]

    model.fit(rows, labels)
    diagnostics = model.training_diagnostics()

    assert diagnostics["scaled_features"] is True
    assert diagnostics["max_iterations"] == 5000
    assert diagnostics["converged"] is True
    assert len(model.predict_proba(rows)) == len(rows)
