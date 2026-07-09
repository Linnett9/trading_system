from __future__ import annotations

from dataclasses import asdict, dataclass

from core.research.ml.stock_level.stock_alpha_model_sets import (
    TABULAR_MODELS,
    VALIDATED_STANDARD_SEQUENCE_MODELS,
)
from core.research.ml.stock_level_benchmark_types import (
    BASELINE_COLUMNS,
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    SEQUENCE_MODEL_NAMES,
)


FULLY_RUNNABLE = "fully_runnable"
PARTIALLY_WIRED = "partially_wired"
SCAFFOLD_ONLY = "scaffold_only"


@dataclass(frozen=True)
class StockRankingModelInventoryRow:
    registry_key: str
    status: str
    implementation_path: str
    input_data_shape: str
    sequence_length_requirements: str
    target_support: str
    cpu_support: bool
    gpu_support: bool
    oos_benchmark_support: bool
    prediction_artifact_support: bool
    model_persistence_support: str
    test_coverage: str
    known_limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["known_limitations"] = list(self.known_limitations)
        return payload


def stock_ranking_model_inventory() -> list[StockRankingModelInventoryRow]:
    rows = []
    for model in MODEL_NAMES:
        if model in TABULAR_MODELS:
            rows.append(_tabular_row(model))
        elif model in VALIDATED_STANDARD_SEQUENCE_MODELS:
            rows.append(_sequence_row(model))
        else:
            rows.append(_partial_sequence_row(model))
    return rows


def stock_ranking_model_sets() -> dict[str, tuple[str, ...]]:
    sequence_smoke = ("dlinear", "patchtst")
    transformer_smoke = (
        "transformer",
        "itransformer",
        "momentum_transformer",
        "multitask_transformer",
    )
    context_smoke = ("market_context_encoder", "temporal_fusion_transformer")
    full_candidates = tuple(
        model
        for model in (*TABULAR_MODELS, *VALIDATED_STANDARD_SEQUENCE_MODELS)
        if model != "news_analysis_transformer"
    )
    return {
        "fast_tabular": TABULAR_MODELS,
        "sequence_smoke": sequence_smoke,
        "transformer_smoke": transformer_smoke,
        "context_smoke": context_smoke,
        "full_research_candidates": full_candidates,
    }


def score_column_for_model(model_key: str) -> str:
    if model_key in BASELINE_COLUMNS:
        return BASELINE_COLUMNS[model_key]
    return f"stock_level_predicted_forward_return_10d_{model_key}"


def inventory_as_dicts() -> list[dict]:
    return [row.to_dict() for row in stock_ranking_model_inventory()]


def _tabular_row(model: str) -> StockRankingModelInventoryRow:
    return StockRankingModelInventoryRow(
        registry_key=model,
        status=FULLY_RUNNABLE,
        implementation_path="core/research/ml/stock_level_benchmark_models.py",
        input_data_shape="tabular rows: rebalance_date x symbol x feature_columns",
        sequence_length_requirements="none",
        target_support="actual_forward_return_10d",
        cpu_support=True,
        gpu_support=False,
        oos_benchmark_support=True,
        prediction_artifact_support=True,
        model_persistence_support="sklearn object; benchmark writes OOS predictions, not live model artifact",
        test_coverage="stock-level benchmark/model-set tests",
        known_limitations=(
            "research-only stock-ranker path",
            "no live inference service in this slice",
        ),
    )


def _sequence_row(model: str) -> StockRankingModelInventoryRow:
    needs_context = model in {
        "market_context_encoder",
        "temporal_fusion_transformer",
    }
    return StockRankingModelInventoryRow(
        registry_key=model,
        status=FULLY_RUNNABLE,
        implementation_path=(
            "core/research/ml/stock_level/stock_level_sequence_regressors.py"
        ),
        input_data_shape=(
            "per-symbol chronological sequences of feature rows"
            + (" plus context columns" if needs_context else "")
        ),
        sequence_length_requirements="stock_ranker_sequence_length >= 2",
        target_support="actual_forward_return_10d; multitask also consumes auxiliary targets",
        cpu_support=True,
        gpu_support=True,
        oos_benchmark_support=True,
        prediction_artifact_support=True,
        model_persistence_support="research adapter in benchmark path; no live model persistence contract",
        test_coverage="model-set and sequence benchmark tests",
        known_limitations=(
            "more expensive than fast tabular models",
            "requires PyTorch for execution",
        ),
    )


def _partial_sequence_row(model: str) -> StockRankingModelInventoryRow:
    limitations = (
        "requires validated point-in-time news feature contract",
        "excluded from validated_standard model set",
    ) if model == "news_analysis_transformer" else (
        "not selected by current validated smoke sets",
    )
    return StockRankingModelInventoryRow(
        registry_key=model,
        status=PARTIALLY_WIRED,
        implementation_path=(
            "core/research/ml/stock_level/stock_level_sequence_regressors.py"
        ),
        input_data_shape="conditional sequence/context feature rows",
        sequence_length_requirements="stock_ranker_sequence_length >= 2",
        target_support="actual_forward_return_10d where feature contract is satisfied",
        cpu_support=True,
        gpu_support=True,
        oos_benchmark_support=model in SEQUENCE_MODEL_NAMES,
        prediction_artifact_support=model in SEQUENCE_MODEL_NAMES,
        model_persistence_support="not promoted to live persistence contract",
        test_coverage="contract-level tests only or conditional coverage",
        known_limitations=limitations,
    )
