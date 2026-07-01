from __future__ import annotations

from dataclasses import dataclass

BASELINE_MODELS = ("momentum_120d", "risk_adjusted_momentum")
TABULAR_MODELS = ("ridge", "elastic_net", "random_forest", "gradient_boosting")
ULTRAFAST_TARGET_MODELS = ("ridge", "elastic_net")
STANDARD_SEQUENCE_MODELS = ("dlinear", "market_context_encoder")
FULL_SEQUENCE_MODELS = ("dlinear", "patchtst", "transformer", "itransformer", "momentum_transformer", "multitask_transformer", "market_context_encoder", "news_analysis_transformer", "temporal_fusion_transformer")
VALIDATED_STANDARD_SEQUENCE_MODELS = tuple(model for model in FULL_SEQUENCE_MODELS if model != "news_analysis_transformer")
MODEL_SETS = {"fast": TABULAR_MODELS, "standard": TABULAR_MODELS + STANDARD_SEQUENCE_MODELS, "validated_standard": TABULAR_MODELS + VALIDATED_STANDARD_SEQUENCE_MODELS, "full": TABULAR_MODELS + FULL_SEQUENCE_MODELS}
TARGET_MODEL_SETS = {"ultrafast": ULTRAFAST_TARGET_MODELS, **MODEL_SETS}


@dataclass(frozen=True)
class StockAlphaModelSet:
    requested_model_set: str
    effective_model_set: str
    included_models: tuple[str, ...]
    excluded_models: tuple[dict[str, str], ...]
    baseline_models: tuple[str, ...] = BASELINE_MODELS

    def metadata(self) -> dict[str, object]:
        tree_excluded = any(row["name"] in {"random_forest", "gradient_boosting"} for row in self.excluded_models)
        return {"requested_model_set": self.requested_model_set, "effective_model_set": self.effective_model_set, "included_models": list(self.included_models), "excluded_models": list(self.excluded_models), "baseline_models": list(self.baseline_models), "deep_sequence_models_skipped_intentionally": self.effective_model_set in {"fast", "ultrafast"}, "tree_models_skipped_intentionally": tree_excluded}


def default_model_set(run_size: str) -> str:
    return "fast" if run_size == "dev" else "full"


def resolve_stock_alpha_model_set(name: str, *, include_sequence_models: bool = True) -> StockAlphaModelSet:
    requested = str(name).lower()
    if requested not in MODEL_SETS:
        raise ValueError("stock-alpha model set must be fast, standard, validated_standard, or full")
    desired = MODEL_SETS[requested]
    included = tuple(model for model in desired if include_sequence_models or model in TABULAR_MODELS)
    excluded = tuple({"name": model, "exclusion_reason": "conditional_news_features_unavailable" if requested == "validated_standard" and model == "news_analysis_transformer" else "not_selected_by_model_set" if model not in desired else "sequence_models_disabled"} for model in MODEL_SETS["full"] if model not in included)
    return StockAlphaModelSet(requested, requested, included, excluded)


def resolve_stock_alpha_target_model_set(
    name: str,
    *,
    include_sequence_models: bool = True,
) -> StockAlphaModelSet:
    requested = str(name).lower()
    if requested not in TARGET_MODEL_SETS:
        raise ValueError("stock-alpha target comparison model set must be ultrafast, fast, standard, validated_standard, or full")
    desired = TARGET_MODEL_SETS[requested]
    included = tuple(model for model in desired if include_sequence_models or model in TABULAR_MODELS)
    excluded = []
    for model in MODEL_SETS["full"]:
        if model in included:
            continue
        if requested == "ultrafast" and model in {"random_forest", "gradient_boosting"}:
            reason = "tree_model_excluded_by_ultrafast_target_comparison"
        elif requested == "validated_standard" and model == "news_analysis_transformer":
            reason = "conditional_news_features_unavailable"
        elif model not in desired:
            reason = "not_selected_by_model_set"
        else:
            reason = "sequence_models_disabled"
        excluded.append({"name": model, "exclusion_reason": reason})
    return StockAlphaModelSet(requested, requested, included, tuple(excluded))
