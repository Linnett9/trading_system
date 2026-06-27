from __future__ import annotations


ARTIFACT_SCHEMA_VERSION = "ml_prediction_artifact_v1"

REQUIRED_PREDICTION_ARTIFACT_COLUMNS = [
    "artifact_schema_version",
    "profile",
    "model_name",
    "model_type",
    "config_path",
    "dataset_hash",
    "source_dataset_row_count",
    "train_sample_count",
    "prediction_date",
    "symbol",
    "rebalance_date",
    "actual_label",
    "predicted_probability",
]

ALLOWED_OPTIONAL_PREDICTION_COLUMNS = [
    "predicted_forward_return_5d",
    "predicted_forward_return_10d",
    "predicted_future_volatility",
    "predicted_future_drawdown",
    "predicted_max_adverse_excursion",
    "predicted_max_favourable_excursion",
    "predicted_trend_score",
    "predicted_regime_score",
    "predicted_size_multiplier",
    "predicted_rank_score",
    "predicted_context_risk_multiplier",
]


def is_allowed_predicted_column(name: str) -> bool:
    return name.startswith("predicted_")
