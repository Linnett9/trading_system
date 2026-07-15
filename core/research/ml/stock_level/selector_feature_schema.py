from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.selector_dataset import DETERMINISTIC_SIGNAL_COLUMNS
from core.research.ml.stock_level.stock_level_alpha_features import ENGINEERED_FEATURE_COLUMNS

OUTCOME_COLUMNS = {
    "target_observation_count", "target_start_timestamp", "label_start_timestamp",
    "label_end_timestamp", "label_available_timestamp", "benchmark_target_start_timestamp",
    "benchmark_label_start_timestamp", "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp", "target_status",
}
IDENTITY_COLUMNS = {"row_id", "symbol", "asset_id", "canonical_symbol", "industry_id"}
TIMESTAMP_COLUMNS = {
    "rebalance_date", "feature_timestamp", "feature_data_cutoff_timestamp", "decision_timestamp",
    "decision_session_date", "first_actionable_session", "market_context_source_date",
    "market_context_availability_timestamp",
}
CONDITIONAL_COLUMNS = {
    "industry_relative_strength", "relative_momentum_vs_industry", "industry_momentum_percentile",
}
UNAVAILABLE_COLUMNS = {
    "actual_forward_return_5d", "actual_future_volatility", "actual_future_drawdown",
    "actual_rank_normalized_forward_return_10d", "actual_top_decile_label_10d",
}


def classify_column(name: str, *, predictor_names: Sequence[str] = ()) -> tuple[str, str]:
    if name.startswith("actual_") or name in OUTCOME_COLUMNS:
        return "target/outcome", "Observed or label-window information; unavailable at decision time."
    if name in predictor_names:
        if name in CONDITIONAL_COLUMNS:
            return "conditionally available", "Point-in-time safe when industry mapping and peers are available; median imputation handles missing values."
        return "safe predictor", "Explicitly audited point-in-time predictor in the frozen schema."
    if name in IDENTITY_COLUMNS:
        return "identity", "Identifies a row, asset, symbol, or group; not a numeric model signal."
    if name in TIMESTAMP_COLUMNS or name.endswith("_timestamp") or name.endswith("_date"):
        return "timestamp", "Timing control or audit field; excluded from predictors."
    if name in UNAVAILABLE_COLUMNS:
        return "unavailable or all-null", "Frozen artifact contains no usable observations."
    if name.startswith("_stock_"):
        return "provenance/diagnostic", "Intermediate producer value retained for audit, not a frozen predictor."
    if name.startswith("target_") or name.startswith("label_") or name.startswith("benchmark_label_"):
        return "potentially leaking", "Describes the forward target window or label availability."
    return "provenance/diagnostic", "Contract, eligibility, provider, mapping, or audit metadata; excluded."


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "schema_hash"}


def schema_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(canonical_payload(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def load_feature_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_hash") != schema_hash(payload):
        raise RuntimeError(f"Selector feature schema hash mismatch: {path}")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError("Selector feature schema must contain an ordered non-empty features list")
    names = [row.get("name") for row in features]
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise RuntimeError("Selector feature schema feature names must be unique strings")
    if any(name.startswith("actual_") or name in OUTCOME_COLUMNS for name in names):
        raise RuntimeError("Outcome columns cannot be selector features")
    return payload


def expected_final_features() -> tuple[str, ...]:
    return (*DETERMINISTIC_SIGNAL_COLUMNS, *ENGINEERED_FEATURE_COLUMNS)
