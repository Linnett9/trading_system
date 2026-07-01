from __future__ import annotations

from typing import Any

from core.research.framework.registry import FeatureRegistry
from core.research.ml.stock_level.stock_level_alpha_features_math import _number
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    ENGINEERED_FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    RESEARCH_METADATA,
)


def _audit(
    source_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, float | str]]],
    source_path: str | None,
    n_jobs: int,
) -> dict[str, Any]:
    features = []
    for feature in ENGINEERED_FEATURE_COLUMNS:
        populated = sum(_number(row.get(feature)) is not None for row in rows)
        features.append(
            {
                "feature": feature,
                "definition": FEATURE_DEFINITIONS[feature],
                "populated_count": populated,
                "missing_count": len(rows) - populated,
                "availability_rate": populated / len(rows) if rows else 0.0,
            }
        )
    source_columns = list(source_rows[0]) if source_rows else []
    source_by_key = {
        (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper()): row
        for row in source_rows
    }
    return {
        "mode": "stock_level_alpha_features_research_only",
        "source_path": source_path,
        "output_policy": "Write a sibling enriched artifact; never overwrite the source CSV.",
        "row_count": len(rows),
        "source_column_count": len(source_columns),
        "engineered_feature_count": len(ENGINEERED_FEATURE_COLUMNS),
        "source_columns_preserved": all(
            all(row.get(column) == source_by_key.get(
                (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper()),
                {},
            ).get(column) for column in source_columns)
            for row in rows
        ),
        "unique_symbol_date_rows": len(
            {(row.get("rebalance_date"), row.get("symbol")) for row in rows}
        )
        == len(rows),
        "price_history_symbol_count": sum(bool(history) for history in histories.values()),
        "parallelism": {
            "stock_alpha_feature_n_jobs": n_jobs,
            "partition": "symbol_level_time_series_features",
            "cross_sectional_features_after_parallel_assembly": True,
            "output_order": "rebalance_date_symbol",
        },
        "industry_metadata_available": any(str(row.get("industry", "")).strip() for row in rows),
        "features": features,
        **RESEARCH_METADATA,
    }
def alpha_feature_registry() -> FeatureRegistry[str]:
    registry: FeatureRegistry[str] = FeatureRegistry()
    for name in ENGINEERED_FEATURE_COLUMNS:
        registry.register(
            name,
            name,
            metadata={"definition": FEATURE_DEFINITIONS[name]},
        )
    return registry
