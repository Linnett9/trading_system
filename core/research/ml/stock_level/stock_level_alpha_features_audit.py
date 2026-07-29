from __future__ import annotations

from typing import Any

from core.research.framework.registry import FeatureRegistry
from core.research.ml.stock_level.stock_level_alpha_features_math import _number
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    BREADTH_CONTRACT_VERSION,
    ENGINEERED_FEATURE_COLUMNS,
    ENRICHMENT_METADATA_COLUMNS,
    FEATURE_DEFINITIONS,
    INDUSTRY_MAPPING_CONTRACT_VERSION,
    MARKET_CONTEXT_CONTRACT_VERSION,
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
    populated_features = [row["feature"] for row in features if row["populated_count"] > 0]
    if len(populated_features) == len(ENGINEERED_FEATURE_COLUMNS):
        enrichment_status = "enriched"
    elif populated_features:
        enrichment_status = "partially_enriched"
    else:
        enrichment_status = "no_additional_features"
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
        "enrichment_metadata_columns": list(ENRICHMENT_METADATA_COLUMNS),
        "resolved_enriched_columns": populated_features,
        "resolved_enriched_column_count": len(populated_features),
        "enrichment_status": enrichment_status,
        "enrichment_contract_version": "stock_level_alpha_enrichment_contract_v1",
        "enrichment_configuration_identity": {
            "partition": "symbol_level_time_series_features",
            "cross_sectional_features_after_parallel_assembly": True,
            "uses_history_strictly_before_rebalance": True,
            "n_jobs": n_jobs,
        },
        "market_context_contract": {
            "contract_version": MARKET_CONTEXT_CONTRACT_VERSION,
            "benchmark_symbol": "SPY",
            "feature_cutoff_rule": "market observation date < stock rebalance_date via history_before",
            "lookbacks": ["20", "60", "120", "200"],
            "missing_session_behaviour": "remain missing; no forward fill from future observations",
        },
        "daily_price_feature_availability_authority": {
            "authority_version": next(
                (
                    str(row.get("daily_price_availability_authority_version"))
                    for row in rows
                    if row.get("daily_price_availability_authority_version")
                ),
                "",
            ),
            "availability_rule": "daily feature observations must be available before decision_timestamp",
            "status_counts": _count_values(rows, "daily_price_availability_status"),
            "future_feature_inclusion_count": sum(
                1
                for row in rows
                if str(row.get("daily_price_availability_status") or "")
                in {"NOT_YET_AVAILABLE", "REVISED_AFTER_DECISION", "CONFLICTING_EVIDENCE"}
            ),
        },
        "breadth_contract": {
            "contract_version": BREADTH_CONTRACT_VERSION,
            "universe_identity": "source artifact decision-date cross-section",
            "minimum_required_coverage": "reported in breadth_coverage; selector gates decide warn/block",
            "calculation_rule": "date-local eligible rows only",
        },
        "industry_mapping_contract": {
            "contract_version": INDUSTRY_MAPPING_CONTRACT_VERSION,
            "mapping_source": "artifact industry column when present; static/current if supplied upstream",
            "historical_limitation": "not historically versioned by this enrichment stage",
            "missing_mapping_behaviour": "numeric relative values remain missing",
        },
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


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
