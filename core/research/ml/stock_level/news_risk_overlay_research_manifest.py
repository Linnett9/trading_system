from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


def build_news_risk_metrics_and_manifest(
    *,
    price_path: Path,
    news_path: Path,
    output_dir: Path,
    price_score_column: str,
    return_column: str,
    labeled: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    dataset_max_rows: int,
    shadow_max_rows: int,
    price_metrics: Mapping[str, Any],
    news_metrics: Mapping[str, Any],
    price_feature_columns: list[str],
    price_news_feature_columns: list[str],
    label_source_columns: tuple[str, ...],
    limited_rows: Callable[[list[Mapping[str, Any]], int], list[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = {
        "model_type": "in_repo_logistic_regression",
        "chronological_walk_forward": True,
        "transformer_trained": False,
        "paper_trading_enabled": False,
        "live_trading_enabled": False,
        "paper_orders_enabled": False,
        "price_only": price_metrics,
        "price_plus_news": news_metrics,
        "price_feature_columns": price_feature_columns,
        "price_plus_news_feature_columns": price_news_feature_columns,
    }
    manifest = {
        "mode": "ml-stock-alpha-news-risk-overlay-research",
        "research_only": True,
        "trading_impact": "none",
        "price_candidates_path": str(price_path),
        "news_features_path": str(news_path),
        "output_dir": str(output_dir),
        "price_score_column": price_score_column,
        "return_column": return_column,
        "label_source_columns": [column for column in label_source_columns if column in labeled[0]],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_joined_row_count": len(labeled),
        "dataset_csv_row_count": len(limited_rows(labeled, dataset_max_rows)),
        "shadow_csv_row_count": len(limited_rows(decision_rows, shadow_max_rows)),
        "transformer_trained": False,
        "paper_orders_enabled": False,
    }
    return metrics, manifest
