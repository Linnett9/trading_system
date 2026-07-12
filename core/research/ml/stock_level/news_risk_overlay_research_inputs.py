from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.research.ml.stock_level.news_risk_overlay import (
    DECISION_TIMESTAMP_COLUMNS,
    TIMESTAMP_COLUMNS,
    NewsRiskOverlayConfig,
    build_news_risk_labels,
    join_news_to_stock_alpha_observations,
)
from core.research.ml.stock_level.news_risk_overlay_research_parallel import (
    parallel_config as _parallel_config,
    parallel_report_skeleton as _parallel_report_skeleton,
    timed_phase as _timed_phase,
)
from core.research.ml.stock_level.news_risk_overlay_research_utils import (
    EXCLUDED_FEATURE_COLUMNS,
    EXCLUDED_FEATURE_PREFIXES,
    LABEL_SOURCE_COLUMNS,
    PRICE_SCORE_COLUMNS,
    RETURN_COLUMNS,
    _configured_first,
    _existing,
    _number,
    _optional_path,
    _optional_str,
    _read_csv,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir

def _resolve_news_risk_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    parallel_config = _parallel_config(ml)
    parallel_report = _parallel_report_skeleton(parallel_config)
    output_dir = Path(
        str(
            ml.get(
                "stock_alpha_news_risk_overlay_output_dir",
                "research-results/stock_alpha_news_risk_overlay",
            )
        )
    )
    return {
        "ml": ml,
        "parallel_config": parallel_config,
        "parallel_report": parallel_report,
        "output_dir": output_dir,
    }


def _load_news_risk_research_inputs(
    config: Mapping[str, Any],
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> tuple[Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    del ml
    with _timed_phase(parallel_report, "input_loading"):
        price_path = _locate_price_candidates(config)
        news_path = _locate_news_features(config)
        price_rows = _read_csv(price_path)
        news_rows = _read_csv(news_path)
        _validate_source_rows(price_rows, news_rows, price_path, news_path)
    return price_path, news_path, price_rows, news_rows


def _build_news_risk_overlay_config(ml: Mapping[str, Any]) -> NewsRiskOverlayConfig:
    return NewsRiskOverlayConfig(
        decision_timestamp_column=_optional_str(
            ml.get("stock_alpha_news_risk_overlay_decision_timestamp_column")
        ),
        news_timestamp_preference=tuple(dict.fromkeys((*TIMESTAMP_COLUMNS, *DECISION_TIMESTAMP_COLUMNS))),
        adverse_return_threshold=float(
            ml.get("stock_alpha_news_risk_overlay_adverse_return_threshold", -0.05)
        ),
        block_threshold=float(ml.get("stock_alpha_news_risk_overlay_block_threshold", 0.70)),
        reduce_threshold=float(ml.get("stock_alpha_news_risk_overlay_reduce_threshold", 0.50)),
        reduce_multiplier=float(ml.get("stock_alpha_news_risk_overlay_reduce_multiplier", 0.50)),
        model_version=str(ml.get("stock_alpha_news_risk_overlay_model_version", "news-risk-overlay-research-v1")),
    )


def _build_labeled_news_risk_dataset(
    price_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    overlay_config: NewsRiskOverlayConfig,
    ml: Mapping[str, Any],
    parallel_report: dict[str, Any],
) -> dict[str, Any]:
    with _timed_phase(parallel_report, "point_in_time_join"):
        joined, leakage = join_news_to_stock_alpha_observations(price_rows, news_rows, overlay_config)
        labeled = build_news_risk_labels(joined, overlay_config)
    coverage = _coverage_report(labeled, leakage)
    min_coverage = float(ml.get("stock_alpha_news_risk_overlay_min_coverage_ratio", 0.01))
    if coverage["covered_row_count"] <= 0 or coverage["row_coverage_ratio"] < min_coverage:
        raise ValueError(
            "stock-alpha news risk overlay coverage unavailable: "
            f"covered_row_count={coverage['covered_row_count']} "
            f"row_coverage_ratio={coverage['row_coverage_ratio']:.4f} "
            f"required_min={min_coverage:.4f}"
        )
    if leakage.get("leakage_violation_count", 0):
        raise ValueError("timestamp leakage detected in joined news features")
    return {
        "labeled": labeled,
        "leakage": leakage,
        "coverage": coverage,
    }


def _select_news_risk_features(
    labeled: list[dict[str, Any]],
    ml: Mapping[str, Any],
) -> dict[str, Any]:
    max_features = int(ml.get("stock_alpha_news_risk_overlay_max_features", 48))
    price_score_column = _choose_column(
        labeled,
        _configured_first(ml.get("stock_alpha_news_risk_overlay_price_score_column"), PRICE_SCORE_COLUMNS),
        "price score",
    )
    return_column = _choose_column(
        labeled,
        _configured_first(ml.get("stock_alpha_news_risk_overlay_return_column"), RETURN_COLUMNS),
        "portfolio return",
    )
    price_feature_columns = _limit_features(
        _feature_columns(labeled, include_news=False),
        labeled,
        max_features=max_features,
    )
    price_news_feature_columns = _limit_features(
        _feature_columns(labeled, include_news=True),
        labeled,
        max_features=max_features,
        require_news=True,
    )
    if not price_feature_columns:
        raise ValueError("no numeric price candidate features available for price-only baseline")
    if not any(column.startswith("news_") for column in price_news_feature_columns):
        raise ValueError("no numeric joined news features available for price-plus-news baseline")
    return {
        "price_score_column": price_score_column,
        "return_column": return_column,
        "price_feature_columns": price_feature_columns,
        "price_news_feature_columns": price_news_feature_columns,
    }


def _locate_price_candidates(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    configured = _optional_path(ml.get("stock_alpha_news_risk_overlay_price_candidates_path"))
    if configured:
        return _existing(configured, "configured price candidates")
    output = stock_alpha_output_dir(config)
    candidates = [
        output / "enriched" / "stock_level_model_oos_predictions.csv",
        output / "baseline" / "stock_level_model_oos_predictions.csv",
        output / "stock_level_model_oos_predictions.csv",
        output / "stock_level_prediction_artifacts_enriched.parquet",
        output / "stock_level_prediction_artifacts_enriched.csv",
        output / "stock_level_prediction_artifacts.parquet",
        output / "stock_level_prediction_artifacts.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "historical price-strategy candidates not found; set "
        "ml.stock_alpha_news_risk_overlay_price_candidates_path"
    )


def _locate_news_features(config: Mapping[str, Any]) -> Path:
    ml = dict(config.get("ml", {}) or {})
    configured = _optional_path(
        ml.get("stock_alpha_news_risk_overlay_news_features_path")
        or ml.get("stock_alpha_news_features_path")
    )
    if configured:
        return _existing(configured, "configured news features")
    candidates = [
        Path("reports/ml/benchmark/regime_transformer_meta_ensemble_v1/news_transformer_features_120mo_v1/news_transformer_event_features.csv"),
        stock_alpha_output_dir(config) / "news_features" / "stock_alpha_news_features.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "stock-alpha news features not found; set "
        "ml.stock_alpha_news_risk_overlay_news_features_path"
    )


def _validate_source_rows(
    price_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    price_path: Path,
    news_path: Path,
) -> None:
    if not price_rows:
        raise ValueError(f"price candidates are empty: {price_path}")
    if not news_rows:
        raise ValueError(f"news features are empty: {news_path}")
    price_columns = set(price_rows[0])
    news_columns = set(news_rows[0])
    if "symbol" not in price_columns:
        raise ValueError(f"price candidates missing symbol column: {price_path}")
    if "symbol" not in news_columns:
        raise ValueError(f"news features missing symbol column: {news_path}")
    if not price_columns.intersection(DECISION_TIMESTAMP_COLUMNS):
        raise ValueError(f"price candidates missing decision timestamp column: {price_path}")
    if not news_columns.intersection((*TIMESTAMP_COLUMNS, *DECISION_TIMESTAMP_COLUMNS)):
        raise ValueError(f"news features missing point-in-time timestamp column: {news_path}")
    if not price_columns.intersection(LABEL_SOURCE_COLUMNS):
        raise ValueError(f"price candidates missing configurable adverse-outcome label source: {price_path}")


def _feature_columns(rows: list[Mapping[str, Any]], *, include_news: bool) -> list[str]:
    columns = []
    all_columns = list(dict.fromkeys(key for row in rows for key in row))
    for column in all_columns:
        if column in EXCLUDED_FEATURE_COLUMNS or column in LABEL_SOURCE_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES):
            continue
        if column.startswith("news_") != include_news and column.startswith("news_"):
            continue
        values = [_number(row.get(column)) for row in rows[:200]]
        if any(value is not None and math.isfinite(value) for value in values):
            columns.append(column)
    if include_news:
        price = _feature_columns(rows, include_news=False)
        news = [column for column in columns if column.startswith("news_")]
        return [*price, *news]
    return columns


def _limit_features(
    columns: list[str],
    rows: list[Mapping[str, Any]],
    *,
    max_features: int,
    require_news: bool = False,
) -> list[str]:
    if max_features <= 0 or len(columns) <= max_features:
        return columns
    scored = sorted(
        columns,
        key=lambda column: (
            sum(_number(row.get(column)) is not None for row in rows),
            column.startswith("news_"),
        ),
        reverse=True,
    )
    selected = scored[:max_features]
    if require_news and not any(column.startswith("news_") for column in selected):
        first_news = next((column for column in scored if column.startswith("news_")), None)
        if first_news:
            selected[-1] = first_news
    return list(dict.fromkeys(selected))


def _coverage_report(rows: list[Mapping[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    total = len(rows)
    covered = sum(str(row.get("news_coverage_status")) == "COVERED" for row in rows)
    return {
        "stock_row_count": total,
        "covered_row_count": covered,
        "row_coverage_ratio": covered / max(total, 1),
        "label_positive_rate": sum(int(row.get("news_risk_label", 0)) for row in rows) / max(total, 1),
        "symbol_coverage": audit.get("symbol_coverage", {}),
        "date_coverage": audit.get("date_coverage", {}),
        "future_news_rows_rejected": audit.get("future_news_rows_rejected", 0),
        "leakage_violation_count": audit.get("leakage_violation_count", 0),
    }


def _choose_column(rows: list[Mapping[str, Any]], candidates: Iterable[str], label: str) -> str:
    available = set().union(*(row.keys() for row in rows))
    for column in candidates:
        if column in available and any(_number(row.get(column)) is not None for row in rows):
            return column
    raise ValueError(f"no usable {label} column found; tried {list(candidates)}")


__all__ = [name for name in globals() if not name.startswith("__")]
