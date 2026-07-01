from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    REQUIRED_NEWS_AGGREGATE_FEATURES,
    REQUIRED_NEWS_FEATURE_COLUMNS,
)


COUNT_FEATURES = tuple(column for column in REQUIRED_NEWS_AGGREGATE_FEATURES if "count" in column)
SENTIMENT_FEATURES = tuple(column for column in REQUIRED_NEWS_AGGREGATE_FEATURES if "sentiment" in column)
EVENT_COUNT_FEATURES = tuple(column for column in COUNT_FEATURES if column not in {"news_count_1d", "news_count_3d", "news_count_7d"})
LABEL_COLUMNS = ("actual_forward_return_10d", "forward_return_10d", "label", "target")


@dataclass(frozen=True)
class StockAlphaNewsFeatureDiagnosticsPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_feature_diagnostics(config: Mapping[str, Any]) -> StockAlphaNewsFeatureDiagnosticsPaths:
    payload = build_stock_alpha_news_feature_diagnostics(config)
    output = _required_path(config, "stock_alpha_news_feature_diagnostics_report_dir")
    paths = StockAlphaNewsFeatureDiagnosticsPaths(
        output / "stock_alpha_news_feature_diagnostics.json",
        output / "stock_alpha_news_feature_diagnostics.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_feature_diagnostics(config: Mapping[str, Any]) -> dict[str, Any]:
    features_path = _required_path(config, "stock_alpha_news_features_path")
    stock_rows_path = _required_path(config, "stock_alpha_stock_rows_path")
    features_exist = features_path.is_file()
    stock_rows_exist = stock_rows_path.is_file()
    feature_rows = CsvRowRepository().read(features_path) if features_exist else []
    stock_rows = CsvRowRepository().read(stock_rows_path) if stock_rows_exist else []
    columns = list(feature_rows[0]) if feature_rows else []
    missing_required = [column for column in REQUIRED_NEWS_FEATURE_COLUMNS if column not in columns]
    audit_path, audit = _nearby_audit(features_path)
    audit_pit = int(audit.get("pit_violation_count", 0) or 0)
    audit_synthetic = bool(audit.get("synthetic_news_features_created", False))
    blocking = []
    if not features_exist:
        blocking.append("news_features_file_not_found")
    elif not feature_rows:
        blocking.append("news_features_file_empty")
    if not stock_rows_exist:
        blocking.append("stock_rows_file_not_found")
    elif not stock_rows:
        blocking.append("stock_rows_file_empty")
    if feature_rows and missing_required:
        blocking.append("missing_required_news_feature_columns")
    pit = _pit_diagnostics(feature_rows)
    if audit_pit or pit["future_looking_row_count"]:
        blocking.append("pit_violation_detected")
    if audit_synthetic:
        blocking.append("synthetic_news_features_detected")

    coverage = _coverage(feature_rows)
    missingness = _missingness(feature_rows)
    distributions = _distributions(feature_rows)
    usability = _numeric_usability(feature_rows)
    correlations = _correlations(feature_rows, stock_rows)
    warnings = []
    if coverage["sparse_symbols"]:
        warnings.append("sparse_symbol_coverage")
    if coverage["sparse_dates"]:
        warnings.append("sparse_date_coverage")
    if int(audit.get("future_article_excluded_count", 0) or 0):
        warnings.append("future_articles_correctly_excluded")
    suitable = bool(not blocking and usability["all_required_features_numerically_usable"])
    next_action = _next_action(blocking, suitable)
    return {
        "next_action": next_action,
        "blocking_issues": blocking,
        "warning_issues": warnings,
        "input_status": {
            "feature_csv_path": str(features_path),
            "stock_rows_path": str(stock_rows_path),
            "feature_csv_exists": features_exist,
            "stock_rows_exists": stock_rows_exist,
            "feature_row_count": len(feature_rows),
            "stock_row_count": len(stock_rows),
            "feature_symbol_count": _unique_count(feature_rows, "symbol"),
            "stock_symbol_count": _unique_count(stock_rows, "symbol"),
            "feature_date_count": _unique_count(feature_rows, "rebalance_date"),
            "stock_date_count": _unique_count(stock_rows, "rebalance_date"),
            "required_columns_present": not missing_required,
            "missing_required_columns": missing_required,
        },
        "coverage_diagnostics": coverage,
        "missingness_diagnostics": missingness,
        "distribution_diagnostics": distributions,
        "pit_safety_diagnostics": {
            **pit,
            "nearby_feature_audit_path": str(audit_path) if audit_path else "",
            "nearby_feature_audit_available": bool(audit_path),
            "future_article_candidate_count": audit.get("future_article_candidate_count"),
            "future_article_excluded_count": audit.get("future_article_excluded_count"),
            "pit_violation_count": audit_pit,
            "synthetic_news_features_created": audit_synthetic,
        },
        "model_readiness_diagnostics": {
            **usability,
            "suitable_for_disabled_diagnostic_merge_check": suitable,
            "performance_claims_made": False,
        },
        "exploratory_correlations": correlations,
        "inspection_only": True,
        "features_generated": False,
        "files_ingested": False,
        "readiness_invoked": False,
        "diagnostics_invoked": False,
        "model_training_invoked": False,
        "news_transformer_enabled": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _coverage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if _bool(row.get("news_has_coverage_30d"))]
    by_symbol = _group_coverage(rows, "symbol")
    by_date = _group_coverage(rows, "rebalance_date")
    no_news = len(rows) - len(covered)
    return {
        "news_has_coverage_30d_rate": len(covered) / len(rows) if rows else 0.0,
        "coverage_by_symbol": by_symbol,
        "coverage_by_rebalance_date": by_date,
        "no_news_row_count": no_news,
        "no_news_row_rate": no_news / len(rows) if rows else 0.0,
        "sparse_symbols": [key for key, value in by_symbol.items() if value["coverage_rate"] < 0.5],
        "sparse_dates": [key for key, value in by_date.items() if value["coverage_rate"] < 0.5],
    }


def _group_coverage(rows: list[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return {
        name: {"row_count": len(group), "coverage_rate": sum(_bool(row.get("news_has_coverage_30d")) for row in group) / len(group)}
        for name, group in sorted(groups.items()) if name
    }


def _missingness(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    rates = {
        column: sum(_is_missing(row.get(column)) for row in rows) / len(rows) if rows else 0.0
        for column in REQUIRED_NEWS_AGGREGATE_FEATURES
    }
    no_news = [row for row in rows if not _bool(row.get("news_has_coverage_30d"))]
    fake_neutral = sum(
        1 for row in no_news
        if any(_float(row.get(column)) == 0.0 for column in SENTIMENT_FEATURES)
    )
    inconsistent = sum(
        1 for row in rows
        if (not _bool(row.get("news_has_coverage_30d")))
        and any((_float(row.get(column)) or 0.0) > 0 for column in COUNT_FEATURES)
    )
    return {
        "missing_rate_by_feature_column": rates,
        "missing_sentiment_rates": {column: rates[column] for column in SENTIMENT_FEATURES},
        "no_news_fake_neutral_sentiment_row_count": fake_neutral,
        "coverage_flag_inconsistent_row_count": inconsistent,
    }


def _distributions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {column: _summary(rows, column) for column in REQUIRED_NEWS_AGGREGATE_FEATURES}
    extremes = []
    for column, summary in summaries.items():
        if summary["count"] and summary["max"] is not None:
            for row in rows:
                if _float(row.get(column)) == summary["max"]:
                    extremes.append({"column": column, "symbol": row.get("symbol"), "rebalance_date": row.get("rebalance_date"), "value": summary["max"]})
                    break
    return {
        "count_feature_summaries": {column: summaries[column] for column in COUNT_FEATURES},
        "sentiment_feature_summaries": {column: summaries[column] for column in SENTIMENT_FEATURES},
        "event_count_feature_summaries": {column: summaries[column] for column in EVENT_COUNT_FEATURES},
        "extreme_value_rows": extremes,
        "zero_variance_columns": [column for column, value in summaries.items() if value["count"] and value["min"] == value["max"]],
    }


def _summary(rows: list[Mapping[str, Any]], column: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _float(row.get(column))) is not None]
    return {"count": len(values), "min": min(values) if values else None, "max": max(values) if values else None, "mean": mean(values) if values else None}


def _pit_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    timestamp_columns = [column for column in (rows[0] if rows else {}) if ("timestamp" in column.lower() or column.lower().endswith("_at"))]
    future = 0
    for row in rows:
        rebalance = _datetime(row.get("rebalance_date"))
        if rebalance is None:
            continue
        for column in timestamp_columns:
            value = _datetime(row.get(column))
            if value is not None and value > rebalance:
                future += 1
    return {"timestamp_like_columns": timestamp_columns, "future_looking_row_count": future}


def _numeric_usability(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    unusable = [
        column for column in REQUIRED_NEWS_AGGREGATE_FEATURES
        if rows and all(_float(row.get(column)) is None for row in rows)
    ]
    return {"numerically_unusable_required_columns": unusable, "all_required_features_numerically_usable": bool(rows and not unusable)}


def _correlations(features: list[Mapping[str, Any]], stock_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [column for column in LABEL_COLUMNS if stock_rows and column in stock_rows[0]]
    if not labels:
        return {"status": "skipped_labels_absent", "exploratory_only": True, "correlations": {}}
    label = labels[0]
    stock_by_key = {(str(row.get("rebalance_date", ""))[:10], str(row.get("symbol", "")).upper()): row for row in stock_rows}
    result = {}
    for column in REQUIRED_NEWS_AGGREGATE_FEATURES:
        pairs = []
        for row in features:
            stock = stock_by_key.get((str(row.get("rebalance_date", ""))[:10], str(row.get("symbol", "")).upper()))
            x, y = _float(row.get(column)), _float(stock.get(label)) if stock else None
            if x is not None and y is not None:
                pairs.append((x, y))
        result[column] = _correlation(pairs)
    return {"status": "computed", "label_column": label, "exploratory_only": True, "no_alpha_claim": True, "correlations": result}


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _nearby_audit(features_path: Path) -> tuple[Path | None, dict[str, Any]]:
    candidates = [features_path.parent / "news_features" / "stock_alpha_news_features_audit.json", features_path.parent / "stock_alpha_news_features_audit.json"]
    for path in candidates:
        if path.is_file():
            try:
                return path, json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return path, {"pit_violation_count": 1}
    return None, {}


def _next_action(blocking: list[str], suitable: bool) -> str:
    priorities = [
        ("news_features_file_not_found", "provide_news_features"),
        ("stock_rows_file_not_found", "provide_stock_rows"),
        ("missing_required_news_feature_columns", "fix_missing_required_feature_columns"),
        ("pit_violation_detected", "investigate_pit_violation"),
        ("synthetic_news_features_detected", "investigate_synthetic_features"),
    ]
    for issue, action in priorities:
        if issue in blocking:
            return action
    return "ready_for_disabled_diagnostic_merge_check" if suitable else "run_pipeline_preflight"


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _unique_count(rows: list[Mapping[str, Any]], column: str) -> int:
    return len({str(row.get(column, "")).strip().upper() for row in rows if str(row.get(column, "")).strip()})


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _float(value: Any) -> float | None:
    try:
        return float(value) if not _is_missing(value) else None
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
    except ValueError:
        return None


def _markdown(payload: Mapping[str, Any]) -> str:
    status = payload["input_status"]
    pit = payload["pit_safety_diagnostics"]
    return "\n".join([
        "# Stock-Alpha News Feature Diagnostics",
        "",
        f"- Next action: {payload['next_action']}",
        f"- Feature rows: {status['feature_row_count']}",
        f"- Stock rows: {status['stock_row_count']}",
        f"- Required columns present: {status['required_columns_present']}",
        f"- PIT violations: {pit['pit_violation_count']}",
        f"- Future articles correctly excluded: {pit['future_article_excluded_count']}",
        f"- Synthetic features created: {pit['synthetic_news_features_created']}",
        "- Inspection only: true",
        "- Model training invoked: false",
        "- Model diagnostics invoked: false",
        "",
        "Exploratory descriptive diagnostics only. No alpha or performance claim is made.",
    ])
