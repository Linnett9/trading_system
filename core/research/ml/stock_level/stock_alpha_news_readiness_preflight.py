from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    GUARDRAILS,
    REQUIRED_NEWS_FEATURE_COLUMNS,
    check_news_transformer_readiness,
)
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_output_dir


@dataclass(frozen=True)
class StockAlphaNewsReadinessPreflightPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_readiness_preflight(
    config: Mapping[str, Any],
) -> StockAlphaNewsReadinessPreflightPaths:
    payload = build_stock_alpha_news_readiness_preflight(config)
    output = stock_alpha_output_dir(config)
    paths = StockAlphaNewsReadinessPreflightPaths(
        json_path=output / "stock_alpha_news_readiness_preflight.json",
        markdown_path=output / "stock_alpha_news_readiness_preflight.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_readiness_preflight(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    enable_flag = bool(ml.get("stock_alpha_news_enable_transformer", False))
    features_path = _optional_path(ml.get("stock_alpha_news_features_path"))
    stock_rows_path = _stock_rows_path(ml)
    source_features_exists = bool(features_path and features_path.exists())
    stock_rows_exists = bool(stock_rows_path and stock_rows_path.exists())

    feature_rows = _read_rows_if_present(features_path)
    stock_rows = _read_rows_if_present(stock_rows_path)
    readiness = check_news_transformer_readiness(config, stock_rows) if stock_rows_exists else None
    feature_audit = _nearby_audit_payload(features_path)

    required_columns_missing = _required_columns_missing(feature_rows, source_features_exists)
    blocking_issues: list[str] = []
    warning_issues: list[str] = []

    if not enable_flag:
        blocking_issues.append("stock_alpha_news_enable_transformer_false")
    if not features_path:
        blocking_issues.append("missing_ml.stock_alpha_news_features_path")
    elif not source_features_exists:
        blocking_issues.append("news_features_file_not_found")
    if not stock_rows_path:
        blocking_issues.append("missing_stock_rows_path")
    elif not stock_rows_exists:
        blocking_issues.append("stock_rows_file_not_found")
    if required_columns_missing:
        blocking_issues.append("missing_required_news_feature_columns")
    if feature_rows and not _has_columns(feature_rows, ("rebalance_date", "symbol")):
        blocking_issues.append("missing_key_columns")
    if stock_rows_exists and not _has_columns(stock_rows, ("rebalance_date", "symbol")):
        blocking_issues.append("stock_rows_missing_key_columns")
    if readiness is None:
        warning_issues.append("readiness_unavailable_without_stock_rows")
    elif not readiness.transformer_available and readiness.unavailable_reason:
        blocking_issues.append(readiness.unavailable_reason)
    if readiness is not None and readiness.pit_violation_count > 0:
        blocking_issues.append("news_feature_rows_contain_future_timestamps")
    if int(feature_audit.get("pit_violation_count", 0) or 0) > 0:
        blocking_issues.append("news_features_audit_reports_pit_violations")
    if source_features_exists and "stock_alpha_news_features_audit.json" not in _nearby_audit_names(features_path):
        warning_issues.append("pit_audit_metadata_unavailable")
        blocking_issues.append("pit_audit_metadata_unavailable")

    blocking_issues = _dedupe(blocking_issues)
    warning_issues = _dedupe(warning_issues)
    readiness_available = bool(readiness and readiness.transformer_available)
    safe_to_train = bool(enable_flag and readiness_available and not blocking_issues)

    return {
        "safe_to_train_news_transformer": safe_to_train,
        "readiness_available": readiness_available,
        "enable_flag": enable_flag,
        "source_features_exists": source_features_exists,
        "stock_rows_exists": stock_rows_exists,
        "required_columns_missing": required_columns_missing,
        "blocking_issues": blocking_issues,
        "warning_issues": warning_issues,
        "coverage_summary": _coverage_summary(readiness),
        "pit_audit_summary": _pit_audit_summary(readiness, features_path),
        "row_count": len(feature_rows),
        "symbol_count": len({str(row.get("symbol", "")).strip().upper() for row in feature_rows if str(row.get("symbol", "")).strip()}),
        "date_count": len({str(row.get("rebalance_date", ""))[:10] for row in feature_rows if str(row.get("rebalance_date", "")).strip()}),
        "features_path": str(features_path) if features_path else "",
        "stock_rows_path": str(stock_rows_path) if stock_rows_path else "",
        **GUARDRAILS,
    }


def _stock_rows_path(ml: Mapping[str, Any]) -> Path | None:
    value = ml.get(
        "stock_alpha_news_stock_rows_path",
        ml.get("stock_level_prediction_artifacts_path"),
    )
    return _optional_path(value)


def _optional_path(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def _read_rows_if_present(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return CsvRowRepository().read(path)


def _required_columns_missing(
    rows: list[Mapping[str, Any]],
    source_features_exists: bool,
) -> list[str]:
    if not source_features_exists or not rows:
        return list(REQUIRED_NEWS_FEATURE_COLUMNS)
    return [column for column in REQUIRED_NEWS_FEATURE_COLUMNS if column not in rows[0]]


def _has_columns(rows: list[Mapping[str, Any]], columns: tuple[str, ...]) -> bool:
    return bool(rows) and all(column in rows[0] for column in columns)


def _coverage_summary(readiness: Any) -> dict[str, Any]:
    if readiness is None:
        return {
            "symbol_coverage": 0.0,
            "date_coverage": 0.0,
            "aligned_stock_row_count": 0,
        }
    return {
        "symbol_coverage": readiness.symbol_coverage,
        "date_coverage": readiness.date_coverage,
        "aligned_stock_row_count": readiness.aligned_stock_row_count,
        "feature_row_count": readiness.feature_row_count,
    }


def _pit_audit_summary(readiness: Any, features_path: Path | None) -> dict[str, Any]:
    audit_paths = _nearby_audit_paths(features_path)
    audit_payload = _nearby_audit_payload(features_path)
    return {
        "pit_violation_count": readiness.pit_violation_count if readiness else 0,
        "audit_metadata_available": bool(audit_paths),
        "nearby_audit_files": [path.name for path in audit_paths],
        "audit_pit_violation_count": audit_payload.get("pit_violation_count"),
        "audit_synthetic_news_features_created": audit_payload.get("synthetic_news_features_created"),
        "audit_point_in_time_filters": audit_payload.get("point_in_time_filters", {}),
    }


def _nearby_audit_payload(features_path: Path | None) -> dict[str, Any]:
    audit_paths = _nearby_audit_paths(features_path)
    return _read_audit_payload(audit_paths[0]) if audit_paths else {}


def _nearby_audit_names(features_path: Path | None) -> list[str]:
    return [path.name for path in _nearby_audit_paths(features_path)]


def _nearby_audit_paths(features_path: Path | None) -> list[Path]:
    if features_path is None:
        return []
    candidates = [
        features_path.parent / "news_features" / "stock_alpha_news_features_audit.json",
        features_path.parent / "stock_alpha_news_features_audit.json",
    ]
    return [path for path in candidates if path.exists()]


def _read_audit_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _markdown(payload: Mapping[str, Any]) -> str:
    coverage = dict(payload.get("coverage_summary", {}) or {})
    pit = dict(payload.get("pit_audit_summary", {}) or {})
    blocking = payload.get("blocking_issues", []) or ["none"]
    warnings = payload.get("warning_issues", []) or ["none"]
    return "\n".join(
        [
            "# Stock-Alpha News Readiness Preflight",
            "",
            f"- Safe to train news transformer: {payload['safe_to_train_news_transformer']}",
            f"- Readiness available: {payload['readiness_available']}",
            f"- Enable flag: {payload['enable_flag']}",
            f"- Source features exists: {payload['source_features_exists']}",
            f"- Stock rows exists: {payload['stock_rows_exists']}",
            f"- Feature rows: {payload['row_count']}",
            f"- Symbols: {payload['symbol_count']}",
            f"- Dates: {payload['date_count']}",
            f"- Symbol coverage: {coverage.get('symbol_coverage', 0.0)}",
            f"- Date coverage: {coverage.get('date_coverage', 0.0)}",
            f"- PIT audit metadata available: {pit.get('audit_metadata_available', False)}",
            f"- PIT violation count: {pit.get('pit_violation_count', 0)}",
            "",
            "## Blocking Issues",
            *[f"- {issue}" for issue in blocking],
            "",
            "## Warnings",
            *[f"- {issue}" for issue in warnings],
            "",
            "Research-only preflight. No models were trained and no trading or execution paths were touched.",
        ]
    )
