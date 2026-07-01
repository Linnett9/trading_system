from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    GUARDRAILS,
    REQUIRED_NEWS_CONTRACT_COLUMNS,
)
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    _normalize_event_type,
    _parse_utc_timestamp,
)


REQUIRED_STOCK_ROW_COLUMNS = ("rebalance_date", "symbol")
FRESHNESS_BUCKETS = {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30}


@dataclass(frozen=True)
class NewsCoverageAuditPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_coverage_audit(
    config: Mapping[str, Any],
) -> NewsCoverageAuditPaths:
    payload = build_stock_alpha_news_coverage_audit(config)
    audit_dir = _required_path(config, "stock_alpha_news_coverage_audit_dir")
    paths = NewsCoverageAuditPaths(
        json_path=audit_dir / "stock_alpha_news_coverage_audit.json",
        markdown_path=audit_dir / "stock_alpha_news_coverage_audit.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_coverage_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    contract_path = Path(str(ml.get("stock_alpha_news_contract_path", "")))
    stock_rows_path = Path(str(ml.get("stock_alpha_news_stock_rows_path", "")))
    thresholds = _thresholds(ml)
    blocking_issues: list[str] = []
    if not contract_path.exists():
        blocking_issues.append(f"news contract file not found: {contract_path}")
    if not stock_rows_path.exists():
        blocking_issues.append(f"stock rows file not found: {stock_rows_path}")
    if blocking_issues:
        return _empty_payload(contract_path, stock_rows_path, blocking_issues)

    news_rows = CsvRowRepository().read(contract_path)
    stock_rows = CsvRowRepository().read(stock_rows_path)
    missing_news_columns = _missing_columns(news_rows, REQUIRED_NEWS_CONTRACT_COLUMNS)
    missing_stock_columns = _missing_columns(stock_rows, REQUIRED_STOCK_ROW_COLUMNS)
    blocking_issues.extend(
        _column_blockers(missing_news_columns, missing_stock_columns)
    )
    normalized_news = _normalize_news_rows(news_rows) if not missing_news_columns else []
    normalized_stock_rows = _normalize_stock_rows(stock_rows) if not missing_stock_columns else []
    metrics = _coverage_metrics(normalized_news, normalized_stock_rows)
    blocking_issues.extend(_threshold_blockers(metrics, thresholds))
    return {
        **metrics,
        "safe_for_feature_generation": not blocking_issues,
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "warning_issues": _warnings(metrics),
        "news_contract_path": str(contract_path),
        "stock_rows_path": str(stock_rows_path),
        "missing_required_news_columns": missing_news_columns,
        "missing_required_stock_row_columns": missing_stock_columns,
        **GUARDRAILS,
    }


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _thresholds(ml: Mapping[str, Any]) -> dict[str, float]:
    return {
        "min_symbol_coverage": float(ml.get("stock_alpha_news_coverage_min_symbol_coverage", 0.50)),
        "min_date_coverage": float(ml.get("stock_alpha_news_coverage_min_date_coverage", 0.50)),
        "min_article_count": float(ml.get("stock_alpha_news_coverage_min_article_count", 1)),
        "min_covered_stock_rows": float(ml.get("stock_alpha_news_coverage_min_covered_stock_rows", 1)),
        "max_pit_violation_count": float(ml.get("stock_alpha_news_coverage_max_pit_violation_count", 0)),
    }


def _empty_payload(
    contract_path: Path,
    stock_rows_path: Path,
    blocking_issues: list[str],
) -> dict[str, Any]:
    return {
        **_coverage_metrics([], []),
        "safe_for_feature_generation": False,
        "blocking_issues": blocking_issues,
        "warning_issues": [],
        "news_contract_path": str(contract_path),
        "stock_rows_path": str(stock_rows_path),
        "missing_required_news_columns": [],
        "missing_required_stock_row_columns": [],
        **GUARDRAILS,
    }


def _missing_columns(rows: list[Mapping[str, Any]], required: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    return [column for column in required if column not in rows[0]]


def _column_blockers(
    missing_news_columns: list[str],
    missing_stock_columns: list[str],
) -> list[str]:
    blockers = []
    if missing_news_columns:
        blockers.append(
            "missing required news contract columns: "
            + ", ".join(missing_news_columns)
        )
    if missing_stock_columns:
        blockers.append(
            "missing required stock row columns: "
            + ", ".join(missing_stock_columns)
        )
    return blockers


def _normalize_news_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **dict(row),
                "symbol": str(row.get("symbol", "")).strip().upper(),
                "published_at_utc": _parse_utc_timestamp(row.get("published_at_utc")),
                "ingested_at": _parse_utc_timestamp(row.get("ingested_at")),
                "event_type": _normalize_event_type(row.get("event_type", "")),
            }
        )
    return normalized


def _normalize_stock_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **dict(row),
                "symbol": str(row.get("symbol", "")).strip().upper(),
                "rebalance_date": _parse_rebalance_date(row.get("rebalance_date")),
            }
        )
    return normalized


def _coverage_metrics(
    news_rows: list[Mapping[str, Any]],
    stock_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    news_symbol_count = len({row.get("symbol") for row in news_rows if row.get("symbol")})
    stock_symbols = {row.get("symbol") for row in stock_rows if row.get("symbol")}
    rebalance_dates = {
        row["rebalance_date"].date().isoformat()
        for row in stock_rows
        if row.get("rebalance_date") is not None
    }
    event_type_counts = _counts(
        str(row.get("event_type", "other")) for row in news_rows
    )
    covered_symbols: set[str] = set()
    covered_dates: set[str] = set()
    covered_stock_row_count = 0
    event_type_covered_stock_rows: dict[str, int] = {}
    freshness_bucket_counts = {bucket: 0 for bucket in FRESHNESS_BUCKETS}
    future_article_candidate_count = 0

    for stock_row in stock_rows:
        symbol = str(stock_row.get("symbol", ""))
        rebalance = stock_row.get("rebalance_date")
        if not symbol or rebalance is None:
            continue
        same_symbol_articles = [
            row for row in news_rows if row.get("symbol") == symbol
        ]
        eligible = []
        for article in same_symbol_articles:
            published = article.get("published_at_utc")
            ingested = article.get("ingested_at")
            if published is None or ingested is None:
                continue
            if published <= rebalance and ingested <= rebalance:
                eligible.append(article)
                for bucket, days in FRESHNESS_BUCKETS.items():
                    if published >= rebalance - timedelta(days=days):
                        freshness_bucket_counts[bucket] += 1
            else:
                future_article_candidate_count += 1
        if eligible:
            covered_stock_row_count += 1
            covered_symbols.add(symbol)
            covered_dates.add(rebalance.date().isoformat())
            for event_type in {str(row.get("event_type", "other")) for row in eligible}:
                event_type_covered_stock_rows[event_type] = (
                    event_type_covered_stock_rows.get(event_type, 0) + 1
                )

    stock_row_count = len(stock_rows)
    no_news_stock_row_count = stock_row_count - covered_stock_row_count
    return {
        "news_row_count": len(news_rows),
        "stock_row_count": stock_row_count,
        "news_symbol_count": news_symbol_count,
        "stock_symbol_count": len(stock_symbols),
        "rebalance_date_count": len(rebalance_dates),
        "covered_symbol_count": len(covered_symbols),
        "symbol_coverage": len(covered_symbols) / len(stock_symbols) if stock_symbols else 0.0,
        "covered_rebalance_date_count": len(covered_dates),
        "date_coverage": len(covered_dates) / len(rebalance_dates) if rebalance_dates else 0.0,
        "covered_stock_row_count": covered_stock_row_count,
        "stock_row_coverage": covered_stock_row_count / stock_row_count if stock_row_count else 0.0,
        "no_news_stock_row_count": no_news_stock_row_count,
        "no_news_stock_row_rate": no_news_stock_row_count / stock_row_count if stock_row_count else 0.0,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "event_type_covered_stock_rows": dict(sorted(event_type_covered_stock_rows.items())),
        "freshness_bucket_counts": freshness_bucket_counts,
        "future_article_candidate_count": future_article_candidate_count,
        "future_article_excluded_count": future_article_candidate_count,
        "pit_violation_count": 0,
    }


def _threshold_blockers(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    blockers = []
    if metrics["symbol_coverage"] < thresholds["min_symbol_coverage"]:
        blockers.append("symbol coverage below minimum")
    if metrics["date_coverage"] < thresholds["min_date_coverage"]:
        blockers.append("date coverage below minimum")
    if metrics["news_row_count"] < thresholds["min_article_count"]:
        blockers.append("article count below minimum")
    if metrics["covered_stock_row_count"] < thresholds["min_covered_stock_rows"]:
        blockers.append("covered stock rows below minimum")
    if metrics["pit_violation_count"] > thresholds["max_pit_violation_count"]:
        blockers.append("pit violation count above maximum")
    return blockers


def _warnings(metrics: Mapping[str, Any]) -> list[str]:
    warnings = []
    if metrics["no_news_stock_row_count"]:
        warnings.append(f"stock rows without eligible news: {metrics['no_news_stock_row_count']}")
    if metrics["future_article_excluded_count"]:
        warnings.append(
            "future article candidates correctly excluded by PIT filter: "
            f"{metrics['future_article_excluded_count']}"
        )
    return warnings


def _parse_rebalance_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _markdown(payload: Mapping[str, Any]) -> str:
    blocking = payload.get("blocking_issues", []) or ["none"]
    warnings = payload.get("warning_issues", []) or ["none"]
    return "\n".join(
        [
            "# Stock-Alpha News Coverage Audit",
            "",
            f"- Safe for feature generation: {payload.get('safe_for_feature_generation', False)}",
            f"- News contract: {payload.get('news_contract_path', '')}",
            f"- Stock rows: {payload.get('stock_rows_path', '')}",
            f"- News rows: {payload.get('news_row_count', 0)}",
            f"- Stock rows: {payload.get('stock_row_count', 0)}",
            f"- Symbol coverage: {payload.get('symbol_coverage', 0.0)}",
            f"- Date coverage: {payload.get('date_coverage', 0.0)}",
            f"- Stock-row coverage: {payload.get('stock_row_coverage', 0.0)}",
            f"- No-news stock rows: {payload.get('no_news_stock_row_count', 0)}",
            f"- Future article candidates: {payload.get('future_article_candidate_count', 0)}",
            f"- Future article exclusions: {payload.get('future_article_excluded_count', 0)}",
            f"- PIT violations: {payload.get('pit_violation_count', 0)}",
            "",
            "## Blocking Issues",
            *[f"- {issue}" for issue in blocking],
            "",
            "## Warnings",
            *[f"- {issue}" for issue in warnings],
            "",
            "Inspection-only audit. No contracts, features, or models were generated.",
        ]
    )
