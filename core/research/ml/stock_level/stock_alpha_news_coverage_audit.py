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
from core.research.ml.stock_level.stock_alpha_news_pit_policy import (
    PROVIDER_AVAILABLE_AT,
    StockAlphaNewsPitPolicy,
    article_is_pit_eligible,
    article_pit_exclusion_flags,
    enrich_news_row_with_pit_timestamps,
    pit_policy_payload,
    resolve_stock_alpha_news_pit_policy,
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
    pit_policy = resolve_stock_alpha_news_pit_policy(config)
    blocking_issues: list[str] = []
    if not contract_path.exists():
        blocking_issues.append(f"news contract file not found: {contract_path}")
    if not stock_rows_path.exists():
        blocking_issues.append(f"stock rows file not found: {stock_rows_path}")
    if blocking_issues:
        return _empty_payload(contract_path, stock_rows_path, blocking_issues, pit_policy)

    news_rows = CsvRowRepository().read(contract_path)
    stock_rows = CsvRowRepository().read(stock_rows_path)
    missing_news_columns = _missing_columns(news_rows, REQUIRED_NEWS_CONTRACT_COLUMNS)
    missing_stock_columns = _missing_columns(stock_rows, REQUIRED_STOCK_ROW_COLUMNS)
    blocking_issues.extend(
        _column_blockers(missing_news_columns, missing_stock_columns)
    )
    normalized_news = _normalize_news_rows(news_rows, pit_policy) if not missing_news_columns else []
    normalized_stock_rows = _normalize_stock_rows(stock_rows) if not missing_stock_columns else []
    metrics = _coverage_metrics(normalized_news, normalized_stock_rows, pit_policy)
    blocking_issues.extend(_threshold_blockers(metrics, thresholds, pit_policy))
    return {
        **metrics,
        "safe_for_feature_generation": not blocking_issues,
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "warning_issues": _warnings(metrics),
        "news_contract_path": str(contract_path),
        "stock_rows_path": str(stock_rows_path),
        "missing_required_news_columns": missing_news_columns,
        "missing_required_stock_row_columns": missing_stock_columns,
        **pit_policy_payload(pit_policy),
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
    pit_policy: StockAlphaNewsPitPolicy,
) -> dict[str, Any]:
    return {
        **_coverage_metrics([], [], pit_policy),
        "safe_for_feature_generation": False,
        "blocking_issues": blocking_issues,
        "warning_issues": [],
        "news_contract_path": str(contract_path),
        "stock_rows_path": str(stock_rows_path),
        "missing_required_news_columns": [],
        "missing_required_stock_row_columns": [],
        **pit_policy_payload(pit_policy),
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


def _normalize_news_rows(
    rows: list[Mapping[str, Any]],
    pit_policy: StockAlphaNewsPitPolicy,
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            enrich_news_row_with_pit_timestamps(
                {
                    **dict(row),
                    "symbol": str(row.get("symbol", "")).strip().upper(),
                    "published_at_utc": _parse_utc_timestamp(row.get("published_at_utc")),
                    "ingested_at": _parse_utc_timestamp(row.get("ingested_at")),
                    "event_type": _normalize_event_type(row.get("event_type", "")),
                },
                pit_policy,
            )
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
    pit_policy: StockAlphaNewsPitPolicy,
) -> dict[str, Any]:
    published_values = [
        row.get("published_at_utc")
        for row in news_rows
        if row.get("published_at_utc") is not None
    ]
    ingested_values = [
        row.get("ingested_at")
        for row in news_rows
        if row.get("ingested_at") is not None
    ]
    collected_values = [
        row.get("collected_at_utc")
        for row in news_rows
        if row.get("collected_at_utc") is not None
    ]
    available_values = [
        row.get("available_at_utc")
        for row in news_rows
        if row.get("available_at_utc") is not None
    ]
    rebalance_values = [
        row.get("rebalance_date")
        for row in stock_rows
        if row.get("rebalance_date") is not None
    ]
    news_symbol_count = len({row.get("symbol") for row in news_rows if row.get("symbol")})
    stock_symbols = {row.get("symbol") for row in stock_rows if row.get("symbol")}
    news_symbols = {row.get("symbol") for row in news_rows if row.get("symbol")}
    pre_pit_symbol_overlap = stock_symbols & news_symbols
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
    same_symbol_stock_article_pair_count = 0
    published_after_rebalance_count = 0
    ingested_after_rebalance_count = 0
    available_after_rebalance_count = 0
    eligibility_after_rebalance_count = 0
    published_and_ingested_after_rebalance_count = 0
    published_and_available_after_rebalance_count = 0

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
            same_symbol_stock_article_pair_count += 1
            if published is None or ingested is None:
                continue
            if article_is_pit_eligible(article, rebalance, pit_policy):
                eligible.append(article)
                for bucket, days in FRESHNESS_BUCKETS.items():
                    if published >= rebalance - timedelta(days=days):
                        freshness_bucket_counts[bucket] += 1
            else:
                future_article_candidate_count += 1
                flags = article_pit_exclusion_flags(article, rebalance, pit_policy)
                if flags["published_after_rebalance"]:
                    published_after_rebalance_count += 1
                if flags["collected_after_rebalance"]:
                    ingested_after_rebalance_count += 1
                if flags["available_after_rebalance"]:
                    available_after_rebalance_count += 1
                if flags["eligibility_after_rebalance"]:
                    eligibility_after_rebalance_count += 1
                if flags["published_after_rebalance"] and flags["collected_after_rebalance"]:
                    published_and_ingested_after_rebalance_count += 1
                if flags["published_after_rebalance"] and flags["available_after_rebalance"]:
                    published_and_available_after_rebalance_count += 1
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
        "pre_pit_symbol_overlap_count": len(pre_pit_symbol_overlap),
        "pre_pit_symbol_overlap": sorted(pre_pit_symbol_overlap),
        "rebalance_date_count": len(rebalance_dates),
        "news_published_at_utc_min": _iso_min(published_values),
        "news_published_at_utc_max": _iso_max(published_values),
        "news_ingested_at_min": _iso_min(ingested_values),
        "news_ingested_at_max": _iso_max(ingested_values),
        "news_collected_at_utc_min": _iso_min(collected_values),
        "news_collected_at_utc_max": _iso_max(collected_values),
        "news_available_at_utc_min": _iso_min(available_values),
        "news_available_at_utc_max": _iso_max(available_values),
        "stock_rebalance_date_min": _date_min(rebalance_values),
        "stock_rebalance_date_max": _date_max(rebalance_values),
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
        "same_symbol_stock_article_pair_count": same_symbol_stock_article_pair_count,
        "future_article_candidate_count": future_article_candidate_count,
        "future_article_excluded_count": future_article_candidate_count,
        "published_after_rebalance_count": published_after_rebalance_count,
        "ingested_after_rebalance_count": ingested_after_rebalance_count,
        "collected_after_rebalance_count": ingested_after_rebalance_count,
        "available_after_rebalance_count": available_after_rebalance_count,
        "eligibility_after_rebalance_count": eligibility_after_rebalance_count,
        "published_and_ingested_after_rebalance_count": published_and_ingested_after_rebalance_count,
        "published_and_available_after_rebalance_count": published_and_available_after_rebalance_count,
        "pit_violation_count": 0,
    }


def _threshold_blockers(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
    pit_policy: StockAlphaNewsPitPolicy,
) -> list[str]:
    blockers = []
    if (
        pit_policy.mode == PROVIDER_AVAILABLE_AT
        and not pit_policy.historical_provider_availability_assumed
    ):
        blockers.append(
            "provider_available_at PIT policy requires "
            "stock_alpha_news_historical_provider_availability_enabled=true"
        )
    if metrics["symbol_coverage"] < thresholds["min_symbol_coverage"]:
        blockers.append("symbol coverage below minimum")
    if metrics["date_coverage"] < thresholds["min_date_coverage"]:
        blockers.append("date coverage below minimum")
    if metrics["news_row_count"] < thresholds["min_article_count"]:
        blockers.append("article count below minimum")
    if metrics["covered_stock_row_count"] < thresholds["min_covered_stock_rows"]:
        blockers.append("covered stock rows below minimum")
    if (
        metrics["same_symbol_stock_article_pair_count"]
        and metrics["covered_stock_row_count"] == 0
    ):
        if (
            pit_policy.mode == PROVIDER_AVAILABLE_AT
            and metrics["available_after_rebalance_count"]
            == metrics["same_symbol_stock_article_pair_count"]
        ):
            blockers.append("all same-symbol articles fail PIT available_at alignment")
        elif (
            metrics["ingested_after_rebalance_count"]
            == metrics["same_symbol_stock_article_pair_count"]
            and pit_policy.mode != PROVIDER_AVAILABLE_AT
        ):
            blockers.append("all same-symbol articles fail PIT ingested_at alignment")
        elif (
            metrics["published_after_rebalance_count"]
            == metrics["same_symbol_stock_article_pair_count"]
        ):
            blockers.append("all same-symbol articles fail PIT published_at alignment")
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
    if metrics.get("published_after_rebalance_count", 0):
        warnings.append(
            "same-symbol article pairs excluded because published_at_utc is after "
            f"rebalance_date: {metrics['published_after_rebalance_count']}"
        )
    if metrics.get("ingested_after_rebalance_count", 0):
        warnings.append(
            "same-symbol article pairs excluded because ingested_at is after "
            f"rebalance_date: {metrics['ingested_after_rebalance_count']}"
        )
    if metrics.get("available_after_rebalance_count", 0):
        warnings.append(
            "same-symbol article pairs excluded because available_at_utc is after "
            f"rebalance_date: {metrics['available_after_rebalance_count']}"
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


def _iso_min(values: list[datetime]) -> str | None:
    return min(values).isoformat().replace("+00:00", "Z") if values else None


def _iso_max(values: list[datetime]) -> str | None:
    return max(values).isoformat().replace("+00:00", "Z") if values else None


def _date_min(values: list[datetime]) -> str | None:
    return min(values).date().isoformat() if values else None


def _date_max(values: list[datetime]) -> str | None:
    return max(values).date().isoformat() if values else None


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
            f"- PIT policy: {payload.get('pit_policy', 'strict_collected_at')}",
            f"- Eligibility timestamp: {payload.get('eligibility_timestamp_field', 'collected_at_utc')}",
            f"- Availability lag hours: {payload.get('availability_lag_hours', 0.0)}",
            f"- Historical provider availability assumed: {payload.get('historical_provider_availability_assumed', False)}",
            f"- Production PIT validated: {payload.get('production_pit_validated', False)}",
            f"- News published range: {payload.get('news_published_at_utc_min') or 'n/a'} to {payload.get('news_published_at_utc_max') or 'n/a'}",
            f"- News ingested range: {payload.get('news_ingested_at_min') or 'n/a'} to {payload.get('news_ingested_at_max') or 'n/a'}",
            f"- News available range: {payload.get('news_available_at_utc_min') or 'n/a'} to {payload.get('news_available_at_utc_max') or 'n/a'}",
            f"- Stock rebalance range: {payload.get('stock_rebalance_date_min') or 'n/a'} to {payload.get('stock_rebalance_date_max') or 'n/a'}",
            f"- Pre-PIT symbol overlap: {payload.get('pre_pit_symbol_overlap_count', 0)}",
            f"- Symbol coverage: {payload.get('symbol_coverage', 0.0)}",
            f"- Date coverage: {payload.get('date_coverage', 0.0)}",
            f"- Stock-row coverage: {payload.get('stock_row_coverage', 0.0)}",
            f"- No-news stock rows: {payload.get('no_news_stock_row_count', 0)}",
            f"- Same-symbol stock/article pairs: {payload.get('same_symbol_stock_article_pair_count', 0)}",
            f"- Future article candidates: {payload.get('future_article_candidate_count', 0)}",
            f"- Future article exclusions: {payload.get('future_article_excluded_count', 0)}",
            f"- Published-after-rebalance exclusions: {payload.get('published_after_rebalance_count', 0)}",
            f"- Ingested-after-rebalance exclusions: {payload.get('ingested_after_rebalance_count', 0)}",
            f"- Available-after-rebalance exclusions: {payload.get('available_after_rebalance_count', 0)}",
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
