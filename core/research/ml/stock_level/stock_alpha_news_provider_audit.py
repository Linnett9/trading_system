from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    GUARDRAILS,
    REQUIRED_NEWS_CONTRACT_COLUMNS,
)
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    REQUIRED_NON_EMPTY_FIELDS,
    _apply_provider_column_map,
    _normalize_event_type,
    _parse_utc_timestamp,
    _provider_column_map,
)


@dataclass(frozen=True)
class NewsProviderAuditPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_provider_audit(
    config: Mapping[str, Any],
) -> NewsProviderAuditPaths:
    payload = build_stock_alpha_news_provider_audit(config)
    audit_dir = _required_path(config, "stock_alpha_news_provider_audit_dir")
    paths = NewsProviderAuditPaths(
        json_path=audit_dir / "stock_alpha_news_provider_audit.json",
        markdown_path=audit_dir / "stock_alpha_news_provider_audit.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_provider_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw_path = Path(str(ml.get("stock_alpha_news_raw_path", "")))
    provider_column_map = _provider_column_map(ml)
    thresholds = _thresholds(ml)
    if not raw_path.exists():
        return _payload(
            raw_path=raw_path,
            rows=[],
            provider_column_map=provider_column_map,
            mapping_audit=_empty_mapping_audit(provider_column_map),
            thresholds=thresholds,
            blocking_issues=[f"raw source file not found: {raw_path}"],
        )

    raw_rows = CsvRowRepository().read(raw_path)
    mapped_rows, mapping_audit = _apply_provider_column_map(raw_rows, provider_column_map)
    blocking_issues: list[str] = []
    if mapping_audit["missing_mapped_provider_columns"]:
        blocking_issues.append(
            "missing mapped provider columns: "
            + ", ".join(mapping_audit["missing_mapped_provider_columns"])
        )
    missing_canonical = [
        column for column in REQUIRED_NEWS_CONTRACT_COLUMNS if mapped_rows and column not in mapped_rows[0]
    ]
    if missing_canonical:
        blocking_issues.append(
            "missing required canonical columns after mapping: "
            + ", ".join(missing_canonical)
        )
    return _payload(
        raw_path=raw_path,
        rows=mapped_rows,
        provider_column_map=provider_column_map,
        mapping_audit=mapping_audit,
        thresholds=thresholds,
        blocking_issues=blocking_issues,
    )


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _thresholds(ml: Mapping[str, Any]) -> dict[str, float]:
    return {
        "min_symbol_count": float(ml.get("stock_alpha_news_provider_audit_min_symbol_count", 1)),
        "min_article_count": float(ml.get("stock_alpha_news_provider_audit_min_article_count", 1)),
        "max_missing_body_rate": float(ml.get("stock_alpha_news_provider_audit_max_missing_body_rate", 0.25)),
        "max_duplicate_headline_rate": float(ml.get("stock_alpha_news_provider_audit_max_duplicate_headline_rate", 0.10)),
        "max_invalid_timestamp_rate": float(ml.get("stock_alpha_news_provider_audit_max_invalid_timestamp_rate", 0.0)),
        "max_ingested_before_published_rate": float(ml.get("stock_alpha_news_provider_audit_max_ingested_before_published_rate", 0.0)),
    }


def _empty_mapping_audit(provider_column_map: Mapping[str, str]) -> dict[str, Any]:
    return {
        "provider_column_map_used": bool(provider_column_map),
        "provider_column_map": dict(provider_column_map),
        "missing_mapped_provider_columns": [],
        "unmapped_provider_columns": [],
        "canonical_columns_after_mapping": [],
    }


def _payload(
    *,
    raw_path: Path,
    rows: list[Mapping[str, Any]],
    provider_column_map: Mapping[str, str],
    mapping_audit: Mapping[str, Any],
    thresholds: Mapping[str, float],
    blocking_issues: list[str],
) -> dict[str, Any]:
    raw_row_count = len(rows)
    symbols = [_clean_symbol(row.get("symbol")) for row in rows if _clean_symbol(row.get("symbol"))]
    sources = [str(row.get("source", "")).strip() for row in rows if str(row.get("source", "")).strip()]
    languages = [str(row.get("language", "")).strip().lower() for row in rows if str(row.get("language", "")).strip()]
    event_types = [_normalize_event_type(row.get("event_type", "")) for row in rows]
    article_ids = [str(row.get("article_id", "")).strip() for row in rows if str(row.get("article_id", "")).strip()]
    headlines = [str(row.get("headline", "")).strip().lower() for row in rows if str(row.get("headline", "")).strip()]
    missing_required = {
        field: sum(1 for row in rows if not str(row.get(field, "")).strip())
        for field in REQUIRED_NON_EMPTY_FIELDS
    }
    missing_body_count = sum(1 for row in rows if not str(row.get("body_or_summary", "")).strip())
    duplicate_article_id_count = _duplicate_count(article_ids)
    duplicate_headline_count = _duplicate_count(headlines)
    invalid_timestamp_count = 0
    ingested_before_published_count = 0
    published_values = []
    ingested_values = []
    for row in rows:
        published = _parse_utc_timestamp(row.get("published_at_utc"))
        ingested = _parse_utc_timestamp(row.get("ingested_at"))
        if published is None or ingested is None:
            invalid_timestamp_count += 1
            continue
        published_values.append(published)
        ingested_values.append(ingested)
        if ingested < published:
            ingested_before_published_count += 1

    duplicate_headline_rate = _rate(duplicate_headline_count, raw_row_count)
    missing_body_rate = _rate(missing_body_count, raw_row_count)
    invalid_timestamp_rate = _rate(invalid_timestamp_count, raw_row_count)
    ingested_before_published_rate = _rate(ingested_before_published_count, raw_row_count)
    warning_issues: list[str] = []
    if missing_body_count:
        warning_issues.append(f"missing body_or_summary rows: {missing_body_count}")
    if duplicate_headline_count:
        warning_issues.append(f"duplicate headline rows: {duplicate_headline_count}")
    blocking_issues = list(blocking_issues)
    if raw_row_count < thresholds["min_article_count"]:
        blocking_issues.append("article count below minimum")
    if len(set(symbols)) < thresholds["min_symbol_count"]:
        blocking_issues.append("symbol count below minimum")
    if any(missing_required.values()):
        blocking_issues.append("missing required canonical fields")
    if invalid_timestamp_rate > thresholds["max_invalid_timestamp_rate"]:
        blocking_issues.append("invalid timestamp rate above maximum")
    if ingested_before_published_rate > thresholds["max_ingested_before_published_rate"]:
        blocking_issues.append("ingested-before-published rate above maximum")
    if missing_body_rate > thresholds["max_missing_body_rate"]:
        blocking_issues.append("missing body_or_summary rate above maximum")
    if duplicate_headline_rate > thresholds["max_duplicate_headline_rate"]:
        blocking_issues.append("duplicate headline rate above maximum")

    return {
        "safe_for_pit_research": not blocking_issues,
        "blocking_issues": list(dict.fromkeys(blocking_issues)),
        "warning_issues": list(dict.fromkeys(warning_issues)),
        "raw_path": str(raw_path),
        "provider_column_map_used": bool(provider_column_map),
        "provider_column_map": dict(provider_column_map),
        "missing_mapped_provider_columns": list(mapping_audit.get("missing_mapped_provider_columns", [])),
        "raw_row_count": raw_row_count,
        "article_id_count": len(set(article_ids)),
        "duplicate_article_id_count": duplicate_article_id_count,
        "duplicate_headline_count": duplicate_headline_count,
        "duplicate_headline_rate": duplicate_headline_rate,
        "symbol_count": len(set(symbols)),
        "source_count": len(set(sources)),
        "language_counts": _counts(languages),
        "event_type_counts": _counts(event_types),
        "article_count_by_symbol": _counts(symbols),
        "article_count_by_source": _counts(sources),
        "earliest_published_at_utc": min(published_values).isoformat().replace("+00:00", "Z") if published_values else None,
        "latest_published_at_utc": max(published_values).isoformat().replace("+00:00", "Z") if published_values else None,
        "earliest_ingested_at": min(ingested_values).isoformat().replace("+00:00", "Z") if ingested_values else None,
        "latest_ingested_at": max(ingested_values).isoformat().replace("+00:00", "Z") if ingested_values else None,
        "missing_required_field_counts": missing_required,
        "missing_body_or_summary_count": missing_body_count,
        "missing_body_or_summary_rate": missing_body_rate,
        "invalid_timestamp_count": invalid_timestamp_count,
        "invalid_timestamp_rate": invalid_timestamp_rate,
        "ingested_before_published_count": ingested_before_published_count,
        "ingested_before_published_rate": ingested_before_published_rate,
        "sentiment_present_count": _present_count(rows, "sentiment_score"),
        "sentiment_missing_count": raw_row_count - _present_count(rows, "sentiment_score"),
        "relevance_present_count": _present_count(rows, "relevance_score"),
        "novelty_present_count": _present_count(rows, "novelty_score"),
        **GUARDRAILS,
    }


def _clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _present_count(rows: list[Mapping[str, Any]], column: str) -> int:
    return sum(1 for row in rows if str(row.get(column, "")).strip())


def _duplicate_count(values: list[str]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for value in values:
        if value in seen:
            duplicates += 1
        else:
            seen.add(value)
    return duplicates


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _markdown(payload: Mapping[str, Any]) -> str:
    blocking = payload.get("blocking_issues", []) or ["none"]
    warnings = payload.get("warning_issues", []) or ["none"]
    return "\n".join(
        [
            "# Stock-Alpha News Provider Audit",
            "",
            f"- Safe for PIT research: {payload.get('safe_for_pit_research', False)}",
            f"- Raw path: {payload.get('raw_path', '')}",
            f"- Raw rows: {payload.get('raw_row_count', 0)}",
            f"- Article IDs: {payload.get('article_id_count', 0)}",
            f"- Symbols: {payload.get('symbol_count', 0)}",
            f"- Sources: {payload.get('source_count', 0)}",
            f"- Duplicate article IDs: {payload.get('duplicate_article_id_count', 0)}",
            f"- Duplicate headline rate: {payload.get('duplicate_headline_rate', 0.0)}",
            f"- Invalid timestamp rate: {payload.get('invalid_timestamp_rate', 0.0)}",
            f"- Ingested-before-published rate: {payload.get('ingested_before_published_rate', 0.0)}",
            f"- Provider column map used: {payload.get('provider_column_map_used', False)}",
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
