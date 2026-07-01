from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import (
    GUARDRAILS,
    REQUIRED_NEWS_CONTRACT_COLUMNS,
)


REQUIRED_INPUT_COLUMNS = REQUIRED_NEWS_CONTRACT_COLUMNS
REQUIRED_NON_EMPTY_FIELDS = (
    "article_id",
    "symbol",
    "published_at_utc",
    "source",
    "headline",
    "ingested_at",
)
EVENT_TYPE_BUCKETS = (
    "earnings",
    "analyst",
    "guidance",
    "litigation",
    "mna",
    "macro",
    "product",
    "management",
    "other",
)


@dataclass(frozen=True)
class NewsContractIngestPaths:
    contract_path: Path
    audit_json_path: Path
    audit_markdown_path: Path


def write_stock_alpha_news_contract_ingest(
    config: Mapping[str, Any],
) -> NewsContractIngestPaths:
    ml = dict(config.get("ml", {}) or {})
    raw_path = _required_path(ml, "stock_alpha_news_raw_path")
    contract_path = _required_path(ml, "stock_alpha_news_contract_path")
    audit_dir = _required_path(ml, "stock_alpha_news_contract_ingest_audit_dir")
    provider_column_map = _provider_column_map(ml)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"stock-alpha news raw source file not found: {raw_path}"
        )

    raw_rows = CsvRowRepository().read(raw_path)
    normalized_rows, audit = build_stock_alpha_news_contract_rows(
        raw_rows,
        provider_column_map=provider_column_map,
    )
    audit.update(
        {
            "raw_source_path": str(raw_path),
            "output_contract_path": str(contract_path),
            **GUARDRAILS,
        }
    )
    if audit["missing_mapped_provider_columns"]:
        _write_audit(audit_dir, audit)
        raise ValueError(
            "stock-alpha news raw source missing mapped provider columns: "
            + ", ".join(audit["missing_mapped_provider_columns"])
        )
    if audit["missing_required_column_names"]:
        _write_audit(audit_dir, audit)
        raise ValueError(
            "stock-alpha news raw source missing required columns: "
            + ", ".join(audit["missing_required_column_names"])
        )

    writer = ResearchArtifactWriter()
    writer.write_csv(
        contract_path,
        normalized_rows,
        fieldnames=REQUIRED_NEWS_CONTRACT_COLUMNS,
    )
    paths = _write_audit(audit_dir, audit)
    return NewsContractIngestPaths(
        contract_path=contract_path,
        audit_json_path=paths.audit_json_path,
        audit_markdown_path=paths.audit_markdown_path,
    )


def build_stock_alpha_news_contract_rows(
    raw_rows: list[Mapping[str, Any]],
    *,
    provider_column_map: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = dict(provider_column_map or {})
    mapped_rows, mapping_audit = _apply_provider_column_map(raw_rows, mapping)
    missing_columns = _missing_required_columns(mapped_rows)
    if missing_columns:
        return [], _audit_payload(
            raw_rows,
            [],
            duplicate_article_id_count=0,
            missing_required_column_names=missing_columns,
            missing_required_field_counts={field: 0 for field in REQUIRED_NON_EMPTY_FIELDS},
            invalid_timestamp_count=0,
            ingested_before_published_count=0,
            **mapping_audit,
        )

    valid_rows: list[dict[str, Any]] = []
    seen_article_ids: set[str] = set()
    duplicate_article_id_count = 0
    missing_required_field_counts = {field: 0 for field in REQUIRED_NON_EMPTY_FIELDS}
    invalid_timestamp_count = 0
    ingested_before_published_count = 0

    for raw_row in mapped_rows:
        row = {column: str(raw_row.get(column, "")).strip() for column in REQUIRED_NEWS_CONTRACT_COLUMNS}
        missing_fields = [field for field in REQUIRED_NON_EMPTY_FIELDS if not row[field]]
        for field in missing_fields:
            missing_required_field_counts[field] += 1
        if missing_fields:
            continue

        published = _parse_utc_timestamp(row["published_at_utc"])
        ingested = _parse_utc_timestamp(row["ingested_at"])
        if published is None or ingested is None:
            invalid_timestamp_count += 1
            continue
        if ingested < published:
            ingested_before_published_count += 1
            continue

        article_id = row["article_id"]
        if article_id in seen_article_ids:
            duplicate_article_id_count += 1
            continue
        seen_article_ids.add(article_id)

        row["symbol"] = row["symbol"].upper()
        row["published_at_utc"] = _format_utc_timestamp(published)
        row["ingested_at"] = _format_utc_timestamp(ingested)
        row["event_type"] = _normalize_event_type(row["event_type"])
        valid_rows.append(row)

    audit = _audit_payload(
        raw_rows,
        valid_rows,
        duplicate_article_id_count=duplicate_article_id_count,
        missing_required_column_names=[],
        missing_required_field_counts=missing_required_field_counts,
        invalid_timestamp_count=invalid_timestamp_count,
        ingested_before_published_count=ingested_before_published_count,
        **mapping_audit,
    )
    return valid_rows, audit


def _required_path(ml: Mapping[str, Any], key: str) -> Path:
    value = ml.get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _provider_column_map(ml: Mapping[str, Any]) -> dict[str, str]:
    raw = ml.get("stock_alpha_news_provider_column_map", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("ml.stock_alpha_news_provider_column_map must be a mapping")
    return {
        str(canonical).strip(): str(provider).strip()
        for canonical, provider in raw.items()
        if str(canonical).strip() and str(provider).strip()
    }


def _apply_provider_column_map(
    raw_rows: list[Mapping[str, Any]],
    provider_column_map: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not provider_column_map:
        return [dict(row) for row in raw_rows], _mapping_audit(
            raw_rows,
            provider_column_map={},
            missing_mapped_provider_columns=[],
            canonical_columns_after_mapping=list(raw_rows[0]) if raw_rows else [],
        )

    raw_columns = set(raw_rows[0]) if raw_rows else set()
    missing_provider_columns = [
        provider_column
        for provider_column in provider_column_map.values()
        if provider_column not in raw_columns
    ]
    mapped_rows = [
        {
            canonical_column: row.get(provider_column, "")
            for canonical_column, provider_column in provider_column_map.items()
        }
        for row in raw_rows
    ]
    return mapped_rows, _mapping_audit(
        raw_rows,
        provider_column_map=provider_column_map,
        missing_mapped_provider_columns=missing_provider_columns,
        canonical_columns_after_mapping=list(mapped_rows[0]) if mapped_rows else list(provider_column_map),
    )


def _mapping_audit(
    raw_rows: list[Mapping[str, Any]],
    *,
    provider_column_map: Mapping[str, str],
    missing_mapped_provider_columns: list[str],
    canonical_columns_after_mapping: list[str],
) -> dict[str, Any]:
    raw_columns = set(raw_rows[0]) if raw_rows else set()
    mapped_provider_columns = set(provider_column_map.values())
    return {
        "provider_column_map_used": bool(provider_column_map),
        "provider_column_map": dict(provider_column_map),
        "missing_mapped_provider_columns": list(missing_mapped_provider_columns),
        "unmapped_provider_columns": sorted(raw_columns - mapped_provider_columns),
        "canonical_columns_after_mapping": list(canonical_columns_after_mapping),
    }


def _missing_required_columns(raw_rows: list[Mapping[str, Any]]) -> list[str]:
    if not raw_rows:
        return []
    return [column for column in REQUIRED_INPUT_COLUMNS if column not in raw_rows[0]]


def _parse_utc_timestamp(value: Any) -> datetime | None:
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_event_type(value: Any) -> str:
    raw = str(value).strip().lower().replace("&", "and")
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in EVENT_TYPE_BUCKETS:
        return normalized
    if any(marker in normalized for marker in ("earnings", "eps", "revenue")):
        return "earnings"
    if any(marker in normalized for marker in ("analyst", "upgrade", "downgrade", "rating")):
        return "analyst"
    if any(marker in normalized for marker in ("guidance", "outlook", "forecast")):
        return "guidance"
    if any(marker in normalized for marker in ("litigation", "lawsuit", "legal")):
        return "litigation"
    if any(marker in normalized for marker in ("mna", "manda", "merger", "acquisition", "takeover")):
        return "mna"
    if any(marker in normalized for marker in ("macro", "fed", "inflation", "rates", "gdp")):
        return "macro"
    if any(marker in normalized for marker in ("product", "launch", "approval")):
        return "product"
    if any(marker in normalized for marker in ("management", "ceo", "cfo", "executive")):
        return "management"
    return "other"


def _audit_payload(
    raw_rows: list[Mapping[str, Any]],
    valid_rows: list[Mapping[str, Any]],
    *,
    duplicate_article_id_count: int,
    missing_required_column_names: list[str],
    missing_required_field_counts: dict[str, int],
    invalid_timestamp_count: int,
    ingested_before_published_count: int,
    provider_column_map_used: bool,
    provider_column_map: dict[str, str],
    missing_mapped_provider_columns: list[str],
    unmapped_provider_columns: list[str],
    canonical_columns_after_mapping: list[str],
) -> dict[str, Any]:
    invalid_row_count = (
        len(raw_rows)
        - len(valid_rows)
        - duplicate_article_id_count
    )
    event_type_counts: dict[str, int] = {}
    for row in valid_rows:
        event_type = str(row.get("event_type", "other"))
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
    return {
        "raw_row_count": len(raw_rows),
        "valid_row_count": len(valid_rows),
        "invalid_row_count": max(0, invalid_row_count),
        "duplicate_article_id_count": duplicate_article_id_count,
        "missing_required_column_names": list(missing_required_column_names),
        "provider_column_map_used": provider_column_map_used,
        "provider_column_map": dict(provider_column_map),
        "missing_mapped_provider_columns": list(missing_mapped_provider_columns),
        "unmapped_provider_columns": list(unmapped_provider_columns),
        "canonical_columns_after_mapping": list(canonical_columns_after_mapping),
        "missing_required_field_counts": dict(missing_required_field_counts),
        "invalid_timestamp_count": invalid_timestamp_count,
        "ingested_before_published_count": ingested_before_published_count,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "symbol_count": len({str(row.get("symbol", "")).strip().upper() for row in valid_rows}),
        "source_count": len({str(row.get("source", "")).strip() for row in valid_rows}),
        "output_contract_path": "",
        "safe_to_generate_features": bool(valid_rows and not missing_required_column_names),
    }


def _write_audit(audit_dir: Path, audit: Mapping[str, Any]) -> NewsContractIngestPaths:
    paths = NewsContractIngestPaths(
        contract_path=Path(str(audit.get("output_contract_path", ""))),
        audit_json_path=audit_dir / "stock_alpha_news_contract_ingest_audit.json",
        audit_markdown_path=audit_dir / "stock_alpha_news_contract_ingest_audit.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.audit_json_path, audit)
    writer.write_markdown(paths.audit_markdown_path, _markdown(audit))
    return paths


def _markdown(audit: Mapping[str, Any]) -> str:
    missing_columns = audit.get("missing_required_column_names", []) or ["none"]
    missing_mapped_columns = audit.get("missing_mapped_provider_columns", []) or ["none"]
    missing_fields = dict(audit.get("missing_required_field_counts", {}) or {})
    event_counts = dict(audit.get("event_type_counts", {}) or {})
    return "\n".join(
        [
            "# Stock-Alpha News Contract Ingest Audit",
            "",
            f"- Raw rows: {audit.get('raw_row_count', 0)}",
            f"- Valid rows: {audit.get('valid_row_count', 0)}",
            f"- Invalid rows: {audit.get('invalid_row_count', 0)}",
            f"- Duplicate article IDs: {audit.get('duplicate_article_id_count', 0)}",
            f"- Invalid timestamps: {audit.get('invalid_timestamp_count', 0)}",
            f"- Ingested before published: {audit.get('ingested_before_published_count', 0)}",
            f"- Symbols: {audit.get('symbol_count', 0)}",
            f"- Sources: {audit.get('source_count', 0)}",
            f"- Safe to generate features: {audit.get('safe_to_generate_features', False)}",
            f"- Provider column map used: {audit.get('provider_column_map_used', False)}",
            f"- Output contract: {audit.get('output_contract_path', '')}",
            "",
            "## Missing Mapped Provider Columns",
            *[f"- {column}" for column in missing_mapped_columns],
            "",
            "## Missing Required Columns",
            *[f"- {column}" for column in missing_columns],
            "",
            "## Missing Required Fields",
            *[f"- {field}: {count}" for field, count in missing_fields.items()],
            "",
            "## Event Types",
            *[f"- {event_type}: {count}" for event_type, count in event_counts.items()],
            "",
            "Research-only ingest. No models were trained and no trading paths were touched.",
        ]
    )
