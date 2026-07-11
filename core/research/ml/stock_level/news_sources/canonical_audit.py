"""Dry-run canonical conversion audit for manually supplied news rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.canonical import (
    CANONICAL_NEWS_SCHEMA_VERSION,
    CanonicalNewsRecord,
    canonical_from_compatibility_row,
)


CANONICAL_CONVERSION_AUDIT_SCHEMA_VERSION = "stock_alpha_news.canonical_conversion_audit.v1"


@dataclass(frozen=True)
class CanonicalConversionAuditPaths:
    """Report files written by the explicit dry-run audit helper."""

    audit_json_path: Path
    canonical_rows_json_path: Path
    markdown_path: Path


def build_canonical_conversion_audit(
    compatibility_rows: Sequence[Mapping[str, Any]],
    *,
    artifact_uri: str | None = None,
    partition_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert caller-supplied compatibility rows without collecting data."""

    canonical_rows: list[dict[str, Any]] = []
    row_diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(compatibility_rows, start=1):
        diagnostic = _row_diagnostic(row, row_number=index)
        try:
            record = canonical_from_compatibility_row(
                row,
                artifact_uri=artifact_uri,
                row_number=index,
                partition_id=partition_id,
            )
        except (TypeError, ValueError) as exc:
            diagnostic.update(
                {
                    "converted": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        else:
            diagnostic["converted"] = True
            canonical_rows.append(_canonical_record_payload(record))
        row_diagnostics.append(diagnostic)

    payload = _audit_payload(
        compatibility_rows,
        canonical_rows=canonical_rows,
        row_diagnostics=row_diagnostics,
    )
    return payload, canonical_rows


def write_canonical_conversion_audit(
    compatibility_rows: Sequence[Mapping[str, Any]],
    report_dir: str | Path,
    *,
    artifact_uri: str | None = None,
    partition_id: str | None = None,
) -> CanonicalConversionAuditPaths:
    """Write a dry-run canonical conversion report to a caller-supplied directory."""

    report_root = Path(report_dir)
    audit_payload, canonical_rows = build_canonical_conversion_audit(
        compatibility_rows,
        artifact_uri=artifact_uri,
        partition_id=partition_id,
    )
    paths = CanonicalConversionAuditPaths(
        audit_json_path=report_root / "canonical_conversion_audit.json",
        canonical_rows_json_path=report_root / "canonical_rows.json",
        markdown_path=report_root / "canonical_conversion_audit.md",
    )
    output_files = {
        "audit_json": str(paths.audit_json_path),
        "canonical_rows_json": str(paths.canonical_rows_json_path),
        "markdown": str(paths.markdown_path),
    }
    audit_payload["output_files"] = output_files
    writer = ResearchArtifactWriter()
    writer.write_json(paths.audit_json_path, audit_payload)
    writer.write_json(paths.canonical_rows_json_path, canonical_rows)
    writer.write_markdown(paths.markdown_path, _markdown(audit_payload))
    return paths


def _audit_payload(
    compatibility_rows: Sequence[Mapping[str, Any]],
    *,
    canonical_rows: list[dict[str, Any]],
    row_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    providers = sorted({str(row.get("provider", "") or row.get("source", "")).strip() for row in compatibility_rows} - {""})
    symbols = sorted({str(row.get("symbol", "")).strip().upper() for row in compatibility_rows} - {""})
    return {
        "schema_version": CANONICAL_CONVERSION_AUDIT_SCHEMA_VERSION,
        "audit_type": "manual_compatibility_rows_to_canonical_dry_run",
        "row_count": len(compatibility_rows),
        "converted_row_count": len(canonical_rows),
        "conversion_error_count": len(compatibility_rows) - len(canonical_rows),
        "missing_publication_timestamp_count": sum(
            not str(row.get("published_at_utc", "")).strip()
            for row in compatibility_rows
        ),
        "missing_provider_article_id_count": sum(
            not str(row.get("provider_article_id", "") or row.get("article_id", "")).strip()
            for row in compatibility_rows
        ),
        "missing_symbol_count": sum(not str(row.get("symbol", "")).strip() for row in compatibility_rows),
        "event_type_count": sum(bool(str(row.get("event_type", "")).strip()) for row in compatibility_rows),
        "form_type_only_count": sum(
            bool(str(row.get("form_type", "")).strip())
            and not str(row.get("event_type", "")).strip()
            for row in compatibility_rows
        ),
        "providers": providers,
        "symbols": symbols,
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "row_diagnostics": row_diagnostics,
        "output_files": {},
        "provider_collection_invoked": False,
        "network_invoked": False,
        "canonical_ingest_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "trading_impact": "none",
    }


def _row_diagnostic(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    return {
        "row_number": row_number,
        "provider": str(row.get("provider", "") or row.get("source", "")).strip(),
        "symbol": str(row.get("symbol", "")).strip().upper(),
        "provider_article_id": str(row.get("provider_article_id", "") or row.get("article_id", "")).strip(),
        "missing_publication_timestamp": not str(row.get("published_at_utc", "")).strip(),
        "missing_provider_article_id": not str(row.get("provider_article_id", "") or row.get("article_id", "")).strip(),
        "missing_symbol": not str(row.get("symbol", "")).strip(),
        "event_type": str(row.get("event_type", "")).strip(),
        "form_type": str(row.get("form_type", "")).strip(),
        "form_type_only": bool(str(row.get("form_type", "")).strip())
        and not str(row.get("event_type", "")).strip(),
        "converted": False,
    }


def _canonical_record_payload(record: CanonicalNewsRecord) -> dict[str, Any]:
    payload = asdict(record)
    return _json_ready(payload)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Canonical Conversion Audit",
            "",
            f"- Schema version: {payload['schema_version']}",
            f"- Audit type: {payload['audit_type']}",
            f"- Rows inspected: {payload['row_count']}",
            f"- Rows converted: {payload['converted_row_count']}",
            f"- Conversion errors: {payload['conversion_error_count']}",
            f"- Missing publication timestamps: {payload['missing_publication_timestamp_count']}",
            f"- Missing provider article IDs: {payload['missing_provider_article_id_count']}",
            f"- Missing symbols: {payload['missing_symbol_count']}",
            f"- Explicit event types: {payload['event_type_count']}",
            f"- SEC form type only rows: {payload['form_type_only_count']}",
            f"- Providers: {payload['providers']}",
            f"- Symbols: {payload['symbols']}",
            f"- Canonical schema version: {payload['canonical_schema_version']}",
            f"- Provider collection invoked: {payload['provider_collection_invoked']}",
            f"- Canonical ingest invoked: {payload['canonical_ingest_invoked']}",
            f"- Feature generation invoked: {payload['feature_generation_invoked']}",
            f"- Trading impact: {payload['trading_impact']}",
            "",
        ]
    )
