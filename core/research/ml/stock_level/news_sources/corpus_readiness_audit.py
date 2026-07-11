"""Dry-run corpus-readiness audit for caller-supplied news rows."""

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


CORPUS_READINESS_AUDIT_SCHEMA_VERSION = "stock_alpha_news.corpus_readiness_audit.v1"
READY_FOR_TINY_CORPUS_DRY_RUN = "READY_FOR_TINY_CORPUS_DRY_RUN"
NEEDS_MORE_METADATA = "NEEDS_MORE_METADATA"
NEEDS_TIMESTAMPS = "NEEDS_TIMESTAMPS"
NEEDS_TEXT = "NEEDS_TEXT"
NOT_READY = "NOT_READY"


@dataclass(frozen=True)
class CorpusReadinessAuditPaths:
    """Report files written by the explicit dry-run readiness helper."""

    audit_json_path: Path
    markdown_path: Path


def build_corpus_readiness_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_are_canonical: bool = False,
) -> dict[str, Any]:
    """Assess sample news rows without collecting, assembling, or modeling."""

    canonical_rows, row_diagnostics = _canonical_rows(rows, rows_are_canonical=rows_are_canonical)
    row_count = len(canonical_rows)
    metrics = _coverage_metrics(canonical_rows)
    blockers, warnings = _readiness_messages(metrics, row_count=row_count)
    recommendation = _recommendation(blockers, metrics, row_count=row_count)
    return {
        "schema_version": CORPUS_READINESS_AUDIT_SCHEMA_VERSION,
        "audit_type": "sample_news_corpus_readiness_dry_run",
        "row_count": row_count,
        **metrics,
        "blockers": blockers,
        "warnings": warnings,
        "recommendation": recommendation,
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "row_diagnostics": row_diagnostics,
        "output_files": {},
        "provider_collection_invoked": False,
        "network_invoked": False,
        "canonical_ingest_invoked": False,
        "corpus_assembly_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "trading_impact": "none",
    }


def write_corpus_readiness_audit(
    rows: Sequence[Mapping[str, Any]],
    report_dir: str | Path,
    *,
    rows_are_canonical: bool = False,
) -> CorpusReadinessAuditPaths:
    """Write a sample corpus-readiness report to a caller-supplied directory."""

    report_root = Path(report_dir)
    payload = build_corpus_readiness_audit(rows, rows_are_canonical=rows_are_canonical)
    paths = CorpusReadinessAuditPaths(
        audit_json_path=report_root / "corpus_readiness_audit.json",
        markdown_path=report_root / "corpus_readiness_audit.md",
    )
    payload["output_files"] = {
        "audit_json": str(paths.audit_json_path),
        "markdown": str(paths.markdown_path),
    }
    writer = ResearchArtifactWriter()
    writer.write_json(paths.audit_json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def _canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_are_canonical: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        diagnostic = {
            "row_number": index,
            "converted": False,
            "error_type": "",
            "error_message": "",
        }
        try:
            canonical = _canonical_payload(row) if rows_are_canonical else _compatibility_payload(row, row_number=index)
        except (TypeError, ValueError) as exc:
            diagnostic.update(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        else:
            diagnostic.update(
                {
                    "converted": True,
                    "provider": _text(canonical.get("provider")),
                    "symbol": _text(canonical.get("symbol")),
                    "has_any_text": _has_any_text(canonical),
                    "has_publication_timestamp": bool(_text(canonical.get("published_at_utc"))),
                    "has_provider_available_timestamp": bool(_text(canonical.get("provider_available_at_utc"))),
                    "has_collection_timestamp": bool(_text(canonical.get("collected_at_utc"))),
                    "has_event_type": bool(_text(canonical.get("event_type"))),
                    "has_relevance_label": _has_relevance_label(canonical),
                }
            )
            canonical_rows.append(canonical)
        diagnostics.append(diagnostic)
    return canonical_rows, diagnostics


def _compatibility_payload(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    record = canonical_from_compatibility_row(row, row_number=row_number)
    return _json_ready(asdict(record))


def _canonical_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(row))


def _coverage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    usable_text_count = sum(_has_any_text(row) for row in rows)
    publication_count = sum(bool(_text(row.get("published_at_utc"))) for row in rows)
    provider_available_count = sum(bool(_text(row.get("provider_available_at_utc"))) for row in rows)
    collection_count = sum(bool(_text(row.get("collected_at_utc"))) for row in rows)
    return {
        "usable_text_row_count": usable_text_count,
        "headline_coverage": _coverage(sum(bool(_text(row.get("headline"))) for row in rows), row_count),
        "summary_coverage": _coverage(sum(bool(_text(row.get("summary"))) for row in rows), row_count),
        "body_coverage": _coverage(sum(bool(_text(row.get("body_or_full_text"))) for row in rows), row_count),
        "any_text_coverage": _coverage(usable_text_count, row_count),
        "publication_timestamp_coverage": _coverage(publication_count, row_count),
        "provider_availability_timestamp_coverage": _coverage(provider_available_count, row_count),
        "collection_timestamp_coverage": _coverage(collection_count, row_count),
        "point_in_time_safe_timestamp_count": publication_count,
        "symbol_coverage": _coverage(sum(bool(_text(row.get("symbol"))) for row in rows), row_count),
        "provider_coverage": _coverage(sum(bool(_text(row.get("provider"))) for row in rows), row_count),
        "language_coverage": _coverage(sum(bool(_text(row.get("language"))) for row in rows), row_count),
        "duplicate_group_coverage": _coverage(sum(bool(_text(row.get("duplicate_group_id"))) for row in rows), row_count),
        "event_type_coverage": _coverage(sum(bool(_text(row.get("event_type"))) for row in rows), row_count),
        "relevance_label_coverage": _coverage(sum(_has_relevance_label(row) for row in rows), row_count),
    }


def _readiness_messages(metrics: Mapping[str, Any], *, row_count: int) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if row_count == 0:
        blockers.append("no_rows_supplied")
    if int(metrics.get("usable_text_row_count", 0) or 0) == 0:
        blockers.append("no_usable_text")
    if int(metrics.get("point_in_time_safe_timestamp_count", 0) or 0) == 0:
        blockers.append("publication_timestamps_missing")
    if metrics.get("symbol_coverage") == 0.0:
        blockers.append("symbols_missing")
    elif metrics.get("symbol_coverage") not in {None, 1.0}:
        warnings.append("some_symbols_missing")
    if metrics.get("provider_coverage") == 0.0:
        blockers.append("providers_missing")
    elif metrics.get("provider_coverage") not in {None, 1.0}:
        warnings.append("some_providers_missing")
    if metrics.get("provider_availability_timestamp_coverage") in {None, 0.0}:
        warnings.append("provider_availability_timestamps_missing")
    if metrics.get("collection_timestamp_coverage") in {None, 0.0}:
        warnings.append("collection_timestamps_missing")
    if metrics.get("language_coverage") in {None, 0.0}:
        warnings.append("language_missing")
    if metrics.get("duplicate_group_coverage") in {None, 0.0}:
        warnings.append("duplicate_groups_not_available")
    if metrics.get("event_type_coverage") in {None, 0.0}:
        warnings.append("event_labels_missing_for_supervised_modeling")
    if metrics.get("relevance_label_coverage") in {None, 0.0}:
        warnings.append("relevance_labels_missing_for_supervised_modeling")
    return blockers, warnings


def _recommendation(blockers: list[str], metrics: Mapping[str, Any], *, row_count: int) -> str:
    if row_count == 0:
        return NOT_READY
    if "no_usable_text" in blockers:
        return NEEDS_TEXT
    if "publication_timestamps_missing" in blockers:
        return NEEDS_TIMESTAMPS
    if blockers:
        return NEEDS_MORE_METADATA
    if metrics.get("symbol_coverage") == 1.0 and metrics.get("provider_coverage") == 1.0:
        return READY_FOR_TINY_CORPUS_DRY_RUN
    return NEEDS_MORE_METADATA


def _coverage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _has_any_text(row: Mapping[str, Any]) -> bool:
    return any(
        bool(_text(row.get(field)))
        for field in ("headline", "summary", "body_or_full_text")
    )


def _has_relevance_label(row: Mapping[str, Any]) -> bool:
    return any(
        bool(_text(row.get(field)))
        for field in (
            "relevance_status",
            "heuristic_relevance_status",
            "human_reviewed_relevance_label",
            "model_predicted_relevance_label",
        )
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CanonicalNewsRecord):
        return _json_ready(asdict(value))
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
            "# Corpus Readiness Audit",
            "",
            f"- Schema version: {payload['schema_version']}",
            f"- Audit type: {payload['audit_type']}",
            f"- Rows inspected: {payload['row_count']}",
            f"- Usable text rows: {payload['usable_text_row_count']}",
            f"- Any text coverage: {payload['any_text_coverage']}",
            f"- Publication timestamp coverage: {payload['publication_timestamp_coverage']}",
            f"- Point-in-time safe timestamp count: {payload['point_in_time_safe_timestamp_count']}",
            f"- Symbol coverage: {payload['symbol_coverage']}",
            f"- Provider coverage: {payload['provider_coverage']}",
            f"- Event type coverage: {payload['event_type_coverage']}",
            f"- Relevance label coverage: {payload['relevance_label_coverage']}",
            f"- Blockers: {payload['blockers']}",
            f"- Warnings: {payload['warnings']}",
            f"- Recommendation: {payload['recommendation']}",
            f"- Provider collection invoked: {payload['provider_collection_invoked']}",
            f"- Corpus assembly invoked: {payload['corpus_assembly_invoked']}",
            f"- Feature generation invoked: {payload['feature_generation_invoked']}",
            f"- Model training invoked: {payload['model_training_invoked']}",
            f"- Model inference invoked: {payload['model_inference_invoked']}",
            f"- Trading impact: {payload['trading_impact']}",
            "",
        ]
    )
