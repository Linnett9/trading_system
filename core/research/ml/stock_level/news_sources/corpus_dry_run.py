"""Tiny explicit corpus dry-run builder for caller-supplied news rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.canonical import (
    CANONICAL_NEWS_SCHEMA_VERSION,
    CanonicalNewsRecord,
    canonical_from_compatibility_row,
)
from core.research.ml.stock_level.news_sources.corpus_readiness_audit import (
    build_corpus_readiness_audit,
)


CORPUS_DRY_RUN_SCHEMA_VERSION = "stock_alpha_news.corpus_dry_run.v1"


@dataclass(frozen=True)
class CorpusDryRunPaths:
    """Scratch artifacts written by the explicit corpus dry-run helper."""

    rows_jsonl_path: Path
    manifest_json_path: Path
    summary_markdown_path: Path


def build_corpus_dry_run(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_are_canonical: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build deterministic scratch corpus rows without assembling production data."""

    canonical_rows, conversion_diagnostics = _canonical_rows(rows, rows_are_canonical=rows_are_canonical)
    readiness = build_corpus_readiness_audit(rows, rows_are_canonical=rows_are_canonical)
    corpus_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(canonical_rows, start=1):
        reasons = _skip_reasons(row)
        if reasons:
            skipped_rows.append(
                {
                    "row_number": row_number,
                    "provider": _text(row.get("provider")),
                    "symbol": _text(row.get("symbol")),
                    "provider_article_id": _text(row.get("provider_article_id")),
                    "reasons": reasons,
                }
            )
            continue
        corpus_rows.append(_corpus_row(row))
    corpus_rows = sorted(corpus_rows, key=_corpus_sort_key)
    corpus_rows = [
        {"corpus_row_id": f"corpus-row-{index:06d}", **row}
        for index, row in enumerate(corpus_rows, start=1)
    ]
    manifest = _manifest(
        rows,
        corpus_rows=corpus_rows,
        skipped_rows=skipped_rows,
        readiness=readiness,
        conversion_diagnostics=conversion_diagnostics,
    )
    return manifest, corpus_rows


def write_corpus_dry_run(
    rows: Sequence[Mapping[str, Any]],
    report_dir: str | Path,
    *,
    rows_are_canonical: bool = False,
) -> CorpusDryRunPaths:
    """Write JSONL rows, manifest JSON, and Markdown summary to a scratch directory."""

    report_root = Path(report_dir)
    manifest, corpus_rows = build_corpus_dry_run(rows, rows_are_canonical=rows_are_canonical)
    paths = CorpusDryRunPaths(
        rows_jsonl_path=report_root / "corpus_rows.jsonl",
        manifest_json_path=report_root / "corpus_manifest.json",
        summary_markdown_path=report_root / "corpus_summary.md",
    )
    manifest["output_files"] = {
        "corpus_rows_jsonl": str(paths.rows_jsonl_path),
        "manifest_json": str(paths.manifest_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    writer = ResearchArtifactWriter()
    writer.write_text(paths.rows_jsonl_path, _jsonl(corpus_rows))
    writer.write_json(paths.manifest_json_path, manifest)
    writer.write_markdown(paths.summary_markdown_path, _markdown(manifest))
    return paths


def _canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_are_canonical: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        diagnostic = {
            "row_number": row_number,
            "converted": False,
            "error_type": "",
            "error_message": "",
        }
        try:
            payload = _canonical_payload(row) if rows_are_canonical else _compatibility_payload(row, row_number=row_number)
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
                    "provider": _text(payload.get("provider")),
                    "symbol": _text(payload.get("symbol")),
                    "provider_article_id": _text(payload.get("provider_article_id")),
                }
            )
            canonical_rows.append(payload)
        diagnostics.append(diagnostic)
    return canonical_rows, diagnostics


def _compatibility_payload(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    record = canonical_from_compatibility_row(row, row_number=row_number)
    return _json_ready(asdict(record))


def _canonical_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(dict(row))


def _skip_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _text(row.get("symbol")):
        reasons.append("missing_symbol")
    if not _text(row.get("provider")) and not _text(row.get("source")):
        reasons.append("missing_provider")
    if not _text(row.get("published_at_utc")):
        reasons.append("missing_publication_timestamp")
    if not _text_for_model(row)[0]:
        reasons.append("missing_text")
    return reasons


def _corpus_row(row: Mapping[str, Any]) -> dict[str, Any]:
    text_for_model, text_field_used = _text_for_model(row)
    return {
        "canonical_story_id": _text(row.get("canonical_story_id")),
        "story_symbol_id": _text(row.get("story_symbol_id")),
        "provider": _text(row.get("provider")),
        "provider_article_id": _none_if_blank(row.get("provider_article_id")),
        "symbol": _text(row.get("symbol")),
        "published_at_utc": _text(row.get("published_at_utc")),
        "available_at_utc": _none_if_blank(row.get("provider_available_at_utc")),
        "language": _none_if_blank(row.get("language")),
        "headline": _none_if_blank(row.get("headline")),
        "summary": _none_if_blank(row.get("summary")),
        "body": _none_if_blank(row.get("body_or_full_text")),
        "text_for_model": text_for_model,
        "text_field_used": text_field_used,
        "event_type": _none_if_blank(row.get("event_type")),
        "relevance_label": _relevance_label(row),
        "source_type": _text(row.get("source_type")),
    }


def _text_for_model(row: Mapping[str, Any]) -> tuple[str, str]:
    for field, label in (
        ("body_or_full_text", "body"),
        ("summary", "summary"),
        ("headline", "headline"),
    ):
        value = _text(row.get(field))
        if value:
            return value, label
    return "", ""


def _relevance_label(row: Mapping[str, Any]) -> str | None:
    for field in (
        "human_reviewed_relevance_label",
        "model_predicted_relevance_label",
        "heuristic_relevance_status",
        "relevance_status",
    ):
        value = _text(row.get(field))
        if value:
            return value
    return None


def _manifest(
    input_rows: Sequence[Mapping[str, Any]],
    *,
    corpus_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    readiness: Mapping[str, Any],
    conversion_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    published_values = sorted(_text(row.get("published_at_utc")) for row in corpus_rows if _text(row.get("published_at_utc")))
    return {
        "schema_version": CORPUS_DRY_RUN_SCHEMA_VERSION,
        "artifact_type": "sample_corpus_dry_run",
        "input_row_count": len(input_rows),
        "corpus_row_count": len(corpus_rows),
        "skipped_row_count": len(skipped_rows),
        "skip_reasons": _skip_reason_counts(skipped_rows),
        "skipped_rows": skipped_rows,
        "symbols": sorted({_text(row.get("symbol")) for row in corpus_rows} - {""}),
        "providers": sorted({_text(row.get("provider")) for row in corpus_rows} - {""}),
        "languages": sorted({_text(row.get("language")) for row in corpus_rows} - {""}),
        "start_published_at_utc": published_values[0] if published_values else None,
        "end_published_at_utc": published_values[-1] if published_values else None,
        "readiness_recommendation": readiness.get("recommendation"),
        "readiness_blockers": list(readiness.get("blockers", []) or []),
        "readiness_warnings": list(readiness.get("warnings", []) or []),
        "conversion_diagnostics": conversion_diagnostics,
        "output_files": {},
        "safety_flags": {
            "provider_collection_invoked": False,
            "network_invoked": False,
            "canonical_ingest_invoked": False,
            "historical_backfill_invoked": False,
            "corpus_assembly_invoked": False,
            "feature_generation_invoked": False,
            "model_training_invoked": False,
            "model_inference_invoked": False,
            "trading_impact": "none",
        },
    }


def _skip_reason_counts(skipped_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in skipped_rows:
        for reason in row.get("reasons", []) or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _corpus_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row.get("published_at_utc")),
        _text(row.get("provider")),
        _text(row.get("provider_article_id")),
        _text(row.get("symbol")),
    )


def _jsonl(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"


def _markdown(manifest: Mapping[str, Any]) -> str:
    safety = dict(manifest.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Corpus Dry Run",
            "",
            f"- Schema version: {manifest['schema_version']}",
            f"- Artifact type: {manifest['artifact_type']}",
            f"- Input rows: {manifest['input_row_count']}",
            f"- Corpus rows: {manifest['corpus_row_count']}",
            f"- Skipped rows: {manifest['skipped_row_count']}",
            f"- Skip reasons: {manifest['skip_reasons']}",
            f"- Symbols: {manifest['symbols']}",
            f"- Providers: {manifest['providers']}",
            f"- Languages: {manifest['languages']}",
            f"- Start published at UTC: {manifest['start_published_at_utc']}",
            f"- End published at UTC: {manifest['end_published_at_utc']}",
            f"- Readiness recommendation: {manifest['readiness_recommendation']}",
            f"- Readiness blockers: {manifest['readiness_blockers']}",
            f"- Readiness warnings: {manifest['readiness_warnings']}",
            f"- Provider collection invoked: {safety['provider_collection_invoked']}",
            f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
            f"- Corpus assembly invoked: {safety['corpus_assembly_invoked']}",
            f"- Feature generation invoked: {safety['feature_generation_invoked']}",
            f"- Model training invoked: {safety['model_training_invoked']}",
            f"- Model inference invoked: {safety['model_inference_invoked']}",
            f"- Trading impact: {safety['trading_impact']}",
            "",
        ]
    )


def _none_if_blank(value: Any) -> str | None:
    text = _text(value)
    return text or None


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
