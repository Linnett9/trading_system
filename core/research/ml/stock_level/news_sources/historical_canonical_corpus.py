"""Derived canonical corpus materialisation for frozen stock-alpha news assemblies."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.canonical import (
    CANONICAL_NEWS_SCHEMA_VERSION,
    SourceType,
    canonical_from_compatibility_row,
)
from core.research.ml.stock_level.news_sources.normalization import (
    normalize_source_name,
    normalize_url,
)


HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION = "stock_alpha_news.historical_canonical_corpus.v2"
LEGACY_HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION = "stock_alpha_news.historical_canonical_corpus.v1"
HISTORICAL_CANONICAL_TRANSFORMATION_VERSION = "stock_alpha_news.historical_assembly_to_canonical.v1"
CANONICAL_ROW_ORDERING_POLICY = (
    "canonical_story_id,story_symbol_id,provider,provider_article_id,symbol,"
    "published_at_utc,canonical_row_logical_payload"
)
CHECKSUM_ALGORITHM = "sha256"

CANONICAL_CORPUS_CSV = "stock_alpha_news_canonical_corpus.csv"
CANONICAL_CORPUS_MANIFEST_JSON = "stock_alpha_news_canonical_corpus_manifest.json"
CANONICAL_CORPUS_AUDIT_JSON = "stock_alpha_news_canonical_corpus_audit.json"
CANONICAL_CORPUS_SUMMARY_MD = "stock_alpha_news_canonical_corpus_summary.md"


class HistoricalCanonicalCorpusError(RuntimeError):
    """Raised when materialisation would be lossy or unsafe."""


@dataclass(frozen=True)
class HistoricalCanonicalCorpusConfig:
    """Configuration for a derived canonical corpus materialisation run."""

    source_assembly_csv_path: str | Path
    source_assembly_metadata_json_path: str | Path
    output_dir: str | Path
    expected_source_checksum: str
    canonical_schema_version: str = CANONICAL_NEWS_SCHEMA_VERSION
    transformation_version: str = HISTORICAL_CANONICAL_TRANSFORMATION_VERSION
    write_enabled: bool = False
    ingested_at_utc: str | None = None


def materialize_historical_canonical_corpus_from_config(
    config: HistoricalCanonicalCorpusConfig,
) -> dict[str, Any]:
    """Materialise a canonical corpus from an explicit config object."""

    if config.canonical_schema_version != CANONICAL_NEWS_SCHEMA_VERSION:
        raise HistoricalCanonicalCorpusError("unsupported canonical_schema_version")
    return materialize_historical_canonical_corpus(
        source_assembly_csv_path=config.source_assembly_csv_path,
        source_assembly_metadata_json_path=config.source_assembly_metadata_json_path,
        output_dir=config.output_dir,
        expected_source_checksum=config.expected_source_checksum,
        transformation_version=config.transformation_version,
        write_enabled=config.write_enabled,
        ingested_at_utc=config.ingested_at_utc,
    )


def build_historical_canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_assembly_path: str | Path,
    source_assembly_checksum: str,
    ingested_at_utc: str,
    transformation_version: str = HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert frozen historical assembly rows into lossless canonical rows."""

    canonical_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        diagnostic = {
            "source_row_number": row_number,
            "converted": False,
            "provider": _text(row.get("provider")),
            "symbol": _text(row.get("symbol")),
            "provider_article_id": _text(row.get("provider_article_id") or row.get("article_id")),
            "error_type": "",
            "error_message": "",
        }
        try:
            compatibility_row = historical_assembly_row_to_compatibility(
                row,
                source_assembly_path=source_assembly_path,
                source_assembly_checksum=source_assembly_checksum,
                source_row_number=row_number,
            )
            record = canonical_from_compatibility_row(
                compatibility_row,
                artifact_uri=str(source_assembly_path),
                row_number=row_number,
                partition_id=_text(row.get("partition_id")) or None,
            )
            canonical_rows.append(
                _materialized_row(
                    record=record,
                    compatibility_row=compatibility_row,
                    source_row=row,
                    source_row_number=row_number,
                    source_assembly_path=source_assembly_path,
                    source_assembly_checksum=source_assembly_checksum,
                    ingested_at_utc=ingested_at_utc,
                    transformation_version=transformation_version,
                )
            )
            diagnostic["converted"] = True
        except Exception as exc:  # fail closed; caller should not receive partial output
            diagnostic["error_type"] = type(exc).__name__
            diagnostic["error_message"] = str(exc)
            blockers.append(dict(diagnostic))
        diagnostics.append(diagnostic)

    audit = {
        "schema_version": HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "transformation_version": transformation_version,
        "source_assembly_path": str(source_assembly_path),
        "source_assembly_checksum": source_assembly_checksum,
        "source_row_count": len(rows),
        "canonical_row_count": len(canonical_rows),
        "failed_row_count": len(blockers),
        "row_count_reconciled": len(rows) == len(canonical_rows) and not blockers,
        "conversion_diagnostics": diagnostics,
        "blocking_rows": blockers,
        "duplicate_group_method": "provider_original_article_id_or_story_identity_non_destructive",
        "duplicate_group_version": transformation_version,
        "relevance_method": "not_run_phase1_lossless_materialisation",
        "relevance_version": transformation_version,
        "safety_flags": _safety_flags(),
    }
    return canonical_rows, audit


def historical_assembly_row_to_compatibility(
    row: Mapping[str, Any],
    *,
    source_assembly_path: str | Path,
    source_assembly_checksum: str,
    source_row_number: int,
) -> dict[str, Any]:
    """Map a frozen assembly CSV row into the existing compatibility contract."""

    provider = _first_text(row, "provider", "delivery_provider", "source_provider")
    provider_article_id = _first_text(row, "provider_article_id", "article_id", "id")
    symbol = _first_text(row, "symbol", "ticker")
    published_at = _first_text(row, "published_at_utc", "published_at", "created_at")
    if not provider:
        raise HistoricalCanonicalCorpusError("provider is required")
    if not provider_article_id:
        raise HistoricalCanonicalCorpusError("provider_article_id or article_id is required")
    if not symbol:
        raise HistoricalCanonicalCorpusError("symbol is required")
    if not published_at:
        raise HistoricalCanonicalCorpusError("published_at_utc is required")

    provider_url = _first_text(row, "provider_url", "url", "article_url", "source_url")
    provider_symbols = _first_text(row, "provider_symbols", "symbols") or symbol
    source_type = _first_text(row, "source_type") or _infer_source_type(provider, row)
    compatibility = dict(row)
    compatibility.update(
        {
            "article_id": _first_text(row, "article_id") or f"{provider}:{provider_article_id}:{symbol}",
            "provider": provider,
            "provider_article_id": provider_article_id,
            "provider_original_article_id": _first_text(row, "provider_original_article_id") or provider_article_id,
            "provider_symbols": provider_symbols,
            "symbol": symbol,
            "published_at_utc": published_at,
            "provider_available_at_utc": _first_text(row, "provider_available_at_utc", "available_at_utc"),
            "updated_at_utc": _first_text(row, "updated_at_utc", "updated_at"),
            "collected_at_utc": _first_text(row, "collected_at_utc", "collected_at", "ingested_at"),
            "headline": _first_text(row, "headline", "headline_or_title", "title"),
            "summary": _first_text(row, "summary", "body_or_summary", "description"),
            "body_or_full_text": _first_text(row, "body_or_full_text", "full_text", "body"),
            "source": _first_text(row, "source", "raw_source"),
            "source_type": source_type,
            "delivery_provider": _first_text(row, "delivery_provider"),
            "original_source": _first_text(row, "original_source", "raw_source"),
            "publisher": _first_text(row, "publisher"),
            "author": _first_text(row, "author"),
            "provider_url": provider_url,
            "source_url": provider_url,
            "language": _first_text(row, "language"),
            "duplicate_group_id": _first_text(row, "duplicate_group_id"),
            "relevance_status": _first_text(row, "relevance_status"),
            "event_type": _first_text(row, "event_type"),
            "raw_source_artifact": str(source_assembly_path),
            "raw_source_row_number": str(source_row_number),
            "source_assembly_path": str(source_assembly_path),
            "source_assembly_checksum": source_assembly_checksum,
        }
    )
    return compatibility


def materialize_historical_canonical_corpus(
    *,
    source_assembly_csv_path: str | Path,
    source_assembly_metadata_json_path: str | Path,
    output_dir: str | Path,
    expected_source_checksum: str,
    transformation_version: str = HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
    write_enabled: bool = False,
    ingested_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write a separate canonical corpus after validating source integrity."""

    if not write_enabled:
        raise HistoricalCanonicalCorpusError("canonical corpus materialisation is disabled by default")
    source_path = Path(source_assembly_csv_path)
    metadata_path = Path(source_assembly_metadata_json_path)
    output_root = Path(output_dir)
    if source_path.resolve(strict=False).parent == output_root.resolve(strict=False):
        raise HistoricalCanonicalCorpusError("canonical output_dir must be separate from source assembly directory")

    source_checksum = sha256_file(source_path)
    metadata = _read_json(metadata_path)
    metadata_checksum = _text(
        metadata.get("assembly_checksum")
        or metadata.get("checksum")
        or metadata.get("source_assembly_checksum")
    )
    if expected_source_checksum and source_checksum != expected_source_checksum:
        raise HistoricalCanonicalCorpusError("source checksum does not match expected_source_checksum")
    if metadata_checksum and metadata_checksum != source_checksum:
        raise HistoricalCanonicalCorpusError("source checksum does not match metadata checksum")

    rows = _read_csv_rows(source_path)
    if ingested_at_utc is None:
        ingested_at_utc = _text(metadata.get("ingested_at_utc"))
    if not ingested_at_utc:
        raise HistoricalCanonicalCorpusError(
            "ingested_at_utc is required for deterministic canonical publication"
        )
    ingested = _format_utc(ingested_at_utc)
    canonical_rows, audit = build_historical_canonical_rows(
        rows,
        source_assembly_path=source_path,
        source_assembly_checksum=source_checksum,
        ingested_at_utc=ingested,
        transformation_version=transformation_version,
    )
    if audit["failed_row_count"]:
        raise HistoricalCanonicalCorpusError("canonical conversion failed for one or more source rows")
    canonical_rows = sorted(canonical_rows, key=_canonical_row_order_key)
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_path = output_root / CANONICAL_CORPUS_CSV
    manifest_path = output_root / CANONICAL_CORPUS_MANIFEST_JSON
    audit_path = output_root / CANONICAL_CORPUS_AUDIT_JSON
    summary_path = output_root / CANONICAL_CORPUS_SUMMARY_MD
    corpus_bytes = _canonical_csv_bytes(canonical_rows)
    corpus_checksum = hashlib.sha256(corpus_bytes).hexdigest()
    rows_logical_checksum = canonical_rows_logical_checksum(canonical_rows)
    schema_checksum = _hash_payload(
        {
            "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
            "fieldnames": CANONICAL_CORPUS_FIELDNAMES,
        }
    )
    manifest = _manifest(
        source_path=source_path,
        metadata_path=metadata_path,
        output_root=output_root,
        source_checksum=source_checksum,
        source_metadata=metadata,
        canonical_rows=canonical_rows,
        audit=audit,
        ingested_at_utc=ingested,
        transformation_version=transformation_version,
        corpus_path=corpus_path,
        corpus_checksum=corpus_checksum,
        rows_logical_checksum=rows_logical_checksum,
        schema_checksum=schema_checksum,
    )
    manifest["output_files"] = {
        "canonical_corpus_csv": str(corpus_path),
        "manifest_json": str(manifest_path),
        "audit_json": str(audit_path),
        "summary_markdown": str(summary_path),
    }
    manifest["logical_manifest_checksum"] = _logical_manifest_checksum(manifest)
    audit["output_files"] = dict(manifest["output_files"])
    publication_result = _publish_canonical_bundle(
        corpus_path=corpus_path,
        corpus_bytes=corpus_bytes,
        manifest_path=manifest_path,
        manifest=manifest,
        audit_path=audit_path,
        audit=audit,
        summary_path=summary_path,
        summary=_markdown(manifest, audit),
    )
    return {**manifest, "publication_result": publication_result}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_logical_checksum(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    payloads = sorted(
        (_canonical_row_logical_payload(row) for row in rows),
        key=_canonical_json,
    )
    return _hash_payload(payloads)


def verify_canonical_corpus_inventory(
    manifest: Mapping[str, Any], *, corpus_path: Path | None = None
) -> dict[str, Any]:
    required = (
        "canonical_corpus_contract",
        "canonical_corpus_identity",
        "canonical_corpus_checksum",
        "canonical_rows_logical_checksum",
        "canonical_schema_checksum",
        "canonical_row_ordering_policy",
        "logical_manifest_checksum",
    )
    reasons = []
    if manifest.get("schema_version") not in {
        HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        LEGACY_HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
    }:
        reasons.append("CORPUS_MANIFEST_CONTRACT_UNKNOWN")
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        reasons.append("STRENGTHENED_INVENTORY_IDENTITY_MISSING")
    elif manifest.get("logical_manifest_checksum") != _logical_manifest_checksum(manifest):
        reasons.append("LOGICAL_MANIFEST_CHECKSUM_MISMATCH")
    if corpus_path is not None and not missing:
        try:
            if sha256_file(corpus_path) != manifest["canonical_corpus_checksum"]:
                reasons.append("CANONICAL_CORPUS_ARTIFACT_CHECKSUM_MISMATCH")
            rows = _read_csv_rows(corpus_path)
            if (
                canonical_rows_logical_checksum(rows)
                != manifest["canonical_rows_logical_checksum"]
            ):
                reasons.append("CANONICAL_ROWS_LOGICAL_CHECKSUM_MISMATCH")
        except OSError:
            reasons.append("CANONICAL_CORPUS_ARTIFACT_MISSING")
    return {
        "readable": manifest.get("schema_version") in {
            HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
            LEGACY_HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        },
        "inventory_certified": not reasons,
        "reasons": sorted(set(reasons)),
    }


CANONICAL_CORPUS_FIELDNAMES = [
    "canonical_schema_version",
    "transformation_version",
    "canonical_story_id",
    "story_symbol_id",
    "provider",
    "provider_article_id",
    "provider_original_article_id",
    "provider_symbols",
    "symbol",
    "published_at_utc",
    "provider_available_at_utc",
    "updated_at_utc",
    "collected_at_utc",
    "ingested_at_utc",
    "headline",
    "normalized_headline",
    "summary",
    "body_or_full_text",
    "source",
    "normalized_source",
    "source_type",
    "delivery_provider",
    "original_source",
    "publisher",
    "normalized_publisher",
    "author",
    "provider_url",
    "normalized_url",
    "language",
    "duplicate_group_id",
    "duplicate_group_method",
    "duplicate_group_version",
    "relevance_status",
    "relevance_evidence",
    "relevance_method",
    "relevance_version",
    "event_type",
    "event_status",
    "raw_source_artifact",
    "raw_source_row_number",
    "source_assembly_path",
    "source_assembly_checksum",
    "conversion_status",
    "missing_field_states",
    "raw_provider_values",
]


def _materialized_row(
    *,
    record: Any,
    compatibility_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_row_number: int,
    source_assembly_path: str | Path,
    source_assembly_checksum: str,
    ingested_at_utc: str,
    transformation_version: str,
) -> dict[str, Any]:
    payload = asdict(record)
    provider_symbols = ",".join(payload.get("provider_symbols", ()) or ())
    raw_values = json.dumps(dict(source_row), sort_keys=True)
    duplicate_group_id = _text(payload.get("duplicate_group_id")) or _duplicate_group_id(payload)
    return {
        "canonical_schema_version": payload["schema_version"],
        "transformation_version": transformation_version,
        "canonical_story_id": payload["canonical_story_id"],
        "story_symbol_id": payload["story_symbol_id"],
        "provider": payload["provider"],
        "provider_article_id": payload.get("provider_article_id") or "",
        "provider_original_article_id": payload.get("provider_original_article_id") or "",
        "provider_symbols": provider_symbols,
        "symbol": payload["symbol"],
        "published_at_utc": payload.get("published_at_utc") or "",
        "provider_available_at_utc": payload.get("provider_available_at_utc") or "",
        "updated_at_utc": payload.get("updated_at_utc") or "",
        "collected_at_utc": payload.get("collected_at_utc") or "",
        "ingested_at_utc": ingested_at_utc,
        "headline": payload.get("headline") or "",
        "normalized_headline": _normalize_text(payload.get("headline")),
        "summary": payload.get("summary") or "",
        "body_or_full_text": payload.get("body_or_full_text") or "",
        "source": payload.get("source") or "",
        "normalized_source": normalize_source_name(payload.get("source")) or "",
        "source_type": _source_type_value(payload.get("source_type")),
        "delivery_provider": payload.get("delivery_provider") or "",
        "original_source": payload.get("original_source") or "",
        "publisher": payload.get("publisher") or "",
        "normalized_publisher": normalize_source_name(payload.get("publisher")) or "",
        "author": payload.get("author") or "",
        "provider_url": payload.get("provider_url") or "",
        "normalized_url": payload.get("normalized_provider_url") or normalize_url(payload.get("provider_url")) or "",
        "language": payload.get("language") or "",
        "duplicate_group_id": duplicate_group_id,
        "duplicate_group_method": "provider_original_article_id_or_story_identity_non_destructive",
        "duplicate_group_version": transformation_version,
        "relevance_status": payload.get("relevance_status") or "",
        "relevance_evidence": json.dumps(payload.get("relevance_evidence") or {}, sort_keys=True),
        "relevance_method": "not_run_phase1_lossless_materialisation",
        "relevance_version": transformation_version,
        "event_type": payload.get("event_type") or "",
        "event_status": "not_modelled_phase1",
        "raw_source_artifact": _text(compatibility_row.get("raw_source_artifact")) or str(source_assembly_path),
        "raw_source_row_number": str(source_row_number),
        "source_assembly_path": str(source_assembly_path),
        "source_assembly_checksum": source_assembly_checksum,
        "conversion_status": "converted",
        "missing_field_states": json.dumps(_missing_field_states(payload), sort_keys=True),
        "raw_provider_values": raw_values,
    }


def _manifest(
    *,
    source_path: Path,
    metadata_path: Path,
    output_root: Path,
    source_checksum: str,
    source_metadata: Mapping[str, Any],
    canonical_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    ingested_at_utc: str,
    transformation_version: str,
    corpus_path: Path,
    corpus_checksum: str,
    rows_logical_checksum: str,
    schema_checksum: str,
) -> dict[str, Any]:
    identity_payload = {
        "contract": HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "canonical_schema_checksum": schema_checksum,
        "transformation_version": transformation_version,
        "source_assembly_checksum": source_checksum,
        "canonical_rows_logical_checksum": rows_logical_checksum,
    }
    return {
        "schema_version": HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        "canonical_corpus_contract": HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        "canonical_corpus_identity": _hash_payload(identity_payload),
        "canonical_corpus_checksum": corpus_checksum,
        "canonical_rows_logical_checksum": rows_logical_checksum,
        "canonical_artifact_path": str(corpus_path),
        "canonical_artifact_checksum_algorithm": CHECKSUM_ALGORITHM,
        "canonical_row_ordering_policy": CANONICAL_ROW_ORDERING_POLICY,
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "canonical_schema_checksum": schema_checksum,
        "transformation_contract": HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
        "transformation_version": transformation_version,
        "source_assembly_identity": _hash_payload(
            {
                "source_assembly_checksum": source_checksum,
                "source_row_count": int(audit["source_row_count"]),
            }
        ),
        "source_assembly_path": str(source_path),
        "source_assembly_metadata_path": str(metadata_path),
        "source_assembly_checksum": source_checksum,
        "source_metadata": dict(source_metadata),
        "output_dir": str(output_root),
        "source_row_count": int(audit["source_row_count"]),
        "canonical_row_count": len(canonical_rows),
        "row_count_reconciled": bool(audit["row_count_reconciled"]),
        "unique_canonical_story_count": len({row["canonical_story_id"] for row in canonical_rows}),
        "unique_story_symbol_count": len({row["story_symbol_id"] for row in canonical_rows}),
        "duplicate_group_count": len({row["duplicate_group_id"] for row in canonical_rows}),
        "ingested_at_utc": ingested_at_utc,
        "publication_timestamp": ingested_at_utc,
        "source_git_commit": _git_commit(),
        "features_generated": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "output_files": {},
        "safety_flags": _safety_flags(),
    }


def _canonical_row_logical_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "canonical_schema_version",
        "transformation_version",
        "canonical_story_id",
        "story_symbol_id",
        "provider",
        "provider_article_id",
        "provider_original_article_id",
        "provider_symbols",
        "symbol",
        "published_at_utc",
        "provider_available_at_utc",
        "updated_at_utc",
        "collected_at_utc",
        "headline",
        "normalized_headline",
        "summary",
        "body_or_full_text",
        "source",
        "normalized_source",
        "source_type",
        "delivery_provider",
        "original_source",
        "publisher",
        "normalized_publisher",
        "author",
        "provider_url",
        "normalized_url",
        "language",
        "duplicate_group_id",
        "duplicate_group_method",
        "duplicate_group_version",
        "relevance_status",
        "relevance_evidence",
        "relevance_method",
        "relevance_version",
        "event_type",
        "event_status",
        "conversion_status",
        "missing_field_states",
    )
    return {field: row.get(field, "") for field in fields}


def _canonical_row_order_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _text(row.get("canonical_story_id")),
        _text(row.get("story_symbol_id")),
        _text(row.get("provider")),
        _text(row.get("provider_article_id")),
        _text(row.get("symbol")),
        _text(row.get("published_at_utc")),
        _canonical_json(_canonical_row_logical_payload(row)),
    )


def _canonical_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CANONICAL_CORPUS_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _publish_canonical_bundle(
    *,
    corpus_path: Path,
    corpus_bytes: bytes,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    audit_path: Path,
    audit: Mapping[str, Any],
    summary_path: Path,
    summary: str,
) -> str:
    desired = {
        corpus_path: corpus_bytes,
        manifest_path: json.dumps(
            manifest, indent=2, sort_keys=True
        ).encode("utf-8"),
        audit_path: json.dumps(audit, indent=2, sort_keys=True).encode("utf-8"),
        summary_path: summary.encode("utf-8"),
    }
    existing = [path for path in desired if path.exists()]
    if existing:
        if len(existing) != len(desired) or any(
            path.read_bytes() != payload for path, payload in desired.items()
        ):
            raise HistoricalCanonicalCorpusError(
                "incompatible existing canonical corpus publication"
            )
        return "SKIPPED_COMPATIBLE"
    for path, payload in desired.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return "PUBLISHED"


def _logical_manifest_checksum(manifest: Mapping[str, Any]) -> str:
    excluded = {
        "logical_manifest_checksum",
        "output_dir",
        "output_files",
        "canonical_artifact_path",
        "source_assembly_path",
        "source_assembly_metadata_path",
        "publication_timestamp",
        "ingested_at_utc",
        "publication_result",
    }
    return _hash_payload(
        {key: value for key, value in manifest.items() if key not in excluded}
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise HistoricalCanonicalCorpusError("metadata JSON must be an object")
    return dict(payload)


def _first_text(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = _text(row.get(name))
        if value:
            return value
    return ""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _infer_source_type(provider: str, row: Mapping[str, Any]) -> str:
    provider_lower = provider.lower()
    if "alpaca" in provider_lower or "benzinga" in provider_lower:
        return SourceType.MARKET_DATA_PROVIDER.value
    if _first_text(row, "form_type"):
        return SourceType.SEC_FILING.value
    return SourceType.UNKNOWN.value


def _source_type_value(value: Any) -> str:
    if isinstance(value, SourceType):
        return value.value
    return _text(value)


def _duplicate_group_id(payload: Mapping[str, Any]) -> str:
    provider = _text(payload.get("provider"))
    original_id = _text(payload.get("provider_original_article_id") or payload.get("provider_article_id"))
    if provider and original_id:
        return hashlib.sha256(f"{provider}\x1f{original_id}".encode("utf-8")).hexdigest()
    return _text(payload.get("canonical_story_id"))


def _missing_field_states(payload: Mapping[str, Any]) -> dict[str, str]:
    fields = {
        "headline": payload.get("headline"),
        "summary": payload.get("summary"),
        "body_or_full_text": payload.get("body_or_full_text"),
        "provider_available_at_utc": payload.get("provider_available_at_utc"),
        "updated_at_utc": payload.get("updated_at_utc"),
        "collected_at_utc": payload.get("collected_at_utc"),
        "provider_url": payload.get("provider_url"),
        "publisher": payload.get("publisher"),
        "author": payload.get("author"),
        "language": payload.get("language"),
    }
    return {name: ("present" if _text(value) else "missing_or_unavailable") for name, value in fields.items()}


def _normalize_text(value: Any) -> str:
    return " ".join(_text(value).lower().split())


def _format_utc(value: str) -> str:
    text = _text(value)
    if text.endswith("Z"):
        return text
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safety_flags() -> dict[str, Any]:
    return {
        "source_assembly_mutated": False,
        "raw_partitions_mutated": False,
        "contract_ingest_invoked": False,
        "canonical_ingest_invoked": False,
        "features_generated": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "backfill_invoked": False,
        "trading_impact": "none",
    }


def _markdown(manifest: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stock Alpha News Canonical Corpus",
            "",
            f"- Schema version: {manifest['schema_version']}",
            f"- Canonical schema version: {manifest['canonical_schema_version']}",
            f"- Transformation version: {manifest['transformation_version']}",
            f"- Source assembly path: {manifest['source_assembly_path']}",
            f"- Source checksum: {manifest['source_assembly_checksum']}",
            f"- Source rows: {manifest['source_row_count']}",
            f"- Canonical rows: {manifest['canonical_row_count']}",
            f"- Row count reconciled: {manifest['row_count_reconciled']}",
            f"- Failed row count: {audit['failed_row_count']}",
            f"- Unique canonical stories: {manifest['unique_canonical_story_count']}",
            f"- Unique story-symbol rows: {manifest['unique_story_symbol_count']}",
            f"- Features generated: {manifest['features_generated']}",
            f"- Model training invoked: {manifest['model_training_invoked']}",
            "",
        ]
    )
