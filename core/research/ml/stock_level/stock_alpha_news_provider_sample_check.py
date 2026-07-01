from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import REQUIRED_NEWS_CONTRACT_COLUMNS
from core.research.ml.stock_level.stock_alpha_news_contract_ingest import (
    _apply_provider_column_map,
    _normalize_event_type,
    _parse_utc_timestamp,
    _provider_column_map,
)


NUMERIC_COLUMNS = ("sentiment_score", "relevance_score", "novelty_score")


@dataclass(frozen=True)
class StockAlphaNewsProviderSampleCheckPaths:
    json_path: Path
    markdown_path: Path


def write_stock_alpha_news_provider_sample_check(
    config: Mapping[str, Any],
) -> StockAlphaNewsProviderSampleCheckPaths:
    payload = build_stock_alpha_news_provider_sample_check(config)
    output = _required_path(config, "stock_alpha_news_provider_sample_check_output_dir")
    paths = StockAlphaNewsProviderSampleCheckPaths(
        output / "stock_alpha_news_provider_sample_check.json",
        output / "stock_alpha_news_provider_sample_check.md",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(paths.json_path, payload)
    writer.write_markdown(paths.markdown_path, _markdown(payload))
    return paths


def build_stock_alpha_news_provider_sample_check(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    raw_path = _required_path(config, "stock_alpha_news_raw_path")
    provider_map = _provider_column_map(ml)
    if not raw_path.is_file():
        return _payload(
            raw_path=raw_path,
            raw_columns=[],
            mapped_rows=[],
            provider_map=provider_map,
            missing_mapped_columns=[],
            next_action="provide_raw_news_file",
        )

    raw_rows = CsvRowRepository().read(raw_path)
    raw_columns = list(raw_rows[0]) if raw_rows else []
    mapped_rows, mapping_audit = _apply_provider_column_map(raw_rows, provider_map)
    missing_mapped = list(mapping_audit["missing_mapped_provider_columns"])
    if missing_mapped:
        next_action = "fix_missing_provider_columns"
    elif not provider_map and _missing_canonical(mapped_rows):
        next_action = "add_provider_column_map"
    else:
        next_action = ""
    return _payload(
        raw_path=raw_path,
        raw_columns=raw_columns,
        mapped_rows=mapped_rows,
        provider_map=provider_map,
        missing_mapped_columns=missing_mapped,
        next_action=next_action,
    )


def _payload(
    *,
    raw_path: Path,
    raw_columns: list[str],
    mapped_rows: list[Mapping[str, Any]],
    provider_map: Mapping[str, str],
    missing_mapped_columns: list[str],
    next_action: str,
) -> dict[str, Any]:
    canonical_present_directly = [column for column in REQUIRED_NEWS_CONTRACT_COLUMNS if column in raw_columns]
    missing_canonical = _missing_canonical(mapped_rows)
    timestamp_parseability = {
        column: _parseability(mapped_rows, column, timestamp=True)
        for column in ("published_at_utc", "ingested_at")
    }
    ingested_before_published = 0
    for row in mapped_rows:
        published = _parse_utc_timestamp(row.get("published_at_utc"))
        ingested = _parse_utc_timestamp(row.get("ingested_at"))
        if published is not None and ingested is not None and ingested < published:
            ingested_before_published += 1
    numeric_parseability = {
        column: _parseability(mapped_rows, column, timestamp=False)
        for column in NUMERIC_COLUMNS
    }
    raw_events = [str(row.get("event_type", "")).strip() for row in mapped_rows]
    normalized_events = [_normalize_event_type(value) for value in raw_events]
    article_ids = [str(row.get("article_id", "")).strip() for row in mapped_rows]
    non_empty_ids = [value for value in article_ids if value]
    symbols = [str(row.get("symbol", "")).strip() for row in mapped_rows]
    timestamp_failures = sum(item["invalid_count"] for item in timestamp_parseability.values())
    numeric_failures = sum(item["invalid_count"] for item in numeric_parseability.values())
    compatible = bool(
        mapped_rows
        and not missing_mapped_columns
        and not missing_canonical
        and not timestamp_failures
        and not numeric_failures
        and len(non_empty_ids) == len(mapped_rows)
    )
    if not next_action:
        if timestamp_failures:
            next_action = "fix_timestamp_columns"
        elif missing_canonical:
            next_action = "add_provider_column_map"
        elif compatible:
            next_action = "run_provider_audit"
        else:
            next_action = "fix_sample_values"
    return {
        "raw_csv_path": str(raw_path),
        "raw_file_exists": raw_path.is_file(),
        "raw_row_count": len(mapped_rows),
        "raw_column_names": raw_columns,
        "canonical_columns_present_directly": canonical_present_directly,
        "provider_mapping_used": bool(provider_map),
        "provider_column_map": dict(provider_map),
        "missing_mapped_provider_columns": missing_mapped_columns,
        "missing_canonical_fields_after_mapping": missing_canonical,
        "timestamp_parseability": timestamp_parseability,
        "ingested_before_published_count": ingested_before_published,
        "symbol_normalization_preview": [
            {"raw": symbol, "normalized": symbol.upper()} for symbol in list(dict.fromkeys(symbols))[:10]
        ],
        "article_id_uniqueness_preview": {
            "non_empty_count": len(non_empty_ids),
            "unique_count": len(set(non_empty_ids)),
            "duplicate_count": len(non_empty_ids) - len(set(non_empty_ids)),
            "sample": non_empty_ids[:10],
        },
        "numeric_parseability": numeric_parseability,
        "event_types": {
            "raw_counts": _counts(raw_events),
            "normalized_bucket_counts": _counts(normalized_events),
        },
        "language_counts": _counts(str(row.get("language", "")).strip().lower() for row in mapped_rows),
        "source_counts": _counts(str(row.get("source", "")).strip() for row in mapped_rows),
        "compatible_with_contract_ingest": compatible,
        "next_action": next_action,
        "inspection_only": True,
        "canonical_contract_written": False,
        "features_generated": False,
        "model_training_invoked": False,
        "diagnostics_invoked": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _missing_canonical(rows: list[Mapping[str, Any]]) -> list[str]:
    columns = set(rows[0]) if rows else set()
    return [column for column in REQUIRED_NEWS_CONTRACT_COLUMNS if column not in columns]


def _parseability(rows: list[Mapping[str, Any]], column: str, *, timestamp: bool) -> dict[str, int]:
    present = 0
    valid = 0
    for row in rows:
        value = row.get(column)
        if str(value or "").strip():
            present += 1
            is_valid = _parse_utc_timestamp(value) is not None if timestamp else _is_float(value)
            if is_valid:
                valid += 1
    return {"present_count": present, "valid_count": valid, "invalid_count": present - valid}


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = dict(config.get("ml", {}) or {}).get(key)
    if not value:
        raise ValueError(f"missing ml.{key}")
    return Path(str(value))


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Stock-Alpha News Provider Sample Check",
        "",
        f"- Raw CSV: {payload['raw_csv_path']}",
        f"- Raw file exists: {payload['raw_file_exists']}",
        f"- Compatible with contract ingest: {payload['compatible_with_contract_ingest']}",
        f"- Next action: {payload['next_action']}",
        f"- Missing mapped provider columns: {payload['missing_mapped_provider_columns']}",
        f"- Missing canonical fields after mapping: {payload['missing_canonical_fields_after_mapping']}",
        f"- Ingested before published: {payload['ingested_before_published_count']}",
        "- Inspection only: true",
        "- Canonical contract written: false",
        "- Features generated: false",
        "- Model training invoked: false",
        "- Diagnostics invoked: false",
        "",
        "Read-only compatibility check. No pipeline stage was executed.",
    ])
