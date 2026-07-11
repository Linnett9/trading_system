"""Explicit sample selection for tiny stock-alpha news corpus dry-runs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.canonical import (
    CANONICAL_NEWS_SCHEMA_VERSION,
    CanonicalNewsRecord,
    canonical_from_compatibility_row,
)


CORPUS_SAMPLE_SELECTOR_SCHEMA_VERSION = "stock_alpha_news.corpus_sample_selector.v1"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_MAX_INPUT_BYTES = 512_000
DEFAULT_MAX_INPUT_ROWS = 1_000
PROTECTED_ACTIVE_BACKFILL_PATH = (
    "reports/ml/benchmark/regime_transformer_meta_ensemble_v1/"
    "stock_alpha_news_historical_backfill_alpaca_benzinga_full/dev"
)


@dataclass(frozen=True)
class CorpusSampleSelectionPaths:
    """Scratch artifacts written by the explicit sample selector."""

    sample_rows_json_path: Path
    audit_json_path: Path
    summary_markdown_path: Path


def build_corpus_sample_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_are_canonical: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    input_source_type: str = "in_memory",
    input_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a deterministic eligible sample without provider or corpus wiring."""

    if sample_size < 0:
        raise ValueError("sample_size must be greater than or equal to zero")
    canonical_rows, conversion_diagnostics = _canonical_rows(rows, rows_are_canonical=rows_are_canonical)
    eligible_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(canonical_rows, start=1):
        reasons = _exclusion_reasons(row)
        if reasons:
            excluded_rows.append(_excluded_row(row, row_number=row_number, reasons=reasons))
            continue
        eligible_rows.append({"input_row_number": row_number, "row": row})

    selected_wrapped = _select_diverse_rows(eligible_rows, sample_size=sample_size)
    selected_rows = [dict(item["row"]) for item in selected_wrapped]
    selected_input_numbers = {int(item["input_row_number"]) for item in selected_wrapped}
    for item in eligible_rows:
        row_number = int(item["input_row_number"])
        if row_number not in selected_input_numbers:
            excluded_rows.append(
                _excluded_row(
                    item["row"],
                    row_number=row_number,
                    reasons=["not_selected_sample_size_limit"],
                )
            )

    selected_rows = sorted(selected_rows, key=_selection_sort_key)
    selected_rows = [
        {"sample_row_id": f"corpus-sample-row-{index:06d}", **row}
        for index, row in enumerate(selected_rows, start=1)
    ]
    audit = _audit(
        input_rows=rows,
        eligible_rows=eligible_rows,
        selected_rows=selected_rows,
        excluded_rows=sorted(excluded_rows, key=lambda row: int(row["row_number"])),
        conversion_diagnostics=conversion_diagnostics,
        sample_size=sample_size,
        input_source_type=input_source_type,
        input_path=input_path,
    )
    return audit, selected_rows


def write_corpus_sample_selection(
    rows: Sequence[Mapping[str, Any]] | None,
    report_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    rows_are_canonical: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_input_rows: int = DEFAULT_MAX_INPUT_ROWS,
) -> CorpusSampleSelectionPaths:
    """Write selected rows, audit JSON, and Markdown summary to a scratch directory."""

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")
    loaded_rows = list(rows) if rows is not None else None
    input_source_type = "in_memory"
    resolved_input_path: Path | None = None
    if input_path is not None:
        resolved_input_path = _validate_input_path(input_path, max_input_bytes=max_input_bytes)
        if loaded_rows is not None:
            raise ValueError("Provide either rows or input_path, not both")
        loaded_rows = load_rows_from_path(resolved_input_path, max_input_rows=max_input_rows)
        input_source_type = _input_source_type(resolved_input_path)
    if loaded_rows is None:
        raise ValueError("rows or input_path must be supplied explicitly")
    if len(loaded_rows) > max_input_rows:
        raise ValueError(f"input row count exceeds max_input_rows={max_input_rows}")

    audit, selected_rows = build_corpus_sample_selection(
        loaded_rows,
        rows_are_canonical=rows_are_canonical,
        sample_size=sample_size,
        input_source_type=input_source_type,
        input_path=resolved_input_path,
    )
    paths = CorpusSampleSelectionPaths(
        sample_rows_json_path=report_root / "corpus_sample_rows.json",
        audit_json_path=report_root / "corpus_sample_selection_audit.json",
        summary_markdown_path=report_root / "corpus_sample_selection_summary.md",
    )
    _ensure_output_paths_under_report_dir(report_root, paths)
    audit["output_files"] = {
        "sample_rows_json": str(paths.sample_rows_json_path),
        "audit_json": str(paths.audit_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
    }
    writer = ResearchArtifactWriter()
    writer.write_json(paths.sample_rows_json_path, selected_rows)
    writer.write_json(paths.audit_json_path, audit)
    writer.write_markdown(paths.summary_markdown_path, _markdown(audit))
    return paths


def load_rows_from_path(
    input_path: str | Path,
    *,
    max_input_rows: int = DEFAULT_MAX_INPUT_ROWS,
) -> list[dict[str, Any]]:
    """Load a caller-supplied small JSON, JSONL, or CSV fixture."""

    path = _validate_input_path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of row objects")
        rows = [_mapping_row(row, row_number=index) for index, row in enumerate(payload, start=1)]
    elif suffix == ".jsonl":
        rows = [
            _mapping_row(json.loads(line), row_number=index)
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        raise ValueError("Unsupported input format; expected .json, .jsonl, or .csv")
    if len(rows) > max_input_rows:
        raise ValueError(f"input row count exceeds max_input_rows={max_input_rows}")
    return rows


def _validate_input_path(
    input_path: str | Path,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> Path:
    path = Path(input_path)
    if _contains_protected_path(path):
        raise ValueError("input_path must not reference the protected active backfill path")
    if not path.exists():
        raise FileNotFoundError(f"input_path does not exist: {path}")
    if path.is_dir():
        raise ValueError("input_path must be an explicit file, not a directory")
    if path.stat().st_size > max_input_bytes:
        raise ValueError(f"input_path exceeds max_input_bytes={max_input_bytes}")
    return path


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
        except (AttributeError, TypeError, ValueError) as exc:
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


def _exclusion_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _text(row.get("symbol")):
        reasons.append("missing_symbol")
    if not _text(row.get("provider")) and not _text(row.get("source")):
        reasons.append("missing_provider")
    if not _text(row.get("published_at_utc")):
        reasons.append("missing_publication_timestamp")
    if not _has_text_for_model(row):
        reasons.append("missing_text")
    return reasons


def _excluded_row(row: Mapping[str, Any], *, row_number: int, reasons: list[str]) -> dict[str, Any]:
    return {
        "row_number": row_number,
        "provider": _text(row.get("provider")),
        "symbol": _text(row.get("symbol")),
        "provider_article_id": _text(row.get("provider_article_id")),
        "reasons": reasons,
    }


def _select_diverse_rows(
    eligible_rows: list[dict[str, Any]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    if sample_size == 0:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in sorted(eligible_rows, key=lambda value: _selection_sort_key(value["row"])):
        row = item["row"]
        key = (_text(row.get("symbol")), _text(row.get("provider")))
        grouped.setdefault(key, []).append(item)
    selected: list[dict[str, Any]] = []
    group_keys = sorted(grouped)
    while len(selected) < sample_size and any(grouped.values()):
        for key in group_keys:
            if grouped[key]:
                selected.append(grouped[key].pop(0))
                if len(selected) >= sample_size:
                    break
    return selected


def _audit(
    input_rows: Sequence[Mapping[str, Any]],
    *,
    eligible_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    conversion_diagnostics: list[dict[str, Any]],
    sample_size: int,
    input_source_type: str,
    input_path: str | Path | None,
) -> dict[str, Any]:
    published_values = sorted(
        _text(row.get("published_at_utc"))
        for row in selected_rows
        if _text(row.get("published_at_utc"))
    )
    return {
        "schema_version": CORPUS_SAMPLE_SELECTOR_SCHEMA_VERSION,
        "artifact_type": "corpus_sample_selection",
        "canonical_schema_version": CANONICAL_NEWS_SCHEMA_VERSION,
        "input_row_count": len(input_rows),
        "eligible_row_count": len(eligible_rows),
        "selected_row_count": len(selected_rows),
        "excluded_row_count": len(excluded_rows),
        "sample_size": sample_size,
        "selection_strategy": "deterministic_round_robin_by_symbol_provider_then_publication",
        "skip_reasons": _reason_counts(excluded_rows),
        "excluded_rows": excluded_rows,
        "symbols": sorted({_text(row.get("symbol")) for row in selected_rows} - {""}),
        "providers": sorted({_text(row.get("provider")) for row in selected_rows} - {""}),
        "start_published_at_utc": published_values[0] if published_values else None,
        "end_published_at_utc": published_values[-1] if published_values else None,
        "input_source_type": input_source_type,
        "input_path": str(input_path) if input_path is not None else None,
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
            "protected_active_backfill_path_rejected": True,
        },
    }


def _reason_counts(excluded_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in excluded_rows:
        for reason in row.get("reasons", []) or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _selection_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row.get("published_at_utc")),
        _text(row.get("provider")),
        _text(row.get("provider_article_id")),
        _text(row.get("symbol")),
    )


def _ensure_output_paths_under_report_dir(
    report_root: Path,
    paths: CorpusSampleSelectionPaths,
) -> None:
    root = report_root.resolve(strict=False)
    for path in (
        paths.sample_rows_json_path,
        paths.audit_json_path,
        paths.summary_markdown_path,
    ):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("output paths must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _input_source_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return f"{suffix}_file" if suffix else "file"


def _mapping_row(row: Any, *, row_number: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {row_number} is not an object")
    return dict(row)


def _has_text_for_model(row: Mapping[str, Any]) -> bool:
    return any(
        bool(_text(row.get(field)))
        for field in ("body_or_full_text", "summary", "headline")
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


def _markdown(audit: Mapping[str, Any]) -> str:
    safety = dict(audit.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Corpus Sample Selection",
            "",
            f"- Schema version: {audit['schema_version']}",
            f"- Artifact type: {audit['artifact_type']}",
            f"- Input source type: {audit['input_source_type']}",
            f"- Input rows: {audit['input_row_count']}",
            f"- Eligible rows: {audit['eligible_row_count']}",
            f"- Selected rows: {audit['selected_row_count']}",
            f"- Excluded rows: {audit['excluded_row_count']}",
            f"- Sample size: {audit['sample_size']}",
            f"- Selection strategy: {audit['selection_strategy']}",
            f"- Skip reasons: {audit['skip_reasons']}",
            f"- Symbols: {audit['symbols']}",
            f"- Providers: {audit['providers']}",
            f"- Start published at UTC: {audit['start_published_at_utc']}",
            f"- End published at UTC: {audit['end_published_at_utc']}",
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
