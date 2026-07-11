"""Guarded scratch-only provider-like dry-run for stock-alpha news rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.news_sources.corpus_composition_smoke import (
    write_corpus_composition_smoke_report,
)
from core.research.ml.stock_level.news_sources.corpus_sample_selector import (
    PROTECTED_ACTIVE_BACKFILL_PATH,
)


PROVIDER_SCRATCH_DRY_RUN_SCHEMA_VERSION = "stock_alpha_news.provider_scratch_dry_run.v1"
MAX_SYMBOL_CAP = 10
MAX_ROW_CAP = 100
MAX_REQUEST_CAP = 10


class ProviderScratchDryRunAdapter(Protocol):
    """Caller-supplied provider-like adapter used only by explicit dry-runs."""

    provider_id: str
    provider_family: str

    def collect(
        self,
        *,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return caller-supplied compatibility rows without network side effects."""


@dataclass(frozen=True)
class ProviderScratchDryRunPaths:
    """Scratch artifacts written by the guarded provider-like dry-run."""

    report_json_path: Path
    summary_markdown_path: Path
    composition_dir: Path


def write_provider_scratch_dry_run_report(
    adapter: ProviderScratchDryRunAdapter,
    report_dir: str | Path,
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    max_symbols: int,
    max_rows: int,
    max_requests: int,
    enabled: bool = False,
    network_allowed: bool = False,
) -> tuple[dict[str, Any], ProviderScratchDryRunPaths]:
    """Run a disabled-by-default scratch dry-run with caller-supplied rows only."""

    if not enabled:
        raise ValueError("provider scratch dry-run is disabled by default; pass enabled=True explicitly")
    if network_allowed:
        raise ValueError("network_allowed must remain False for provider scratch dry-run")
    _validate_caps(max_symbols=max_symbols, max_rows=max_rows, max_requests=max_requests)

    report_root = Path(report_dir)
    if _contains_protected_path(report_root):
        raise ValueError("report_dir must not reference the protected active backfill path")
    paths = ProviderScratchDryRunPaths(
        report_json_path=report_root / "provider_scratch_dry_run_report.json",
        summary_markdown_path=report_root / "provider_scratch_dry_run_summary.md",
        composition_dir=report_root / "composition",
    )
    _ensure_paths_under_report_dir(report_root, paths)

    normalized_symbols = _normalized_symbols(symbols)
    capped_symbols = normalized_symbols[:max_symbols]
    warnings: list[str] = []
    if len(normalized_symbols) > len(capped_symbols):
        warnings.append("symbols_capped_to_max_symbols")

    adapter_rows = [
        dict(row)
        for row in adapter.collect(
            symbols=capped_symbols,
            start_date=start_date,
            end_date=end_date,
            limit=max_rows,
        )
    ]
    sorted_rows = sorted(adapter_rows, key=_row_sort_key)
    capped_rows = sorted_rows[:max_rows]
    if len(sorted_rows) > len(capped_rows):
        warnings.append("adapter_rows_capped_to_max_rows")

    composition_report, composition_paths = write_corpus_composition_smoke_report(
        capped_rows,
        paths.composition_dir,
        sample_size=max_rows,
    )
    report = _report(
        adapter=adapter,
        symbols=capped_symbols,
        start_date=start_date,
        end_date=end_date,
        max_symbols=max_symbols,
        max_rows=max_rows,
        max_requests=max_requests,
        adapter_row_count=len(capped_rows),
        raw_adapter_row_count=len(sorted_rows),
        composition_report=composition_report,
        composition_report_path=composition_paths.report_json_path,
        warnings=warnings,
    )
    report["output_files"] = {
        "report_json": str(paths.report_json_path),
        "summary_markdown": str(paths.summary_markdown_path),
        "composition_report_json": str(composition_paths.report_json_path),
        "composition_summary_markdown": str(composition_paths.summary_markdown_path),
    }

    writer = ResearchArtifactWriter()
    writer.write_json(paths.report_json_path, report)
    writer.write_markdown(paths.summary_markdown_path, _markdown(report))
    return report, paths


def _report(
    *,
    adapter: ProviderScratchDryRunAdapter,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    max_symbols: int,
    max_rows: int,
    max_requests: int,
    adapter_row_count: int,
    raw_adapter_row_count: int,
    composition_report: Mapping[str, Any],
    composition_report_path: Path,
    warnings: Sequence[str],
) -> dict[str, Any]:
    selected_row_count = int(composition_report.get("selected_row_count", 0) or 0)
    corpus_row_count = int(composition_report.get("corpus_row_count", 0) or 0)
    skipped_row_count = int(composition_report.get("skipped_row_count", 0) or 0)
    excluded_row_count = int(composition_report.get("sample_excluded_row_count", 0) or 0)
    return {
        "schema_version": PROVIDER_SCRATCH_DRY_RUN_SCHEMA_VERSION,
        "artifact_type": "provider_scratch_dry_run_report",
        "provider_id": _text(getattr(adapter, "provider_id", "caller_supplied_adapter")),
        "provider_family": _text(getattr(adapter, "provider_family", "unknown")),
        "enabled": True,
        "network_allowed": False,
        "symbols": list(symbols),
        "start_date": start_date,
        "end_date": end_date,
        "max_symbols": max_symbols,
        "max_rows": max_rows,
        "max_requests": max_requests,
        "adapter_collect_invoked": True,
        "raw_adapter_row_count": raw_adapter_row_count,
        "adapter_row_count": adapter_row_count,
        "selected_row_count": selected_row_count,
        "corpus_row_count": corpus_row_count,
        "skipped_row_count": skipped_row_count,
        "excluded_row_count": excluded_row_count,
        "sample_skip_reasons": dict(composition_report.get("sample_skip_reasons", {}) or {}),
        "corpus_skip_reasons": dict(composition_report.get("corpus_skip_reasons", {}) or {}),
        "composition_blockers": list(composition_report.get("blockers", []) or []),
        "composition_warnings": list(composition_report.get("warnings", []) or []),
        "composition_report_path": str(composition_report_path),
        "guards": _guards(),
        "blockers": [],
        "warnings": sorted(set(warnings)),
        "safety_flags": _safety_flags(),
        "output_files": {},
    }


def _validate_caps(*, max_symbols: int, max_rows: int, max_requests: int) -> None:
    if not 0 < int(max_symbols) <= MAX_SYMBOL_CAP:
        raise ValueError(f"max_symbols must be between 1 and {MAX_SYMBOL_CAP}")
    if not 0 < int(max_rows) <= MAX_ROW_CAP:
        raise ValueError(f"max_rows must be between 1 and {MAX_ROW_CAP}")
    if not 0 < int(max_requests) <= MAX_REQUEST_CAP:
        raise ValueError(f"max_requests must be between 1 and {MAX_REQUEST_CAP}")


def _guards() -> list[str]:
    return [
        "explicit_enable_flag_required",
        "network_disabled_by_default",
        "caller_supplied_adapter_only",
        "scratch_output_directory_required",
        "protected_active_backfill_path_rejected",
        "max_symbol_cap_enforced",
        "max_row_cap_enforced",
        "max_request_cap_enforced",
        "composition_outputs_confined_to_report_dir",
        "no_config_or_api_key_read",
        "no_feature_generation",
        "no_model_training_or_inference",
        "no_replay_or_trading",
    ]


def _safety_flags() -> dict[str, Any]:
    return {
        "caller_supplied_adapter_used": True,
        "provider_collection_invoked": False,
        "real_provider_object_instantiated": False,
        "network_invoked": False,
        "download_invoked": False,
        "api_keys_read": False,
        "config_read": False,
        "canonical_ingest_invoked": False,
        "historical_backfill_invoked": False,
        "active_backfill_path_read": False,
        "corpus_assembly_invoked": False,
        "feature_generation_invoked": False,
        "model_training_invoked": False,
        "model_inference_invoked": False,
        "trading_impact": "none",
        "protected_active_backfill_path_rejected": True,
    }


def _ensure_paths_under_report_dir(
    report_root: Path,
    paths: ProviderScratchDryRunPaths,
) -> None:
    root = report_root.resolve(strict=False)
    for path in (paths.report_json_path, paths.summary_markdown_path, paths.composition_dir):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("provider scratch dry-run outputs must stay under report_dir") from exc


def _contains_protected_path(path: Path) -> bool:
    normalized = path.as_posix()
    resolved = path.resolve(strict=False).as_posix()
    return PROTECTED_ACTIVE_BACKFILL_PATH in normalized or PROTECTED_ACTIVE_BACKFILL_PATH in resolved


def _normalized_symbols(symbols: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        value = _text(symbol).upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row.get("published_at_utc")),
        _text(row.get("provider")),
        _text(row.get("provider_article_id")),
        _text(row.get("symbol")),
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _markdown(report: Mapping[str, Any]) -> str:
    safety = dict(report.get("safety_flags", {}) or {})
    return "\n".join(
        [
            "# Provider Scratch Dry-Run",
            "",
            f"- Schema version: {report['schema_version']}",
            f"- Artifact type: {report['artifact_type']}",
            f"- Provider: {report['provider_id']}",
            f"- Provider family: {report['provider_family']}",
            f"- Enabled: {report['enabled']}",
            f"- Network allowed: {report['network_allowed']}",
            f"- Symbols: {report['symbols']}",
            f"- Start date: {report['start_date']}",
            f"- End date: {report['end_date']}",
            f"- Max symbols: {report['max_symbols']}",
            f"- Max rows: {report['max_rows']}",
            f"- Max requests: {report['max_requests']}",
            f"- Adapter rows: {report['adapter_row_count']}",
            f"- Selected rows: {report['selected_row_count']}",
            f"- Corpus rows: {report['corpus_row_count']}",
            f"- Skipped rows: {report['skipped_row_count']}",
            f"- Excluded rows: {report['excluded_row_count']}",
            f"- Sample skip reasons: {report['sample_skip_reasons']}",
            f"- Corpus skip reasons: {report['corpus_skip_reasons']}",
            f"- Composition report: {report['composition_report_path']}",
            f"- Blockers: {report['blockers']}",
            f"- Warnings: {report['warnings']}",
            f"- Caller-supplied adapter used: {safety['caller_supplied_adapter_used']}",
            f"- Provider collection invoked: {safety['provider_collection_invoked']}",
            f"- Network invoked: {safety['network_invoked']}",
            f"- API keys read: {safety['api_keys_read']}",
            f"- Config read: {safety['config_read']}",
            f"- Historical backfill invoked: {safety['historical_backfill_invoked']}",
            f"- Active backfill path read: {safety['active_backfill_path_read']}",
            f"- Feature generation invoked: {safety['feature_generation_invoked']}",
            f"- Model training invoked: {safety['model_training_invoked']}",
            f"- Model inference invoked: {safety['model_inference_invoked']}",
            f"- Trading impact: {safety['trading_impact']}",
            "",
        ]
    )
