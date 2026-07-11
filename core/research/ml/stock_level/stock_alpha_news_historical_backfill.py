from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from core.research.framework.data import CsvRowRepository
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.stock_level.stock_alpha_news_contract import REQUIRED_NEWS_CONTRACT_COLUMNS
from core.research.ml.stock_level.stock_alpha_news_free_source_collect import (
    PROVENANCE_COLUMNS,
    build_stock_alpha_news_free_source_collect,
)


SCHEMA_VERSION = "alpaca_benzinga_historical_backfill_v1"
MANIFEST_FILENAME = "stock_alpha_news_historical_backfill_manifest.json"
SUMMARY_FILENAME = "stock_alpha_news_historical_backfill_summary.json"
SUMMARY_MARKDOWN_FILENAME = "stock_alpha_news_historical_backfill_summary.md"
ASSEMBLY_FILENAME = "stock_alpha_news_historical_corpus_assembly.csv"
ASSEMBLY_JSON_FILENAME = "stock_alpha_news_historical_corpus_assembly.json"
ASSEMBLY_MARKDOWN_FILENAME = "stock_alpha_news_historical_corpus_assembly.md"
LOGGER = logging.getLogger(__name__)


class HistoricalBackfillManifestIntegrityError(ValueError):
    """Raised when a complete partition manifest record cannot be safely assembled."""


@dataclass(frozen=True)
class StockAlphaNewsHistoricalBackfillPaths:
    manifest_path: Path
    summary_json_path: Path
    summary_markdown_path: Path
    assembly_csv_path: Path | None = None
    assembly_json_path: Path | None = None
    assembly_markdown_path: Path | None = None


@dataclass(frozen=True)
class SymbolUniverseResolution:
    symbols: list[str]
    raw_symbol_row_count: int
    unique_symbol_count: int
    duplicate_symbol_row_count: int
    expected_base_partition_count: int


def write_stock_alpha_news_historical_backfill(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> StockAlphaNewsHistoricalBackfillPaths:
    settings = _settings(config)
    action = str(settings.get("action", "collect")).strip().lower()
    if action == "assemble":
        return write_stock_alpha_news_historical_corpus_assembly(config)
    if action in {"collect_until_done", "collect_until_drained"} or bool(settings.get("run_until_drained", False)):
        payload = build_stock_alpha_news_historical_backfill_until_done(config, sources=sources)
    elif action in {"collect", "backfill"}:
        payload = build_stock_alpha_news_historical_backfill(config, sources=sources)
    else:
        raise ValueError(f"unsupported historical backfill action: {action}")
    paths = _paths(settings)
    writer = ResearchArtifactWriter()
    writer.write_json(paths.manifest_path, payload["manifest"])
    writer.write_json(paths.summary_json_path, payload["summary"])
    writer.write_markdown(paths.summary_markdown_path, _summary_markdown(payload["summary"]))
    return paths


def write_stock_alpha_news_historical_corpus_assembly(
    config: Mapping[str, Any],
) -> StockAlphaNewsHistoricalBackfillPaths:
    settings = _settings(config)
    paths = _paths(settings)
    payload, rows = build_stock_alpha_news_historical_corpus_assembly(config)
    writer = ResearchArtifactWriter()
    assembly_checksum = ""
    if rows:
        writer.write_csv(
            paths.assembly_csv_path,
            rows,
            fieldnames=_fieldnames(rows),
            extrasaction="ignore",
        )
        assembly_checksum = _sha256_file(paths.assembly_csv_path)
    payload["assembly_checksum"] = assembly_checksum
    payload["checksum"] = assembly_checksum
    writer.write_json(paths.assembly_json_path, payload)
    writer.write_markdown(paths.assembly_markdown_path, _assembly_markdown(payload))
    return paths


def build_stock_alpha_news_historical_backfill_until_done(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = _settings(config)
    max_iterations = max(1, int(settings.get("max_iterations", 100) or 100))
    sleep_seconds = max(0.0, float(settings.get("sleep_between_iterations_seconds", 5.0) or 0.0))
    iterations: list[dict[str, Any]] = []
    total_processed = 0
    total_completed = 0
    total_partial = 0
    total_failed = 0
    last_payload: dict[str, Any] | None = None
    stopped_reason = "max_iterations"

    for iteration in range(1, max_iterations + 1):
        last_payload = build_stock_alpha_news_historical_backfill(config, sources=sources)
        summary = dict(last_payload["summary"])
        manifest = dict(last_payload["manifest"])
        iteration_summary = {
            "iteration": iteration,
            "status_counts": dict(summary.get("status_counts", {}) or {}),
            "processed_this_run": int(summary.get("processed_this_run", 0) or 0),
            "skipped_complete": int(summary.get("skipped_complete", 0) or 0),
            "completed_this_run": int(summary.get("completed_this_run", 0) or 0),
            "partial_this_run": int(summary.get("partial_this_run", 0) or 0),
            "failed_this_run": int(summary.get("failed_this_run", 0) or 0),
        }
        iterations.append(iteration_summary)
        total_processed += iteration_summary["processed_this_run"]
        total_completed += iteration_summary["completed_this_run"]
        total_partial += iteration_summary["partial_this_run"]
        total_failed += iteration_summary["failed_this_run"]
        LOGGER.info(
            "historical_backfill_collect_until_done iteration=%s status_counts=%s "
            "processed_this_run=%s skipped_complete=%s completed_this_run=%s "
            "partial_this_run=%s failed_this_run=%s",
            iteration_summary["iteration"],
            iteration_summary["status_counts"],
            iteration_summary["processed_this_run"],
            iteration_summary["skipped_complete"],
            iteration_summary["completed_this_run"],
            iteration_summary["partial_this_run"],
            iteration_summary["failed_this_run"],
        )
        stop_reason = _collect_until_done_stop_reason(summary, manifest)
        if stop_reason:
            stopped_reason = stop_reason
            break
        if iteration < max_iterations and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if last_payload is None:
        last_payload = build_stock_alpha_news_historical_backfill(config, sources=sources)
    final_summary = dict(last_payload["summary"])
    final_summary.update({
        "action": "collect_until_done",
        "run_until_drained": True,
        "collect_until_done": True,
        "iteration_count": len(iterations),
        "max_iterations": max_iterations,
        "sleep_between_iterations_seconds": sleep_seconds,
        "stopped_reason": stopped_reason,
        "final_status_counts": dict(final_summary.get("status_counts", {}) or {}),
        "iteration_summaries": iterations,
        "total_processed_across_iterations": total_processed,
        "total_completed_across_iterations": total_completed,
        "total_partial_across_iterations": total_partial,
        "total_failed_across_iterations": total_failed,
    })
    return {"manifest": last_payload["manifest"], "summary": final_summary}


def build_stock_alpha_news_historical_backfill(
    config: Mapping[str, Any],
    *,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = _settings(config)
    paths = _paths(settings)
    partitions = generate_historical_news_partitions(config)
    symbol_resolution = _symbol_universe_resolution(config)
    manifest = _load_manifest(paths.manifest_path)
    manifest_records = {str(row.get("partition_id")): dict(row) for row in manifest.get("partitions", [])}
    for existing in manifest.get("partitions", []) or []:
        if str(existing.get("parent_partition_id", "")).strip():
            partition = _partition_from_manifest_record(existing)
            if partition["partition_id"] not in {item["partition_id"] for item in partitions}:
                partitions.append(partition)
    planned = [_initial_manifest_record(partition) for partition in partitions]
    for record in planned:
        manifest_records.setdefault(record["partition_id"], record)
    max_partitions = max(0, int(settings.get("max_partitions_per_run", 0) or 0))
    dry_run = bool(settings.get("dry_run", False))
    processed = 0
    skipped_complete = 0
    failed = 0
    partial = 0
    completed = 0
    for partition in _partition_processing_order(partitions, manifest_records):
        record = manifest_records[partition["partition_id"]]
        if record.get("status") == "complete" and _complete_record_valid(record):
            skipped_complete += 1
            continue
        if _partial_partition_superseded(record, manifest_records.values()):
            continue
        if max_partitions and processed >= max_partitions:
            break
        if dry_run:
            record.update({"status": "pending", "last_error": ""})
            continue
        processed += 1
        _run_partition(config, settings, partition, record, sources=sources)
        if record["status"] == "partial":
            for child in _child_partitions_for_dense_partition(partition, settings):
                if child["partition_id"] in manifest_records:
                    continue
                child_record = _initial_manifest_record(child)
                child_record["parent_partition_id"] = partition["partition_id"]
                child_record["parent_split_reason"] = "stopped_with_more_results_available"
                manifest_records[child["partition_id"]] = child_record
                partitions.append(child)
            record["child_partition_ids"] = [
                child["partition_id"]
                for child in _child_partitions_for_dense_partition(partition, settings)
            ]
            record["continuation_strategy"] = "adaptive_date_split" if record["child_partition_ids"] else "manual_review"
        _atomic_write_json(paths.manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _now_utc(),
            "action": "collect",
            "dry_run": dry_run,
            "partition_count": len(partitions),
            "partitions": [manifest_records[item["partition_id"]] for item in partitions],
        })
        if record["status"] == "complete":
            completed += 1
        elif record["status"] == "partial":
            partial += 1
        elif record["status"] == "failed":
            failed += 1
    ordered_records = [manifest_records[partition["partition_id"]] for partition in partitions]
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _now_utc(),
        "action": "collect",
        "dry_run": dry_run,
        "partition_count": len(ordered_records),
        "partitions": ordered_records,
    }
    summary = _backfill_summary(
        output_manifest,
        processed=processed,
        skipped_complete=skipped_complete,
        completed=completed,
        partial=partial,
        failed=failed,
        paths=paths,
        symbol_resolution=symbol_resolution,
    )
    return {"manifest": output_manifest, "summary": summary}


def build_stock_alpha_news_historical_corpus_assembly(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = _settings(config)
    paths = _paths(settings)
    manifest = _load_manifest(paths.manifest_path)
    incomplete = [
        row for row in manifest.get("partitions", [])
        if not _partition_resolved_for_assembly(row, manifest.get("partitions", []))
    ]
    if incomplete and bool(settings.get("assembly_require_all_complete", True)):
        return _assembly_payload(manifest, [], incomplete, paths), []
    rows: list[dict[str, Any]] = []
    for record in manifest.get("partitions", []):
        if record.get("status") != "complete":
            continue
        rows.extend(_read_complete_partition_rows(record))
    valid_rows, invalid_rows = _valid_contract_rows(rows)
    deduplicated = _deduplicate_rows(valid_rows)
    return _assembly_payload(manifest, deduplicated, incomplete, paths, invalid_row_count=invalid_rows), deduplicated


def generate_historical_news_partitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    settings = _settings(config)
    provider = str(settings.get("provider", "alpaca_benzinga")).strip()
    months = _monthly_windows(str(settings["start_date"]), str(settings["end_date"]))
    symbols = _symbol_universe_resolution(config).symbols
    batch_size = max(1, int(settings.get("symbol_batch_size", settings.get("symbols_per_batch", 25)) or 25))
    batches = [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]
    partitions: list[dict[str, Any]] = []
    for month_start, month_end in months:
        for batch_index, batch_symbols in enumerate(batches, start=1):
            identity = _partition_identity(
                provider=provider,
                start_date=month_start,
                end_date=month_end,
                symbol_batch_id=f"symbol_batch_{batch_index:03d}",
                symbols=batch_symbols,
                settings=settings,
            )
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            partition_id = f"{provider}_{month_start[:7]}_symbol_batch_{batch_index:03d}_{digest}"
            partitions.append({
                **identity,
                "partition_id": partition_id,
                "identity_hash": digest,
            })
    return partitions


def _partition_identity(
    *,
    provider: str,
    start_date: str,
    end_date: str,
    symbol_batch_id: str,
    symbols: list[str],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "provider_config_hash": _provider_config_hash(settings),
        "start_date": start_date,
        "end_date": end_date,
        "symbol_batch_id": symbol_batch_id,
        "symbols": symbols,
    }


def _provider_config_hash(settings: Mapping[str, Any]) -> str:
    relevant = {
        "provider_config": dict(settings.get("provider_config", {}) or {}),
        "provider_request_limit": int(settings.get("provider_request_limit", 250)),
        "max_rows_per_partition": int(settings.get("max_rows_per_partition", 10_000)),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _partition_from_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(record.get("schema_version", SCHEMA_VERSION)),
        "provider": str(record.get("provider", "")),
        "provider_config_hash": str(record.get("provider_config_hash", "")),
        "start_date": str(record.get("start_date", "")),
        "end_date": str(record.get("end_date", "")),
        "symbol_batch_id": str(record.get("symbol_batch_id", "")),
        "symbols": list(record.get("symbols", []) or []),
        "partition_id": str(record.get("partition_id", "")),
        "identity_hash": str(record.get("identity_hash", "")),
        "parent_partition_id": str(record.get("parent_partition_id", "")),
    }


def _partial_partition_superseded(
    record: Mapping[str, Any],
    records: Any,
) -> bool:
    if str(record.get("status", "")) != "partial":
        return False
    partition_id = str(record.get("partition_id", ""))
    if not partition_id:
        return False
    child_ids = [str(value) for value in record.get("child_partition_ids", []) or [] if str(value).strip()]
    if child_ids:
        return True
    return any(
        str(candidate.get("parent_partition_id", "")) == partition_id
        for candidate in records or []
    )


def _partition_processing_order(
    partitions: list[dict[str, Any]],
    manifest_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = list(manifest_records.values())
    child_partitions: list[dict[str, Any]] = []
    other_partitions: list[dict[str, Any]] = []
    for partition in partitions:
        record = manifest_records.get(str(partition["partition_id"]), {})
        if (
            str(record.get("parent_partition_id", "")).strip()
            and not _partial_partition_superseded(record, records)
        ):
            child_partitions.append(partition)
        else:
            other_partitions.append(partition)
    return [*child_partitions, *other_partitions]


def _collect_until_done_stop_reason(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    provider_reason = _blocking_provider_stop_reason(manifest)
    if provider_reason:
        return provider_reason
    if int(summary.get("failed_this_run", 0) or 0) > 0:
        return "failed_partitions"
    if int(summary.get("processed_this_run", 0) or 0) == 0:
        return "drained"
    return ""


def _blocking_provider_stop_reason(manifest: Mapping[str, Any]) -> str:
    blocking_fields = [
        ("provider_skipped_missing_key", "provider_skipped_missing_key"),
        ("provider_entitlement_failed", "provider_entitlement_failed"),
        ("provider_rate_limited", "provider_rate_limited"),
        ("provider_failed", "provider_failed"),
    ]
    for record in manifest.get("partitions", []) or []:
        if str(record.get("status", "")) != "failed":
            continue
        for field, reason in blocking_fields:
            if bool(record.get(field, False)):
                return reason
        if bool(record.get("provider_requested", False)) and not bool(record.get("provider_attempted", False)):
            return "provider_not_attempted"
        last_error = str(record.get("last_error", "")).strip()
        if last_error in {
            "provider_not_requested",
            "provider_not_attempted",
            "provider_skipped_missing_key",
            "rate_limited",
            "entitlement_error",
            "provider_zero_batches",
            "provider_attempted_without_pages",
            "provider_attempted_without_termination_reason",
        }:
            return last_error
    return ""


def _partition_resolved_for_assembly(
    record: Mapping[str, Any],
    records: Any,
    *,
    _seen: set[str] | None = None,
) -> bool:
    status = str(record.get("status", ""))
    if status == "complete":
        return True
    if status != "partial":
        return False
    partition_id = str(record.get("partition_id", ""))
    if not partition_id:
        return False
    seen = set(_seen or set())
    if partition_id in seen:
        return False
    seen.add(partition_id)
    children = [
        child for child in records or []
        if str(child.get("parent_partition_id", "")) == partition_id
    ]
    if not children:
        return False
    return all(
        _partition_resolved_for_assembly(child, records, _seen=seen)
        for child in children
    )


def _child_partitions_for_dense_partition(
    partition: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    start = date.fromisoformat(str(partition["start_date"])[:10])
    end = date.fromisoformat(str(partition["end_date"])[:10])
    if start >= end:
        return []
    midpoint = start + timedelta(days=(end - start).days // 2)
    windows = [
        (start, midpoint),
        (midpoint + timedelta(days=1), end),
    ]
    children: list[dict[str, Any]] = []
    for split_index, (child_start, child_end) in enumerate(windows, start=1):
        if child_start > child_end:
            continue
        identity = _partition_identity(
            provider=str(partition["provider"]),
            start_date=child_start.isoformat(),
            end_date=child_end.isoformat(),
            symbol_batch_id=f"{partition['symbol_batch_id']}_split_{split_index:02d}",
            symbols=list(partition.get("symbols", []) or []),
            settings=settings,
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        children.append({
            **identity,
            "partition_id": (
                f"{identity['provider']}_{identity['start_date'][:7]}_"
                f"{identity['symbol_batch_id']}_{digest}"
            ),
            "identity_hash": digest,
            "parent_partition_id": partition["partition_id"],
        })
    return children


def _run_partition(
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    partition: Mapping[str, Any],
    record: dict[str, Any],
    *,
    sources: Mapping[str, Any] | None,
) -> None:
    record["status"] = "running"
    record["attempt_count"] = int(record.get("attempt_count", 0) or 0) + 1
    record["started_at"] = _now_utc()
    record["completed_at"] = ""
    record["last_error"] = ""
    partition_config = _partition_collect_config(config, settings, partition)
    try:
        payload, rows = build_stock_alpha_news_free_source_collect(partition_config, sources=sources)
        diagnostic = _provider_diagnostic(payload, str(partition["provider"]))
        _update_record_from_payload(record, payload, diagnostic, rows)
        provider_failure = _provider_failure_reason(payload, str(partition["provider"]))
        if provider_failure:
            record["status"] = "failed"
            record["last_error"] = provider_failure
            _write_partition_audit(settings, partition, record, payload, diagnostic)
            return
        completion_blocker = _provider_completion_blocker(
            payload,
            diagnostic,
            str(partition["provider"]),
        )
        if completion_blocker:
            record["status"] = "failed"
            record["last_error"] = completion_blocker
            _write_partition_audit(settings, partition, record, payload, diagnostic)
            return
        if bool(diagnostic.get(f"{partition['provider']}_stopped_with_more_results_available", False)):
            record["status"] = "partial"
            record["last_error"] = "partition stopped with more results available"
            _write_partition_audit(settings, partition, record, payload, diagnostic)
            return
        artifact = _partition_artifact_path(settings, partition)
        if rows:
            _atomic_write_csv(artifact, rows)
            record["output_artifact"] = str(artifact)
            record["output_row_count"] = len(rows)
            record["artifact_size"] = artifact.stat().st_size
            record["checksum"] = _sha256_file(artifact)
        else:
            record["output_artifact"] = ""
            record["output_row_count"] = 0
            record["artifact_size"] = 0
            record["checksum"] = ""
        record["status"] = "complete"
        _write_partition_audit(settings, partition, record, payload, diagnostic)
    except Exception as exc:
        record["status"] = "failed"
        record["last_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["completed_at"] = _now_utc()


def _partition_collect_config(
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    provider = str(partition["provider"])
    provider_config = dict(settings.get("provider_config", {}) or {})
    provider_config.setdefault("enabled", True)
    provider_config.setdefault("api_key_env", "ALPACA_API_KEY_ID")
    provider_config.setdefault("secret_key_env", "ALPACA_SECRET_KEY")
    collect = {
        "enabled": True,
        "dry_run": True,
        "allow_overwrite": False,
        "merge_existing": False,
        "backup_existing": False,
        "start_date": partition["start_date"],
        "end_date": partition["end_date"],
        "symbols": list(partition["symbols"]),
        "symbols_per_batch": len(partition["symbols"]),
        "max_articles_per_provider": int(settings.get("provider_request_limit", 250)),
        "provider_request_limit": int(settings.get("provider_request_limit", 250)),
        "max_rows_per_provider": int(settings.get("max_rows_per_partition", 10_000)),
        "max_symbols_per_run": len(partition["symbols"]),
        "request_timeout_seconds": int(settings.get("request_timeout_seconds", 20)),
        "rate_limit_sleep_seconds": float(settings.get("rate_limit_sleep_seconds", 0.0)),
        "providers": {provider: provider_config},
    }
    return {
        **dict(config),
        "ml": {
            **dict(config.get("ml", {}) or {}),
            "stock_alpha_news_collect_report_dir": str(Path(settings["work_dir"]) / "partition_reports" / partition["partition_id"]),
            "stock_alpha_news_collect_output_path": str(_partition_artifact_path(settings, partition)),
            "stock_alpha_news_collect": collect,
            "stock_alpha_news_enable_transformer": False,
        },
    }


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    settings = dict(ml.get("stock_alpha_news_historical_backfill", {}) or {})
    if not settings:
        raise ValueError("missing ml.stock_alpha_news_historical_backfill")
    return settings


def _paths(settings: Mapping[str, Any]) -> StockAlphaNewsHistoricalBackfillPaths:
    work_dir = Path(str(settings.get("work_dir", "")))
    if not work_dir:
        raise ValueError("missing ml.stock_alpha_news_historical_backfill.work_dir")
    return StockAlphaNewsHistoricalBackfillPaths(
        manifest_path=work_dir / MANIFEST_FILENAME,
        summary_json_path=work_dir / SUMMARY_FILENAME,
        summary_markdown_path=work_dir / SUMMARY_MARKDOWN_FILENAME,
        assembly_csv_path=work_dir / ASSEMBLY_FILENAME,
        assembly_json_path=work_dir / ASSEMBLY_JSON_FILENAME,
        assembly_markdown_path=work_dir / ASSEMBLY_MARKDOWN_FILENAME,
    )


def _resolved_symbols(config: Mapping[str, Any]) -> list[str]:
    return _symbol_universe_resolution(config).symbols


def _symbol_universe_resolution(config: Mapping[str, Any]) -> SymbolUniverseResolution:
    settings = _settings(config)
    raw_values = list(settings.get("symbols", []) or [])
    if not raw_values and bool(settings.get("use_canonical_universe", False)):
        stock_path = Path(str(dict(config.get("ml", {}) or {}).get("stock_alpha_stock_rows_path", "")))
        rows = CsvRowRepository().read(stock_path) if stock_path.is_file() else []
        raw_values = [row.get("symbol", "") for row in rows]
    raw_symbol_row_count = len(raw_values)
    source_symbols = _normalized_symbols(raw_values)
    source_unique_symbols = _unique_preserving_order(source_symbols)
    duplicate_symbol_row_count = len(source_symbols) - len(source_unique_symbols)
    symbols = list(source_unique_symbols)
    only = set(_symbols(settings.get("only_symbols", [])))
    if only:
        symbols = [symbol for symbol in symbols if symbol in only]
    max_symbols = int(settings.get("max_symbols", 0) or 0)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    if not symbols:
        raise ValueError("historical backfill resolved zero symbols")
    months = _monthly_windows(str(settings["start_date"]), str(settings["end_date"]))
    batch_size = max(1, int(settings.get("symbol_batch_size", settings.get("symbols_per_batch", 25)) or 25))
    batch_count = (len(symbols) + batch_size - 1) // batch_size
    return SymbolUniverseResolution(
        symbols=symbols,
        raw_symbol_row_count=raw_symbol_row_count,
        unique_symbol_count=len(symbols),
        duplicate_symbol_row_count=duplicate_symbol_row_count,
        expected_base_partition_count=len(months) * batch_count,
    )


def _monthly_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_date[:10])
    end = date.fromisoformat(end_date[:10])
    if start > end:
        raise ValueError("historical backfill start_date must be <= end_date")
    windows: list[tuple[str, str]] = []
    current = start.replace(day=1)
    while current <= end:
        next_month = _next_month(current)
        month_start = max(start, current)
        month_end = min(end, next_month - timedelta(days=1))
        windows.append((month_start.isoformat(), month_end.isoformat()))
        current = next_month
    return windows


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _initial_manifest_record(partition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "partition_id": partition["partition_id"],
        "identity_hash": partition["identity_hash"],
        "schema_version": SCHEMA_VERSION,
        "provider": partition["provider"],
        "provider_config_hash": partition.get("provider_config_hash", ""),
        "parent_partition_id": partition.get("parent_partition_id", ""),
        "symbol_batch_id": partition["symbol_batch_id"],
        "symbols": list(partition["symbols"]),
        "start_date": partition["start_date"],
        "end_date": partition["end_date"],
        "status": "pending",
        "attempt_count": 0,
        "started_at": "",
        "completed_at": "",
        "last_error": "",
        "pages_requested": 0,
        "pages_completed": 0,
        "provider_records_returned": 0,
        "unique_provider_articles": 0,
        "article_symbol_rows": 0,
        "multi_symbol_expansion_rows": 0,
        "out_of_window_rejected_count": 0,
        "out_of_window_before_start_count": 0,
        "out_of_window_after_end_count": 0,
        "earliest_accepted_published_at": "",
        "latest_accepted_published_at": "",
        "termination_reason": "",
        "stopped_with_more_results_available": False,
        "provider_requested": False,
        "provider_attempted": False,
        "provider_skipped_missing_key": False,
        "provider_failed": False,
        "provider_rate_limited": False,
        "provider_entitlement_failed": False,
        "provider_returned_zero_rows": False,
        "provider_zero_row_reason": "",
        "provider_batch_count": 0,
        "output_artifact": "",
        "audit_artifact": "",
        "output_row_count": 0,
        "artifact_size": 0,
        "checksum": "",
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "partitions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_valid(record: Mapping[str, Any]) -> bool:
    artifact = Path(str(record.get("output_artifact", "")))
    if int(record.get("output_row_count", 0) or 0) == 0:
        return True
    if not artifact.is_file():
        return False
    checksum = str(record.get("checksum", ""))
    return not checksum or _sha256_file(artifact) == checksum


def _complete_record_valid(record: Mapping[str, Any]) -> bool:
    if not _artifact_valid(record):
        return False
    provider = str(record.get("provider", "")).strip()
    if provider == "alpaca_benzinga":
        return _alpaca_benzinga_completion_evidence_valid(record)
    return True


def _alpaca_benzinga_completion_evidence_valid(record: Mapping[str, Any]) -> bool:
    if bool(record.get("provider_skipped_missing_key", False)):
        return False
    if bool(record.get("provider_failed", False)):
        return False
    if bool(record.get("provider_rate_limited", False)):
        return False
    if bool(record.get("provider_entitlement_failed", False)):
        return False
    if not bool(record.get("provider_requested", False)):
        return False
    if not bool(record.get("provider_attempted", False)):
        return False
    if int(record.get("provider_batch_count", 0) or 0) < 1:
        return False
    if int(record.get("pages_requested", 0) or 0) < 1:
        return False
    if not str(record.get("termination_reason", "")).strip():
        return False
    return True


def _read_complete_partition_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    partition_id = str(record.get("partition_id", "<missing>"))
    if bool(record.get("stopped_with_more_results_available", False)):
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} has stopped_with_more_results_available=true"
        )
    output_row_count = int(record.get("output_row_count", 0) or 0)
    artifact_value = record.get("output_artifact")
    artifact_text = "" if artifact_value is None else str(artifact_value).strip()
    if output_row_count == 0 and not artifact_text:
        return []
    if not artifact_text:
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} has blank output_artifact"
        )
    artifact = Path(artifact_text)
    if artifact.name == "." or artifact_text in {".", "./"}:
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} has invalid output_artifact: {artifact_text}"
        )
    if artifact.suffix.lower() != ".csv":
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} output_artifact is not a CSV path: {artifact}"
        )
    if not artifact.exists():
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} output_artifact does not exist: {artifact}"
        )
    if not artifact.is_file():
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} output_artifact is not a regular file: {artifact}"
        )
    checksum = str(record.get("checksum", "")).strip()
    if output_row_count > 0 and not checksum:
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} has blank checksum"
        )
    if checksum and _sha256_file(artifact) != checksum:
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} checksum mismatch for output_artifact: {artifact}"
        )
    rows = CsvRowRepository().read(artifact)
    if len(rows) != output_row_count:
        raise HistoricalBackfillManifestIntegrityError(
            f"complete partition {partition_id} output_row_count={output_row_count} "
            f"does not match artifact row count={len(rows)}"
        )
    return rows


def _provider_diagnostic(payload: Mapping[str, Any], provider: str) -> dict[str, Any]:
    for diagnostic in payload.get("provider_batch_diagnostics", []) or []:
        if str(diagnostic.get("provider", "")) == provider:
            return dict(diagnostic)
    return {}


def _provider_failure_reason(payload: Mapping[str, Any], provider: str) -> str:
    failures = dict(payload.get("providers_failed", {}) or {})
    if provider in failures:
        return str(failures[provider])
    if provider in set(payload.get("providers_rate_limited", []) or []):
        return "rate_limited"
    if provider in set(payload.get("providers_entitlement_failed", []) or []):
        return "entitlement_error"
    return ""


def _provider_completion_blocker(
    payload: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    provider: str,
) -> str:
    requested = set(payload.get("providers_requested", []) or [])
    attempted = set(payload.get("providers_attempted", []) or [])
    skipped = set(payload.get("providers_skipped_missing_key", []) or [])
    failures = dict(payload.get("providers_failed", {}) or {})
    rate_limited = set(payload.get("providers_rate_limited", []) or [])
    entitlement_failed = set(payload.get("providers_entitlement_failed", []) or [])
    batch_counts = dict(payload.get("provider_batch_counts", {}) or {})
    if provider not in requested:
        return "provider_not_requested"
    if provider in skipped:
        return "provider_skipped_missing_key"
    if provider in failures:
        return str(failures[provider])
    if provider in rate_limited:
        return "rate_limited"
    if provider in entitlement_failed:
        return "entitlement_error"
    if provider not in attempted:
        return "provider_not_attempted"
    if int(batch_counts.get(provider, 0) or 0) < 1:
        return "provider_zero_batches"
    if provider == "alpaca_benzinga":
        pages_requested = int(diagnostic.get(f"{provider}_pages_requested", 0) or 0)
        termination_reason = str(diagnostic.get(f"{provider}_termination_reason", "")).strip()
        if pages_requested < 1:
            return "provider_attempted_without_pages"
        if not termination_reason:
            return "provider_attempted_without_termination_reason"
    return ""


def _body_full_text_availability(rows: list[dict[str, Any]]) -> float:
    return _field_availability(rows, "body_or_full_text")


def _field_availability(rows: list[dict[str, Any]], field: str) -> float:
    return (
        sum(1 for row in rows if str(row.get(field, "")).strip()) / len(rows)
        if rows else 0.0
    )


def _update_record_from_payload(
    record: dict[str, Any],
    payload: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    provider = str(record["provider"])
    published = sorted(str(row.get("published_at_utc", "")) for row in rows if row.get("published_at_utc"))
    requested = set(payload.get("providers_requested", []) or [])
    attempted = set(payload.get("providers_attempted", []) or [])
    skipped = set(payload.get("providers_skipped_missing_key", []) or [])
    failures = dict(payload.get("providers_failed", {}) or {})
    rate_limited = set(payload.get("providers_rate_limited", []) or [])
    entitlement_failed = set(payload.get("providers_entitlement_failed", []) or [])
    returned_zero = set(payload.get("providers_returned_zero_rows", []) or [])
    zero_reasons = dict(payload.get("provider_zero_row_reasons", {}) or {})
    batch_counts = dict(payload.get("provider_batch_counts", {}) or {})
    record.update({
        "pages_requested": int(diagnostic.get(f"{provider}_pages_requested", 0) or 0),
        "pages_completed": int(diagnostic.get(f"{provider}_pages_completed", 0) or 0),
        "termination_reason": str(diagnostic.get(f"{provider}_termination_reason", "")),
        "stopped_with_more_results_available": bool(diagnostic.get(f"{provider}_stopped_with_more_results_available", False)),
        "provider_records_returned": int(diagnostic.get(f"{provider}_provider_records_returned", 0) or 0),
        "unique_provider_articles": int(diagnostic.get(f"{provider}_unique_provider_articles", 0) or 0),
        "article_symbol_rows": len(rows),
        "multi_symbol_expansion_rows": int(diagnostic.get(f"{provider}_multi_symbol_expansion_row_count", 0) or 0),
        "out_of_window_rejected_count": int(payload.get("out_of_window_rejected_count", 0) or 0),
        "out_of_window_before_start_count": int(payload.get("out_of_window_before_start_count", 0) or 0),
        "out_of_window_after_end_count": int(payload.get("out_of_window_after_end_count", 0) or 0),
        "exact_duplicate_provider_record_count": int(diagnostic.get(f"{provider}_exact_duplicate_provider_record_count", 0) or 0),
        "headline_availability_rate": float(diagnostic.get(f"{provider}_headline_availability_rate", 0.0) or 0.0),
        "summary_availability_rate": float(diagnostic.get(f"{provider}_summary_availability_rate", 0.0) or 0.0),
        "body_or_full_text_availability_rate": _body_full_text_availability(rows),
        "source_availability_rate": _field_availability(rows, "source"),
        "publisher_availability_rate": _field_availability(rows, "publisher"),
        "author_availability_rate": _field_availability(rows, "author"),
        "earliest_accepted_published_at": published[0] if published else "",
        "latest_accepted_published_at": published[-1] if published else "",
        "provider_requested": provider in requested,
        "provider_attempted": provider in attempted,
        "provider_skipped_missing_key": provider in skipped,
        "provider_failed": provider in failures,
        "provider_rate_limited": provider in rate_limited,
        "provider_entitlement_failed": provider in entitlement_failed,
        "provider_returned_zero_rows": provider in returned_zero,
        "provider_zero_row_reason": str(zero_reasons.get(provider, "")),
        "provider_batch_count": int(batch_counts.get(provider, 0) or 0),
    })


def _partition_artifact_path(settings: Mapping[str, Any], partition: Mapping[str, Any]) -> Path:
    return (
        Path(str(settings["work_dir"]))
        / "partitions"
        / str(partition["provider"])
        / f"year={str(partition['start_date'])[:4]}"
        / f"month={str(partition['start_date'])[5:7]}"
        / f"symbol_batch={str(partition['symbol_batch_id']).replace('symbol_batch_', '')}"
        / f"{partition['partition_id']}.csv"
    )


def _partition_audit_path(settings: Mapping[str, Any], partition: Mapping[str, Any]) -> Path:
    return _partition_artifact_path(settings, partition).with_suffix(".audit.json")


def _write_partition_audit(
    settings: Mapping[str, Any],
    partition: Mapping[str, Any],
    record: dict[str, Any],
    payload: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> None:
    audit_path = _partition_audit_path(settings, partition)
    record["audit_artifact"] = str(audit_path)
    _atomic_write_json(audit_path, _partition_audit(record, payload, diagnostic))


def _partition_audit(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "partition": dict(record),
        "diagnostic": dict(diagnostic),
        "provider_execution": {
            "providers_requested": list(payload.get("providers_requested", []) or []),
            "providers_attempted": list(payload.get("providers_attempted", []) or []),
            "providers_skipped_missing_key": list(payload.get("providers_skipped_missing_key", []) or []),
            "providers_returned_zero_rows": list(payload.get("providers_returned_zero_rows", []) or []),
            "provider_zero_row_reasons": dict(payload.get("provider_zero_row_reasons", {}) or {}),
            "provider_batch_counts": dict(payload.get("provider_batch_counts", {}) or {}),
        },
        "providers_failed": dict(payload.get("providers_failed", {}) or {}),
        "providers_rate_limited": list(payload.get("providers_rate_limited", []) or []),
        "providers_entitlement_failed": list(payload.get("providers_entitlement_failed", []) or []),
    }


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    tmp = path.with_name(f".{path.name}.tmp")
    ResearchArtifactWriter().write_csv(tmp, rows, fieldnames=fieldnames, extrasaction="ignore")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    ordered = [*REQUIRED_NEWS_CONTRACT_COLUMNS, *PROVENANCE_COLUMNS]
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered and key not in extras:
                extras.append(str(key))
    return [*ordered, *extras]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_symbols(values: Any) -> list[str]:
    return [
        str(value).strip().upper()
        for value in values or []
        if str(value).strip()
    ]


def _unique_preserving_order(values: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _symbols(values: Any) -> list[str]:
    return _unique_preserving_order(_normalized_symbols(values))


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("provider", "")).strip(),
            str(row.get("provider_article_id", "")).strip(),
            str(row.get("symbol", "")).strip().upper(),
            str(row.get("published_at_utc", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _backfill_summary(
    manifest: Mapping[str, Any],
    *,
    processed: int,
    skipped_complete: int,
    completed: int,
    partial: int,
    failed: int,
    paths: StockAlphaNewsHistoricalBackfillPaths,
    symbol_resolution: SymbolUniverseResolution,
) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "")) for row in manifest.get("partitions", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "collect",
        "manifest_path": str(paths.manifest_path),
        "partition_count": len(manifest.get("partitions", [])),
        "raw_symbol_row_count": symbol_resolution.raw_symbol_row_count,
        "unique_symbol_count": symbol_resolution.unique_symbol_count,
        "duplicate_symbol_row_count": symbol_resolution.duplicate_symbol_row_count,
        "expected_base_partition_count": symbol_resolution.expected_base_partition_count,
        "status_counts": dict(sorted(statuses.items())),
        "processed_this_run": processed,
        "skipped_complete": skipped_complete,
        "completed_this_run": completed,
        "partial_this_run": partial,
        "failed_this_run": failed,
        "collection_only": True,
        "contract_ingest_invoked": False,
        "features_generated": False,
        "model_training_invoked": False,
        "news_transformer_enabled": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _assembly_payload(
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    incomplete: list[Mapping[str, Any]],
    paths: StockAlphaNewsHistoricalBackfillPaths,
    *,
    invalid_row_count: int = 0,
) -> dict[str, Any]:
    symbols = sorted({str(row.get("symbol", "")).strip().upper() for row in rows if str(row.get("symbol", "")).strip()})
    published = sorted(str(row.get("published_at_utc", "")) for row in rows if row.get("published_at_utc"))
    year_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        published_at = str(row.get("published_at_utc", ""))
        if len(published_at) >= 4:
            year_rows[published_at[:4]].append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "action": "assemble",
        "manifest_path": str(paths.manifest_path),
        "assembly_csv_path": str(paths.assembly_csv_path),
        "assembly_checksum": "",
        "checksum": "",
        "complete_partition_count": sum(1 for row in manifest.get("partitions", []) if row.get("status") == "complete"),
        "incomplete_partition_count": len(incomplete),
        "incomplete_partitions": [row.get("partition_id") for row in incomplete],
        "row_count": len(rows),
        "invalid_required_field_row_count": invalid_row_count,
        "unique_provider_article_count": len({(row.get("provider"), row.get("provider_article_id")) for row in rows}),
        "symbol_count": len(symbols),
        "symbols": symbols,
        "min_published_at_utc": published[0] if published else "",
        "max_published_at_utc": published[-1] if published else "",
        "text_availability_by_year": {
            year: _text_availability(year_values)
            for year, year_values in sorted(year_rows.items())
        },
        "source_distribution": dict(sorted(Counter(str(row.get("source", "")).strip() for row in rows).items())),
        "publisher_availability_rate": _availability(rows, "publisher"),
        "author_availability_rate": _availability(rows, "author"),
        "raw_source_availability_rate": _availability(rows, "raw_source"),
        "contract_ingest_invoked": False,
        "features_generated": False,
        "model_training_invoked": False,
        "trading_impact": "none",
        "production_validated": False,
    }


def _text_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "headline_availability_rate": _availability(rows, "headline"),
        "summary_availability_rate": _availability(rows, "summary"),
        "body_or_full_text_availability_rate": _availability(rows, "body_or_full_text"),
        "article_symbol_rows": len(rows),
        "unique_article_count": len({str(row.get("provider_article_id", "")).strip() for row in rows if row.get("provider_article_id")}),
    }


def _valid_contract_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    for row in rows:
        if all(field in row for field in REQUIRED_NEWS_CONTRACT_COLUMNS):
            valid.append(row)
        else:
            invalid += 1
    return valid, invalid


def _availability(rows: list[dict[str, Any]], field: str) -> float:
    return (
        sum(1 for row in rows if str(row.get(field, "")).strip()) / len(rows)
        if rows else 0.0
    )


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Stock-Alpha Historical News Backfill",
        "",
        f"- Action: {payload['action']}",
        f"- Manifest: {payload['manifest_path']}",
        f"- Partitions: {payload['partition_count']}",
        f"- Raw symbol rows: {payload['raw_symbol_row_count']}",
        f"- Unique symbols: {payload['unique_symbol_count']}",
        f"- Duplicate symbol rows: {payload['duplicate_symbol_row_count']}",
        f"- Expected base partitions: {payload['expected_base_partition_count']}",
        f"- Status counts: {payload['status_counts']}",
        f"- Processed this run: {payload['processed_this_run']}",
        f"- Skipped complete: {payload['skipped_complete']}",
        f"- Completed this run: {payload['completed_this_run']}",
        f"- Partial this run: {payload['partial_this_run']}",
        f"- Failed this run: {payload['failed_this_run']}",
        "- Model training invoked: false",
        "- Trading impact: none",
    ])


def _assembly_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Stock-Alpha Historical News Corpus Assembly",
        "",
        f"- Rows: {payload['row_count']}",
        f"- Complete partitions: {payload['complete_partition_count']}",
        f"- Incomplete partitions: {payload['incomplete_partition_count']}",
        f"- Symbols: {payload['symbol_count']}",
        f"- Published range: {payload['min_published_at_utc']} to {payload['max_published_at_utc']}",
        f"- Text availability by year: {payload['text_availability_by_year']}",
        "- Contract ingest invoked: false",
        "- Model training invoked: false",
        "- Trading impact: none",
    ])


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
