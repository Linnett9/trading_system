from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from core.research.ml.reference.canonical_assets import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    build_dataset_manifest,
    daily_spine_row_id,
    file_sha256,
    read_aliases_csv,
    read_assets_csv,
)
from core.research.ml.stock_level.prediction_artifacts.types import (
    ACTUAL_COLUMNS,
    CONTEXT_COLUMNS,
    PREDICTION_COLUMNS,
    TARGET_PROVENANCE_COLUMNS,
)
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    ENGINEERED_FEATURE_COLUMNS,
    ENRICHMENT_METADATA_COLUMNS,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    iter_stock_level_artifact_batches,
    read_stock_level_artifact,
)


STATUS_READY = "READY"
STATUS_BLOCKED = "BLOCKED"
DEFAULT_STREAM_BATCH_SIZE = 65536
MAX_REPORT_ROWS = 1000
SPINE_SCHEMA_VERSION = "canonical_daily_stock_spine.v1"
PRICE_FEATURE_SCHEMA_VERSION = "daily_price_feature_registration.v1"
ROW_IDENTITY_VERSION = "canonical_daily_stock_row_id.v1"
SPINE_COLUMNS = (
    "row_id",
    "source_row_id",
    "asset_id",
    "canonical_symbol",
    "source_symbol",
    "session_date",
    "decision_timestamp",
    "feature_cutoff_timestamp",
    "universe_version",
    "eligible_at_decision",
    "eligibility_reason",
    "daily_price_dataset_version",
    "symbol_registry_version",
    "calendar_version",
    "target_horizon_sessions",
    "target_start_timestamp",
    "target_end_timestamp",
    "target_available_timestamp",
    "target_definition_version",
    "benchmark_asset_id",
    "stock_target",
    "benchmark_return",
    "excess_return",
    "source_artifact_path",
    "source_artifact_dataset_id",
)
IDENTITY_COLUMNS = {
    "row_id",
    "source_row_id",
    "asset_id",
    "canonical_symbol",
    "source_symbol",
    "symbol",
    "rebalance_date",
}
TIMING_COLUMNS = {
    "session_date",
    "decision_session_date",
    "decision_timestamp",
    "feature_timestamp",
    "feature_data_cutoff_timestamp",
    "first_actionable_session",
    "target_start_timestamp",
    "label_start_timestamp",
    "label_end_timestamp",
    "label_available_timestamp",
    "benchmark_target_start_timestamp",
    "benchmark_label_start_timestamp",
    "benchmark_label_end_timestamp",
    "benchmark_label_available_timestamp",
}
TARGET_COLUMNS = set(ACTUAL_COLUMNS) | {"stock_target", "target_horizon", "target_horizon_trading_days", "target_status"}
BENCHMARK_COLUMNS = {"benchmark_symbol", "actual_benchmark_return_10d"}
PROVENANCE_COLUMNS = set(TARGET_PROVENANCE_COLUMNS) | {
    "source",
    "source_feature_id",
    "source_model_type",
    "source_split",
    "source_dataset_hash",
    "target_provenance_contract_version",
    "decision_grid_version",
    "decision_grid_identity",
    "exchange_calendar_identity",
}
FEATURE_COLUMNS = set(PREDICTION_COLUMNS) | set(CONTEXT_COLUMNS) | set(ENGINEERED_FEATURE_COLUMNS)
DIAGNOSTIC_COLUMNS = set(ENRICHMENT_METADATA_COLUMNS) | {
    "average_dollar_volume_21d",
    "average_dollar_volume_63d",
    "sector",
    "true_stock_level_row",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify/register canonical daily stock spine and daily price features.")
    parser.add_argument("--base-artifact", type=Path)
    parser.add_argument("--enriched-artifact", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("data/reference/assets/canonical_asset_registry.csv"))
    parser.add_argument("--aliases", type=Path, default=Path("data/reference/assets/provider_symbol_aliases.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/canonical/spines/daily_stock"))
    parser.add_argument("--feature-output-root", type=Path, default=Path("data/features/daily_price"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/data_lineage/daily_stock_spine"))
    parser.add_argument("--expected-config", type=Path)
    parser.add_argument("--expected-run-manifest", type=Path)
    parser.add_argument("--daily-archive-manifest", type=Path)
    parser.add_argument("--registry-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stream-batch-size", type=int, default=DEFAULT_STREAM_BATCH_SIZE)
    parser.add_argument("--stream-temp-root", type=Path)
    args = parser.parse_args(argv)
    if not args.daily_archive_manifest or not args.registry_manifest:
        parser.error("--daily-archive-manifest and --registry-manifest are required")
    result = verify_and_register(
        base_artifact=args.base_artifact,
        enriched_artifact=args.enriched_artifact,
        registry=args.registry,
        aliases=args.aliases,
        output_root=args.output_root,
        feature_output_root=args.feature_output_root,
        report_root=args.report_root,
        expected_config=args.expected_config,
        expected_run_manifest=args.expected_run_manifest,
        daily_archive_manifest=args.daily_archive_manifest,
        registry_manifest=args.registry_manifest,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
        stream_batch_size=args.stream_batch_size,
        stream_temp_root=args.stream_temp_root,
    )
    print(json.dumps(_summary(result), indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == STATUS_READY else 2


def verify_and_register(
    *,
    base_artifact: Path | None = None,
    enriched_artifact: Path | None = None,
    registry: Path = Path("data/reference/assets/canonical_asset_registry.csv"),
    aliases: Path = Path("data/reference/assets/provider_symbol_aliases.csv"),
    output_root: Path = Path("data/canonical/spines/daily_stock"),
    feature_output_root: Path = Path("data/features/daily_price"),
    report_root: Path = Path("reports/data_lineage/daily_stock_spine"),
    expected_config: Path | None = None,
    expected_run_manifest: Path | None = None,
    daily_archive_manifest: Path | None = None,
    registry_manifest: Path | None = None,
    dry_run: bool = False,
    verify_only: bool = False,
    stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
    stream_temp_root: Path | None = None,
) -> dict[str, Any]:
    config = _read_yaml(expected_config) if expected_config else {}
    run_manifest = _read_json(expected_run_manifest) if expected_run_manifest else {}
    selected = _select_sources(base_artifact, enriched_artifact, config, run_manifest)
    completion = _completion_status(selected, run_manifest)
    blockers = list(completion["blockers"])
    if not selected["base_artifact"]:
        blockers.append("base_artifact_path_not_supplied")
    if not selected["enriched_artifact"]:
        blockers.append("enriched_artifact_path_not_supplied")
    if selected["base_artifact"] and not Path(selected["base_artifact"]).exists():
        blockers.append(f"missing_base_artifact:{selected['base_artifact']}")
    if selected["enriched_artifact"] and not Path(selected["enriched_artifact"]).exists():
        blockers.append(f"missing_enriched_artifact:{selected['enriched_artifact']}")
    lineage = _lineage(config, run_manifest)
    if lineage["daily_price_provider"] not in {"stooq_parquet", "stooq", "canonical_daily_v2"}:
        blockers.append("daily_source_not_stooq_parquet")
    expected_sources = {"data/processed/stooq_parquet", "data/processed/market_data/canonical_daily_v2/full"}
    if lineage["source_path"] and _norm_path(lineage["source_path"]) not in {_norm_path(value) for value in expected_sources}:
        blockers.append("daily_source_path_not_expected_stooq_parquet")
    archive = _read_json(daily_archive_manifest) if daily_archive_manifest else {}
    if daily_archive_manifest:
        if not archive:
            blockers.append("daily_archive_manifest_missing")
        elif archive.get("status") != "COMPLETE" or int(archive.get("row_count", 0)) <= 0 or int(archive.get("symbol_count", 0)) != 514:
            blockers.append("daily_archive_manifest_not_complete")
        elif _norm_path(archive.get("dataset_root")) != _norm_path(lineage["source_path"]):
            blockers.append("daily_archive_source_mismatch")
    registry_parent = _read_json(registry_manifest) if registry_manifest else {}
    if registry_manifest:
        if not registry_parent:
            blockers.append("registry_manifest_missing")
        elif registry_parent.get("status") != "READY" or registry_parent.get("validation_status") != "VERIFIED":
            blockers.append("registry_manifest_not_verified")
        elif Path(str(registry_parent.get("registry_path", ""))).resolve() != registry.resolve() or str(registry_parent.get("registry_content_checksum", "")).lower() != file_sha256(registry).lower():
            blockers.append("registry_manifest_source_mismatch")

    base_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    base_meta = _missing_meta(selected.get("base_artifact"))
    enriched_meta = _missing_meta(selected.get("enriched_artifact"))
    if not blockers and not verify_only:
        base_path = Path(str(selected["base_artifact"]))
        enriched_path = Path(str(selected["enriched_artifact"]))
        base_rows = read_stock_level_artifact(base_path, required_columns={"rebalance_date", "symbol"})
        enriched_rows = read_stock_level_artifact(enriched_path, required_columns={"rebalance_date", "symbol"})
        base_meta = inspect_artifact(base_path, base_rows)
        enriched_meta = inspect_artifact(enriched_path, enriched_rows)
    assets = read_assets_csv(registry) if registry.exists() else []
    provider_aliases = read_aliases_csv(aliases) if aliases.exists() else []
    if verify_only and not blockers:
        try:
            streamed = _streaming_preflight(
                base_path=Path(str(selected["base_artifact"])),
                enriched_path=Path(str(selected["enriched_artifact"])),
                assets=assets,
                aliases=provider_aliases,
                lineage=lineage,
                batch_size=stream_batch_size,
                temp_root=stream_temp_root,
            )
        except (OSError, ValueError, sqlite3.Error, MemoryError) as exc:
            blockers.append(f"stream_read_failure:{exc}")
            streamed = None
        if streamed is not None:
            blockers.extend(streamed["blockers"])
            status = STATUS_BLOCKED if blockers else STATUS_READY
            spine_dataset_id = streamed["spine_dataset_id"]
            price_feature_dataset_id = streamed["price_feature_dataset_id"]
            verification = {
                "schema_version": "daily_stock_spine_verification.v1",
                "status": status,
                "blockers": sorted(set(blockers)),
                "selected_sources": selected,
                "source_selection_policy": "explicit_cli_then_supplied_manifest_or_config_then_no_fallback",
                "run_manifest_completion": completion,
                "lineage": lineage,
                "daily_archive_manifest": {"path": str(daily_archive_manifest) if daily_archive_manifest else None, "identity": archive.get("dataset_logical_partition_hash"), "status": archive.get("status"), "row_count": archive.get("row_count"), "symbol_count": archive.get("symbol_count"), "date_min": archive.get("date_min"), "date_max": archive.get("date_max")},
                "canonical_registry_parent": {"path": str(registry_manifest) if registry_manifest else None, "identity": registry_parent.get("dataset_id"), "version": registry_parent.get("symbol_registry_version"), "checksum": file_sha256(registry_manifest) if registry_manifest and registry_manifest.exists() else None},
                "base_artifact": streamed["base_artifact"],
                "enriched_artifact": streamed["enriched_artifact"],
                "symbol_resolution": streamed["symbol_resolution"],
                "row_grain": streamed["row_grain"],
                "duplicate_economic_row_count": streamed["duplicate_economic_row_count"],
                "temporal_validation": streamed["temporal_validation"],
                "alignment": streamed["alignment"],
                "target_alignment": streamed["target_alignment"],
                "unknown_columns": streamed["unknown_columns"],
                "spine_dataset_id": spine_dataset_id,
                "price_feature_dataset_id": price_feature_dataset_id,
                "row_identity_version": ROW_IDENTITY_VERSION,
                "dry_run": dry_run,
                "verify_only": verify_only,
                "streaming_diagnostics": streamed["streaming_diagnostics"],
                "existing_owners_reused": [
                    "core.research.ml.stock_level.stock_level_artifact_io.iter_stock_level_artifact_batches",
                    "core.research.ml.reference.daily_stock_spine identity and validation helpers",
                    "temporary SQLite exact duplicate and population verification",
                ],
                "spine_path": None,
                "price_feature_registration_path": None,
            }
            if not dry_run:
                write_verification_reports(verification, report_dir=report_root / spine_dataset_id)
            return verification
    resolution = _resolve_symbols(base_rows, assets, provider_aliases) if base_rows else _empty_resolution()
    if resolution["unresolved_symbols"]:
        blockers.append("unresolved_symbols")
    if resolution["ambiguous_symbols"]:
        blockers.append("ambiguous_symbols")
    if resolution["validity_violations"]:
        blockers.append("asset_validity_violations")

    base_augmented = _augment_rows(base_rows, resolution, lineage) if base_rows else []
    enriched_augmented = _augment_rows(enriched_rows, resolution, lineage) if enriched_rows and not resolution["unresolved_symbols"] else []
    row_grain = _row_grain(base_augmented)
    duplicates = _duplicate_rows(base_augmented)
    if duplicates:
        blockers.append("duplicate_economic_rows")
    temporal = _temporal_validation(base_augmented)
    if temporal["violation_count"]:
        blockers.append("temporal_violations")
    alignment = _alignment(base_augmented, enriched_augmented) if base_augmented and enriched_augmented else _empty_alignment()
    if alignment["base_only_count"] or alignment["enriched_only_count"] or alignment["duplicate_base_count"] or alignment["duplicate_enriched_count"]:
        blockers.append("row_alignment_failures")
    target_alignment = _target_alignment(base_augmented, enriched_augmented) if base_augmented and enriched_augmented else _empty_target_alignment()
    if target_alignment["target_mismatch_count"] or target_alignment["benchmark_mismatch_count"] or target_alignment["timestamp_mismatch_count"]:
        blockers.append("target_or_benchmark_alignment_failures")
    unknown_columns = _unknown_columns(enriched_meta.get("columns", []))
    status = STATUS_BLOCKED if blockers else STATUS_READY

    spine_dataset_id = _dataset_id("canonical_daily_stock_spine", base_augmented, lineage)
    price_feature_dataset_id = _dataset_id("daily_price_features", enriched_augmented or base_augmented, {**lineage, "spine_dataset_id": spine_dataset_id})
    report_dir = report_root / spine_dataset_id
    verification = {
        "schema_version": "daily_stock_spine_verification.v1",
        "status": status,
        "blockers": sorted(set(blockers)),
        "selected_sources": selected,
        "source_selection_policy": "explicit_cli_then_supplied_manifest_or_config_then_no_fallback",
        "run_manifest_completion": completion,
        "lineage": lineage,
        "daily_archive_manifest": {"path": str(daily_archive_manifest) if daily_archive_manifest else None, "identity": archive.get("dataset_logical_partition_hash"), "status": archive.get("status"), "row_count": archive.get("row_count"), "symbol_count": archive.get("symbol_count"), "date_min": archive.get("date_min"), "date_max": archive.get("date_max")},
        "canonical_registry_parent": {"path": str(registry_manifest) if registry_manifest else None, "identity": registry_parent.get("dataset_id"), "version": registry_parent.get("symbol_registry_version"), "checksum": file_sha256(registry_manifest) if registry_manifest and registry_manifest.exists() else None},
        "base_artifact": base_meta,
        "enriched_artifact": enriched_meta,
        "symbol_resolution": resolution,
        "row_grain": row_grain,
        "duplicate_economic_row_count": len(duplicates),
        "temporal_validation": temporal,
        "alignment": alignment,
        "target_alignment": target_alignment,
        "unknown_columns": unknown_columns,
        "spine_dataset_id": spine_dataset_id,
        "price_feature_dataset_id": price_feature_dataset_id,
        "row_identity_version": ROW_IDENTITY_VERSION,
        "dry_run": dry_run,
        "verify_only": verify_only,
        "existing_owners_reused": [
            "core.research.ml.stock_level.stock_level_artifact_io.read_stock_level_artifact",
            "core.research.ml.stock_level.stock_level_artifact_io.file_sha256",
            "core.research.ml.stock_level.prediction_artifacts.types target/provenance constants",
            "core.research.ml.stock_level.stock_level_alpha_features_types feature metadata constants",
            "core.research.ml.reference.canonical_assets manifest and row-id helpers",
        ],
    }
    if not dry_run:
        write_verification_reports(verification, report_dir=report_dir)
    spine_path = None
    feature_path = None
    if status == STATUS_READY and not dry_run and not verify_only:
        spine_dir = output_root / f"version={spine_dataset_id}"
        feature_dir = feature_output_root / f"version={price_feature_dataset_id}"
        spine_path = materialize_spine(base_augmented, spine_dir=spine_dir, verification=verification)
        feature_path = register_price_features(enriched_augmented, feature_dir=feature_dir, verification=verification)
    verification["spine_path"] = str(spine_path) if spine_path else None
    verification["price_feature_registration_path"] = str(feature_path) if feature_path else None
    return verification


def _streaming_preflight(
    *,
    base_path: Path,
    enriched_path: Path,
    assets: Sequence[Any],
    aliases: Sequence[Any],
    lineage: Mapping[str, Any],
    batch_size: int,
    temp_root: Path | None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("stream batch size must be positive")
    base_parquet, enriched_parquet = pq.ParquetFile(base_path), pq.ParquetFile(enriched_path)
    required = {"rebalance_date", "symbol"}
    for path, parquet in ((base_path, base_parquet), (enriched_path, enriched_parquet)):
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Stock-level artifact {path} missing required columns: {missing}")
    candidate_columns = {
        "row_id", "source_row_id", "symbol", "rebalance_date", "decision_session_date",
        "session_date", "decision_timestamp", "feature_data_cutoff_timestamp",
        "feature_timestamp", "target_horizon_trading_days", "target_horizon_sessions",
        "target_horizon", "target_start_timestamp", "label_start_timestamp",
        "label_end_timestamp", "label_available_timestamp", "actual_forward_return_10d",
        "actual_benchmark_return_10d", "actual_market_residual_return_10d",
        "source_dataset_hash", "exchange_calendar_identity",
    }
    base_columns = [name for name in base_parquet.schema_arrow.names if name in candidate_columns]
    enriched_columns = [name for name in enriched_parquet.schema_arrow.names if name in candidate_columns]
    symbols = set()
    first_pass_batches = 0
    for batch in iter_stock_level_artifact_batches(
        base_path, required_columns=["symbol"], batch_size=batch_size,
    ):
        first_pass_batches += 1
        for row in batch:
            if row.get("symbol") not in (None, ""):
                symbols.add(str(row["symbol"]).upper())
    resolution = _resolve_symbols([{"symbol": symbol} for symbol in sorted(symbols)], assets, aliases)
    blockers = []
    if resolution["unresolved_symbols"]:
        blockers.append("unresolved_symbols")
    if resolution["ambiguous_symbols"]:
        blockers.append("ambiguous_symbols")
    if resolution["validity_violations"]:
        blockers.append("asset_validity_violations")
    temp_parent = None
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_parent = str(temp_root.resolve())
    maximum_batch_rows = 0
    batch_counts = {"base": 0, "enriched": 0}
    row_counts = {"base": 0, "enriched": 0}
    temporal_violations = []
    temporal_missing = Counter()
    dates = {"base": set(), "enriched": set()}
    artifact_symbols = {"base": set(), "enriched": set()}
    decision_bounds = {"base": [None, None], "enriched": [None, None]}
    null_counts = {"base": Counter(), "enriched": Counter()}
    with tempfile.TemporaryDirectory(prefix="daily-spine-stream-", dir=temp_parent) as temp_dir:
        database_path = Path(temp_dir) / "verification.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """CREATE TABLE rows (
                    side TEXT NOT NULL, sequence INTEGER NOT NULL, row_id TEXT NOT NULL,
                    asset_id TEXT, decision_timestamp TEXT, target_horizon TEXT,
                    stock_target TEXT, benchmark_asset_id TEXT, benchmark_return TEXT,
                    excess_return TEXT, feature_cutoff_timestamp TEXT,
                    target_start_timestamp TEXT, target_end_timestamp TEXT,
                    target_available_timestamp TEXT, source_symbol TEXT
                )"""
            )
            for side, path, columns in (
                ("base", base_path, base_columns),
                ("enriched", enriched_path, enriched_columns),
            ):
                sequence = 0
                for batch in iter_stock_level_artifact_batches(
                    path, required_columns=columns, batch_size=batch_size,
                ):
                    batch_counts[side] += 1
                    maximum_batch_rows = max(maximum_batch_rows, len(batch))
                    augmented = _augment_rows(batch, resolution, lineage)
                    inserts = []
                    for source, row in zip(batch, augmented):
                        sequence += 1
                        row_counts[side] += 1
                        symbol = str(source.get("symbol") or "").upper()
                        date_value = _date_value(source)
                        decision = row["decision_timestamp"]
                        if not symbol:
                            null_counts[side]["symbol"] += 1
                        else:
                            artifact_symbols[side].add(symbol)
                        if not date_value:
                            null_counts[side]["rebalance_date"] += 1
                        else:
                            dates[side].add(date_value)
                        if not decision:
                            null_counts[side]["decision_timestamp"] += 1
                        else:
                            bounds = decision_bounds[side]
                            bounds[0] = decision if bounds[0] is None or decision < bounds[0] else bounds[0]
                            bounds[1] = decision if bounds[1] is None or decision > bounds[1] else bounds[1]
                        if side == "base":
                            _stream_temporal(row, temporal_violations, temporal_missing)
                        inserts.append((
                            side, sequence, row["row_id"], row["asset_id"], decision,
                            row["target_horizon_sessions"], _string_or_none(row["stock_target"]),
                            row["benchmark_asset_id"], _string_or_none(row["benchmark_return"]),
                            _string_or_none(row["excess_return"]), row["feature_cutoff_timestamp"],
                            row["target_start_timestamp"], row["target_end_timestamp"],
                            row["target_available_timestamp"], row["source_symbol"],
                        ))
                    connection.executemany(
                        "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        inserts,
                    )
                connection.commit()
            connection.execute("CREATE INDEX rows_side_row_id ON rows(side, row_id)")
            connection.execute(
                "CREATE INDEX rows_side_key ON rows(side, asset_id, decision_timestamp, target_horizon)"
            )
            duplicates = connection.execute(
                """SELECT asset_id, decision_timestamp, target_horizon, COUNT(*)
                   FROM rows WHERE side='base'
                   GROUP BY asset_id, decision_timestamp, target_horizon HAVING COUNT(*) > 1"""
            ).fetchall()
            if row_counts["base"] == 0 or row_counts["enriched"] == 0:
                blockers.append("invalid_row:empty_artifact")
            if any(null_counts[side]["symbol"] for side in ("base", "enriched")):
                blockers.append("invalid_row:missing_symbol")
            if any(null_counts[side]["rebalance_date"] for side in ("base", "enriched")):
                blockers.append("invalid_row:missing_date")
            if duplicates:
                blockers.append("duplicate_economic_rows")
            alignment = _stream_alignment(connection, row_counts)
            if any(alignment[key] for key in (
                "base_only_count", "enriched_only_count", "duplicate_base_count", "duplicate_enriched_count",
            )):
                blockers.append("row_alignment_failures")
            target_alignment = _stream_target_alignment(connection)
            if any(target_alignment[key] for key in (
                "target_mismatch_count", "benchmark_mismatch_count", "timestamp_mismatch_count",
            )):
                blockers.append("target_or_benchmark_alignment_failures")
            if temporal_violations:
                blockers.append("temporal_violations")
            base_row_ids_checksum = _stream_json_array_checksum(connection, "base")
            enriched_row_ids_checksum = _stream_json_array_checksum(connection, "enriched")
            spine_dataset_id = _stream_dataset_id(
                connection, "base", "canonical_daily_stock_spine", lineage,
                SPINE_SCHEMA_VERSION,
            )
            price_feature_dataset_id = _stream_dataset_id(
                connection, "enriched", "daily_price_features",
                {**lineage, "spine_dataset_id": spine_dataset_id},
                PRICE_FEATURE_SCHEMA_VERSION,
            )
            temp_bytes = database_path.stat().st_size if database_path.exists() else 0
        finally:
            connection.close()
    base_meta = _stream_artifact_meta(
        base_path, base_parquet, base_columns, row_counts["base"], artifact_symbols["base"],
        dates["base"], decision_bounds["base"], null_counts["base"], base_row_ids_checksum,
        batch_counts["base"],
    )
    enriched_meta = _stream_artifact_meta(
        enriched_path, enriched_parquet, enriched_columns, row_counts["enriched"],
        artifact_symbols["enriched"], dates["enriched"], decision_bounds["enriched"],
        null_counts["enriched"], enriched_row_ids_checksum, batch_counts["enriched"],
    )
    return {
        "blockers": blockers,
        "base_artifact": base_meta,
        "enriched_artifact": enriched_meta,
        "symbol_resolution": resolution,
        "row_grain": {
            "grain": "one asset at one selector decision timestamp",
            "includes_target_horizon": False,
            "candidate_key": ["asset_id", "decision_timestamp", "target_horizon_sessions"],
            "row_count": row_counts["base"],
            "unique_key_count": row_counts["base"] - sum(row[3] - 1 for row in duplicates),
        },
        "duplicate_economic_row_count": len(duplicates),
        "temporal_validation": {
            "violation_count": len(temporal_violations),
            "violations": temporal_violations[:MAX_REPORT_ROWS],
            "missing_field_counts": dict(temporal_missing),
        },
        "alignment": alignment,
        "target_alignment": target_alignment,
        "unknown_columns": _unknown_columns(enriched_parquet.schema_arrow.names),
        "spine_dataset_id": spine_dataset_id,
        "price_feature_dataset_id": price_feature_dataset_id,
        "streaming_diagnostics": {
            "streaming_mode": "pyarrow_record_batches_with_temporary_sqlite",
            "whole_table_materialisation": False,
            "whole_table_to_pylist_used": False,
            "configured_batch_size": batch_size,
            "projected_columns": {"base": base_columns, "enriched": enriched_columns},
            "projected_column_count": {"base": len(base_columns), "enriched": len(enriched_columns)},
            "source_row_groups": {
                "base": base_parquet.metadata.num_row_groups,
                "enriched": enriched_parquet.metadata.num_row_groups,
            },
            "batches_processed": {**batch_counts, "base_symbol_first_pass": first_pass_batches},
            "rows_processed": row_counts,
            "maximum_batch_row_count": maximum_batch_rows,
            "temporary_resource": "cleaned_temporary_sqlite",
            "maximum_temporary_bytes": temp_bytes,
            "worker_count": 1,
        },
    }


def _stream_temporal(row, violations, missing):
    checks = [
        ("feature_cutoff_timestamp", "decision_timestamp"),
        ("decision_timestamp", "target_start_timestamp"),
        ("target_start_timestamp", "target_end_timestamp"),
        ("target_end_timestamp", "target_available_timestamp"),
    ]
    for left, right in checks:
        if not row.get(left) or not row.get(right):
            missing[f"{left}:{right}"] += 1
        elif _parse_dt(row[left]) > _parse_dt(row[right]):
            if len(violations) < MAX_REPORT_ROWS:
                violations.append({
                    "row_id": row["row_id"], "left": left, "left_value": row[left],
                    "right": right, "right_value": row[right],
                })


def _stream_alignment(connection, counts):
    base_only_count = connection.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT row_id FROM rows WHERE side='base' EXCEPT SELECT DISTINCT row_id FROM rows WHERE side='enriched')"
    ).fetchone()[0]
    enriched_only_count = connection.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT row_id FROM rows WHERE side='enriched' EXCEPT SELECT DISTINCT row_id FROM rows WHERE side='base')"
    ).fetchone()[0]
    duplicate_base_count = connection.execute(
        "SELECT COUNT(*) FROM (SELECT row_id FROM rows WHERE side='base' GROUP BY row_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    duplicate_enriched_count = connection.execute(
        "SELECT COUNT(*) FROM (SELECT row_id FROM rows WHERE side='enriched' GROUP BY row_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    base_only = connection.execute(
        "SELECT row_id FROM (SELECT DISTINCT row_id FROM rows WHERE side='base' EXCEPT SELECT DISTINCT row_id FROM rows WHERE side='enriched') ORDER BY row_id LIMIT ?",
        (MAX_REPORT_ROWS,),
    ).fetchall()
    enriched_only = connection.execute(
        "SELECT row_id FROM (SELECT DISTINCT row_id FROM rows WHERE side='enriched' EXCEPT SELECT DISTINCT row_id FROM rows WHERE side='base') ORDER BY row_id LIMIT ?",
        (MAX_REPORT_ROWS,),
    ).fetchall()
    return {
        "same_row_id_set": base_only_count == 0 and enriched_only_count == 0,
        "base_row_count": counts["base"], "enriched_row_count": counts["enriched"],
        "base_only_count": base_only_count, "enriched_only_count": enriched_only_count,
        "duplicate_base_count": duplicate_base_count,
        "duplicate_enriched_count": duplicate_enriched_count,
        "base_only_rows": [{"row_id": row[0]} for row in base_only],
        "enriched_only_rows": [{"row_id": row[0]} for row in enriched_only],
    }


def _stream_target_alignment(connection):
    fields = (
        ("target", "stock_target"), ("target", "target_horizon"),
        ("target", "target_start_timestamp"), ("target", "target_end_timestamp"),
        ("target", "target_available_timestamp"), ("benchmark", "benchmark_asset_id"),
        ("benchmark", "benchmark_return"), ("benchmark", "excess_return"),
        ("timestamp", "decision_timestamp"), ("timestamp", "feature_cutoff_timestamp"),
    )
    output = {"target": [], "benchmark": [], "timestamp": []}
    counts = Counter()
    for category, field in fields:
        query = f"""SELECT b.row_id, b.{field}, e.{field}
                    FROM rows b JOIN rows e ON b.row_id=e.row_id
                    WHERE b.side='base' AND e.side='enriched'
                    AND COALESCE(b.{field}, '') <> COALESCE(e.{field}, '')"""
        rows = connection.execute(query).fetchall()
        counts[category] += len(rows)
        remaining = MAX_REPORT_ROWS - len(output[category])
        output[category].extend({
            "row_id": row_id, "field": field, "base": base, "enriched": enriched,
        } for row_id, base, enriched in rows[:remaining])
    return {
        "target_mismatch_count": counts["target"],
        "benchmark_mismatch_count": counts["benchmark"],
        "timestamp_mismatch_count": counts["timestamp"],
        "target_mismatches": output["target"],
        "benchmark_mismatches": output["benchmark"],
        "timestamp_mismatches": output["timestamp"],
    }


def _stream_json_array_checksum(connection, side):
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for (row_id,) in connection.execute(
        "SELECT row_id FROM rows WHERE side=? ORDER BY row_id, sequence", (side,),
    ):
        if not first:
            digest.update(b",")
        digest.update(json.dumps(str(row_id), separators=(",", ":")).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _stream_dataset_id(connection, side, kind, identity, schema):
    digest = hashlib.sha256()
    digest.update(b'{"identity":')
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    digest.update(b',"kind":')
    digest.update(json.dumps(kind).encode("utf-8"))
    digest.update(b',"row_ids":[')
    first = True
    for (row_id,) in connection.execute(
        "SELECT row_id FROM rows WHERE side=? ORDER BY row_id, sequence", (side,),
    ):
        if not first:
            digest.update(b",")
        digest.update(json.dumps(str(row_id)).encode("utf-8"))
        first = False
    digest.update(b'],"schema":')
    digest.update(json.dumps(schema).encode("utf-8"))
    digest.update(b"}")
    return f"{kind}-{digest.hexdigest()[:16]}"


def _stream_artifact_meta(
    path, parquet, projected, loaded_count, symbols, dates, decision_bounds,
    null_counts, population_checksum, batches,
):
    columns = list(parquet.schema_arrow.names)
    return {
        "path": str(path), "exists": True, "file_size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": file_sha256(path), "row_count": parquet.metadata.num_rows,
        "loaded_row_count": loaded_count, "column_count": len(columns), "columns": columns,
        "symbol_count": len(symbols), "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "decision_timestamp_min": decision_bounds[0],
        "decision_timestamp_max": decision_bounds[1],
        "target_horizons": [],
        "target_fields": [column for column in columns if column in TARGET_COLUMNS],
        "benchmark_fields": [column for column in columns if column in BENCHMARK_COLUMNS],
        "identity_fields": [column for column in columns if column in IDENTITY_COLUMNS],
        "provenance_fields": [column for column in columns if column in PROVENANCE_COLUMNS],
        "null_counts": dict(null_counts),
        "duplicate_candidate_keys": None,
        "row_population_checksum": population_checksum,
        "projected_columns": projected,
        "streamed_batch_count": batches,
        "parquet": {
            "num_row_groups": parquet.metadata.num_row_groups,
            "compression_codecs": sorted({
                str(parquet.metadata.row_group(group).column(column).compression)
                for group in range(parquet.metadata.num_row_groups)
                for column in range(parquet.metadata.row_group(group).num_columns)
            }),
            "row_groups": [
                {"index": index, "num_rows": parquet.metadata.row_group(index).num_rows}
                for index in range(parquet.metadata.num_row_groups)
            ],
        },
    }


def inspect_artifact(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema_arrow.names)
    dates = [_date_value(row) for row in rows if _date_value(row)]
    decisions = [_decision_timestamp(row) for row in rows if _decision_timestamp(row)]
    symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")})
    null_counts = {field: sum(row.get(field) in (None, "") for row in rows) for field in ("symbol", "rebalance_date", "decision_timestamp") if field in columns}
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "sha256": file_sha256(path),
        "row_count": parquet.metadata.num_rows,
        "loaded_row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "symbol_count": len(symbols),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "decision_timestamp_min": min(decisions) if decisions else None,
        "decision_timestamp_max": max(decisions) if decisions else None,
        "target_horizons": sorted({str(row.get("target_horizon_trading_days") or row.get("target_horizon") or "") for row in rows if row.get("target_horizon_trading_days") or row.get("target_horizon")}),
        "target_fields": [column for column in columns if column in TARGET_COLUMNS],
        "benchmark_fields": [column for column in columns if column in BENCHMARK_COLUMNS],
        "identity_fields": [column for column in columns if column in IDENTITY_COLUMNS],
        "provenance_fields": [column for column in columns if column in PROVENANCE_COLUMNS],
        "null_counts": null_counts,
        "duplicate_candidate_keys": _duplicate_candidate_key_count(rows),
        "parquet": {
            "num_row_groups": parquet.metadata.num_row_groups,
            "compression_codecs": sorted({
                str(parquet.metadata.row_group(group).column(column).compression)
                for group in range(parquet.metadata.num_row_groups)
                for column in range(parquet.metadata.row_group(group).num_columns)
            }),
            "row_groups": [
                {"index": index, "num_rows": parquet.metadata.row_group(index).num_rows}
                for index in range(parquet.metadata.num_row_groups)
            ],
        },
    }


def write_verification_reports(verification: Mapping[str, Any], *, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "verification_report.json", verification)
    (report_dir / "verification_report.md").write_text(_markdown(verification), encoding="utf-8")
    _write_csv(report_dir / "symbol_resolution.csv", verification["symbol_resolution"]["rows"], ("source_symbol", "asset_id", "canonical_symbol", "status"))
    _write_csv(report_dir / "unknown_columns.csv", [{"column": col, "classification": "unknown"} for col in verification["unknown_columns"]], ("column", "classification"))
    _write_mismatch_csv(report_dir / "base_only_rows.csv", verification["alignment"].get("base_only_rows", []))
    _write_mismatch_csv(report_dir / "enriched_only_rows.csv", verification["alignment"].get("enriched_only_rows", []))
    _write_mismatch_csv(report_dir / "target_mismatches.csv", verification["target_alignment"].get("target_mismatches", []))
    _write_mismatch_csv(report_dir / "benchmark_mismatches.csv", verification["target_alignment"].get("benchmark_mismatches", []))
    _write_mismatch_csv(report_dir / "temporal_violations.csv", verification["temporal_validation"].get("violations", []))


def materialize_spine(rows: Sequence[Mapping[str, Any]], *, spine_dir: Path, verification: Mapping[str, Any]) -> Path:
    spine_rows = [{column: row.get(column) for column in SPINE_COLUMNS} for row in rows]
    tmp_dir = spine_dir.with_name(f".{spine_dir.name}.{os.getpid()}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    schema = pa.schema([(column, pa.string()) for column in SPINE_COLUMNS])
    table = pa.Table.from_pylist([{k: _string_or_none(v) for k, v in row.items()} for row in spine_rows], schema=schema)
    pq.write_table(table, tmp_dir / "spine.parquet", compression="zstd")
    _write_json(tmp_dir / "schema.json", {"schema_version": SPINE_SCHEMA_VERSION, "columns": list(SPINE_COLUMNS)})
    _write_json(tmp_dir / "source_pointer.json", _source_pointer(verification, derivative="canonical spine"))
    _write_json(tmp_dir / "quality_report.json", verification)
    _write_csv(tmp_dir / "symbol_coverage.csv", verification["symbol_resolution"]["rows"], ("source_symbol", "asset_id", "canonical_symbol", "status"))
    pq.write_table(pa.Table.from_pylist([{"row_id": row["row_id"], "identity_hash": _row_identity_hash(row)} for row in rows]), tmp_dir / "row_identity_audit.parquet", compression="zstd")
    manifest = _manifest_for_spine(rows, verification)
    manifest_payload = asdict(manifest)
    source_path = Path(str(verification["selected_sources"]["base_artifact"]))
    manifest_payload.update({
        "spine_artifact_path": str(spine_dir / "spine.parquet"),
        "spine_artifact_checksum": file_sha256(tmp_dir / "spine.parquet"),
        "source_artifact_path": str(source_path),
        "source_artifact_checksum": file_sha256(source_path),
        "source_price_artifact_identities": [verification.get("price_feature_dataset_id")],
        "point_in_time_feature_store_identities": [],
        "canonical_symbol_registry_identity": verification.get("canonical_registry_parent", {}).get("identity"),
        "canonical_symbol_registry_version": verification.get("canonical_registry_parent", {}).get("version"),
        "canonical_symbol_registry_manifest_checksum": verification.get("canonical_registry_parent", {}).get("checksum"),
        "daily_archive_identity": verification.get("daily_archive_manifest", {}).get("identity"),
        "daily_archive_manifest_path": verification.get("daily_archive_manifest", {}).get("path"),
        "publication_status": "complete", "validation_status": verification.get("status"),
    })
    _write_json(tmp_dir / "manifest.json", manifest_payload)
    if spine_dir.exists():
        existing = _read_json(spine_dir / "manifest.json")
        if existing.get("dataset_id") == manifest_payload.get("dataset_id") and existing.get("spine_artifact_checksum") == manifest_payload.get("spine_artifact_checksum"):
            shutil.rmtree(tmp_dir)
            return spine_dir / "spine.parquet"
        raise FileExistsError(f"Canonical daily spine destination is immutable: {spine_dir}")
    os.replace(tmp_dir, spine_dir)
    return spine_dir / "spine.parquet"


def register_price_features(rows: Sequence[Mapping[str, Any]], *, feature_dir: Path, verification: Mapping[str, Any]) -> Path:
    tmp_dir = feature_dir.with_name(f".{feature_dir.name}.{os.getpid()}.tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    columns = verification["enriched_artifact"].get("columns", [])
    classifications = [{"column": column, "classification": classify_column(column)} for column in columns]
    _write_json(tmp_dir / "schema.json", {"schema_version": PRICE_FEATURE_SCHEMA_VERSION, "columns": classifications})
    _write_json(tmp_dir / "source_pointer.json", _source_pointer(verification, derivative="daily price feature registration"))
    _write_json(tmp_dir / "row_alignment_report.json", verification["alignment"])
    _write_json(tmp_dir / "quality_report.json", verification)
    manifest = _manifest_for_features(rows, verification)
    _write_json(tmp_dir / "manifest.json", asdict(manifest))
    if feature_dir.exists():
        existing = _read_json(feature_dir / "manifest.json")
        if existing.get("dataset_id") == manifest.dataset_id:
            shutil.rmtree(tmp_dir)
            return feature_dir / "manifest.json"
        raise FileExistsError(f"Daily price feature destination is immutable: {feature_dir}")
    os.replace(tmp_dir, feature_dir)
    return feature_dir / "manifest.json"


def classify_column(column: str) -> str:
    if column in IDENTITY_COLUMNS:
        return "identity"
    if column in TIMING_COLUMNS:
        return "timing"
    if column in TARGET_COLUMNS:
        return "target"
    if column in BENCHMARK_COLUMNS:
        return "benchmark"
    if column in PROVENANCE_COLUMNS:
        return "provenance"
    if column in FEATURE_COLUMNS:
        return "feature"
    if column in DIAGNOSTIC_COLUMNS:
        return "diagnostic"
    return "unknown"


def _select_sources(base: Path | None, enriched: Path | None, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, str | None]:
    manifest_base, manifest_enriched = _manifest_paths(manifest)
    config_base, config_enriched = _config_paths(config)
    return {
        "base_artifact": str(base or manifest_base or config_base) if (base or manifest_base or config_base) else None,
        "base_source": "explicit" if base else ("run_manifest" if manifest_base else ("config" if config_base else "none")),
        "enriched_artifact": str(enriched or manifest_enriched or config_enriched) if (enriched or manifest_enriched or config_enriched) else None,
        "enriched_source": "explicit" if enriched else ("run_manifest" if manifest_enriched else ("config" if config_enriched else "none")),
    }


def _manifest_paths(manifest: Mapping[str, Any]) -> tuple[Path | None, Path | None]:
    base = enriched = None
    for stage in manifest.get("stages", []) or []:
        if not isinstance(stage, Mapping):
            continue
        outputs = stage.get("output_paths", {}) if isinstance(stage.get("output_paths"), Mapping) else {}
        if stage.get("name") == "stock_artifact" and outputs.get("parquet_path"):
            base = Path(str(outputs["parquet_path"]))
        if stage.get("name") == "alpha_features" and outputs.get("enriched_parquet_path"):
            enriched = Path(str(outputs["enriched_parquet_path"]))
    return base, enriched


def _config_paths(config: Mapping[str, Any]) -> tuple[Path | None, Path | None]:
    ml = config.get("ml", {}) if isinstance(config.get("ml"), Mapping) else {}
    output_dir = Path(str(ml.get("output_dir"))) if ml.get("output_dir") else None
    base = Path(str(ml["stock_level_prediction_artifacts_path"])) if ml.get("stock_level_prediction_artifacts_path") else None
    enriched = Path(str(ml["stock_level_enriched_prediction_artifacts_path"])) if ml.get("stock_level_enriched_prediction_artifacts_path") else None
    if output_dir:
        base = base or output_dir / "stock_level_prediction_artifacts.parquet"
        enriched = enriched or output_dir / "stock_level_prediction_artifacts_enriched.parquet"
    return base, enriched


def _completion_status(selected: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    blockers = []
    stages = {stage.get("name"): stage for stage in manifest.get("stages", []) or [] if isinstance(stage, Mapping)}
    stock = stages.get("stock_artifact", {})
    alpha = stages.get("alpha_features", {})
    if manifest:
        if stock.get("status") != "completed":
            blockers.append(f"stock_artifact_not_completed:{stock.get('status')}")
        if alpha.get("status") != "completed":
            blockers.append(f"alpha_features_not_completed:{alpha.get('status')}")
        for stage in stages.values():
            if stage.get("status") in {"running", "pending", "interrupted"}:
                blockers.append(f"incomplete_stage:{stage.get('name')}:{stage.get('status')}")
    for key in ("base_artifact", "enriched_artifact"):
        path = selected.get(key)
        if path and Path(str(path)).name.startswith("."):
            blockers.append(f"temporary_source_path:{path}")
    return {"status": "BLOCKED" if blockers else "READY", "blockers": blockers, "stage_statuses": {name: stage.get("status") for name, stage in stages.items()}}


def _lineage(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    ml = config.get("ml", {}) if isinstance(config.get("ml"), Mapping) else {}
    return {
        "daily_price_provider": str(ml.get("historical_data_provider", "stooq_parquet")),
        "source_path": str(ml.get("stooq_parquet_dir", ml.get("parquet_dir", "data/processed/stooq_parquet"))),
        "source_format": "parquet",
        "configured_universe": ml.get("stock_alpha_artifact_universe_paths", []),
        "decision_frequency": ml.get("stock_level_decision_frequency"),
        "minimum_history": ml.get("stock_level_decision_min_history_sessions", ml.get("minimum_history_years")),
        "target_horizon": ml.get("prediction_horizon", ml.get("label_horizon_days")),
        "benchmark_symbol": ml.get("stock_ranker_market_symbol", ml.get("stock_ranker_spy_symbol", "SPY")),
        "config_hash": _hash_json(config) if config else None,
        "code_commit": manifest.get("code_commit") or None,
        "run_manifest_mode": manifest.get("mode"),
    }


def _resolve_symbols(rows: Sequence[Mapping[str, Any]], assets: Sequence[Any], aliases: Sequence[Any]) -> dict[str, Any]:
    active_assets = {asset.asset_id: asset for asset in assets if asset.is_active}
    by_provider_symbol: dict[str, list[Any]] = {}
    for alias in aliases:
        if alias.provider in {"canonical", "stooq"}:
            by_provider_symbol.setdefault(alias.provider_symbol.upper(), []).append(alias)
    result = []
    unresolved = []
    ambiguous = []
    validity_violations = []
    for symbol in sorted({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}):
        matches = list(by_provider_symbol.get(symbol, []))
        asset_ids = sorted({alias.asset_id for alias in matches})
        if not matches:
            unresolved.append(symbol)
            result.append({"source_symbol": symbol, "asset_id": "", "canonical_symbol": "", "status": "unresolved"})
        elif len(asset_ids) > 1:
            ambiguous.append(symbol)
            result.append({"source_symbol": symbol, "asset_id": ",".join(asset_ids), "canonical_symbol": "", "status": "ambiguous"})
        elif asset_ids[0] not in active_assets:
            unresolved.append(symbol)
            result.append({"source_symbol": symbol, "asset_id": asset_ids[0], "canonical_symbol": "", "status": "unresolved_asset"})
        else:
            asset = active_assets[asset_ids[0]]
            result.append({"source_symbol": symbol, "asset_id": asset.asset_id, "canonical_symbol": asset.canonical_symbol, "status": "resolved"})
    map_by_symbol = {row["source_symbol"]: row for row in result if row["status"] == "resolved"}
    return {
        "rows": result,
        "map_by_symbol": map_by_symbol,
        "resolved_symbol_count": len(map_by_symbol),
        "unresolved_symbols": unresolved,
        "ambiguous_symbols": ambiguous,
        "validity_violations": validity_violations,
    }


def _augment_rows(rows: Sequence[Mapping[str, Any]], resolution: Mapping[str, Any], lineage: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = resolution.get("map_by_symbol", {})
    output = []
    registry_version = _registry_version_from_resolution(resolution)
    daily_price_version = _hash_json({"source": lineage.get("source_path"), "provider": lineage.get("daily_price_provider")})[:16]
    universe_version = _hash_json(lineage.get("configured_universe", []))[:16]
    target_definition_version = "stock_level_target_provenance_v1"
    benchmark_asset_id = mapping.get(str(lineage.get("benchmark_symbol", "SPY")).upper(), {}).get("asset_id", "")
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        resolved = mapping.get(symbol, {})
        decision = _decision_timestamp(row)
        horizon = _target_horizon(row)
        row_id = daily_spine_row_id(
            asset_id=resolved.get("asset_id", ""),
            decision_timestamp=decision,
            target_horizon_sessions=horizon,
            universe_version=universe_version,
            daily_price_dataset_version=daily_price_version,
            target_definition_version=target_definition_version,
        )
        output.append({
            **dict(row),
            "row_id": row_id,
            "source_row_id": str(row.get("row_id") or row.get("source_row_id") or ""),
            "asset_id": resolved.get("asset_id", ""),
            "canonical_symbol": resolved.get("canonical_symbol", symbol),
            "source_symbol": symbol,
            "session_date": _date_value(row),
            "decision_timestamp": decision,
            "feature_cutoff_timestamp": _timestamp_value(row, "feature_data_cutoff_timestamp") or _timestamp_value(row, "feature_timestamp"),
            "universe_version": universe_version,
            "eligible_at_decision": "true",
            "eligibility_reason": "present_in_source_artifact",
            "daily_price_dataset_version": daily_price_version,
            "symbol_registry_version": registry_version,
            "calendar_version": str(row.get("exchange_calendar_identity") or ""),
            "target_horizon_sessions": str(horizon),
            "target_start_timestamp": _timestamp_value(row, "target_start_timestamp") or _timestamp_value(row, "label_start_timestamp"),
            "target_end_timestamp": _timestamp_value(row, "label_end_timestamp"),
            "target_available_timestamp": _timestamp_value(row, "label_available_timestamp"),
            "target_definition_version": target_definition_version,
            "benchmark_asset_id": benchmark_asset_id,
            "stock_target": _string_or_none(row.get("actual_forward_return_10d")),
            "benchmark_return": _string_or_none(row.get("actual_benchmark_return_10d")),
            "excess_return": _string_or_none(row.get("actual_market_residual_return_10d")),
            "source_artifact_path": "",
            "source_artifact_dataset_id": _string_or_none(row.get("source_dataset_hash")),
        })
    return output


def _row_grain(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    horizons = {str(row.get("target_horizon_sessions", "")) for row in rows}
    keys = [(row.get("asset_id"), row.get("decision_timestamp"), row.get("target_horizon_sessions")) for row in rows]
    return {
        "grain": "one asset at one selector decision timestamp",
        "includes_target_horizon": len(horizons) > 1,
        "candidate_key": ["asset_id", "decision_timestamp", "target_horizon_sessions"],
        "row_count": len(rows),
        "unique_key_count": len(set(keys)),
    }


def _duplicate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = [(row.get("asset_id"), row.get("decision_timestamp"), row.get("target_horizon_sessions")) for row in rows]
    counts = Counter(keys)
    return [{"asset_id": k[0], "decision_timestamp": k[1], "target_horizon_sessions": k[2], "count": c} for k, c in counts.items() if c > 1]


def _temporal_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations = []
    missing = Counter()
    checks = [
        ("feature_cutoff_timestamp", "<=", "decision_timestamp"),
        ("decision_timestamp", "<=", "target_start_timestamp"),
        ("target_start_timestamp", "<=", "target_end_timestamp"),
        ("target_end_timestamp", "<=", "target_available_timestamp"),
    ]
    for row in rows:
        for left, _op, right in checks:
            if not row.get(left) or not row.get(right):
                missing[f"{left}:{right}"] += 1
                continue
            if _parse_dt(row[left]) > _parse_dt(row[right]):
                violations.append({"row_id": row.get("row_id"), "left": left, "left_value": row.get(left), "right": right, "right_value": row.get(right)})
    return {"violation_count": len(violations), "violations": violations[:MAX_REPORT_ROWS], "missing_field_counts": dict(missing)}


def _alignment(base: Sequence[Mapping[str, Any]], enriched: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_counts = Counter(row["row_id"] for row in base)
    enriched_counts = Counter(row["row_id"] for row in enriched)
    base_set = set(base_counts)
    enriched_set = set(enriched_counts)
    return {
        "same_row_id_set": base_set == enriched_set,
        "base_row_count": len(base),
        "enriched_row_count": len(enriched),
        "base_only_count": len(base_set - enriched_set),
        "enriched_only_count": len(enriched_set - base_set),
        "duplicate_base_count": sum(1 for count in base_counts.values() if count > 1),
        "duplicate_enriched_count": sum(1 for count in enriched_counts.values() if count > 1),
        "base_only_rows": [{"row_id": row_id} for row_id in sorted(base_set - enriched_set)[:MAX_REPORT_ROWS]],
        "enriched_only_rows": [{"row_id": row_id} for row_id in sorted(enriched_set - base_set)[:MAX_REPORT_ROWS]],
    }


def _target_alignment(base: Sequence[Mapping[str, Any]], enriched: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_enriched = {row["row_id"]: row for row in enriched}
    target_mismatches = []
    benchmark_mismatches = []
    timestamp_mismatches = []
    target_fields = ["stock_target", "target_horizon_sessions", "target_start_timestamp", "target_end_timestamp", "target_available_timestamp"]
    benchmark_fields = ["benchmark_asset_id", "benchmark_return", "excess_return"]
    for row in base:
        other = by_enriched.get(row["row_id"])
        if not other:
            continue
        for field in target_fields:
            if _string_or_none(row.get(field)) != _string_or_none(other.get(field)):
                target_mismatches.append({"row_id": row["row_id"], "field": field, "base": row.get(field), "enriched": other.get(field)})
        for field in benchmark_fields:
            if _string_or_none(row.get(field)) != _string_or_none(other.get(field)):
                benchmark_mismatches.append({"row_id": row["row_id"], "field": field, "base": row.get(field), "enriched": other.get(field)})
        for field in ["decision_timestamp", "feature_cutoff_timestamp"]:
            if _string_or_none(row.get(field)) != _string_or_none(other.get(field)):
                timestamp_mismatches.append({"row_id": row["row_id"], "field": field, "base": row.get(field), "enriched": other.get(field)})
    return {
        "target_mismatch_count": len(target_mismatches),
        "benchmark_mismatch_count": len(benchmark_mismatches),
        "timestamp_mismatch_count": len(timestamp_mismatches),
        "target_mismatches": target_mismatches[:MAX_REPORT_ROWS],
        "benchmark_mismatches": benchmark_mismatches[:MAX_REPORT_ROWS],
        "timestamp_mismatches": timestamp_mismatches[:MAX_REPORT_ROWS],
    }


def _unknown_columns(columns: Sequence[str]) -> list[str]:
    return sorted(column for column in columns if classify_column(column) == "unknown")


def _dataset_id(kind: str, rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]) -> str:
    payload = {
        "kind": kind,
        "row_ids": sorted(str(row.get("row_id")) for row in rows),
        "identity": identity,
        "schema": SPINE_SCHEMA_VERSION if kind == "canonical_daily_stock_spine" else PRICE_FEATURE_SCHEMA_VERSION,
    }
    return f"{kind}-{_hash_json(payload)[:16]}"


def _manifest_for_spine(rows: Sequence[Mapping[str, Any]], verification: Mapping[str, Any]) -> DatasetManifest:
    return build_dataset_manifest(
        dataset_type="canonical_daily_stock_spine",
        row_grain="one asset at one selector decision timestamp",
        primary_keys=("row_id",),
        source_paths=(verification["selected_sources"]["base_artifact"],),
        symbol_registry_version=rows[0].get("symbol_registry_version") if rows else "",
        calendar_version=rows[0].get("calendar_version") if rows else "",
        universe_version=rows[0].get("universe_version") if rows else "",
        row_count=len(rows),
        symbol_count=len({row.get("asset_id") for row in rows}),
        date_min=verification["base_artifact"].get("date_min"),
        date_max=verification["base_artifact"].get("date_max"),
        row_identity_checksum=_row_id_checksum(rows),
        provider="stooq",
        timeframe="1d",
        adjustment_policy="source_artifact",
    )


def _manifest_for_features(rows: Sequence[Mapping[str, Any]], verification: Mapping[str, Any]) -> DatasetManifest:
    manifest = build_dataset_manifest(
        dataset_type="daily_price_features",
        row_grain="one row per canonical daily stock spine row",
        primary_keys=("row_id",),
        source_paths=(verification["selected_sources"]["enriched_artifact"],),
        source_dataset_ids=(verification["spine_dataset_id"],),
        symbol_registry_version=rows[0].get("symbol_registry_version") if rows else "",
        calendar_version=rows[0].get("calendar_version") if rows else "",
        universe_version=rows[0].get("universe_version") if rows else "",
        row_count=len(rows),
        symbol_count=len({row.get("asset_id") for row in rows}),
        date_min=verification["enriched_artifact"].get("date_min"),
        date_max=verification["enriched_artifact"].get("date_max"),
        row_identity_checksum=_row_id_checksum(rows),
        feature_versions={"daily_price_features": PRICE_FEATURE_SCHEMA_VERSION},
        provider="stooq",
        timeframe="1d",
        adjustment_policy="source_artifact",
    )
    return manifest


def _source_pointer(verification: Mapping[str, Any], *, derivative: str) -> dict[str, Any]:
    return {
        "schema_version": "canonical_source_pointer.v1",
        "derivative": derivative,
        "base_source_artifact": _repo_rel(verification["selected_sources"].get("base_artifact")),
        "enriched_source_artifact": _repo_rel(verification["selected_sources"].get("enriched_artifact")),
        "canonical_spine_is_slim_governed_derivative": True,
        "price_feature_registration_points_to_existing_enriched_artifact": True,
        "legacy_fallback_permitted": False,
        "source_artifacts_modified": False,
    }


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "blockers": result["blockers"],
        "base_artifact": result["selected_sources"].get("base_artifact"),
        "enriched_artifact": result["selected_sources"].get("enriched_artifact"),
        "base_rows": result["base_artifact"].get("row_count"),
        "enriched_rows": result["enriched_artifact"].get("row_count"),
        "spine_dataset_id": result["spine_dataset_id"],
        "price_feature_dataset_id": result["price_feature_dataset_id"],
        "spine_path": result.get("spine_path"),
        "price_feature_registration_path": result.get("price_feature_registration_path"),
    }


def _missing_meta(path: Any) -> dict[str, Any]:
    return {"path": str(path) if path else None, "exists": bool(path and Path(str(path)).exists()), "row_count": None, "symbol_count": None, "date_min": None, "date_max": None}


def _empty_resolution() -> dict[str, Any]:
    return {"rows": [], "map_by_symbol": {}, "resolved_symbol_count": 0, "unresolved_symbols": [], "ambiguous_symbols": [], "validity_violations": []}


def _empty_alignment() -> dict[str, Any]:
    return {"same_row_id_set": False, "base_row_count": 0, "enriched_row_count": 0, "base_only_count": 0, "enriched_only_count": 0, "duplicate_base_count": 0, "duplicate_enriched_count": 0, "base_only_rows": [], "enriched_only_rows": []}


def _empty_target_alignment() -> dict[str, Any]:
    return {"target_mismatch_count": 0, "benchmark_mismatch_count": 0, "timestamp_mismatch_count": 0, "target_mismatches": [], "benchmark_mismatches": [], "timestamp_mismatches": []}


def _duplicate_candidate_key_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [(str(row.get("symbol", "")).upper(), _decision_timestamp(row), str(_target_horizon(row))) for row in rows]
    return len(keys) - len(set(keys))


def _target_horizon(row: Mapping[str, Any]) -> int:
    for field in ("target_horizon_trading_days", "target_horizon_sessions", "target_horizon"):
        value = row.get(field)
        if value not in (None, ""):
            try:
                return int(float(value))
            except ValueError:
                continue
    return 10


def _decision_timestamp(row: Mapping[str, Any]) -> str:
    return _timestamp_value(row, "decision_timestamp") or _date_to_close_utc(_date_value(row))


def _date_value(row: Mapping[str, Any]) -> str:
    return str(row.get("rebalance_date") or row.get("decision_session_date") or row.get("session_date") or "")[:10]


def _timestamp_value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).replace("+00:00", "Z")
    return text


def _date_to_close_utc(value: str) -> str:
    if not value:
        return ""
    return datetime.combine(datetime.fromisoformat(value).date(), time(21, 0), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_dt(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_id_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    return _hash_json(sorted(str(row.get("row_id")) for row in rows))


def _row_identity_hash(row: Mapping[str, Any]) -> str:
    return _hash_json({key: row.get(key) for key in ("asset_id", "decision_timestamp", "target_horizon_sessions", "universe_version", "daily_price_dataset_version", "target_definition_version")})


def _registry_version_from_resolution(resolution: Mapping[str, Any]) -> str:
    return _hash_json(resolution.get("rows", []))[:16]


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_mismatch_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) or ["row_id"]
    _write_csv(path, rows, fieldnames)


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Daily Stock Spine Verification",
        "",
        f"- Status: {payload['status']}",
        f"- Blockers: {', '.join(payload['blockers']) or 'none'}",
        f"- Base artifact: {payload['selected_sources'].get('base_artifact')}",
        f"- Enriched artifact: {payload['selected_sources'].get('enriched_artifact')}",
        f"- Base rows: {payload['base_artifact'].get('row_count')}",
        f"- Enriched rows: {payload['enriched_artifact'].get('row_count')}",
        f"- Symbols resolved: {payload['symbol_resolution'].get('resolved_symbol_count')}",
        f"- Base-only rows: {payload['alignment'].get('base_only_count')}",
        f"- Enriched-only rows: {payload['alignment'].get('enriched_only_count')}",
        f"- Target mismatches: {payload['target_alignment'].get('target_mismatch_count')}",
        f"- Benchmark mismatches: {payload['target_alignment'].get('benchmark_mismatch_count')}",
        f"- Temporal violations: {payload['temporal_validation'].get('violation_count')}",
        f"- Spine dataset ID: {payload['spine_dataset_id']}",
        f"- Price-feature dataset ID: {payload['price_feature_dataset_id']}",
        "",
    ])


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _norm_path(value: Any) -> str:
    return str(value).replace("\\", "/").strip().rstrip("/")


def _repo_rel(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
