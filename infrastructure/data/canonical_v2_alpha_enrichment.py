from __future__ import annotations

import csv
import hashlib
import json
import math
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow.lib import ArrowTypeError

from core.research.framework.config import StockLevelResearchConfig
from core.research.framework.reporting import ResearchArtifactWriter
from core.research.ml.runtime_parallelism import apply_stock_alpha_worker_caps
from core.research.ml.stock_level.stock_alpha_paths import stock_alpha_report_metadata
from core.research.ml.stock_level.stock_level_alpha_features_audit import _audit
from core.research.ml.stock_level.stock_level_alpha_features_builder import (
    _add_cross_sectional_features,
    _build_symbol_rows,
    _prepare_history,
)
from core.research.ml.stock_level.stock_level_alpha_features_io import (
    _load_price_histories,
    _markdown,
    _write_audit_csv,
)
from core.research.ml.stock_level.stock_level_alpha_features_types import (
    ENGINEERED_FEATURE_COLUMNS,
    ENRICHMENT_METADATA_COLUMNS,
    StockLevelAlphaFeaturePaths,
)
from core.research.ml.stock_level.stock_level_artifact_io import (
    canonical_artifact_path,
    read_stock_level_artifact,
)


DEFAULT_REPORT_ROOT = Path("reports/ml/development/ticket_7b3_daily_large_history/regeneration_canonical_v2")
EXPECTED_CANONICAL_HASH = "c2ab57992c9363c118d854f01da18ea34122b9c0775af3d0676afe5ff80bad56"
EXPECTED_BASE_HASH = "739a2b984cdd0a160d65ea546d9523b75637be3921c14734dd5483a093357e89"
BOOL_COLUMNS = {"selector_eligible", "provider_transition_flag"}
INT_COLUMNS = {
    "target_horizon_trading_days",
    "required_purge_horizon_trading_days",
    "target_observation_count",
    "breadth_eligible_symbol_count",
    "breadth_observed_symbol_count",
    "industry_peer_count",
    "fundamentals_data_age_days",
}
INTERMEDIATE_NUMERIC_COLUMNS = {
    "_stock_above_200d_average",
    "_stock_momentum_20d",
    "_stock_momentum_60d",
}
NUMERIC_COLUMNS = {
    *ENGINEERED_FEATURE_COLUMNS,
    *INTERMEDIATE_NUMERIC_COLUMNS,
    "model_close",
    "actual_forward_return_10d",
    "actual_forward_return_5d",
    "actual_future_volatility",
    "actual_future_drawdown",
    "actual_benchmark_return_10d",
    "actual_market_residual_return_10d",
    "actual_vol_adjusted_forward_return_10d",
    "actual_drawdown_adjusted_forward_return_10d",
    "actual_rank_normalized_forward_return_10d",
    "actual_top_decile_label_10d",
}


class PartitionBuildError(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        super().__init__(str(payload.get("exception_message", "")))


def write_partitioned_canonical_v2_alpha_features(config: dict[str, Any]) -> StockLevelAlphaFeaturePaths:
    settings = StockLevelResearchConfig.from_mapping(config)
    apply_stock_alpha_worker_caps(config)
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    report_root.mkdir(parents=True, exist_ok=True)
    input_resolution = resolve_inputs(config)
    _write_json(report_root / "input_resolution.json", input_resolution)
    if not input_resolution["gates_passed"]:
        raise ValueError(f"canonical-v2 alpha input gates failed: {input_resolution['blocking_issues']}")

    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_root = report_root / "partitions"
    manifest_root = report_root / "partition_manifests"
    base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
    partition_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    base_path = settings.base_artifact_path
    spine_index = _symbol_spine_index(config)

    configured_symbols = ml.get("canonical_v2_alpha_symbols")
    if configured_symbols:
        requested = {str(symbol).upper() for symbol in configured_symbols}
        symbols = [symbol for symbol in sorted(requested) if symbol in spine_index]
        missing = sorted(requested - set(symbols))
        if missing:
            raise ValueError(f"missing labeled spine partition for requested symbols: {missing[:10]}")
    else:
        symbols = sorted(spine_index)
    completed_before = _completed_symbols(manifest_root)
    pending = [symbol for symbol in symbols if symbol not in completed_before]
    workers = settings.alpha_feature_n_jobs
    plan = {
        "partition_unit": "symbol",
        "planned_partitions": len(symbols),
        "completed_before": len(completed_before),
        "pending_partitions": len(pending),
        "requested_workers": workers,
        "effective_workers": min(max(1, workers), max(1, len(pending))),
        "partition_root": str(partition_root),
        "manifest_root": str(manifest_root),
        "base_artifact_path": str(base_path),
        "base_partition_root": str(base_partition_root),
        "source_mode": "labeled_spine_partition",
        "labeled_spine_manifest_path": str(ml.get("canonical_v2_labeled_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/labeled_spine_manifest.json")),
        "canonical_price_root": str(settings.parquet_dir),
        "monolithic_base_read": False,
        "resume_enabled": True,
        "retry_only_failed_command": "python .\\main.py --mode ml-stock-level-alpha-features --config .\\config\\config.ticket_7b3_daily_large_history_regeneration_canonical_v2.yaml",
    }
    _write_json(report_root / "partition_plan.json", plan)
    _write_json(report_root / "failed_partitions.json", {"failed_partition_count": 0, "failed_partitions": []})
    started = time.perf_counter()
    failed: list[dict[str, Any]] = []
    rows_processed = 0
    aborted_early = False
    abort_reason = ""
    dominant_signature = ""
    tasks_cancelled = 0
    fail_fast = _fail_fast_settings(ml)
    if pending:
        prepared_spy = _prepare_history(_load_price_histories(settings.parquet_dir, [settings.spy_symbol]).get(settings.spy_symbol, []))
        effective_workers = min(max(1, workers), len(pending))
        if effective_workers == 1:
            for symbol in pending:
                try:
                    result = _build_partition(symbol, config, prepared_spy, partition_root, manifest_root)
                    rows_processed += int(result["row_count"])
                except Exception as exc:
                    failed.append(_failure_record(symbol, exc))
                    _write_json(report_root / "failed_partitions.json", {"failed_partition_count": len(failed), "failed_partitions": failed})
                    _progress(report_root, len(symbols), len(_completed_symbols(manifest_root)), len(failed), rows_processed, started)
                    raise
                _progress(report_root, len(symbols), len(_completed_symbols(manifest_root)), len(failed), rows_processed, started)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(_build_partition, symbol, config, prepared_spy, partition_root, manifest_root): symbol
                    for symbol in pending
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        rows_processed += int(result["row_count"])
                    except Exception as exc:  # pragma: no cover - exercised by integration failures.
                        failed.append(_failure_record(symbol, exc))
                        _write_json(report_root / "failed_partitions.json", {"failed_partition_count": len(failed), "failed_partitions": failed})
                        should_abort, abort_reason, dominant_signature = _should_abort_fail_fast(
                            failed,
                            completed=len(_completed_symbols(manifest_root)),
                            settings=fail_fast,
                        )
                        if should_abort:
                            aborted_early = True
                            for pending_future in futures:
                                if not pending_future.done() and pending_future.cancel():
                                    tasks_cancelled += 1
                    _progress(
                        report_root,
                        len(symbols),
                        len(_completed_symbols(manifest_root)),
                        len(failed),
                        rows_processed,
                        started,
                        aborted_early=aborted_early,
                        abort_reason=abort_reason,
                        dominant_failure_signature=dominant_signature,
                        tasks_cancelled=tasks_cancelled,
                    )
                    if aborted_early:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    _write_json(report_root / "failed_partitions.json", {"failed_partition_count": len(failed), "failed_partitions": failed})
    if aborted_early:
        raise RuntimeError(f"canonical-v2 alpha aborted early: {abort_reason}; dominant_signature={dominant_signature}")
    if failed:
        raise RuntimeError(f"canonical-v2 alpha partitions failed: {failed[:5]}")

    partition_paths = _completed_partition_paths(manifest_root, expected_symbols=symbols)
    partition_validation = _validate_partition_dataset(partition_paths, report_root=report_root)
    _write_json(
        report_root / "partition_dataset_status.json",
        {
            "partition_processing_status": "complete",
            "partition_validation_status": "complete",
            "consolidation_status": "pending",
            "partitions_reused": len(completed_before) if not pending else len(_completed_symbols(manifest_root)) - len(pending),
            "partitions_recomputed": len(pending),
            "consolidation_retried": True,
            **partition_validation,
        },
    )
    rows = _read_partition_rows(partition_root)
    rows.sort(key=lambda row: (str(row.get("rebalance_date", "")), str(row.get("symbol", "")).upper()))
    _add_cross_sectional_features(rows)
    source_rows = _read_partition_rows(base_partition_root)
    price_histories = _load_price_histories(settings.parquet_dir, sorted({*symbols, settings.spy_symbol}))
    prepared_histories = {symbol.upper(): _prepare_history(history) for symbol, history in price_histories.items()}
    audit = _audit(source_rows, rows, prepared_histories, str(base_path), workers)
    audit["parallelism"].update(
        {
            "requested_workers": workers,
            "effective_workers": plan["effective_workers"],
            "partition": "symbol",
            "symbol_count": len(symbols),
            "partitioned_resume": True,
        }
    )
    audit.update(stock_alpha_report_metadata(config, output_dir, source_artifact_path=base_path))
    audit["canonical_v2_input_resolution"] = input_resolution
    paths = StockLevelAlphaFeaturePaths(
        enriched_parquet_path=canonical_artifact_path(output_dir, "stock_level_prediction_artifacts_enriched", config),
        audit_csv_path=output_dir / "stock_level_alpha_feature_audit.csv",
        audit_json_path=output_dir / "stock_level_alpha_feature_audit.json",
        audit_markdown_path=output_dir / "stock_level_alpha_feature_audit.md",
        enriched_sample_csv_path=output_dir / "stock_level_prediction_artifacts_enriched_sample.csv",
    )
    try:
        identity = _consolidate_partition_parquets(
            partition_paths,
            paths.enriched_parquet_path,
            config=config,
            sample_path=paths.enriched_sample_csv_path,
            expected_row_count=int(partition_validation["row_count"]),
            report_root=report_root,
        )
    except Exception as exc:
        _write_json(
            report_root / "partition_dataset_status.json",
            {
                "partition_processing_status": "complete",
                "partition_validation_status": "complete",
                "consolidation_status": "failed",
                "consolidation_error": f"{type(exc).__name__}: {exc}",
                "partitions_reused": len(completed_before) if not pending else len(_completed_symbols(manifest_root)) - len(pending),
                "partitions_recomputed": len(pending),
                "consolidation_retried": True,
                **partition_validation,
            },
        )
        raise
    audit["canonical_artifact"] = identity
    audit["artifact_format"] = identity["artifact_format"]
    audit["artifact_path"] = identity["resolved_artifact_path"]
    audit["artifact_sha256"] = identity["sha256"]
    audit["logical_content_sha256"] = identity["logical_content_sha256"]
    audit["schema_fingerprint"] = identity["schema_fingerprint"]
    audit["target_contract_version"] = identity.get("target_contract_version")
    audit["benchmark_contract_version"] = "stock_level_benchmark_return_10d_v1"
    _write_audit_csv(paths.audit_csv_path, audit["features"])
    writer = ResearchArtifactWriter()
    writer.write_json(paths.audit_json_path, audit)
    writer.write_markdown(paths.audit_markdown_path, _markdown(audit))
    feature_coverage = audit["features"]
    _write_csv(report_root / "feature_coverage.csv", feature_coverage, ["feature", "definition", "populated_count", "missing_count", "availability_rate"])
    validation = validate_enriched_artifact(paths.enriched_parquet_path, input_resolution=input_resolution)
    validation.update(
        {
            "completed_partitions": len(_completed_symbols(manifest_root)),
            "failed_partitions": len(failed),
            "planned_partitions": len(symbols),
            "worker_count": workers,
            "artifact_hash": identity["sha256"],
            "feature_count": len(ENGINEERED_FEATURE_COLUMNS),
            "populated_feature_count": sum(1 for row in feature_coverage if int(row.get("populated_count", 0) or 0) > 0),
        }
    )
    _write_json(
        report_root / "partition_dataset_status.json",
        {
            "partition_processing_status": "complete",
            "partition_validation_status": "complete",
            "consolidation_status": "complete",
            "partitions_reused": len(completed_before) if not pending else len(_completed_symbols(manifest_root)) - len(pending),
            "partitions_recomputed": len(pending),
            "consolidation_retried": True,
            "final_artifact": identity,
            **partition_validation,
        },
    )
    _write_json(report_root / "final_validation.json", validation)
    _progress(report_root, len(symbols), len(_completed_symbols(manifest_root)), len(failed), validation["row_count"], started)
    return paths


def resolve_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    canonical_manifest_path = Path(str(ml.get("canonical_daily_v2_manifest_path", "reports/data_lineage/canonical_daily_v2/build_manifest.json")))
    canonical_root = Path(str(ml.get("canonical_daily_v2_root", "data/processed/market_data/canonical_daily_v2/full")))
    labeled_manifest_path = Path(str(ml.get("canonical_v2_labeled_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/labeled_spine_manifest.json")))
    inference_manifest_path = Path(str(ml.get("canonical_v2_inference_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/inference_spine_manifest.json")))
    recovered = Path("reports/ml/development/ticket_7b3_daily_large_history/regeneration/benchmark/stock_level_prediction_artifacts.parquet")
    settings = StockLevelResearchConfig.from_mapping(config)
    blocking: list[str] = []
    canonical_manifest = _read_json(canonical_manifest_path)
    labeled_manifest = _read_json(labeled_manifest_path)
    inference_manifest = _read_json(inference_manifest_path)
    canonical_hash = canonical_manifest.get("dataset_logical_partition_hash")
    if canonical_manifest.get("status") != "COMPLETE":
        blocking.append("canonical_manifest_not_complete")
    if int(canonical_manifest.get("completed_partitions", 0) or 0) != 514:
        blocking.append("canonical_completed_partitions_not_514")
    if canonical_hash != EXPECTED_CANONICAL_HASH:
        blocking.append("canonical_hash_mismatch")
    validation_path = canonical_manifest_path.with_name("validation.json")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True:
        blocking.append("canonical_validation_not_valid")
    if labeled_manifest.get("status") != "BUILT":
        blocking.append("labeled_spine_not_built")
    if inference_manifest.get("status") != "BUILT":
        blocking.append("inference_spine_not_built")
    recovered_hash = _file_sha256(recovered) if recovered.exists() else None
    if recovered_hash != EXPECTED_BASE_HASH:
        blocking.append("recovered_artifact_hash_mismatch")
    if str(ml.get("stock_selector_market_data_source", "")).lower() != "canonical_daily_v2":
        blocking.append("selector_source_not_canonical_v2")
    if Path(str(ml.get("stooq_parquet_dir", ""))) != canonical_root:
        blocking.append("alpha_price_root_not_canonical_v2")
    if "expanded_rebalance_dataset" in str(settings.base_artifact_path).lower():
        blocking.append("base_artifact_points_to_expanded_rebalance_cache")
    return {
        "gates_passed": not blocking,
        "blocking_issues": blocking,
        "canonical_dataset": {
            "path": str(canonical_root),
            "manifest_path": str(canonical_manifest_path),
            "hash": canonical_hash,
            "row_count": canonical_manifest.get("row_count"),
            "symbol_count": canonical_manifest.get("symbol_count"),
            "date_min": canonical_manifest.get("date_min"),
            "date_max": canonical_manifest.get("date_max"),
            "completed_partitions": canonical_manifest.get("completed_partitions"),
            "validation_path": str(validation_path),
            "validation_valid": validation.get("valid"),
        },
        "labeled_spine": _manifest_summary(labeled_manifest, labeled_manifest_path),
        "inference_spine": _manifest_summary(inference_manifest, inference_manifest_path),
        "base_artifact": {
            "path": str(settings.base_artifact_path),
            "exists": settings.base_artifact_path.exists(),
            "recovered_reference_path": str(recovered),
            "recovered_reference_hash": recovered_hash,
        },
        "worker_configuration": {
            "stock_alpha_feature_n_jobs": settings.alpha_feature_n_jobs,
            "stock_level_dataset_workers": settings.dataset_workers,
        },
        "output_paths": {
            "output_dir": str(settings.output_dir),
            "enriched_artifact": str(canonical_artifact_path(settings.output_dir, "stock_level_prediction_artifacts_enriched", config)),
        },
        "resume_manifest_paths": {
            "partition_plan": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "partition_plan.json"),
            "progress_manifest": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "progress_manifest.json"),
            "failed_partitions": str(Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment"))) / "failed_partitions.json"),
        },
        "stooq_fallback_used": False,
    }


def build_base_artifact_from_labeled_spines(config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    rows: list[dict[str, Any]] = []
    spy_returns = _spine_return_by_date(root / "symbol=SPY" / "spine.parquet")
    for path in sorted(root.glob("symbol=*/spine.parquet")):
        for row in _read_parquet_file(path):
            if not row.get("selector_eligible") or not row.get("is_labeled"):
                continue
            rows.append(_base_row(row, spy_returns, input_resolution=input_resolution))
    rows.sort(key=lambda row: (row["rebalance_date"], row["symbol"]))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["rebalance_date", "symbol"]
    identity = _write_large_parquet_artifact(
        settings.base_artifact_path,
        rows,
        fieldnames=fieldnames,
        config=config,
        inspection_sample_path=settings.output_dir / "stock_level_prediction_artifacts_sample.csv",
    )
    writer = ResearchArtifactWriter()
    writer.write_json(
        settings.output_dir / "stock_level_prediction_artifacts.json",
        {
            "mode": "canonical_v2_labeled_spine_base_artifact",
            "row_count": len(rows),
            "symbol_count": len({row["symbol"] for row in rows}),
            "date_min": min((row["rebalance_date"] for row in rows), default=None),
            "date_max": max((row["rebalance_date"] for row in rows), default=None),
            "canonical_artifact": identity,
            "source_spine_root": str(root),
            "research_only": True,
            "trading_impact": "none",
            "production_validated": False,
        },
    )
    return identity


def ensure_base_symbol_partitions(config: Mapping[str, Any], *, input_resolution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
    labeled_root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    configured_symbols = ml.get("canonical_v2_alpha_symbols")
    requested = {str(symbol).upper() for symbol in configured_symbols} if configured_symbols else None
    manifest_path = report_root / "base_partition_manifest.json"
    existing = sorted(base_partition_root.glob("symbol=*/rows.parquet"))
    if existing:
        return _read_json(manifest_path) or {"status": "BUILT", "partition_count": len(existing), "path": str(base_partition_root)}
    base_partition_root.mkdir(parents=True, exist_ok=True)
    spy_returns = _spine_return_by_date(labeled_root / "symbol=SPY" / "spine.parquet")
    partitions = []
    started = time.perf_counter()
    for spine_path in sorted(labeled_root.glob("symbol=*/spine.parquet")):
        symbol = spine_path.parent.name.split("=", 1)[1]
        if requested is not None and symbol.upper() not in requested:
            continue
        rows = [
            _base_row(row, spy_returns, input_resolution=input_resolution)
            for row in _read_parquet_file(spine_path)
            if row.get("selector_eligible") and row.get("is_labeled")
        ]
        if not rows:
            continue
        target = base_partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
        _write_partition_parquet(target, rows, list(rows[0]))
        partitions.append({"symbol": symbol, "row_count": len(rows), "path": str(target), "sha256": _file_sha256(target)})
    payload = {
        "status": "BUILT",
        "path": str(base_partition_root),
        "partition_count": len(partitions),
        "row_count": sum(int(row["row_count"]) for row in partitions),
        "symbol_count": len(partitions),
        "elapsed_seconds": time.perf_counter() - started,
        "base_artifact_path": str(settings.base_artifact_path),
        "source_labeled_spine_root": str(labeled_root),
        "partitions": partitions,
    }
    _write_json(manifest_path, payload)
    return payload


def validate_enriched_artifact(path: Path, *, input_resolution: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_stock_level_artifact(path, required_columns={"rebalance_date", "symbol", "asset_id", "actual_forward_return_10d"})
    keys = [(row.get("asset_id"), str(row.get("rebalance_date"))[:10]) for row in rows]
    symbol_keys = [(str(row.get("symbol", "")).upper(), str(row.get("rebalance_date"))[:10]) for row in rows]
    feature_missingness = {}
    for feature in ENGINEERED_FEATURE_COLUMNS:
        missing = sum(1 for row in rows if row.get(feature) in (None, "", "nan"))
        feature_missingness[feature] = missing
    tier_d_rows = sum(1 for row in rows if row.get("compatibility_tier") == "TIER_D_SYMBOL_QUARANTINE")
    quarantined_rows = sum(1 for row in rows if str(row.get("eligibility_reason", "")).startswith("quarantined:"))
    label_violations = sum(
        1
        for row in rows
        if str(row.get("label_available_timestamp", ""))[:10] < str(row.get("decision_timestamp", ""))[:10]
    )
    return {
        "valid": len(keys) == len(set(keys)) and len(symbol_keys) == len(set(symbol_keys)) and tier_d_rows == 0 and quarantined_rows == 0 and label_violations == 0,
        "path": str(path),
        "row_count": len(rows),
        "symbol_count": len({str(row.get("symbol", "")).upper() for row in rows if row.get("symbol")}),
        "date_min": min((str(row.get("rebalance_date"))[:10] for row in rows), default=None),
        "date_max": max((str(row.get("rebalance_date"))[:10] for row in rows), default=None),
        "duplicate_asset_session_rows": len(keys) - len(set(keys)),
        "duplicate_symbol_session_rows": len(symbol_keys) - len(set(symbol_keys)),
        "tier_d_rows": tier_d_rows,
        "quarantined_rows": quarantined_rows,
        "label_availability_violations": label_violations,
        "feature_missingness": feature_missingness,
        "canonical_source_identity": dict(input_resolution.get("canonical_dataset", {})),
    }


def _build_partition(
    symbol: str,
    config: Mapping[str, Any],
    prepared_spy: list[dict[str, float | str]],
    partition_root: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    settings = StockLevelResearchConfig.from_mapping(config)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    source = _resolve_symbol_source(config, symbol)
    source_mode = "labeled_spine_partition"
    monolithic_base_read = False
    base_partition_reused = source["base_partition_reused"]
    source_base_partition_path = source["base_partition_path"]
    source_spine_path = source["spine_path"]
    phase = "spine_read"
    spine_rows_read = 0
    price_history_rows_read = 0
    try:
        phase_started = time.perf_counter()
        rows, source_meta = _read_symbol_source_rows_from_spine(
            symbol,
            Path(source_spine_path),
            Path(source_base_partition_path),
            config=config,
            input_resolution=resolve_inputs(config),
        )
        timings["spine_read_seconds"] = float(source_meta.get("spine_read_seconds", time.perf_counter() - phase_started) or 0.0)
        timings["base_derivation_seconds"] = float(source_meta.get("base_derivation_seconds", 0.0) or 0.0)
        base_partition_reused = bool(source_meta["base_partition_reused"])
        spine_rows_read = int(source_meta.get("source_rows_read", len(rows)) or len(rows))

        phase = "price_history_read"
        phase_started = time.perf_counter()
        history = _load_price_histories(settings.parquet_dir, [symbol]).get(symbol, [])
        price_history_rows_read = len(history)
        timings["price_history_read_seconds"] = time.perf_counter() - phase_started

        phase = "feature_compute"
        phase_started = time.perf_counter()
        prepared_history = _prepare_history(history)
        enriched = _build_symbol_rows((rows, prepared_history, prepared_spy))
        timings["feature_compute_seconds"] = time.perf_counter() - phase_started
    except Exception as exc:
        failure = _partition_failure_payload(
            symbol,
            exc,
            phase=phase,
            source_spine_path=source_spine_path,
            source_base_partition_path=source_base_partition_path,
            monolithic_base_read=monolithic_base_read,
            base_partition_reused=base_partition_reused,
            source_rows_read=spine_rows_read,
            price_history_rows_read=price_history_rows_read,
            timings=timings,
        )
        _write_json(manifest_root.parent / "partition_failures" / f"{_safe_symbol(symbol)}.json", failure)
        raise PartitionBuildError(failure) from exc
    path = partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    diagnosis_path = manifest_root.parent / "schema_diagnostics" / f"{_safe_symbol(symbol)}.json"
    inventory = _column_type_inventory(enriched)
    _write_json(diagnosis_path, {"symbol": symbol, "phase": "pre_normalization", "column_type_inventory": inventory, "timestamp": datetime.now(timezone.utc).isoformat()})
    try:
        phase = "normalisation"
        phase_started = time.perf_counter()
        normalized, schema_report = _normalize_partition_rows(enriched)
        _validate_normalized_rows(enriched, normalized, schema_report)
        table = pa.Table.from_pylist(normalized, schema=_schema_for_fieldnames(list(normalized[0]) if normalized else []))
        timings["normalisation_seconds"] = time.perf_counter() - phase_started
    except (ArrowTypeError, TypeError, ValueError) as exc:
        failure = _schema_failure_payload(
            symbol,
            exc,
            inventory,
            phase=phase,
            source_spine_path=source_spine_path,
            source_base_partition_path=source_base_partition_path,
            monolithic_base_read=monolithic_base_read,
            base_partition_reused=base_partition_reused,
            source_rows_read=spine_rows_read,
            price_history_rows_read=price_history_rows_read,
            timings=timings,
        )
        _write_json(manifest_root.parent / "partition_failures" / f"{_safe_symbol(symbol)}.json", failure)
        raise PartitionBuildError(failure) from exc
    _write_json(manifest_root.parent / "schema_validation" / f"{_safe_symbol(symbol)}.json", schema_report)
    phase_started = time.perf_counter()
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    timings["parquet_write_seconds"] = time.perf_counter() - phase_started
    timings["total_seconds"] = time.perf_counter() - started
    manifest = {
        "symbol": symbol,
        "status": "COMPLETE",
        "row_count": len(enriched),
        "path": str(path),
        "sha256": _file_sha256(path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "source_rows_read": spine_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": timings,
    }
    _write_json(manifest_root / f"{_safe_symbol(symbol)}.json", manifest)
    return manifest


def _base_row(row: Mapping[str, Any], spy_returns: Mapping[str, float], *, input_resolution: Mapping[str, Any] | None) -> dict[str, Any]:
    date = str(row["session_date"])[:10]
    end = str(row.get("target_end_session_date") or date)[:10]
    actual = _float(row.get("actual_forward_return_10d"))
    benchmark = spy_returns.get(date)
    benchmark = 0.0 if benchmark is None else benchmark
    return {
        "rebalance_date": date,
        "symbol": str(row.get("symbol") or row.get("canonical_symbol", "")).upper(),
        "asset_id": row.get("asset_id"),
        "canonical_symbol": row.get("canonical_symbol"),
        "source_provider": row.get("source_provider"),
        "compatibility_tier": row.get("compatibility_tier"),
        "eligibility_reason": row.get("eligibility_reason"),
        "selector_eligible": bool(row.get("selector_eligible")),
        "provider_transition_flag": bool(row.get("provider_transition_flag")),
        "provider_transition_id": row.get("provider_transition_id"),
        "target_provenance_contract_version": "stock_level_target_provenance_v1",
        "feature_timestamp": date,
        "feature_data_cutoff_timestamp": f"{date} 20:00:00+00:00",
        "decision_timestamp": f"{date} 20:05:00+00:00",
        "decision_session_date": date,
        "first_actionable_session": "",
        "decision_grid_version": "canonical_daily_v2_labeled_spine_v1",
        "decision_grid_identity": input_resolution.get("canonical_dataset", {}).get("hash") if input_resolution else "",
        "exchange_calendar_identity": "canonical_daily_v2_sessions",
        "decision_frequency": "daily",
        "target_horizon_trading_days": int(row.get("target_horizon_trading_days") or 10),
        "overlapping_targets": "",
        "required_purge_horizon_trading_days": 10,
        "target_horizon": "10_trading_observations",
        "target_observation_count": 10,
        "target_start_timestamp": f"{date} 00:00:00+00:00",
        "label_start_timestamp": f"{date} 00:00:00+00:00",
        "label_end_timestamp": f"{end} 00:00:00+00:00",
        "label_available_timestamp": f"{end} 21:00:00+00:00",
        "target_price_convention": "canonical_daily_v2_model_close_to_close",
        "benchmark_target_start_timestamp": f"{date} 00:00:00+00:00",
        "benchmark_label_start_timestamp": f"{date} 00:00:00+00:00",
        "benchmark_label_end_timestamp": f"{end} 00:00:00+00:00",
        "benchmark_label_available_timestamp": f"{end} 21:00:00+00:00",
        "target_status": "realized",
        "actual_forward_return_10d": actual,
        "actual_forward_return_5d": "",
        "actual_future_volatility": "",
        "actual_future_drawdown": "",
        "actual_benchmark_return_10d": benchmark,
        "actual_market_residual_return_10d": actual - benchmark if actual is not None else "",
        "actual_vol_adjusted_forward_return_10d": actual,
        "actual_drawdown_adjusted_forward_return_10d": actual,
        "actual_rank_normalized_forward_return_10d": "",
        "actual_top_decile_label_10d": "",
        "source_dataset_hash": input_resolution.get("canonical_dataset", {}).get("hash") if input_resolution else "",
    }


def _spine_return_by_date(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {
        str(row["session_date"])[:10]: float(row["actual_forward_return_10d"])
        for row in _read_parquet_file(path, columns=["session_date", "actual_forward_return_10d"])
        if row.get("actual_forward_return_10d") is not None
    }


def _artifact_symbols(path: Path, *, base_partition_root: Path | None = None) -> list[str]:
    if base_partition_root is not None and base_partition_root.exists():
        symbols = [
            p.parent.name.split("=", 1)[1].upper()
            for p in sorted(base_partition_root.glob("symbol=*/rows.parquet"))
        ]
        if symbols:
            return symbols
    table = pq.read_table(path, columns=["symbol"])
    return sorted({str(value).upper() for value in table.column("symbol").to_pylist() if value})


def _symbol_spine_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ml = dict(config.get("ml", {}) or {})
    manifest_path = Path(str(ml.get("canonical_v2_labeled_spine_manifest_path", "reports/ml/readiness/selector_spine_extension/labeled_spine_manifest.json")))
    manifest = _read_json(manifest_path)
    entries = manifest.get("partition_manifests") or []
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("status") != "BUILT":
            continue
        symbol = str(entry.get("canonical_symbol") or "").upper()
        path = Path(str(entry.get("path") or ""))
        if symbol and path.exists():
            index[symbol] = {**entry, "path": str(path)}
    if index:
        return index
    root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    for path in sorted(root.glob("symbol=*/spine.parquet")):
        symbol = path.parent.name.split("=", 1)[1].upper()
        index[symbol] = {"canonical_symbol": symbol, "path": str(path), "status": "BUILT"}
    return index


def _resolve_symbol_source(config: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    ml = dict(config.get("ml", {}) or {})
    report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
    base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
    spine_index = _symbol_spine_index(config)
    canonical_symbol = symbol.upper()
    if canonical_symbol not in spine_index:
        raise FileNotFoundError(f"missing labeled spine partition for symbol {canonical_symbol}")
    base_partition_path = base_partition_root / f"symbol={_safe_symbol(canonical_symbol)}" / "rows.parquet"
    return {
        "symbol": canonical_symbol,
        "source_mode": "labeled_spine_partition",
        "spine_path": spine_index[canonical_symbol]["path"],
        "base_partition_path": str(base_partition_path),
        "base_partition_reused": base_partition_path.exists(),
        "monolithic_base_read": False,
    }


def _read_source_rows(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    if columns:
        return _read_parquet_file(path, columns=columns)
    return _read_parquet_file(path)


def _read_symbol_source_rows_from_spine(
    symbol: str,
    spine_path: Path,
    base_partition_path: Path,
    *,
    config: Mapping[str, Any],
    input_resolution: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spine_path.exists():
        raise FileNotFoundError(f"missing labeled spine partition for symbol {symbol}: {spine_path}")
    if base_partition_path.exists():
        rows = _read_parquet_file(base_partition_path)
        return rows, {
            "base_partition_reused": True,
            "source_rows_read": len(rows),
            "spine_read_seconds": 0.0,
            "base_derivation_seconds": 0.0,
        }
    spine_started = time.perf_counter()
    spine_rows = _read_parquet_file(spine_path)
    spine_seconds = time.perf_counter() - spine_started
    derive_started = time.perf_counter()
    ml = dict(config.get("ml", {}) or {})
    labeled_root = Path(str(ml.get("canonical_v2_labeled_spine_root", "reports/ml/readiness/selector_spine_extension/labeled_selector_spine_partitions")))
    spy_returns = _spine_return_by_date(labeled_root / "symbol=SPY" / "spine.parquet")
    rows = [
        _base_row(row, spy_returns, input_resolution=input_resolution)
        for row in spine_rows
        if row.get("selector_eligible") and row.get("is_labeled")
    ]
    if not rows:
        raise ValueError(f"labeled spine partition for symbol {symbol} produced no eligible labeled rows")
    _write_partition_parquet(base_partition_path, rows, list(rows[0]))
    return rows, {
        "base_partition_reused": False,
        "source_rows_read": len(spine_rows),
        "spine_read_seconds": spine_seconds,
        "base_derivation_seconds": time.perf_counter() - derive_started,
    }


def _read_symbol_source_rows(path: Path, symbol: str, *, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if config is not None:
        ml = dict(config.get("ml", {}) or {})
        report_root = Path(str(ml.get("canonical_v2_alpha_report_root", DEFAULT_REPORT_ROOT / "alpha_enrichment")))
        base_partition_root = Path(str(ml.get("canonical_v2_base_partition_root", report_root / "base_partitions")))
        partition = base_partition_root / f"symbol={_safe_symbol(symbol)}" / "rows.parquet"
        if partition.exists():
            return _read_parquet_file(partition)
    try:
        table = pq.read_table(path, filters=[("symbol", "=", symbol)])
        rows = table.to_pylist()
        if rows:
            return rows
    except Exception:
        pass
    return [
        row
        for row in _read_parquet_file(path)
        if str(row.get("symbol", "")).upper() == symbol
    ]


def _read_partition_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("symbol=*/rows.parquet")):
        rows.extend(_read_parquet_file(path))
    return rows


def _completed_partition_paths(manifest_root: Path, *, expected_symbols: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []
    for symbol in expected_symbols:
        manifest = _read_json(manifest_root / f"{_safe_symbol(symbol)}.json")
        path = Path(str(manifest.get("path") or ""))
        if manifest.get("status") != "COMPLETE" or not path.exists():
            missing.append(symbol)
            continue
        paths.append(path)
    if missing:
        raise FileNotFoundError(f"missing completed alpha partitions for symbols: {missing[:10]}")
    return paths


def _read_parquet_file(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    return pq.ParquetFile(path).read(columns=list(columns) if columns else None).to_pylist()


def _fieldnames(source_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    source_columns = list(source_rows[0]) if source_rows else []
    return list(dict.fromkeys([*source_columns, *ENGINEERED_FEATURE_COLUMNS, *ENRICHMENT_METADATA_COLUMNS]))


def _write_large_parquet_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    config: Mapping[str, Any],
    sample_path: Path | None,
) -> dict[str, Any]:
    compression = str(dict(config.get("ml", {}) or {}).get("stock_level_parquet_compression", "zstd")).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    schema = _schema_for_rows(rows, fieldnames)
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(rows), 100_000):
            chunk = [
                {name: row.get(name) for name in fieldnames}
                for row in rows[start : start + 100_000]
            ]
            table = pa.Table.from_pylist(chunk, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=compression)
            writer.write_table(table)
        if writer is None:
            writer = pq.ParquetWriter(tmp, schema, compression=compression)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(path)
    if sample_path is not None:
        _write_csv(sample_path, [{name: row.get(name) for name in fieldnames} for row in rows[:100]], fieldnames)
    parquet = pq.ParquetFile(path)
    column_order = list(parquet.schema_arrow.names)
    decision_dates = _column_values(path, "rebalance_date")
    symbols = _column_values(path, "symbol")
    target_versions = sorted(set(_column_values(path, "target_provenance_contract_version")))
    dataset_hashes = sorted(set(_column_values(path, "source_dataset_hash")))
    file_hash = _file_sha256(path)
    return {
        "artifact_format": "parquet",
        "compression": compression,
        "resolved_artifact_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": file_hash,
        "logical_content_sha256": file_hash,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
        "stable_column_order": column_order,
        "row_count": parquet.metadata.num_rows,
        "column_count": len(column_order),
        "symbol_count": len({str(value).upper() for value in symbols if value}),
        "decision_date_count": len({str(value)[:10] for value in decision_dates if value}),
        "minimum_decision_timestamp": min((str(value) for value in decision_dates if value), default=None),
        "maximum_decision_timestamp": max((str(value) for value in decision_dates if value), default=None),
        "target_contract_version": target_versions[0] if len(target_versions) == 1 else None,
        "target_contract_versions": target_versions,
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "source_dataset_hash_count": len(dataset_hashes),
        "source_dataset_hashes": dataset_hashes[:10],
        "completion_status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_partition_dataset(partition_paths: Sequence[Path], *, report_root: Path | None = None) -> dict[str, Any]:
    if not partition_paths:
        raise ValueError("no completed alpha partitions available for consolidation")
    first = pq.ParquetFile(partition_paths[0]).read()
    canonical_schema = _schema_for_fieldnames(first.schema.names)
    partition_reports: list[dict[str, Any]] = []
    row_count = 0
    duplicate_keys = 0
    seen_keys: set[tuple[str, str]] = set()
    for path in partition_paths:
        table = pq.ParquetFile(path).read()
        report, casted = _validate_and_cast_partition_table(path, table, canonical_schema)
        partition_reports.append(report)
        row_count += table.num_rows
        symbols = casted["symbol"] if "symbol" in casted.column_names else None
        dates = casted["rebalance_date"] if "rebalance_date" in casted.column_names else None
        if symbols is not None and dates is not None:
            for symbol, date in zip(_iter_chunked_values(symbols), _iter_chunked_values(dates)):
                key = (str(symbol).upper(), str(date)[:10])
                if key in seen_keys:
                    duplicate_keys += 1
                seen_keys.add(key)
    if duplicate_keys:
        raise ValueError(f"duplicate symbol/date keys in partition dataset: {duplicate_keys}")
    payload = {
        "partition_count": len(partition_paths),
        "row_count": row_count,
        "canonical_schema_fingerprint": _schema_fingerprint(canonical_schema),
        "duplicate_symbol_date_keys": duplicate_keys,
        "partitions": partition_reports,
    }
    if report_root is not None:
        _write_json(report_root / "partition_schema_validation.json", payload)
    return payload


def _consolidate_partition_parquets(
    partition_paths: Sequence[Path],
    output_path: Path,
    *,
    config: Mapping[str, Any],
    sample_path: Path | None,
    expected_row_count: int,
    report_root: Path | None = None,
) -> dict[str, Any]:
    if not partition_paths:
        raise ValueError("no completed alpha partitions available for consolidation")
    compression = str(dict(config.get("ml", {}) or {}).get("stock_level_parquet_compression", "zstd")).lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    first = pq.ParquetFile(partition_paths[0]).read()
    canonical_schema = _schema_for_fieldnames(first.schema.names)
    reports: list[dict[str, Any]] = []
    row_count = 0
    seen_keys: set[tuple[str, str]] = set()
    duplicate_keys = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(tmp, canonical_schema, compression=compression)
        for path in partition_paths:
            table = pq.ParquetFile(path).read()
            report, casted = _validate_and_cast_partition_table(path, table, canonical_schema)
            reports.append(report)
            symbols = casted["symbol"] if "symbol" in casted.column_names else None
            dates = casted["rebalance_date"] if "rebalance_date" in casted.column_names else None
            if symbols is not None and dates is not None:
                for symbol, date in zip(_iter_chunked_values(symbols), _iter_chunked_values(dates)):
                    key = (str(symbol).upper(), str(date)[:10])
                    if key in seen_keys:
                        duplicate_keys += 1
                    seen_keys.add(key)
            if duplicate_keys:
                raise ValueError(f"duplicate symbol/date keys during consolidation: {duplicate_keys}")
            writer.write_table(casted)
            row_count += casted.num_rows
        writer.close()
        writer = None
        if row_count != expected_row_count:
            raise ValueError(f"consolidated row count {row_count} does not match expected {expected_row_count}")
        promoted_metadata = pq.read_metadata(tmp)
        promoted_schema = pq.read_schema(tmp)
        if promoted_metadata.num_rows != expected_row_count:
            raise ValueError(f"temporary artifact row count {promoted_metadata.num_rows} does not match expected {expected_row_count}")
        if _schema_fingerprint(promoted_schema) != _schema_fingerprint(canonical_schema):
            raise ValueError("temporary artifact schema fingerprint mismatch")
        tmp.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        raise
    if sample_path is not None:
        _write_parquet_sample_csv(output_path, sample_path, limit=100)
    parquet = pq.ParquetFile(output_path)
    column_order = list(parquet.schema_arrow.names)
    decision_dates = _column_values(output_path, "rebalance_date")
    symbols = _column_values(output_path, "symbol")
    target_versions = sorted(set(_column_values(output_path, "target_provenance_contract_version")))
    dataset_hashes = sorted(set(_column_values(output_path, "source_dataset_hash")))
    file_hash = _file_sha256(output_path)
    identity = {
        "artifact_format": "parquet",
        "compression": compression,
        "resolved_artifact_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "sha256": file_hash,
        "logical_content_sha256": file_hash,
        "schema_fingerprint": _schema_fingerprint(parquet.schema_arrow),
        "stable_column_order": column_order,
        "row_count": parquet.metadata.num_rows,
        "column_count": len(column_order),
        "symbol_count": len({str(value).upper() for value in symbols if value}),
        "decision_date_count": len({str(value)[:10] for value in decision_dates if value}),
        "minimum_decision_timestamp": min((str(value) for value in decision_dates if value), default=None),
        "maximum_decision_timestamp": max((str(value) for value in decision_dates if value), default=None),
        "target_contract_version": target_versions[0] if len(target_versions) == 1 else None,
        "target_contract_versions": target_versions,
        "benchmark_contract_version": "stock_level_benchmark_return_10d_v1",
        "source_dataset_hash_count": len(dataset_hashes),
        "source_dataset_hashes": dataset_hashes[:10],
        "completion_status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": "validated_symbol_partitions",
        "source_partition_count": len(partition_paths),
    }
    if report_root is not None:
        _write_json(
            report_root / "consolidation_manifest.json",
            {
                "status": "COMPLETE",
                "row_count": row_count,
                "partition_count": len(partition_paths),
                "duplicate_symbol_date_keys": duplicate_keys,
                "schema_fingerprint": identity["schema_fingerprint"],
                "partitions": reports,
                "artifact": identity,
            },
        )
    return identity


def _validate_and_cast_partition_table(path: Path, table: pa.Table, canonical_schema: pa.Schema) -> tuple[dict[str, Any], pa.Table]:
    missing = [name for name in canonical_schema.names if name not in table.column_names]
    unexpected = [name for name in table.column_names if name not in canonical_schema.names]
    type_mismatches: list[dict[str, Any]] = []
    casts: list[dict[str, str]] = []
    if missing or unexpected:
        raise ValueError(f"partition schema columns mismatch for {path}: missing={missing[:10]} unexpected={unexpected[:10]}")
    columns = []
    for field in canonical_schema:
        column = table[field.name]
        actual_type = column.type
        if actual_type.equals(field.type):
            columns.append(column)
            continue
        casted = _safe_cast_partition_column(path, field.name, column, field.type)
        type_mismatches.append({"column": field.name, "actual_type": str(actual_type), "expected_type": str(field.type)})
        casts.append({"column": field.name, "from": str(actual_type), "to": str(field.type)})
        columns.append(casted)
    casted_table = pa.Table.from_arrays(columns, schema=canonical_schema)
    report = {
        "partition_path": str(path),
        "row_count": table.num_rows,
        "schema_fingerprint": _schema_fingerprint(table.schema),
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "type_mismatches": type_mismatches,
        "cast_operations": casts,
    }
    return report, casted_table


def _safe_cast_partition_column(path: Path, name: str, column: pa.ChunkedArray, expected_type: pa.DataType) -> pa.ChunkedArray:
    actual_type = column.type
    if pa.types.is_null(actual_type):
        return column.cast(expected_type)
    if pa.types.is_string(expected_type):
        if pa.types.is_dictionary(actual_type) or pa.types.is_large_string(actual_type):
            return column.cast(expected_type)
    if pa.types.is_floating(expected_type):
        if pa.types.is_integer(actual_type) or pa.types.is_floating(actual_type):
            return column.cast(expected_type)
        if pa.types.is_string(actual_type) or pa.types.is_large_string(actual_type):
            representative = _first_non_null_string(column)
            raise ValueError(
                "numeric column contains text during consolidation: "
                f"partition={path} column={name} row_index={representative['row_index']} "
                f"value={representative['value']!r}"
            )
    if pa.types.is_int64(expected_type) and pa.types.is_integer(actual_type):
        return column.cast(expected_type)
    raise ValueError(f"unsafe partition schema cast for {path}: column={name} actual={actual_type} expected={expected_type}")


def _first_non_null_string(column: pa.ChunkedArray) -> dict[str, Any]:
    offset = 0
    for chunk in column.chunks:
        for index, scalar in enumerate(chunk):
            value = scalar.as_py()
            if value is not None:
                return {"row_index": offset + index, "value": value}
        offset += len(chunk)
    return {"row_index": None, "value": None}


def _iter_chunked_values(column: pa.ChunkedArray):
    for chunk in column.chunks:
        for scalar in chunk:
            yield scalar.as_py()


def _write_parquet_sample_csv(path: Path, sample_path: Path, *, limit: int) -> None:
    table = pq.ParquetFile(path).read().slice(0, limit)
    fieldnames = table.column_names
    rows: list[dict[str, Any]] = []
    columns = {name: table[name].to_pylist() for name in fieldnames}
    for index in range(table.num_rows):
        rows.append({name: columns[name][index] for name in fieldnames})
    _write_csv(sample_path, rows, fieldnames)


def _write_partition_parquet(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    schema = _schema_for_rows(rows, fieldnames)
    table = pa.Table.from_pylist([{name: row.get(name) for name in fieldnames} for row in rows], schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _column_type_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = list(dict.fromkeys(name for row in rows for name in row))
    inventory: dict[str, Any] = {}
    for column in columns:
        counts: dict[str, int] = {}
        examples: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            value = row.get(column)
            type_name = "empty_string" if value == "" else ("null" if value is None else type(value).__name__)
            counts[type_name] = counts.get(type_name, 0) + 1
            if type_name not in examples:
                examples[type_name] = {
                    "value": repr(value),
                    "row_index": index,
                    "symbol": row.get("symbol"),
                    "rebalance_date": row.get("rebalance_date"),
                }
        inventory[column] = {
            "python_type_counts": dict(sorted(counts.items())),
            "representative_values": examples,
        }
    return inventory


def _normalize_partition_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    normalized: list[dict[str, Any]] = []
    coerced_nulls: dict[str, int] = {name: 0 for name in fieldnames}
    invalid_values: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        output: dict[str, Any] = {}
        for column in fieldnames:
            value = row.get(column)
            kind = _column_kind(column)
            try:
                normalized_value, coerced = _normalize_value(column, value, kind)
            except ValueError as exc:
                invalid_values.setdefault(column, []).append(
                    {
                        "row_index": row_index,
                        "symbol": row.get("symbol"),
                        "rebalance_date": row.get("rebalance_date"),
                        "value": repr(value),
                        "error": str(exc),
                    }
                )
                normalized_value, coerced = None, False
            if coerced:
                coerced_nulls[column] += 1
            output[column] = normalized_value
        normalized.append(output)
    duplicate_count = _duplicate_symbol_date_count(normalized)
    report = {
        "input_row_count": len(rows),
        "output_row_count": len(normalized),
        "column_count": len(fieldnames),
        "columns": [
            {"name": name, "kind": _column_kind(name), "arrow_type": str(_arrow_type_for_column(name))}
            for name in fieldnames
        ],
        "values_coerced_to_null_by_column": {k: v for k, v in coerced_nulls.items() if v},
        "invalid_values_rejected_by_column": invalid_values,
        "duplicate_symbol_date_keys": duplicate_count,
        "valid": len(rows) == len(normalized) and duplicate_count == 0 and not invalid_values,
    }
    if invalid_values:
        first_column = next(iter(invalid_values))
        first = invalid_values[first_column][0]
        raise ValueError(f"Invalid value for column {first_column}: {first['value']} ({first['error']})")
    return normalized, report


def _normalize_value(column: str, value: Any, kind: str) -> tuple[Any, bool]:
    if value is None:
        return None, False
    if value == "":
        return (None, True) if kind in {"float", "int", "bool"} else ("", False)
    if kind == "string":
        if isinstance(value, str):
            return value, False
        raise ValueError(f"text column {column} received {type(value).__name__}")
    if kind == "bool":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int) and value in {0, 1}:
            return bool(value), False
        if isinstance(value, float) and value in {0.0, 1.0}:
            return bool(value), False
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true", False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ValueError(f"bool column {column} received non-binary numeric value {value!r}")
        raise ValueError(f"bool column {column} received {type(value).__name__}")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"int column {column} received bool")
        if isinstance(value, int):
            return value, False
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value), False
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError as exc:
                raise ValueError(f"int column {column} received nonnumeric text") from exc
            if math.isfinite(parsed) and parsed.is_integer():
                return int(parsed), False
        raise ValueError(f"int column {column} received non-integer value {value!r}")
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"float column {column} received bool")
        if isinstance(value, (int, float)):
            parsed = float(value)
        elif isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError as exc:
                raise ValueError(f"float column {column} received nonnumeric text") from exc
        else:
            raise ValueError(f"float column {column} received {type(value).__name__}")
        return (None, True) if math.isnan(parsed) else (parsed, False)
    raise ValueError(f"unknown column kind {kind} for {column}")


def _validate_normalized_rows(
    source_rows: Sequence[Mapping[str, Any]],
    normalized_rows: Sequence[Mapping[str, Any]],
    schema_report: Mapping[str, Any],
) -> None:
    if len(source_rows) != len(normalized_rows):
        raise ValueError("normalised row count differs from enriched row count")
    before_keys = [(row.get("symbol"), row.get("rebalance_date")) for row in source_rows]
    after_keys = [(row.get("symbol"), row.get("rebalance_date")) for row in normalized_rows]
    if before_keys != after_keys:
        raise ValueError("symbol/date keys changed during normalisation")
    if int(schema_report.get("duplicate_symbol_date_keys", 0) or 0):
        raise ValueError("duplicate symbol/date keys introduced during normalisation")
    before_columns = set(name for row in source_rows for name in row)
    after_columns = set(name for row in normalized_rows for name in row)
    if before_columns - after_columns:
        raise ValueError(f"fields dropped during normalisation: {sorted(before_columns - after_columns)}")
    for before, after in zip(source_rows, normalized_rows):
        old = before.get("actual_forward_return_10d")
        new = after.get("actual_forward_return_10d")
        if old in (None, "") and new is None:
            continue
        if old not in (None, "") and abs(float(old) - float(new)) > 1e-12:
            raise ValueError("actual_forward_return_10d changed during normalisation")


def _schema_failure_payload(
    symbol: str,
    exc: BaseException,
    inventory: Mapping[str, Any],
    *,
    phase: str,
    source_spine_path: str | None = None,
    source_base_partition_path: str | None = None,
    monolithic_base_read: bool | None = None,
    base_partition_reused: bool | None = None,
    source_rows_read: int | None = None,
    price_history_rows_read: int | None = None,
    timings: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
        "traceback": traceback.format_exc(),
        "column_type_inventory": inventory,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "source_rows_read": source_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": dict(timings or {}),
    }


def _partition_failure_payload(
    symbol: str,
    exc: BaseException,
    *,
    phase: str,
    source_spine_path: str | None,
    source_base_partition_path: str | None,
    monolithic_base_read: bool,
    base_partition_reused: bool,
    source_rows_read: int,
    price_history_rows_read: int,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}",
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_spine_path": source_spine_path,
        "source_base_partition_path": source_base_partition_path,
        "monolithic_base_read": monolithic_base_read,
        "base_partition_reused": base_partition_reused,
        "source_rows_read": source_rows_read,
        "price_history_rows_read": price_history_rows_read,
        "phase_timings": dict(timings),
    }


def _duplicate_symbol_date_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [(row.get("symbol"), row.get("rebalance_date")) for row in rows]
    return len(keys) - len(set(keys))


def _column_kind(column: str) -> str:
    if column in BOOL_COLUMNS:
        return "bool"
    if column in INT_COLUMNS:
        return "int"
    if column in NUMERIC_COLUMNS or column.startswith("actual_") or column.startswith("predicted_"):
        return "float"
    return "string"


def _arrow_type_for_column(column: str) -> pa.DataType:
    kind = _column_kind(column)
    if kind == "bool":
        return pa.bool_()
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    return pa.string()


def _schema_for_fieldnames(fieldnames: Sequence[str]) -> pa.Schema:
    return pa.schema([pa.field(name, _arrow_type_for_column(name)) for name in fieldnames])


def _schema_for_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> pa.Schema:
    fields = []
    sample = rows[: min(len(rows), 100_000)]
    for name in fieldnames:
        value_type = pa.string()
        for row in sample:
            value = row.get(name)
            if value in (None, ""):
                continue
            if isinstance(value, bool):
                value_type = pa.bool_()
            elif isinstance(value, int) and not isinstance(value, bool):
                value_type = pa.int64()
            elif isinstance(value, float):
                value_type = pa.float64()
            else:
                value_type = pa.string()
            break
        fields.append(pa.field(name, value_type))
    return pa.schema(fields)


def _column_values(path: Path, column: str) -> list[Any]:
    try:
        return pq.read_table(path, columns=[column]).column(column).to_pylist()
    except Exception:
        return []


def _schema_fingerprint(schema: pa.Schema) -> str:
    payload = "|".join(f"{field.name}:{field.type}" for field in schema)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _completed_symbols(manifest_root: Path) -> set[str]:
    completed = set()
    for path in manifest_root.glob("*.json"):
        payload = _read_json(path)
        if payload.get("status") == "COMPLETE" and payload.get("symbol"):
            completed.add(str(payload["symbol"]).upper())
    return completed


def _failure_record(symbol: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, PartitionBuildError):
        return dict(exc.payload)
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    signature = f"{type(exc).__name__}: {first_line}"
    return {
        "symbol": symbol,
        "phase": "unknown",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_signature": signature,
        "traceback": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _fail_fast_settings(ml: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(ml.get("canonical_v2_alpha_fail_fast", {}) or {})
    return {
        "minimum_failures_before_abort": int(raw.get("minimum_failures_before_abort", 3)),
        "abort_when_completed_is_zero": bool(raw.get("abort_when_completed_is_zero", True)),
        "same_failure_signature_threshold": int(raw.get("same_failure_signature_threshold", 3)),
        "maximum_failure_ratio": float(raw.get("maximum_failure_ratio", 0.25)),
    }


def _should_abort_fail_fast(
    failed: Sequence[Mapping[str, Any]],
    *,
    completed: int,
    settings: Mapping[str, Any],
) -> tuple[bool, str, str]:
    if len(failed) < int(settings["minimum_failures_before_abort"]):
        return False, "", ""
    counts: dict[str, int] = {}
    for row in failed:
        signature = str(row.get("failure_signature") or row.get("signature", ""))
        counts[signature] = counts.get(signature, 0) + 1
    dominant, count = max(counts.items(), key=lambda item: item[1])
    if (
        bool(settings["abort_when_completed_is_zero"])
        and completed == 0
        and count >= int(settings["same_failure_signature_threshold"])
    ):
        return True, f"0 completed partitions and {count} identical failures", dominant
    return False, "", dominant


def _progress(
    report_root: Path,
    planned: int,
    completed: int,
    failed: int,
    rows: int,
    started: float,
    *,
    aborted_early: bool = False,
    abort_reason: str = "",
    dominant_failure_signature: str = "",
    tasks_cancelled: int = 0,
) -> None:
    elapsed = max(0.001, time.perf_counter() - started)
    payload = {
        "planned_partitions": planned,
        "completed_partitions": completed,
        "pending_partitions": max(0, planned - completed - failed),
        "failed_partitions": failed,
        "elapsed_seconds": elapsed,
        "rows_processed": rows,
        "rows_per_second": rows / elapsed,
        "aborted_early": aborted_early,
        "abort_reason": abort_reason,
        "dominant_failure_signature": dominant_failure_signature,
        "tasks_cancelled": tasks_cancelled,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(report_root / "progress_manifest.json", payload)
    print(
        "[canonical-v2-alpha] "
        f"completed={completed}/{planned} pending={payload['pending_partitions']} "
        f"failed={failed} rows={rows} elapsed={elapsed:.1f}s rps={payload['rows_per_second']:.1f}",
        flush=True,
    )


def _manifest_summary(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "manifest_path": str(path),
        "status": payload.get("status"),
        "path": payload.get("path"),
        "row_count": payload.get("row_count"),
        "symbol_count": payload.get("symbol_count"),
        "partition_count": payload.get("partition_count"),
        "date_min": payload.get("date_min"),
        "date_max": payload.get("date_max"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan
