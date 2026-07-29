from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.research.ml.stock_level.prediction_artifacts.math import (
    _trailing_drawdown,
    _trailing_liquidity_score,
    _trailing_return,
    _trailing_volatility,
)
from core.research.ml.stock_level_benchmark_data import (
    SELECTOR_ROW_ID_CONTRACT_VERSION,
    _stable_selector_row_id,
)
from core.research.ml.dataset_build_manifest import (
    build_dataset_build_manifest,
    code_version_hash,
    configuration_hash as lineage_configuration_hash,
    manifest_hash as dataset_build_manifest_hash,
    write_manifest,
)
from core.research.ml.selector_dataset_lineage import logical_manifest_checksum
from infrastructure.data.calendar_authority import calendar_authority_identity

SELECTOR_DATASET_CONTRACT_VERSION = "canonical_v2_selector_dataset_v1"
SELECTOR_DATASET_MANIFEST_VERSION = "authoritative_frozen_selector_dataset_v2"
FROZEN_SELECTOR_DATASET_BUILD_TYPE = "frozen_selector_dataset"
FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_COMMAND = "build-canonical-v2-selector-dataset"
FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_MODULE = (
    "core.research.ml.stock_level.selector_dataset:build_frozen_selector_dataset"
)
BASELINE_CONTRACT_VERSION = "stock_selector_trailing_signals_v1"
DETERMINISTIC_SIGNAL_COLUMNS = (
    "predicted_momentum_20d",
    "predicted_momentum_60d",
    "predicted_momentum_120d",
    "predicted_volatility_20d",
    "predicted_drawdown_60d",
    "predicted_liquidity_score",
    "predicted_risk_adjusted_momentum",
)
BASELINE_CANDIDATES = {
    "momentum_120d": "predicted_momentum_120d",
    "risk_adjusted_momentum": "predicted_risk_adjusted_momentum",
}


@dataclass(frozen=True)
class SelectorDatasetPaths:
    root: Path
    rows: Path
    baseline_scores: Path
    manifest: Path
    quality_report: Path


def read_selector_dataset_rows(root: Path) -> list[dict[str, Any]]:
    """Join immutable source rows to deterministic signals by stable row identity."""
    import pyarrow.parquet as pq

    rows = pq.read_table(root / "rows.parquet").to_pylist()
    scores = {
        str(row["row_id"]): row
        for row in pq.read_table(root / "baseline_scores.parquet").to_pylist()
    }
    output = []
    for row in rows:
        row_id = _stable_selector_row_id(str(row["asset_id"]), str(row["decision_timestamp"]))
        score = scores.get(row_id)
        if score is None:
            raise RuntimeError(f"Missing deterministic selector signals for row_id={row_id}")
        output.append({**row, **{name: score[name] for name in DETERMINISTIC_SIGNAL_COLUMNS}})
    if len(output) != len(scores):
        raise RuntimeError("Selector rows and baseline scores do not have identical populations")
    return output


def deterministic_baseline_scores(
    *, asset_id: str, decision_timestamp: str, decision_date: str,
    close_dates: list[str], close_values: list[float],
    dollar_volume_dates: list[str], dollar_volume_values: list[float],
) -> dict[str, Any]:
    m20 = _trailing_return(close_dates, close_values, decision_date, lookback=20)
    m60 = _trailing_return(close_dates, close_values, decision_date, lookback=60)
    m120 = _trailing_return(close_dates, close_values, decision_date, lookback=120)
    vol20 = _trailing_volatility(close_dates, close_values, decision_date, lookback=20)
    dd60 = _trailing_drawdown(close_dates, close_values, decision_date, lookback=60)
    liquidity = _trailing_liquidity_score(
        dollar_volume_dates, dollar_volume_values, decision_date, lookback=63
    )
    risk = max(abs(float(vol20 or 0.0)), abs(float(dd60 or 0.0)), 1e-6)
    return {
        "row_id": _stable_selector_row_id(asset_id, decision_timestamp),
        "asset_id": asset_id,
        "decision_timestamp": decision_timestamp,
        "baseline_contract_version": BASELINE_CONTRACT_VERSION,
        "predicted_momentum_20d": _nullable(m20),
        "predicted_momentum_60d": _nullable(m60),
        "predicted_momentum_120d": _nullable(m120),
        "predicted_volatility_20d": _nullable(vol20),
        "predicted_drawdown_60d": _nullable(dd60),
        "predicted_liquidity_score": _nullable(liquidity),
        "predicted_risk_adjusted_momentum": (
            None if m60 == "" else float(m60) / risk
        ),
    }


def build_frozen_selector_dataset(
    source_path: Path, market_root: Path, output_root: Path,
    *, symbols: Iterable[str] | None = None, decision_dates: Iterable[str] | None = None,
    copy_source_rows: bool = True, source_sha256: str | None = None,
    config_hash: str | None = None, daily_spine_manifest_path: Path | None = None,
    daily_feature_manifest_path: Path | None = None,
    symbol_registry_manifest_path: Path | None = None,
    base_artifact_path: Path | None = None,
    base_manifest_path: Path | None = None,
    enriched_manifest_path: Path | None = None,
    source_control: dict[str, Any] | None = None,
) -> SelectorDatasetPaths:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if daily_spine_manifest_path is None or daily_feature_manifest_path is None or symbol_registry_manifest_path is None:
        raise ValueError("Authoritative daily-spine, daily-feature, and symbol-registry parent manifests are required")
    if (
        base_artifact_path is None
        or base_manifest_path is None
        or enriched_manifest_path is None
    ):
        raise ValueError(
            "Frozen selector publication requires explicit base artifact, "
            "base manifest, and enriched manifest"
        )
    parents = _validate_parent_manifests(source_path, daily_spine_manifest_path, daily_feature_manifest_path, symbol_registry_manifest_path)
    final_root = output_root
    if final_root.exists():
        raise FileExistsError(f"Frozen selector dataset already exists: {final_root}")
    output_root = final_root.with_name(f".{final_root.name}.{uuid.uuid4().hex}.tmp")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_symbols = sorted({str(x).upper() for x in symbols or ()})
    selected_dates = sorted({str(x) for x in decision_dates or ()})
    bounded = bool(selected_symbols or selected_dates)
    rows_path = output_root / "rows.parquet"
    baseline_path = output_root / "baseline_scores.parquet"
    source_dataset = ds.dataset(source_path, format="parquet")
    source_schema_names = set(source_dataset.schema.names)
    _validate_target_resolution_schema(
        source_schema_names,
        source_control=source_control,
    )
    identity_columns = [
        "symbol",
        "asset_id",
        "decision_timestamp",
        "decision_session_date",
    ]
    if "selector_eligible" in source_schema_names:
        identity_columns.append("selector_eligible")
    filt = None
    if selected_symbols:
        filt = ds.field("symbol").isin(selected_symbols)
    if selected_dates:
        date_filter = ds.field("decision_session_date").isin(selected_dates)
        filt = date_filter if filt is None else filt & date_filter
    source_table = source_dataset.to_table(filter=filt) if bounded else None
    finite_counts = {name: 0 for name in DETERMINISTIC_SIGNAL_COLUMNS}
    row_id_digests: set[bytes] = set()
    baseline_count = 0
    if bounded:
        source_table = source_table.append_column(
            "row_id",
            pa.array([
                _stable_selector_row_id(str(asset), str(timestamp))
                for asset, timestamp in zip(
                    source_table["asset_id"].to_pylist(),
                    source_table["decision_timestamp"].to_pylist(),
                )
            ]),
        )
        pq.write_table(source_table, rows_path, compression="zstd")
        identity = source_table.select(identity_columns).to_pylist()
        score_rows = _score_identity_rows(identity, market_root, ds, market_cache={})
        pq.write_table(pa.Table.from_pylist(score_rows), baseline_path, compression="zstd")
        for row in score_rows:
            row_id_digests.add(bytes.fromhex(row["row_id"]))
            for name in DETERMINISTIC_SIGNAL_COLUMNS:
                finite_counts[name] += row[name] is not None
        baseline_count = len(score_rows)
    else:
        identity = None
        source_file = pq.ParquetFile(source_path)
        writer = None
        rows_writer = None
        market_cache: dict[str, list[dict[str, Any]]] = {}
        try:
            for index in range(source_file.num_row_groups):
                source_group = source_file.read_row_group(index)
                row_ids = [
                    _stable_selector_row_id(str(asset), str(timestamp))
                    for asset, timestamp in zip(
                        source_group["asset_id"].to_pylist(),
                        source_group["decision_timestamp"].to_pylist(),
                    )
                ]
                derived_group = source_group.append_column("row_id", pa.array(row_ids))
                if rows_writer is None:
                    rows_writer = pq.ParquetWriter(rows_path, derived_group.schema, compression="zstd")
                rows_writer.write_table(derived_group)
                identity_rows = source_group.select(identity_columns).to_pylist()
                score_rows = _score_identity_rows(
                    identity_rows,
                    market_root,
                    ds,
                    market_cache=market_cache,
                )
                table = pa.Table.from_pylist(score_rows)
                if writer is None:
                    writer = pq.ParquetWriter(baseline_path, table.schema, compression="zstd")
                writer.write_table(table)
                baseline_count += len(score_rows)
                for row in score_rows:
                    row_id_digests.add(bytes.fromhex(row["row_id"]))
                    for name in DETERMINISTIC_SIGNAL_COLUMNS:
                        finite_counts[name] += row[name] is not None
        finally:
            if writer is not None:
                writer.close()
            if rows_writer is not None:
                rows_writer.close()
    source_count = source_dataset.count_rows()
    derivative_count = source_table.num_rows if bounded else baseline_count
    source_digest = source_sha256 or _sha256(source_path)
    feature_schema = {
        "contract_version": SELECTOR_DATASET_CONTRACT_VERSION,
        "deterministic_signal_columns": list(DETERMINISTIC_SIGNAL_COLUMNS),
        "availability_rule": "all price/volume observations have session_date < decision_session_date",
        "fitted_meta_features": [],
        "missingness_policy": "fail closed for model input; warmup nulls permitted only before eligibility",
    }
    target_columns = [name for name in source_dataset.schema.names if name.startswith("actual_")]
    target_resolution_columns = [
        name for name in (
            "target_resolution_classification",
            "target_resolution_reason",
            "target_is_mature",
            "target_is_realised",
            "target_is_trainable",
            "target_expected_end_timestamp",
            "target_observed_end_timestamp",
            "target_source_cutoff",
            "target_contract_version",
            "target_resolution_policy_version",
        )
        if name in source_schema_names
    ]
    target_resolution_population = _target_resolution_population(rows_path)
    target_schema = {
        "target_columns": target_columns,
        "target_resolution_columns": target_resolution_columns,
        "primary_target": "actual_forward_return_10d",
        "economic_target_id": "forward_return_10d",
        "target_provenance_contract_version": "stock_level_target_provenance_v2",
        "target_registry_schema_version": "selector_target_identity.v1",
        "training_eligibility_rule": (
            "target_is_trainable == true"
            if "target_is_trainable" in source_schema_names
            else "legacy target_status == realized"
        ),
    }
    candidate_schema = {
        "fitted_models": ["ridge", "elastic_net", "random_forest", "gradient_boosting", "dlinear", "patchtst", "transformer", "itransformer", "momentum_transformer", "multitask_transformer", "market_context_encoder", "news_analysis_transformer", "temporal_fusion_transformer"],
        "non_ml_baselines": BASELINE_CANDIDATES,
    }
    _write_json(output_root / "feature_schema.json", feature_schema)
    _write_json(output_root / "target_schema.json", target_schema)
    _write_json(output_root / "candidate_schema.json", candidate_schema)
    checksums = {
        "rows.parquet": _sha256(rows_path),
        "baseline_scores.parquet": _sha256(baseline_path),
        "feature_schema.json": _sha256(output_root / "feature_schema.json"),
        "target_schema.json": _sha256(output_root / "target_schema.json"),
        "candidate_schema.json": _sha256(output_root / "candidate_schema.json"),
    }
    quality = {
        "source_row_count": source_count, "derivative_row_count": derivative_count,
        "row_count_preserved": derivative_count == source_count if not bounded else True,
        "baseline_row_count": baseline_count, "unique_row_ids": len(row_id_digests),
        "row_id_collisions": baseline_count - len(row_id_digests),
        "baseline_finite_counts": finite_counts,
        "target_resolution_population": target_resolution_population,
        "bounded": bounded,
    }
    population = _dataset_population_identity(rows_path)
    _validate_rows_against_parents(rows_path, parents)
    from core.research.ml.registries import RegistryResolver, load_registry_bundle

    target = RegistryResolver(load_registry_bundle()).resolve(
        "target_contracts", "forward_return_10d", role="selector"
    )
    target_contract_version = frozen_selector_target_contract_version(
        target.canonical_id,
        target.entry.entry_hash,
        "stock_level_target_provenance_v2",
    )
    dataset_build_config_hash = frozen_selector_configuration_hash(
        source_digest=source_digest,
        config_hash=config_hash,
        parents=parents,
        selected_symbols=selected_symbols,
        selected_dates=selected_dates,
        copy_source_rows=copy_source_rows,
    )
    from core.research.ml.stock_level.selector_lineage import (
        preflight_frozen_selector_dataset,
    )

    base_rows = pq.read_table(base_artifact_path).to_pylist()
    enriched_rows = pq.read_table(rows_path).to_pylist()
    frozen_preflight = preflight_frozen_selector_dataset(
        daily_spine_manifest=json.loads(
            daily_spine_manifest_path.read_text(encoding="utf-8")
        ),
        base_manifest=json.loads(base_manifest_path.read_text(encoding="utf-8")),
        enriched_manifest=json.loads(
            enriched_manifest_path.read_text(encoding="utf-8")
        ),
        base_rows=base_rows,
        enriched_rows=enriched_rows,
        feature_columns=DETERMINISTIC_SIGNAL_COLUMNS,
    )
    if frozen_preflight["status"] != "READY":
        raise ValueError(
            f"Frozen selector preflight blocked: {frozen_preflight['blockers']}"
        )
    manifest = {
        "manifest_schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "frozen_dataset_version": "v2",
        "dataset_id": SELECTOR_DATASET_CONTRACT_VERSION + ("_bounded" if bounded else ""),
        "dataset_path": str(final_root / "rows.parquet"), "dataset_checksum": checksums["rows.parquet"],
        "row_population_checksum": population["row_population_checksum"], "row_count": population["row_count"],
        "date_coverage": population["date_coverage"], "symbol_count": population["symbol_count"],
        "source_path": str(source_path), "source_sha256": source_digest,
        "source_row_count": source_count, "source_symbol_count": 406,
        "row_id_contract": SELECTOR_ROW_ID_CONTRACT_VERSION,
        "feature_contract": SELECTOR_DATASET_CONTRACT_VERSION,
        "baseline_contract": BASELINE_CONTRACT_VERSION,
        "economic_target_id": target.canonical_id,
        "target_contract": target.canonical_id,
        "target_provenance_contract_version": (
            "stock_level_target_provenance_v2"
        ),
        "target_registry_schema_version": "selector_target_identity.v1",
        "target_registry_entry_checksum": target.entry.entry_hash,
        "target_contract_checksum": target.entry.entry_hash,
        "target_contract_version": target_contract_version,
        "target_resolution_population": target_resolution_population,
        "target_training_eligibility_rule": (
            "target_is_trainable == true"
            if "target_is_trainable" in source_schema_names
            else "legacy target_status == realized"
        ),
        "ranking_contract": "daily_cross_sectional_ranking_problem_v1",
        "daily_stock_spine_identity": parents["daily_spine_identity"],
        "daily_stock_spine_version": parents["daily_spine_version"],
        "daily_stock_spine_checksum": parents["daily_spine_checksum"],
        "daily_feature_store_identity": parents["daily_feature_identity"],
        "daily_feature_store_version": parents["daily_feature_version"],
        "daily_feature_store_checksum": parents["daily_feature_checksum"],
        "symbol_registry_identity": parents["symbol_registry_identity"],
        "symbol_registry_version": parents["symbol_registry_version"],
        "symbol_registry_checksum": parents["symbol_registry_checksum"],
        "parent_manifests": parents["parent_manifests"],
        "source_price_artifact_identities": parents["source_price_artifact_identities"],
        "point_in_time_feature_store_identities": parents["point_in_time_feature_store_identities"],
        "builder_identity": "core.research.ml.stock_level.selector_dataset:build_frozen_selector_dataset",
        "builder_run_identity": canonical_dataset_run_identity(source_digest, config_hash, parents),
        "git_commit": _git_commit(), "config_hash": config_hash,
        "dataset_build_configuration_hash": dataset_build_config_hash,
        "checksums": checksums,
        "feature_schema_checksum": checksums["feature_schema.json"],
        "target_schema_checksum": checksums["target_schema.json"],
        "frozen_preflight": frozen_preflight,
        "bounded_symbols": selected_symbols, "bounded_decision_dates": selected_dates,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(), "publication_status": "complete", "validation_status": "VERIFIED",
    }
    manifest["logical_checksum"] = logical_manifest_checksum(manifest)
    _write_json(output_root / "quality_report.json", quality)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "checksums.json", checksums)
    generic_manifest = frozen_selector_dataset_build_manifest(
        selector_manifest=manifest,
        source_path=source_path,
        output_root=output_root,
        final_root=final_root,
        rows=enriched_rows,
        parents=parents,
        source_manifest_paths=(
            daily_spine_manifest_path,
            daily_feature_manifest_path,
            symbol_registry_manifest_path,
            base_manifest_path,
            enriched_manifest_path,
        ),
        source_artifact_paths=(base_artifact_path,),
        target_contract_version=target_contract_version,
        target_registry_entry_checksum=target.entry.entry_hash,
        source_control=source_control,
    )
    write_manifest(output_root / "rows.parquet.manifest.json", generic_manifest)
    os.replace(output_root, final_root)
    return SelectorDatasetPaths(final_root, final_root / "rows.parquet", final_root / "baseline_scores.parquet", final_root / "manifest.json", final_root / "quality_report.json")


def frozen_selector_dataset_build_manifest(
    *,
    selector_manifest: Mapping[str, Any],
    source_path: Path,
    output_root: Path,
    final_root: Path,
    rows: Sequence[Mapping[str, Any]],
    parents: Mapping[str, Any],
    source_manifest_paths: Sequence[Path | None],
    source_artifact_paths: Sequence[Path | None] = (),
    target_contract_version: str | None = None,
    target_registry_entry_checksum: str | None = None,
    source_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_paths = (
        output_root / "rows.parquet",
        output_root / "baseline_scores.parquet",
        output_root / "feature_schema.json",
        output_root / "target_schema.json",
        output_root / "candidate_schema.json",
        output_root / "manifest.json",
        output_root / "quality_report.json",
        output_root / "checksums.json",
    )
    source_paths = tuple(
        path for path in (source_path, *source_artifact_paths)
        if path is not None
    )
    source_manifests = tuple(path for path in source_manifest_paths if path is not None)
    source_content_hashes = frozen_selector_source_content_hashes(
        selector_manifest=selector_manifest,
        parents=parents,
        source_artifact_paths=source_artifact_paths,
        target_registry_entry_checksum=target_registry_entry_checksum,
    )
    generic = build_dataset_build_manifest(
        dataset_id=str(selector_manifest["dataset_id"]),
        dataset_type=FROZEN_SELECTOR_DATASET_BUILD_TYPE,
        schema_version=SELECTOR_DATASET_MANIFEST_VERSION,
        producer_command=FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_COMMAND,
        producer_module=FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_MODULE,
        output_paths=output_paths,
        source_paths=source_paths,
        source_dataset_ids=frozen_selector_source_dataset_ids(selector_manifest, parents),
        source_manifest_paths=source_manifests,
        source_content_hashes=source_content_hashes,
        canonical_price_authority_version=frozen_selector_canonical_price_authority_version(parents),
        universe_authority_version=frozen_selector_universe_authority_version(parents),
        identity_authority_version=frozen_selector_identity_authority_version(parents),
        corporate_action_authority_version=frozen_selector_corporate_action_authority_version(parents),
        market_calendar_authority_version=frozen_selector_market_calendar_authority_version(rows),
        market_calendar_authority=frozen_selector_market_calendar_authority(rows),
        target_contract_version=target_contract_version
        or str(selector_manifest.get("target_contract_version") or ""),
        feature_code_version=frozen_selector_feature_code_version(),
        label_code_version=frozen_selector_label_code_version(),
        configuration_hash_value=str(
            selector_manifest.get("dataset_build_configuration_hash")
            or selector_manifest.get("config_hash")
            or ""
        ),
        rows=rows,
        key_fields=("row_id",),
        partition_information={
            "format": "parquet",
            "partitioned": False,
            "dataset_role": "promotion_grade_frozen_selector_input",
            "sidecars": [
                "baseline_scores.parquet",
                "feature_schema.json",
                "target_schema.json",
                "candidate_schema.json",
            ],
        },
        parent_artifact_ids=frozen_selector_parent_artifact_ids(selector_manifest, parents),
        source_control=source_control,
    )
    _rewrite_manifest_output_paths(generic, output_root=output_root, final_root=final_root)
    return generic


def frozen_selector_dataset_lineage_expectation(
    selector_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parents = {
        "daily_spine_identity": selector_manifest.get("daily_stock_spine_identity"),
        "daily_spine_version": selector_manifest.get("daily_stock_spine_version"),
        "daily_spine_checksum": selector_manifest.get("daily_stock_spine_checksum"),
        "daily_feature_identity": selector_manifest.get("daily_feature_store_identity"),
        "daily_feature_version": selector_manifest.get("daily_feature_store_version"),
        "daily_feature_checksum": selector_manifest.get("daily_feature_store_checksum"),
        "symbol_registry_identity": selector_manifest.get("symbol_registry_identity"),
        "symbol_registry_version": selector_manifest.get("symbol_registry_version"),
        "symbol_registry_checksum": selector_manifest.get("symbol_registry_checksum"),
        "source_price_artifact_identities": selector_manifest.get("source_price_artifact_identities", []),
    }
    return {
        "dataset_id": selector_manifest.get("dataset_id"),
        "dataset_type": FROZEN_SELECTOR_DATASET_BUILD_TYPE,
        "schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "producer_command": FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_COMMAND,
        "producer_module": FROZEN_SELECTOR_DATASET_BUILD_PRODUCER_MODULE,
        "canonical_price_authority_version": frozen_selector_canonical_price_authority_version(parents),
        "universe_authority_version": frozen_selector_universe_authority_version(parents),
        "identity_authority_version": frozen_selector_identity_authority_version(parents),
        "corporate_action_authority_version": frozen_selector_corporate_action_authority_version(parents),
        "market_calendar_authority_version": str(
            selector_manifest.get("market_calendar_authority_version")
            or selector_manifest.get("exchange_calendar_identity")
            or ""
        )
        or None,
        "target_contract_version": selector_manifest.get("target_contract_version")
        or frozen_selector_target_contract_version(
            str(selector_manifest.get("target_contract") or selector_manifest.get("economic_target_id") or ""),
            str(selector_manifest.get("target_contract_checksum") or selector_manifest.get("target_registry_entry_checksum") or ""),
            str(selector_manifest.get("target_provenance_contract_version") or ""),
        ),
        "feature_code_version": frozen_selector_feature_code_version(),
        "label_code_version": frozen_selector_label_code_version(),
        "configuration_hash": selector_manifest.get("dataset_build_configuration_hash")
        or selector_manifest.get("config_hash"),
        "source_content_hashes": frozen_selector_source_content_hashes(
            selector_manifest=selector_manifest,
            parents=parents,
            source_artifact_paths=(),
            target_registry_entry_checksum=str(
                selector_manifest.get("target_contract_checksum")
                or selector_manifest.get("target_registry_entry_checksum")
                or ""
            ),
        ),
    }


def frozen_selector_configuration_hash(
    *,
    source_digest: str,
    config_hash: str | None,
    parents: Mapping[str, Any],
    selected_symbols: Sequence[str],
    selected_dates: Sequence[str],
    copy_source_rows: bool,
) -> str:
    if config_hash:
        return str(config_hash)
    payload = {
        "dataset_type": FROZEN_SELECTOR_DATASET_BUILD_TYPE,
        "schema_version": SELECTOR_DATASET_MANIFEST_VERSION,
        "source_digest": source_digest,
        "selected_symbols": list(selected_symbols),
        "selected_dates": list(selected_dates),
        "copy_source_rows": bool(copy_source_rows),
        "daily_spine_identity": parents.get("daily_spine_identity"),
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
        "daily_feature_identity": parents.get("daily_feature_identity"),
        "daily_feature_checksum": parents.get("daily_feature_checksum"),
        "symbol_registry_identity": parents.get("symbol_registry_identity"),
        "symbol_registry_checksum": parents.get("symbol_registry_checksum"),
    }
    return lineage_configuration_hash(payload)


def frozen_selector_source_content_hashes(
    *,
    selector_manifest: Mapping[str, Any],
    parents: Mapping[str, Any],
    source_artifact_paths: Sequence[Path | None],
    target_registry_entry_checksum: str | None,
) -> dict[str, str]:
    checksums = dict(selector_manifest.get("checksums") or {})
    hashes = {
        "source_rows": str(selector_manifest.get("source_sha256") or ""),
        "rows_parquet": str(checksums.get("rows.parquet") or ""),
        "baseline_scores_parquet": str(checksums.get("baseline_scores.parquet") or ""),
        "feature_schema": str(selector_manifest.get("feature_schema_checksum") or ""),
        "target_schema": str(selector_manifest.get("target_schema_checksum") or ""),
        "row_population": str(selector_manifest.get("row_population_checksum") or ""),
        "daily_spine_manifest": str(parents.get("daily_spine_checksum") or ""),
        "daily_feature_manifest": str(parents.get("daily_feature_checksum") or ""),
        "symbol_registry_manifest": str(parents.get("symbol_registry_checksum") or ""),
        "target_registry_entry": str(target_registry_entry_checksum or ""),
    }
    for index, path in enumerate(path for path in source_artifact_paths if path is not None):
        if path.exists() and path.is_file():
            hashes[f"source_artifact_{index}"] = _sha256(path)
    return {
        key: value
        for key, value in hashes.items()
        if value
    }


def frozen_selector_source_dataset_ids(
    selector_manifest: Mapping[str, Any],
    parents: Mapping[str, Any],
) -> tuple[str, ...]:
    values = [
        selector_manifest.get("source_path"),
        parents.get("daily_spine_identity"),
        parents.get("daily_feature_identity"),
        parents.get("symbol_registry_identity"),
        selector_manifest.get("target_contract")
        or selector_manifest.get("economic_target_id"),
    ]
    return tuple(str(value) for value in values if value)


def frozen_selector_parent_artifact_ids(
    selector_manifest: Mapping[str, Any],
    parents: Mapping[str, Any],
) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("source_price_artifact_identities", "point_in_time_feature_store_identities"):
        for value in selector_manifest.get(key, ()) or parents.get(key, ()) or ():
            values.append(str(value))
    return tuple(sorted(set(values)))


def frozen_selector_canonical_price_authority_version(parents: Mapping[str, Any]) -> str:
    return lineage_configuration_hash({
        "authority": "canonical_price_authority_v1",
        "daily_spine_identity": parents.get("daily_spine_identity"),
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
        "source_price_artifact_identities": parents.get("source_price_artifact_identities", []),
    })


def frozen_selector_universe_authority_version(parents: Mapping[str, Any]) -> str:
    return lineage_configuration_hash({
        "authority": "frozen_selector_universe_authority_v1",
        "daily_spine_identity": parents.get("daily_spine_identity"),
        "daily_spine_version": parents.get("daily_spine_version"),
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
    })


def frozen_selector_identity_authority_version(parents: Mapping[str, Any]) -> str:
    return lineage_configuration_hash({
        "authority": "frozen_selector_identity_authority_v1",
        "symbol_registry_identity": parents.get("symbol_registry_identity"),
        "symbol_registry_version": parents.get("symbol_registry_version"),
        "symbol_registry_checksum": parents.get("symbol_registry_checksum"),
    })


def frozen_selector_corporate_action_authority_version(parents: Mapping[str, Any]) -> str:
    return lineage_configuration_hash({
        "authority": "frozen_selector_corporate_action_authority_v1",
        "daily_spine_identity": parents.get("daily_spine_identity"),
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
        "source_price_artifact_identities": parents.get("source_price_artifact_identities", []),
    })


def frozen_selector_market_calendar_authority(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dates = sorted(
        str(row.get("decision_session_date") or row.get("session_date") or "")
        for row in rows
        if row.get("decision_session_date") or row.get("session_date")
    )
    if not dates:
        return calendar_authority_identity()
    return calendar_authority_identity(start=dates[0], end=dates[-1])


def frozen_selector_market_calendar_authority_version(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return str(frozen_selector_market_calendar_authority(rows).get("version") or "")


def frozen_selector_target_contract_version(
    target_id: str,
    target_checksum: str,
    target_provenance_contract_version: str,
) -> str:
    return lineage_configuration_hash({
        "authority": "frozen_selector_target_contract_authority_v1",
        "economic_target_id": target_id,
        "target_contract_checksum": target_checksum,
        "target_provenance_contract_version": target_provenance_contract_version,
    })


def frozen_selector_feature_code_version() -> str:
    return code_version_hash((
        Path("core/research/ml/stock_level/selector_dataset.py"),
        Path("core/research/ml/stock_level/selector_lineage.py"),
        Path("core/research/ml/stock_level/prediction_artifacts/math.py"),
        Path("core/research/ml/stock_level_benchmark_data.py"),
    ))


def frozen_selector_label_code_version() -> str:
    return code_version_hash((
        Path("core/research/ml/stock_level/selector_lineage.py"),
        Path("core/research/ml/stock_level/prediction_artifacts/targets.py"),
        Path("core/research/ml/registries/target_identity.py"),
    ))


def _rewrite_manifest_output_paths(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    final_root: Path,
) -> None:
    for row in manifest.get("output_hashes", []) or []:
        path = Path(str(row.get("path", "")))
        try:
            relative = path.relative_to(output_root)
        except ValueError:
            continue
        row["path"] = str(final_root / relative)
    manifest["manifest_hash"] = dataset_build_manifest_hash(manifest)


def canonical_dataset_run_identity(source_digest: str, config_hash: str | None, parents: dict[str, Any]) -> str:
    payload = {
        "source_checksum": source_digest,
        "config_hash": config_hash,
        "daily_spine_identity": parents["daily_spine_identity"],
        "daily_spine_checksum": parents.get("daily_spine_checksum"),
        "daily_feature_identity": parents.get("daily_feature_identity"),
        "daily_feature_checksum": parents.get("daily_feature_checksum"),
        "symbol_registry_identity": parents["symbol_registry_identity"],
        "symbol_registry_checksum": parents.get("symbol_registry_checksum"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()


def _validate_parent_manifests(source_path: Path, spine_path: Path, feature_path: Path, registry_path: Path) -> dict[str, Any]:
    spine = json.loads(spine_path.read_text(encoding="utf-8")); feature = json.loads(feature_path.read_text(encoding="utf-8")); registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if spine.get("status") != "READY" or spine.get("dataset_type") != "canonical_daily_stock_spine": raise ValueError("Unknown or unready authoritative daily-spine identity")
    if feature.get("status") != "READY" or feature.get("dataset_type") != "daily_price_features": raise ValueError("Unknown or unready authoritative daily-feature identity")
    if registry.get("status") != "READY" or registry.get("dataset_type") != "canonical_asset_registry_audit": raise ValueError("Unknown or unready authoritative symbol-registry identity")
    source_checksum = _sha256(source_path)
    source_checksums = {str(key): str(value).upper() for key, value in dict(feature.get("source_checksums", {})).items()}
    matching_feature_source = any(Path(key).resolve() == source_path.resolve() and value == source_checksum for key, value in source_checksums.items())
    if not matching_feature_source: raise ValueError("Daily-feature parent source checksum or path mismatch")
    if spine.get("dataset_id") not in set(feature.get("source_dataset_ids", [])): raise ValueError("Daily-feature parent does not reference the authoritative spine")
    spine_file = Path(str(spine.get("spine_artifact_path", "")))
    if not spine_file.exists() or str(spine.get("spine_artifact_checksum", "")).upper() != _sha256(spine_file): raise ValueError("Daily-spine artifact checksum mismatch")
    registry_file = Path(str(registry.get("registry_path", "")))
    if not registry_file.exists() or str(registry.get("registry_content_checksum", "")).upper() != _sha256(registry_file): raise ValueError("Symbol-registry parent checksum mismatch")
    return {"daily_spine_identity": spine["dataset_id"], "daily_spine_version": spine["schema_version"], "daily_spine_checksum": _sha256(spine_path), "spine_path": spine_file, "daily_feature_identity": feature["dataset_id"], "daily_feature_version": feature["schema_version"], "daily_feature_checksum": _sha256(feature_path), "symbol_registry_identity": registry["dataset_id"], "symbol_registry_version": registry["symbol_registry_version"], "symbol_registry_checksum": _sha256(registry_path), "registry_path": registry_file, "parent_manifests": [{"path": str(spine_path), "checksum": _sha256(spine_path)}, {"path": str(feature_path), "checksum": _sha256(feature_path)}, {"path": str(registry_path), "checksum": _sha256(registry_path)}], "source_price_artifact_identities": spine.get("source_price_artifact_identities", []), "point_in_time_feature_store_identities": [feature["dataset_id"]]}


def _dataset_population_identity(rows_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    table = pq.read_table(rows_path, columns=["row_id", "asset_id", "canonical_symbol", "decision_session_date"])
    rows = table.to_pylist(); keys = [(str(row["asset_id"]), str(row["decision_session_date"])) for row in rows]
    if len(keys) != len(set(keys)): raise ValueError("Duplicate stock-date identity in frozen selector dataset")
    canonical_rows = sorted(
        rows,
        key=lambda row: (
            str(row["decision_session_date"]),
            str(row["asset_id"]),
            str(row["row_id"]),
        ),
    )
    if rows != canonical_rows:
        raise ValueError("Noncanonical frozen selector dataset ordering")
    ordered = [str(row["row_id"]) for row in rows]
    dates = sorted({str(row["decision_session_date"]) for row in rows})
    return {"row_count": len(rows), "symbol_count": len({str(row["asset_id"]) for row in rows}), "date_coverage": {"min": dates[0], "max": dates[-1], "count": len(dates)}, "row_population_checksum": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest().upper()}


def _validate_rows_against_parents(rows_path: Path, parents: dict[str, Any]) -> None:
    import csv
    import pyarrow.parquet as pq
    with parents["registry_path"].open("r", encoding="utf-8", newline="") as handle:
        registry = {row["asset_id"]: row["canonical_symbol"] for row in csv.DictReader(handle)}
    table = pq.read_table(rows_path, columns=["asset_id", "canonical_symbol", "decision_session_date"])
    spine = pq.read_table(parents["spine_path"], columns=["asset_id", "session_date"])
    spine_keys = {(str(row["asset_id"]), str(row["session_date"])) for row in spine.to_pylist()}
    for row in table.to_pylist():
        expected = registry.get(str(row["asset_id"]))
        if expected is None: raise ValueError(f"Unresolved selector asset: {row['asset_id']}")
        if expected != str(row["canonical_symbol"]): raise ValueError(f"Ambiguous canonical symbol mapping: {row['asset_id']}")
        if (str(row["asset_id"]), str(row["decision_session_date"])) not in spine_keys: raise ValueError(f"Selector row absent from authoritative daily spine: {row['asset_id']}:{row['decision_session_date']}")


def _target_resolution_population(rows_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(rows_path).schema_arrow
    names = set(schema.names)
    required = {"target_is_trainable", "target_resolution_classification"}
    if not required.issubset(names):
        return {
            "status": "LEGACY_TARGET_STATUS_ONLY",
            "required_fields_present": False,
            "training_eligibility_rule": "legacy target_status == realized",
        }
    table = pq.read_table(
        rows_path,
        columns=["target_is_trainable", "target_resolution_classification"],
    )
    rows = table.to_pylist()
    classification_counts: dict[str, int] = {}
    trainable_rows = 0
    for row in rows:
        classification = str(row.get("target_resolution_classification") or "")
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
        trainable_rows += bool(row.get("target_is_trainable"))
    return {
        "status": "GOVERNED_TARGET_STATUS_PRESENT",
        "required_fields_present": True,
        "row_count": len(rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "trainable_rows": trainable_rows,
        "non_trainable_rows": len(rows) - trainable_rows,
        "training_eligibility_rule": "target_is_trainable == true",
    }


def _validate_target_resolution_schema(
    source_schema_names: set[str],
    *,
    source_control: Mapping[str, Any] | None,
) -> None:
    required = {"target_is_trainable", "target_resolution_classification"}
    if required.issubset(source_schema_names):
        return
    if _legacy_target_status_diagnostic_allowed(source_control):
        return
    missing = sorted(required - source_schema_names)
    raise ValueError(
        "Frozen selector dataset source missing governed target-resolution "
        f"fields: {missing}. Legacy target_status fallback is diagnostic-only."
    )


def _legacy_target_status_diagnostic_allowed(
    source_control: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(source_control, Mapping):
        return False
    values = {
        str(source_control.get("permitted_use") or "").upper(),
        str(source_control.get("intended_use") or "").upper(),
        str(source_control.get("lineage_intended_use") or "").upper(),
        str(source_control.get("dataset_intended_use") or "").upper(),
    }
    return bool({"DIAGNOSTIC", "DIAGNOSTIC_ONLY", "LEGACY_DIAGNOSTIC"} & values)


def _score_identity_rows(
    identity: list[dict[str, Any]],
    market_root: Path,
    ds: Any,
    *,
    market_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in identity:
        by_symbol.setdefault(str(row["symbol"]).upper(), []).append(row)
    output: list[dict[str, Any]] = []
    for symbol in sorted(by_symbol):
        if market_cache is not None and symbol in market_cache:
            bars = market_cache[symbol]
        else:
            bars = ds.dataset(market_root / f"symbol={symbol}", format="parquet", partitioning="hive").to_table(
                columns=["session_date", "model_close", "raw_volume"]
            ).sort_by("session_date").to_pylist()
            if market_cache is not None:
                market_cache[symbol] = bars
        dates = [str(row["session_date"]) for row in bars]
        closes = [float(row["model_close"]) for row in bars]
        dollars = [float(row["model_close"]) * float(row["raw_volume"] or 0.0) for row in bars]
        precomputed = _precomputed_signals(dates, closes, dollars)
        for row in sorted(by_symbol[symbol], key=lambda x: str(x["decision_timestamp"])):
            decision_date = str(row["decision_session_date"])
            values = precomputed.get(decision_date)
            if values is None:
                if not _truthy(row.get("selector_eligible"), default=True):
                    values = {name: None for name in DETERMINISTIC_SIGNAL_COLUMNS}
                else:
                    raise RuntimeError(f"Canonical bar missing for selector decision: {symbol} {decision_date}")
            output.append({
                "row_id": _stable_selector_row_id(str(row["asset_id"]), str(row["decision_timestamp"])),
                "asset_id": str(row["asset_id"]), "decision_timestamp": str(row["decision_timestamp"]),
                "baseline_contract_version": BASELINE_CONTRACT_VERSION, **values,
            })
    return output


def _precomputed_signals(dates: list[str], closes: list[float], dollars: list[float]) -> dict[str, dict[str, float | None]]:
    """Linear/vectorized equivalent of the authoritative strictly-prior formulas."""
    import numpy as np
    import pandas as pd

    close = pd.Series(closes, dtype="float64")
    dollar = pd.Series(dollars, dtype="float64")
    prior = close.shift(1)
    m20 = prior / close.shift(21) - 1.0
    m60 = prior / close.shift(61) - 1.0
    m120 = prior / close.shift(121) - 1.0
    returns = close.pct_change(fill_method=None)
    vol20 = returns.shift(1).rolling(20, min_periods=20).std(ddof=0)
    liquidity = np.log1p(dollar.shift(1).rolling(63, min_periods=1).mean())
    dd60 = np.full(len(close), np.nan)
    if len(close) >= 60:
        windows = np.lib.stride_tricks.sliding_window_view(close.to_numpy(), 60)
        running_peaks = np.maximum.accumulate(windows, axis=1)
        drawdowns = windows / running_peaks - 1.0
        # Window ending i-1 belongs to decision/bar index i.
        dd60[60:] = drawdowns[:-1].min(axis=1) if len(drawdowns) > 1 else np.array([], dtype=float)
    output: dict[str, dict[str, float | None]] = {}
    for index, date in enumerate(dates):
        risk = max(abs(_finite_or_zero(vol20.iloc[index])), abs(_finite_or_zero(dd60[index])), 1e-6)
        output[date] = {
            "predicted_momentum_20d": _finite_or_none(m20.iloc[index]),
            "predicted_momentum_60d": _finite_or_none(m60.iloc[index]),
            "predicted_momentum_120d": _finite_or_none(m120.iloc[index]),
            "predicted_volatility_20d": _finite_or_none(vol20.iloc[index]),
            "predicted_drawdown_60d": _finite_or_none(dd60[index]),
            "predicted_liquidity_score": _finite_or_none(liquidity.iloc[index]),
            "predicted_risk_adjusted_momentum": (
                None if _finite_or_none(m60.iloc[index]) is None else float(m60.iloc[index]) / risk
            ),
        }
    return output


def _finite_or_none(value: Any) -> float | None:
    import math
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _finite_or_zero(value: Any) -> float:
    return _finite_or_none(value) or 0.0


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _nullable(value: float | str) -> float | None:
    return None if value == "" else float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
