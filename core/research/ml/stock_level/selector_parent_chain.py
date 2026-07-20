from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

from core.research.ml.stock_level.selector_lineage import (
    CURRENT_ECONOMIC_TARGET_ID,
    CURRENT_TARGET_PROVENANCE_VERSION,
    PROTECTED_ENRICHMENT_COLUMNS,
    TARGET_COLUMNS,
    TARGET_TIMESTAMP_COLUMNS,
    build_ready_daily_spine_manifest,
    merge_enrichment_preserving_base,
    preflight_frozen_selector_dataset,
    row_id_checksum,
    selector_row_id,
    target_checksum,
    validate_selector_parent_child_lineage,
)

CONTRACT_VERSION = "selector_parent_chain_publication.v1"
APPROVED_BASE_SHA256 = (
    "c2487d7f378121069ea5e92a1d0cf0444f42dfc1da237566d24c650ae8558d38"
)
APPROVED_BASE_LOGICAL_HASH = (
    "c564a0187ef1a32ae7f979c37ab2cc553959871c747922553ef5e7486b42b446"
)
APPROVED_CANONICAL_DAILY_HASH = (
    "c2ab57992c9363c118d854f01da18ea34122b9c0775af3d0676afe5ff80bad56"
)
APPROVED_ASSET_REGISTRY_VERSION = (
    "79d9b6f8ea2937394f40ef826b69000427412c1af6b89bc0576d1a7e0fbaa2ac"
)
APPROVED_ASSET_REGISTRY_CHECKSUM = (
    "057a09bca438203eca9a0863e6f93033f6385c256f7f3caafdecbf4823387d36"
)
PRODUCTION_ROW_COUNT = 836_074
PROTECTED_OUTPUT_FRAGMENTS = (
    "regeneration_canonical_v2/alpha_enrichment",
    "selector_evaluation_1c_e/frozen_selector_dataset_v2",
)


@dataclass(frozen=True)
class ParentChainInputs:
    run_id: str
    output_root: Path
    base_artifact: Path
    base_manifest: Path
    canonical_daily_manifest: Path
    asset_registry_manifest: Path
    feature_schema: Path
    canonical_daily_root: Path | None = None
    labeled_spine_root: Path | None = None
    labeled_spine_manifest: Path | None = None
    inference_spine_manifest: Path | None = None
    fundamentals_artifact: Path | None = None
    fundamentals_manifest: Path | None = None
    data_root: Path | None = None
    workers: int = 6
    max_in_flight_tasks: int = 12
    memory_budget_gib: float = 24.0
    resume: bool = False
    production: bool = False
    config_path: Path | None = None
    path_length_limit: int = 240


def prepare_parent_chain_plan(inputs: ParentChainInputs) -> dict[str, Any]:
    blockers: list[str] = []
    base_identity = _base_identity(inputs.base_artifact, inputs.base_manifest, blockers)
    canonical = _json(inputs.canonical_daily_manifest, blockers, "PARENT_IDENTITY_MISSING")
    registry = _json(inputs.asset_registry_manifest, blockers, "PARENT_IDENTITY_MISSING")
    schema = _json(inputs.feature_schema, blockers, "FEATURE_SCHEMA_UNRESOLVED")
    if _canonical_hash(canonical) != APPROVED_CANONICAL_DAILY_HASH:
        blockers.append("CANONICAL_DAILY_HASH_MISMATCH")
    registry_values = json.dumps(registry, sort_keys=True)
    if (
        APPROVED_ASSET_REGISTRY_VERSION not in registry_values
        or APPROVED_ASSET_REGISTRY_CHECKSUM not in registry_values
    ):
        blockers.append("ASSET_REGISTRY_IDENTITY_MISMATCH")
    if inputs.workers < 1 or inputs.max_in_flight_tasks < inputs.workers:
        blockers.append("COMPUTE_CONFIGURATION_INVALID")
    if inputs.memory_budget_gib <= 0 or inputs.memory_budget_gib > 28:
        blockers.append("MEMORY_BUDGET_INVALID")
    if inputs.production and (
        inputs.canonical_daily_root is None
        or not inputs.canonical_daily_root.is_dir()
    ):
        blockers.append("CANONICAL_DAILY_ROOT_MISSING")
    output = inputs.output_root.resolve()
    normalized = output.as_posix().lower()
    if any(fragment in normalized for fragment in PROTECTED_OUTPUT_FRAGMENTS):
        blockers.append("PROTECTED_OUTPUT_ROOT")
    owner_path = output / "owner.json"
    if owner_path.exists():
        owner = _json(owner_path, blockers, "OUTPUT_ROOT_CONFLICT")
        if owner.get("run_id") != inputs.run_id:
            blockers.append("ACTIVE_OWNER_CONFLICT")
        elif not inputs.resume:
            blockers.append("OUTPUT_ROOT_CONFLICT")
    elif output.exists() and any(output.iterdir()):
        blockers.append("OUTPUT_ROOT_CONFLICT")
    feature_columns = _feature_columns(schema)
    if not feature_columns:
        blockers.append("FEATURE_SCHEMA_UNRESOLVED")
    forbidden = set(TARGET_COLUMNS) | set(TARGET_TIMESTAMP_COLUMNS)
    if any(
        column in forbidden
        or str(column).startswith(("actual_", "target_", "label_"))
        for column in feature_columns
    ):
        blockers.append("TARGET_COLUMN_IN_FEATURE_ALLOWLIST")
    final_schema_assembly = _final_schema_assembly_plan(inputs)
    if inputs.production and final_schema_assembly["unresolved_columns"]:
        blockers.append("FINAL_SCHEMA_PRODUCER_UNRESOLVED")
    try:
        alpha_config = _parent_alpha_config(
            inputs,
            base_identity=base_identity,
            config_path=inputs.config_path,
        )
    except ValueError as exc:
        blockers.append("PRODUCTION_INPUT_PATH_UNRESOLVED")
        alpha_config = _parent_alpha_config(
            ParentChainInputs(**{**inputs.__dict__, "production": False}),
            base_identity=base_identity,
            config_path=inputs.config_path,
        )
        alpha_config.setdefault("ml", {})["path_resolution_error"] = str(exc)
    resolved_inputs = _resolved_input_report(inputs, alpha_config)
    if owner_path.exists() and inputs.resume:
        expected_owner = {
            "contract_version": CONTRACT_VERSION,
            "run_id": inputs.run_id,
            "base_sha256": base_identity.get("sha256"),
            "feature_schema_checksum": _hash_json(schema),
            "adapted_configuration_sha256": _adapted_configuration_hash(alpha_config),
            "resolved_inputs_sha256": _hash_json(
                _partition_producing_resolved_inputs(resolved_inputs)
            ),
        }
        if owner != expected_owner:
            blockers.append("RESOLVED_INPUT_IDENTITY_CHANGED")
    alpha_input_resolution: dict[str, Any] = {"gates_passed": True, "blocking_issues": []}
    if inputs.production:
        from infrastructure.data.canonical_v2_alpha_enrichment import resolve_inputs

        alpha_input_resolution = resolve_inputs(alpha_config)
        blockers.extend(str(issue).upper() for issue in alpha_input_resolution["blocking_issues"])
    namespace_plan = _alpha_namespace_path_plan(
        inputs,
        base_identity=base_identity,
        config=alpha_config,
    )
    path_budget = _path_budget_preflight(inputs, namespace_plan)
    if path_budget["status"] != "READY":
        blockers.append("PATH_LENGTH_BUDGET_EXCEEDED")
    blockers = list(dict.fromkeys(blockers))
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "mutation_performed": False,
        "run_id": inputs.run_id,
        "output_root": str(output),
        "base": base_identity,
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
        "canonical_daily_logical_checksum": _canonical_hash(canonical),
        "asset_registry_version": APPROVED_ASSET_REGISTRY_VERSION,
        "asset_registry_checksum": APPROVED_ASSET_REGISTRY_CHECKSUM,
        "feature_schema_checksum": _hash_json(schema),
        "final_schema_assembly": final_schema_assembly,
        "resolved_inputs": resolved_inputs,
        "alpha_input_resolution": alpha_input_resolution,
        "adapted_configuration_sha256": _adapted_configuration_hash(alpha_config),
        "alpha_namespaces": namespace_plan,
        "path_budget": path_budget,
        "compute": {
            "workers": inputs.workers,
            "max_in_flight_tasks": inputs.max_in_flight_tasks,
            "memory_budget_gib": inputs.memory_budget_gib,
            "arrow_threads_per_worker": 1,
        },
        "resume": inputs.resume,
        "production": inputs.production,
    }


def publish_bounded_smoke(
    inputs: ParentChainInputs,
    *,
    decision_dates: Sequence[str],
    feature_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = prepare_parent_chain_plan(inputs)
    if plan["status"] != "READY":
        raise ValueError(f"parent-chain preflight blocked: {plan['blockers']}")
    output = inputs.output_root.resolve()
    if inputs.production:
        raise ValueError("bounded smoke cannot run in production mode")
    resumed_incomplete = False
    existing = None
    try:
        existing = _compatible_result(output, plan)
    except ValueError:
        owner = _json(output / "owner.json", [], "OUTPUT_ROOT_CONFLICT")
        if not inputs.resume or owner != _owner(plan):
            raise
        interrupted = output.with_name(
            f".{output.name}.interrupted.{uuid.uuid4().hex}"
        )
        os.replace(output, interrupted)
        resumed_incomplete = True
    if existing is not None:
        return {**existing, "resume_action": "REUSED_COMPATIBLE"}
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.owner.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("ACTIVE_OWNER_CONFLICT") from exc
    os.write(lock_fd, inputs.run_id.encode("utf-8"))
    os.close(lock_fd)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        _write_json(temporary / "owner.json", _owner(plan))
        base_rows = _read_dates(inputs.base_artifact, decision_dates)
        if not base_rows:
            raise ValueError("bounded smoke selection produced no rows")
        sampler: _MemorySampler | None = None
        if feature_rows is not None:
            enriched = merge_enrichment_preserving_base(base_rows, feature_rows)
            feature_columns = sorted(
                set().union(*(row.keys() for row in feature_rows))
                - PROTECTED_ENRICHMENT_COLUMNS
            )
        elif inputs.canonical_daily_root is not None:
            from core.research.ml.stock_level.stock_level_alpha_features_builder import (
                build_stock_level_alpha_features,
            )

            histories = _canonical_histories(
                inputs.canonical_daily_root,
                {str(row["symbol"]).upper() for row in base_rows} | {"SPY"},
                maximum_date=max(decision_dates),
            )
            sampler = _MemorySampler()
            sampler.start()
            started = time.perf_counter()
            try:
                enriched, audit = build_stock_level_alpha_features(
                    base_rows, histories, n_jobs=inputs.workers
                )
            finally:
                sampler.stop()
            elapsed = time.perf_counter() - started
            feature_columns = sorted(
                set(enriched[0]) - set(base_rows[0]) - {"feature_coverage_status"}
            )
        else:
            enriched = merge_enrichment_preserving_base(
                base_rows,
                [
                    {**row, "smoke_feature": float(index % 17)}
                    for index, row in enumerate(base_rows)
                ],
            )
            feature_columns = ["smoke_feature"]
            audit = {}
            elapsed = 0.0
        parent_manifest = _lineage_manifest(inputs, plan, status="READY")
        child_manifest = dict(parent_manifest)
        lineage = validate_selector_parent_child_lineage(
            parent_manifest=parent_manifest,
            child_manifest=child_manifest,
            parent_rows=base_rows,
            child_rows=enriched,
        )
        if not lineage.ready:
            raise ValueError(f"smoke lineage blocked: {lineage.blockers}")
        pq.write_table(
            __import__("pyarrow").Table.from_pylist(base_rows),
            temporary / "smoke_base.parquet",
        )
        if inputs.canonical_daily_root is not None:
            from infrastructure.data.canonical_v2_alpha_enrichment import (
                _normalize_partition_rows,
                _schema_for_fieldnames,
            )

            canonical_enriched = []
            for row in enriched:
                canonical_row = dict(row)
                canonical_row.pop("row_id", None)
                canonical_row.pop("feature_coverage_status", None)
                canonical_enriched.append(canonical_row)
            normalized, _ = _normalize_partition_rows(canonical_enriched)
            enriched_table = pa.Table.from_pylist(
                normalized,
                schema=_schema_for_fieldnames(
                    list(normalized[0]) if normalized else []
                ),
            )
        else:
            enriched_table = pa.Table.from_pylist(enriched)
        pq.write_table(enriched_table, temporary / "smoke_enriched.parquet")
        smoke_physical = _sha256(temporary / "smoke_base.parquet")
        enriched_physical = _sha256(temporary / "smoke_enriched.parquet")
        enriched_logical = _hash_json({"rows": enriched})
        spine = build_ready_daily_spine_manifest(
            base_rows,
            parent_identity={
                "canonical_daily_dataset_version": parent_manifest[
                    "canonical_daily_dataset_version"
                ],
                "canonical_daily_logical_checksum": parent_manifest[
                    "canonical_daily_logical_checksum"
                ],
                "asset_registry_version": parent_manifest["asset_registry_version"],
                "asset_registry_checksum": parent_manifest["asset_registry_checksum"],
                "calendar_version": parent_manifest["calendar_version"],
                "decision_timing_contract": parent_manifest[
                    "decision_timing_contract"
                ],
            },
            physical_sha256=smoke_physical,
            configuration_hash=parent_manifest["configuration_hash"],
            git_commit=parent_manifest["git_commit"],
        )
        parent_manifest["daily_spine_identity"] = spine["dataset_id"]
        parent_manifest["daily_spine_logical_checksum"] = spine["logical_checksum"]
        child_manifest.update(
            {
                "daily_spine_identity": spine["dataset_id"],
                "daily_spine_logical_checksum": spine["logical_checksum"],
            }
        )
        lineage = validate_selector_parent_child_lineage(
            parent_manifest=parent_manifest,
            child_manifest=child_manifest,
            parent_rows=base_rows,
            child_rows=enriched,
        )
        frozen = preflight_frozen_selector_dataset(
            daily_spine_manifest=spine,
            base_manifest=parent_manifest,
            enriched_manifest=child_manifest,
            base_rows=base_rows,
            enriched_rows=enriched,
            feature_columns=feature_columns,
        )
        result = {
            "contract_version": CONTRACT_VERSION,
            "status": "READY" if lineage.ready and frozen["status"] == "READY" else "BLOCKED",
            "plan_checksum": _plan_identity_hash(plan),
            "selection_policy": "complete_decision_date_cross_sections",
            "selected_decision_dates": list(decision_dates),
            "row_count": len(base_rows),
            "symbol_count": len({str(row.get("symbol")) for row in base_rows}),
            "row_id_checksum": row_id_checksum(base_rows),
            "target_checksum": target_checksum(base_rows),
            "enriched_logical_checksum": enriched_logical,
            "enriched_physical_sha256": enriched_physical,
            "lineage": lineage.as_dict(),
            "daily_spine": spine,
            "frozen_preflight": frozen,
            "resume_action": (
                "RESUMED_INCOMPLETE" if resumed_incomplete else "PUBLISHED_NEW"
            ),
            "resume_history": (
                [{"action": "RESTARTED_OWNED_INCOMPLETE_OUTPUT"}]
                if resumed_incomplete
                else []
            ),
            "parent_production_sha256": APPROVED_BASE_SHA256,
            "benchmark": {
                "workers": inputs.workers,
                "elapsed_seconds": elapsed,
                "rows_per_second": len(base_rows) / elapsed if elapsed else None,
                "peak_parent_rss_bytes": (
                    sampler.peak_parent_rss or None if sampler is not None else None
                ),
                "peak_aggregate_worker_rss_bytes": (
                    sampler.peak_worker_rss or None if sampler is not None else None
                ),
                "read_bytes": None,
                "write_bytes": (
                    (temporary / "smoke_base.parquet").stat().st_size
                    + (temporary / "smoke_enriched.parquet").stat().st_size
                ),
                "audit": audit,
            },
        }
        _write_json(temporary / "base_manifest.json", parent_manifest)
        _write_json(temporary / "enriched_manifest.json", child_manifest)
        _write_json(temporary / "daily_spine_manifest.json", spine)
        _write_json(temporary / "result.json", result)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        lock_path.unlink(missing_ok=True)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
        raise


def build_production_enriched_child(
    inputs: ParentChainInputs, *, config_path: Path
) -> dict[str, Any]:
    """Delegate a preflighted full build to the authoritative partitioned owner."""
    if not inputs.production:
        raise ValueError("production enrichment requires --production")
    planning_inputs = ParentChainInputs(
        **{**inputs.__dict__, "config_path": config_path}
    )
    plan = prepare_parent_chain_plan(planning_inputs)
    if plan["status"] != "READY":
        raise ValueError(f"parent-chain preflight blocked: {plan['blockers']}")
    output = inputs.output_root.resolve()
    output.mkdir(parents=True, exist_ok=inputs.resume)
    lock_path = output.with_name(f".{output.name}.owner.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("ACTIVE_OWNER_CONFLICT") from exc
    try:
        os.write(lock_fd, inputs.run_id.encode("utf-8"))
        os.close(lock_fd)
        config = _parent_alpha_config(
            inputs,
            base_identity=plan["base"],
            config_path=config_path,
        )
        _write_json(output / "owner.json", _owner(plan))
        _write_json(
            output / "final_schema_ownership.json",
            plan["final_schema_assembly"],
        )
        from infrastructure.data.canonical_v2_alpha_enrichment import (
            write_partitioned_canonical_v2_alpha_features,
        )

        paths = write_partitioned_canonical_v2_alpha_features(config)
        final_artifact = _assemble_final_selector_parent(
            alpha_child=paths.enriched_parquet_path,
            labeled_spine_root=inputs.labeled_spine_root,
            fundamentals_artifact=inputs.fundamentals_artifact,
            output_path=output / "selector_parent_182.parquet",
            expected_base=plan["base"],
        )
        lifecycle = _json(
            output / "alpha_enrichment" / "partition_dataset_status.json",
            [],
            "PARTITION_LIFECYCLE_MISSING",
        )
        alpha_run = _json(
            output / "alpha_enrichment" / "alpha_only_run_manifest.json",
            [],
            "ALPHA_RUN_MANIFEST_MISSING",
        )
        progress = _json(
            output / "alpha_enrichment" / "progress_manifest.json",
            [],
            "ALPHA_PROGRESS_MISSING",
        )
        result = {
            "status": "COMPLETE",
            "partition_generation_status": "COMPLETE",
            "consolidation_status": "COMPLETE",
            "publication_status": "PUBLISHED",
            "overall_status": "COMPLETE",
            "planned_partitions": int(
                progress.get("planned_partitions", 0) or 0
            ),
            "completed_partitions": int(
                progress.get("completed_partitions", 0) or 0
            ),
            "failed_partitions": int(
                progress.get("failed_partitions", 0) or 0
            ),
            "contract_version": CONTRACT_VERSION,
            "run_id": inputs.run_id,
            "plan_checksum": _plan_identity_hash(plan),
            "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
            "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
            "parent_lineage": {
                "canonical_base_identity": plan["base"],
                "canonical_daily_logical_checksum": plan[
                    "canonical_daily_logical_checksum"
                ],
                "asset_registry_version": plan["asset_registry_version"],
                "asset_registry_checksum": plan["asset_registry_checksum"],
                "feature_schema_identity": plan["feature_schema_checksum"],
                "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
                "target_provenance_contract_version": (
                    CURRENT_TARGET_PROVENANCE_VERSION
                ),
            },
            "alpha_namespaces": plan["alpha_namespaces"],
            "path_budget": plan["path_budget"],
            "resolved_inputs": plan["resolved_inputs"],
            "alpha_input_resolution": plan["alpha_input_resolution"],
            "adapted_configuration_sha256": plan[
                "adapted_configuration_sha256"
            ],
            "outputs": {
                key: str(value)
                for key, value in vars(paths).items()
                if isinstance(value, Path)
            },
            "intermediate_alpha_child": {
                "path": str(paths.enriched_parquet_path),
                "contract_version": "canonical_v2_alpha_child_120.v1",
            },
            "final_selector_parent": final_artifact,
            "resume_finalization": {
                "reused_partition_count": int(
                    lifecycle.get("partitions_reused", 0) or 0
                ),
                "recomputed_partition_count": int(
                    lifecycle.get("partitions_recomputed", 0) or 0
                ),
                "partition_namespace_identity": lifecycle.get(
                    "partition_namespace_identity"
                ),
                "original_source_commit": alpha_run.get(
                    "original_source_commit"
                ),
                "finalization_source_commit": alpha_run.get(
                    "finalization_source_commit"
                ),
                "compatibility_decision": alpha_run.get(
                    "compatibility_decision"
                ),
            },
        }
        if (
            result["resume_finalization"]["recomputed_partition_count"] == 0
            and result["resume_finalization"]["reused_partition_count"] == 0
        ):
            raise ValueError("finalization resume reported no reusable partitions")
        if not (
            result["planned_partitions"]
            == result["completed_partitions"]
            and result["failed_partitions"] == 0
        ):
            raise ValueError(
                "parent-chain partition status is inconsistent with publication"
            )
        _write_json(output / "production_result.json", result)
        _write_json(output / "parent_chain_result.json", result)
        return result
    finally:
        lock_path.unlink(missing_ok=True)


def _base_identity(path: Path, manifest_path: Path, blockers: list[str]) -> dict[str, Any]:
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        _schema_fingerprint,
    )

    manifest = _json(manifest_path, blockers, "PARENT_IDENTITY_MISSING")
    canonical = dict(manifest.get("canonical_artifact", {}) or manifest)
    if not path.is_file():
        blockers.append("PARENT_IDENTITY_MISSING")
        return {}
    observed = _sha256(path)
    logical = str(canonical.get("logical_content_sha256", ""))
    rows = pq.ParquetFile(path).metadata.num_rows
    if observed != APPROVED_BASE_SHA256:
        blockers.append("BASE_HASH_MISMATCH")
    if logical != APPROVED_BASE_LOGICAL_HASH:
        blockers.append("BASE_LOGICAL_HASH_MISMATCH")
    if rows != PRODUCTION_ROW_COUNT:
        blockers.append("BASE_POPULATION_MISMATCH")
    provenance = str(
        manifest.get("target_provenance_contract_version")
        or canonical.get("target_contract_version")
        or ""
    )
    economic = str(manifest.get("economic_target_id") or CURRENT_ECONOMIC_TARGET_ID)
    if (
        economic != CURRENT_ECONOMIC_TARGET_ID
        or provenance != CURRENT_TARGET_PROVENANCE_VERSION
    ):
        blockers.append("TARGET_IDENTITY_MISMATCH")
    required = {
        "symbol",
        "rebalance_date",
        "decision_timestamp",
        "feature_data_cutoff_timestamp",
        "target_provenance_contract_version",
        *TARGET_COLUMNS,
        *TARGET_TIMESTAMP_COLUMNS,
    }
    missing = sorted(required - set(pq.ParquetFile(path).schema_arrow.names))
    if missing:
        blockers.append("PARENT_IDENTITY_MISSING")
    key_hasher = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=65_536, columns=["rebalance_date", "symbol"]
    ):
        for date, symbol in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist()
        ):
            key_hasher.update(
                f"{str(date)[:10]}\x1f{str(symbol).upper()}\n".encode("utf-8")
            )
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "logical_content_sha256": logical,
        "schema_fingerprint": _schema_fingerprint(
            pq.ParquetFile(path).schema_arrow
        ),
        "economic_key_sha256": key_hasher.hexdigest(),
        "column_count": len(pq.ParquetFile(path).schema_arrow.names),
        "target_provenance_contract_versions": [provenance] if provenance else [],
        "row_count": rows,
        "economic_target_id": economic,
        "target_provenance_contract_version": provenance,
        "missing_required_columns": missing,
    }


def _parquet_columns(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return set(pq.ParquetFile(path).schema_arrow.names)


def _representative_spine_columns(root: Path | None) -> set[str]:
    if root is None or not root.is_dir():
        return set()
    representative = next(iter(sorted(root.glob("symbol=*/spine.parquet"))), None)
    return _parquet_columns(representative)


def _final_schema_assembly_plan(inputs: ParentChainInputs) -> dict[str, Any]:
    """Prove physical column ownership; schema declarations are not producers."""
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        ALPHA_OUTPUT_SCHEMA,
        _time_series_features,
    )
    from core.research.ml.stock_level.stock_fundamentals import (
        FUNDAMENTAL_FEATURE_COLUMNS,
        FUNDAMENTAL_METADATA_COLUMNS,
    )

    final_columns = list(ALPHA_OUTPUT_SCHEMA)
    base_columns = _parquet_columns(inputs.base_artifact)
    alpha_columns = set(_time_series_features([], []))
    intermediate_columns = base_columns | alpha_columns
    spine_columns = _representative_spine_columns(inputs.labeled_spine_root)
    spine_aliases = {
        "asset_id",
        "canonical_symbol",
        "model_close",
        "source_provider",
        "compatibility_tier",
        "eligibility_reason",
        "selector_eligible",
        "provider_transition_flag",
        "provider_transition_id",
    } & spine_columns
    fundamental_columns = set(FUNDAMENTAL_FEATURE_COLUMNS) | set(
        FUNDAMENTAL_METADATA_COLUMNS
    )
    supplied_fundamental_columns = _parquet_columns(inputs.fundamentals_artifact)
    fundamentals_binding = _certified_fundamentals_binding(
        inputs.fundamentals_artifact, inputs.fundamentals_manifest
    )
    fundamentals_join_keys_present = (
        "asset_id" in supplied_fundamental_columns
        and bool(
            {"rebalance_date", "decision_session_date"}
            & supplied_fundamental_columns
        )
    )
    fundamentals_population_matches = bool(
        inputs.fundamentals_artifact
        and inputs.fundamentals_artifact.is_file()
        and pq.ParquetFile(inputs.fundamentals_artifact).metadata.num_rows
        == pq.ParquetFile(inputs.base_artifact).metadata.num_rows
    )
    fundamentals_bound = bool(
        fundamentals_binding["valid"]
        and fundamentals_join_keys_present
        and fundamentals_population_matches
    )

    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for column in final_columns:
        kind, nullable = ALPHA_OUTPUT_SCHEMA[column]
        if column in base_columns:
            producer = "immutable_canonical_v2_alpha_base"
            source = str(inputs.base_artifact.resolve())
            join_key = "physical inheritance"
            availability = "preserve certified base value and timestamps"
            stage = "alpha_child"
            resolved = True
        elif column in alpha_columns:
            producer = "canonical_v2_alpha_feature_producer"
            source = (
                "core.research.ml.stock_level.stock_level_alpha_features_builder"
            )
            join_key = "symbol + rebalance_date"
            availability = "trailing inputs strictly before decision timestamp"
            stage = "alpha_child"
            resolved = True
        elif column in spine_aliases:
            producer = "canonical_v2_labeled_spine_registry_bound"
            source = str(inputs.labeled_spine_root.resolve())
            join_key = "canonical_symbol + session_date -> symbol + rebalance_date"
            availability = "spine session must equal decision session"
            stage = "selector_parent_assembly"
            resolved = True
        elif column in fundamental_columns:
            producer = "stock_fundamentals_pit_enrichment"
            source = (
                (
                    str(inputs.fundamentals_artifact.resolve())
                    + " | "
                    + str(inputs.fundamentals_manifest.resolve())
                )
                if inputs.fundamentals_artifact
                and inputs.fundamentals_manifest
                else ""
            )
            join_key = "asset_id + latest fundamentals_available_timestamp <= decision_timestamp"
            availability = (
                "fundamentals_available_timestamp <= decision_timestamp; "
                "latest eligible certified snapshot only"
            )
            stage = "selector_parent_assembly"
            resolved = (
                fundamentals_bound and column in supplied_fundamental_columns
            )
        else:
            producer = "unresolved"
            source = ""
            join_key = ""
            availability = ""
            stage = "selector_parent_assembly"
            resolved = False
        record = {
            "column": column,
            "declared_schema_type": kind,
            "nullable": nullable,
            "authoritative_producer": producer,
            "authoritative_source_path_or_manifest": source,
            "join_key": join_key,
            "availability_timestamp_rule": availability,
            "expected_stage": stage,
            "physically_produced": resolved,
        }
        records.append(record)
        if not resolved:
            unresolved.append(
                {
                    "column": column,
                    "expected_owner": producer,
                    "expected_stage": stage,
                }
            )
    physically_planned = {row["column"] for row in records if row["physically_produced"]}
    missing_from_alpha_child = sorted(set(final_columns) - intermediate_columns)
    return {
        "contract_version": "selector_parent_final_schema_assembly.v1",
        "contract_decision": "B",
        "alpha_child_contract": {
            "column_count": len(intermediate_columns),
            "columns": sorted(intermediate_columns),
        },
        "final_selector_parent_contract": {
            "column_count": len(final_columns),
            "columns": final_columns,
        },
        "base_physical_column_count": len(base_columns),
        "alpha_producer_physical_column_count": len(alpha_columns),
        "alpha_child_physical_column_count": len(intermediate_columns),
        "missing_from_alpha_child_count": len(missing_from_alpha_child),
        "missing_from_alpha_child": missing_from_alpha_child,
        "planned_final_physical_column_count": len(physically_planned),
        "unresolved_column_count": len(unresolved),
        "unresolved_columns": unresolved,
        "ownership": records,
        "fundamentals_binding": fundamentals_binding,
        "fundamentals_join_keys_present": fundamentals_join_keys_present,
        "fundamentals_population_matches": fundamentals_population_matches,
        "status": "READY" if not unresolved else "BLOCKED",
        "blocker": None if not unresolved else "FINAL_SCHEMA_PRODUCER_UNRESOLVED",
    }


def _certified_fundamentals_binding(
    artifact: Path | None, manifest_path: Path | None
) -> dict[str, Any]:
    if (
        artifact is None
        or manifest_path is None
        or not artifact.is_file()
        or not manifest_path.is_file()
    ):
        return {"valid": False, "reason": "artifact_or_manifest_missing"}
    manifest = _json(manifest_path, [], "FUNDAMENTALS_MANIFEST_INVALID")
    canonical = dict(manifest.get("canonical_artifact", {}) or {})
    expected_sha = str(
        canonical.get("sha256")
        or manifest.get("artifact_sha256")
        or manifest.get("sha256")
        or ""
    )
    observed_sha = _sha256(artifact)
    contract_text = json.dumps(manifest, sort_keys=True)
    complete = str(
        manifest.get("status")
        or manifest.get("completion_status")
        or canonical.get("completion_status")
        or ""
    ).upper() == "COMPLETE"
    contract_valid = any(
        token in contract_text
        for token in (
            "stock_level_fundamentals_enrichment_contract_v1",
            "fundamentals_pit_snapshot_contract_v1",
        )
    )
    valid = complete and bool(expected_sha) and expected_sha == observed_sha and contract_valid
    return {
        "valid": valid,
        "reason": (
            "certified"
            if valid
            else "manifest_status_hash_or_contract_mismatch"
        ),
        "artifact_path": str(artifact.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "observed_sha256": observed_sha,
        "manifest_sha256": expected_sha,
        "completion_status_valid": complete,
        "contract_valid": contract_valid,
    }


def _assemble_final_selector_parent(
    *,
    alpha_child: Path,
    labeled_spine_root: Path | None,
    fundamentals_artifact: Path | None,
    output_path: Path,
    expected_base: Mapping[str, Any],
) -> dict[str, Any]:
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        ALPHA_OUTPUT_SCHEMA,
        _rows_to_table,
        _schema_for_fieldnames,
    )
    from core.research.ml.stock_level.stock_fundamentals import (
        FUNDAMENTAL_FEATURE_COLUMNS,
        FUNDAMENTAL_METADATA_COLUMNS,
    )

    if labeled_spine_root is None or not labeled_spine_root.is_dir():
        raise ValueError("FINAL_SCHEMA_PRODUCER_UNRESOLVED: labeled spine")
    if fundamentals_artifact is None or not fundamentals_artifact.is_file():
        raise ValueError("FINAL_SCHEMA_PRODUCER_UNRESOLVED: PIT fundamentals")
    final_columns = list(ALPHA_OUTPUT_SCHEMA)
    fundamental_columns = [
        *FUNDAMENTAL_FEATURE_COLUMNS,
        *FUNDAMENTAL_METADATA_COLUMNS,
    ]
    required_fundamental_columns = {
        "asset_id",
        *fundamental_columns,
    }
    fundamentals_schema = set(pq.ParquetFile(fundamentals_artifact).schema_arrow.names)
    date_column = (
        "rebalance_date"
        if "rebalance_date" in fundamentals_schema
        else "decision_session_date"
        if "decision_session_date" in fundamentals_schema
        else ""
    )
    missing_fundamentals = sorted(
        required_fundamental_columns - fundamentals_schema
    )
    if not date_column:
        missing_fundamentals.append("rebalance_date|decision_session_date")
    if missing_fundamentals:
        raise ValueError(
            "FINAL_SCHEMA_PRODUCER_UNRESOLVED: PIT fundamentals missing "
            + repr(missing_fundamentals)
        )

    read_columns = ["asset_id", date_column, *fundamental_columns]
    fundamentals_parquet = pq.ParquetFile(fundamentals_artifact)
    if fundamentals_parquet.metadata.num_rows != pq.ParquetFile(alpha_child).metadata.num_rows:
        raise ValueError(
            "PIT fundamentals population differs from alpha child; "
            "row loss or multiplication would occur"
        )

    def fundamental_rows():
        for fundamental_batch in fundamentals_parquet.iter_batches(
            batch_size=16_384, columns=read_columns
        ):
            yield from pa.Table.from_batches([fundamental_batch]).to_pylist()

    fundamental_iterator = iter(fundamental_rows())

    alpha_parquet = pq.ParquetFile(alpha_child)
    alpha_columns = set(alpha_parquet.schema_arrow.names)
    if len(alpha_columns) != 120:
        raise ValueError(
            f"alpha-child contract mismatch: expected 120 columns, got {len(alpha_columns)}"
        )
    schema = _schema_for_fieldnames(final_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    key_hasher = hashlib.sha256()
    spine_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        for batch in alpha_parquet.iter_batches(batch_size=16_384):
            output_rows: list[dict[str, Any]] = []
            for alpha_row in pa.Table.from_batches([batch]).to_pylist():
                symbol = str(alpha_row.get("symbol") or "").upper()
                decision_date = str(alpha_row.get("rebalance_date") or "")[:10]
                spine_path = (
                    labeled_spine_root / f"symbol={symbol}" / "spine.parquet"
                )
                if not spine_path.is_file():
                    raise ValueError(f"unmatched required spine identity: {symbol}")
                if symbol not in spine_indexes:
                    by_date: dict[str, dict[str, Any]] = {}
                    for candidate in pq.read_table(spine_path).to_pylist():
                        candidate_date = str(
                            candidate.get("session_date") or ""
                        )[:10]
                        if candidate_date in by_date:
                            raise ValueError(
                                "spine row multiplication risk: "
                                f"symbol={symbol} date={candidate_date}"
                            )
                        by_date[candidate_date] = candidate
                    spine_indexes[symbol] = by_date
                spine = spine_indexes[symbol].get(decision_date)
                if spine is None:
                    raise ValueError(
                        "spine join must preserve one row: "
                        f"symbol={symbol} date={decision_date} matches=0"
                    )
                if str(spine.get("canonical_symbol") or "").upper() != symbol:
                    raise ValueError(f"ambiguous canonical symbol mapping: {symbol}")
                asset_id = str(spine.get("asset_id") or "")
                try:
                    fundamental = next(fundamental_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        "PIT fundamentals row loss during aligned join"
                    ) from exc
                fundamental_key = (
                    str(fundamental.get("asset_id") or ""),
                    str(fundamental.get(date_column) or "")[:10],
                )
                if fundamental_key != (asset_id, decision_date):
                    raise ValueError(
                        "PIT fundamentals alignment mismatch: "
                        f"expected={(asset_id, decision_date)} observed={fundamental_key}"
                    )
                available = str(
                    fundamental.get("fundamentals_available_timestamp") or ""
                )
                decision = str(alpha_row.get("decision_timestamp") or "")
                if available and decision and available > decision:
                    raise ValueError(
                        "future PIT fundamentals availability: "
                        f"asset_id={asset_id} available={available} decision={decision}"
                    )
                assembled = dict(alpha_row)
                for column in (
                    "asset_id",
                    "canonical_symbol",
                    "model_close",
                    "source_provider",
                    "compatibility_tier",
                    "eligibility_reason",
                    "selector_eligible",
                    "provider_transition_flag",
                    "provider_transition_id",
                ):
                    assembled[column] = spine.get(column)
                for column in fundamental_columns:
                    assembled[column] = fundamental.get(column)
                output_rows.append(assembled)
                key_hasher.update(
                    f"{decision_date}\x1f{symbol}\n".encode("utf-8")
                )
            table = _rows_to_table(output_rows, schema)
            writer.write_table(table)
            row_count += table.num_rows
        writer.close()
        writer = None
        if row_count != alpha_parquet.metadata.num_rows:
            raise ValueError(
                f"final assembly row loss: input={alpha_parquet.metadata.num_rows} output={row_count}"
            )
        try:
            next(fundamental_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("PIT fundamentals row multiplication during aligned join")
        expected_rows = int(expected_base.get("row_count", 0) or 0)
        expected_keys = str(expected_base.get("economic_key_sha256", ""))
        if expected_rows and row_count != expected_rows:
            raise ValueError(
                f"final assembly population mismatch: {row_count} != {expected_rows}"
            )
        if expected_keys and key_hasher.hexdigest() != expected_keys:
            raise ValueError("final assembly economic-key checksum mismatch")
        published_schema = pq.ParquetFile(temporary).schema_arrow
        if published_schema.names != final_columns:
            raise ValueError("final selector-parent schema ordering mismatch")
        os.replace(temporary, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "COMPLETE",
        "contract_version": "selector_parent_final_schema_assembly.v1",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "row_count": row_count,
        "column_count": len(final_columns),
        "economic_key_sha256": key_hasher.hexdigest(),
        "atomic_publication": {"status": "COMPLETE"},
    }


def _parent_alpha_config(
    inputs: ParentChainInputs,
    *,
    base_identity: Mapping[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    import yaml

    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config_path is not None and config_path.is_file()
        else {}
    )
    ml = dict(config.get("ml", {}) or {})
    configured = dict(ml)
    output = inputs.output_root.resolve()
    enriched_root = output / "enriched"
    ml.update(
        {
            "stock_alpha_run_size": "benchmark",
            "stock_alpha_output_dir_override": True,
            "output_dir": str(enriched_root),
            "stock_level_base_prediction_artifacts_path": str(
                inputs.base_artifact.resolve()
            ),
            "canonical_v2_alpha_base_manifest_path": str(
                inputs.base_manifest.resolve()
            ),
            "stock_level_prediction_artifacts_path": str(
                enriched_root
                / "stock_level_prediction_artifacts_enriched.parquet"
            ),
            "canonical_daily_v2_root": str(
                (inputs.canonical_daily_root or Path("")).resolve()
            ),
            "stooq_parquet_dir": str(
                (inputs.canonical_daily_root or Path("")).resolve()
            ),
            "canonical_daily_v2_manifest_path": str(
                inputs.canonical_daily_manifest.resolve()
            ),
            "canonical_asset_registry_manifest_path": str(
                inputs.asset_registry_manifest.resolve()
            ),
            "stock_selector_market_data_source": "canonical_daily_v2",
            "selector_feature_schema_path": str(inputs.feature_schema.resolve()),
            "canonical_v2_alpha_report_root": str(output / "alpha_enrichment"),
            "stock_alpha_feature_partitioned_resume": True,
            "stock_alpha_resume_existing_outputs": inputs.resume,
            "stock_alpha_feature_n_jobs": inputs.workers,
            "stock_level_dataset_workers": inputs.workers,
            "stock_level_dataset_inner_threads": 1,
            "canonical_v2_alpha_max_in_flight_tasks": inputs.max_in_flight_tasks,
            "canonical_v2_alpha_memory_budget_gib": inputs.memory_budget_gib,
            "canonical_v2_alpha_validated_base_sha256": base_identity.get(
                "sha256", ""
            ),
            "canonical_v2_alpha_validated_base_key_sha256": base_identity.get(
                "economic_key_sha256", ""
            ),
            "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
            "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
            "pit_lineage_run_id": inputs.run_id,
        }
    )
    if inputs.data_root is not None:
        if not inputs.data_root.is_absolute():
            raise ValueError("data_root explicit authority must be absolute")
        ml["canonical_v2_production_data_root"] = str(inputs.data_root.resolve())
    for key, explicit in {
        "canonical_v2_labeled_spine_root": inputs.labeled_spine_root,
        "canonical_v2_labeled_spine_manifest_path": inputs.labeled_spine_manifest,
        "canonical_v2_inference_spine_manifest_path": inputs.inference_spine_manifest,
        "canonical_v2_fundamentals_artifact_path": inputs.fundamentals_artifact,
        "canonical_v2_fundamentals_manifest_path": inputs.fundamentals_manifest,
    }.items():
        if key.startswith("canonical_v2_fundamentals_"):
            if explicit is not None:
                ml[key] = str(explicit.resolve())
        elif inputs.production or explicit is not None:
            ml[key] = str(_resolve_production_path(explicit, configured.get(key), inputs.data_root, key))
    config["ml"] = ml
    return config


def _resolve_production_path(
    explicit: Path | None, configured: Any, data_root: Path | None, name: str
) -> Path:
    if explicit is not None:
        if not explicit.is_absolute():
            raise ValueError(f"{name} explicit authority must be absolute")
        return explicit.resolve()
    if configured:
        candidate = Path(str(configured))
        if candidate.is_absolute():
            return candidate.resolve()
        if data_root is not None:
            return (data_root.resolve() / candidate).resolve()
    raise ValueError(f"{name} unresolved: supply an absolute CLI path or --data-root")


def _resolved_input_report(
    inputs: ParentChainInputs, config: Mapping[str, Any]
) -> dict[str, Any]:
    import yaml

    ml = dict(config.get("ml", {}) or {})
    source_config = (
        yaml.safe_load(inputs.config_path.read_text(encoding="utf-8")) or {}
        if inputs.config_path is not None and inputs.config_path.is_file()
        else {}
    )
    source_ml = dict(source_config.get("ml", {}) or {})
    specs = {
        "labeled_spine_root": ("canonical_v2_labeled_spine_root", inputs.labeled_spine_root),
        "labeled_spine_manifest": ("canonical_v2_labeled_spine_manifest_path", inputs.labeled_spine_manifest),
        "inference_spine_manifest": ("canonical_v2_inference_spine_manifest_path", inputs.inference_spine_manifest),
        "canonical_daily_root": ("canonical_daily_v2_root", inputs.canonical_daily_root),
        "canonical_daily_manifest": ("canonical_daily_v2_manifest_path", inputs.canonical_daily_manifest),
        "asset_registry_manifest": ("canonical_asset_registry_manifest_path", inputs.asset_registry_manifest),
        "base_artifact": ("stock_level_base_prediction_artifacts_path", inputs.base_artifact),
        "base_publication_manifest": ("canonical_v2_alpha_base_manifest_path", inputs.base_manifest),
        "fundamentals_artifact": ("canonical_v2_fundamentals_artifact_path", inputs.fundamentals_artifact),
        "fundamentals_manifest": ("canonical_v2_fundamentals_manifest_path", inputs.fundamentals_manifest),
    }
    return {
        label: {
            "source": (
                "CLI"
                if explicit is not None
                else (
                    "config"
                    if Path(str(source_ml.get(key, ""))).is_absolute()
                    else "data-root"
                )
            ),
            "original_configured_value": source_ml.get(key),
            "resolved_absolute_value": str(Path(str(ml[key])).resolve()),
            "overridden": (
                explicit is not None
                and str(source_ml.get(key, "")) != str(Path(str(ml[key])))
            ),
        }
        for label, (key, explicit) in specs.items()
        if key in ml
    }


def _adapted_configuration_hash(config: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(config, default=str))
    ml = payload.setdefault("ml", {})
    ml["stock_alpha_resume_existing_outputs"] = False
    ml.pop("canonical_v2_fundamentals_artifact_path", None)
    ml.pop("canonical_v2_fundamentals_manifest_path", None)
    return _hash_json(payload)


def _partition_producing_resolved_inputs(
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in resolved.items()
        if key not in {"fundamentals_artifact", "fundamentals_manifest"}
    }


def _alpha_namespace_path_plan(
    inputs: ParentChainInputs,
    *,
    base_identity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from infrastructure.data.canonical_v2_alpha_enrichment import (
        _planned_bounded_alpha_namespaces,
    )

    base_validation = {
        "sha256": base_identity.get("sha256", ""),
        "logical_content_sha256": base_identity.get(
            "logical_content_sha256", ""
        ),
        "schema_fingerprint": base_identity.get("schema_fingerprint", ""),
        "economic_key_sha256": base_identity.get("economic_key_sha256", ""),
        "row_count": int(base_identity.get("row_count", 0) or 0),
        "column_count": int(base_identity.get("column_count", 0) or 0),
        "target_provenance_contract_versions": list(
            base_identity.get("target_provenance_contract_versions", [])
        ),
    }
    report_root = inputs.output_root.resolve() / "alpha_enrichment"
    return _planned_bounded_alpha_namespaces(
        report_root, config, base_validation=base_validation
    )


def _longest_symbol(path: Path) -> str:
    longest = "REPRESENTATIVE.MAXIMUM.SYMBOL"
    if not path.is_file():
        return longest
    parquet = pq.ParquetFile(path)
    if "symbol" not in parquet.schema_arrow.names:
        return longest
    for batch in parquet.iter_batches(batch_size=65_536, columns=["symbol"]):
        for value in batch.column(0).to_pylist():
            text = str(value or "").upper().replace("/", "_").replace("\\", "_")
            if len(text) > len(longest):
                longest = text
    return longest


def _path_budget_preflight(
    inputs: ParentChainInputs, namespaces: Mapping[str, Any]
) -> dict[str, Any]:
    output = inputs.output_root.resolve()
    base_root = Path(str(namespaces["base"]["path"]))
    partition_root = Path(str(namespaces["partitions"]["path"]))
    symbol = _longest_symbol(inputs.base_artifact)
    namespace_key = str(namespaces["partitions"]["namespace_key"])
    base_sha = str(namespaces["base"]["source_base_sha256"])
    final = (
        output
        / "enriched"
        / "stock_level_prediction_artifacts_enriched.parquet"
    )
    candidates = {
        "absolute_output_root": output,
        "bounded_base_namespace": base_root,
        "bounded_partition_namespace": partition_root,
        "partition": partition_root / "partitions" / f"symbol={symbol}" / "rows.parquet",
        "partition_temporary": partition_root / "partitions" / f"symbol={symbol}" / "rows.parquet.tmp",
        "partition_manifest": partition_root / "partition_manifests" / f"{symbol}.json",
        "partition_failure": partition_root / "partition_failures" / f"{symbol}.json",
        "schema_diagnostic": partition_root / "schema_diagnostics" / f"{symbol}.json",
        "schema_validation": partition_root / "schema_validation" / f"{symbol}.json",
        "partition_attempt": partition_root.parent / f".attempt-{namespace_key}-2147483647-deadbeef" / "namespace_manifest.json",
        "base_attempt": base_root.parent / f".attempt-{base_sha[:8]}-2147483647-deadbeef" / f"symbol={symbol}" / "rows.parquet.identity.json",
        "final_publication": final,
        "final_publication_temporary": final.with_suffix(final.suffix + ".tmp"),
    }
    lengths = {name: len(str(path)) for name, path in candidates.items()}
    longest_name = max(lengths, key=lengths.get)
    maximum = lengths[longest_name]
    limit = int(inputs.path_length_limit)
    return {
        "status": "READY" if maximum <= limit else "BLOCKED",
        "blocker": (
            None if maximum <= limit else "PATH_LENGTH_BUDGET_EXCEEDED"
        ),
        "configured_limit": limit,
        "calculated_maximum_path_length": maximum,
        "longest_path_kind": longest_name,
        "longest_representative_path": str(candidates[longest_name]),
        "representative_paths": {
            name: {"path": str(path), "length": lengths[name]}
            for name, path in sorted(candidates.items())
        },
        "recommended_shorter_output_root": str(
            output.drive + "\\pit\\" + inputs.run_id
            if output.drive
            else Path("/tmp/pit") / inputs.run_id
        ),
    }


def _lineage_manifest(
    inputs: ParentChainInputs, plan: Mapping[str, Any], *, status: str
) -> dict[str, Any]:
    return {
        "status": status,
        "economic_target_id": CURRENT_ECONOMIC_TARGET_ID,
        "target_provenance_contract_version": CURRENT_TARGET_PROVENANCE_VERSION,
        "canonical_daily_dataset_version": "canonical_daily_v2.partitioned.v1",
        "canonical_daily_logical_checksum": APPROVED_CANONICAL_DAILY_HASH,
        "asset_registry_version": APPROVED_ASSET_REGISTRY_VERSION,
        "asset_registry_checksum": APPROVED_ASSET_REGISTRY_CHECKSUM,
        "daily_spine_identity": "pending-smoke-spine",
        "daily_spine_logical_checksum": "pending-smoke-spine",
        "calendar_version": "canonical_daily_v2_sessions",
        "decision_timing_contract": "close-plus-five-minutes-v1",
        "configuration_hash": _hash_json(plan["compute"]),
        "git_commit": _git_commit(),
        "builder_contract_version": CONTRACT_VERSION,
        "feature_schema_version": str(plan["feature_schema_checksum"]),
        "source_base_sha256": APPROVED_BASE_SHA256,
    }


def _compatible_result(output: Path, plan: Mapping[str, Any]) -> dict[str, Any] | None:
    if not output.exists():
        return None
    result = _json(output / "result.json", [], "OUTPUT_ROOT_CONFLICT")
    owner = _json(output / "owner.json", [], "OUTPUT_ROOT_CONFLICT")
    if (
        owner == _owner(plan)
        and result.get("status") == "READY"
        and result.get("plan_checksum") == _plan_identity_hash(plan)
    ):
        return result
    raise ValueError("OUTPUT_ROOT_CONFLICT")


def _owner(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": plan["run_id"],
        "base_sha256": plan["base"].get("sha256"),
        "feature_schema_checksum": plan["feature_schema_checksum"],
        "adapted_configuration_sha256": plan.get("adapted_configuration_sha256"),
        "resolved_inputs_sha256": _hash_json(
            _partition_producing_resolved_inputs(
                plan.get("resolved_inputs", {})
            )
        ),
    }


def _read_dates(path: Path, dates: Sequence[str]) -> list[dict[str, Any]]:
    selected = pa.array(list(dates), type=pa.string())
    rows: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65_536):
        names = batch.schema.names
        date_column = (
            batch.column(names.index("decision_session_date"))
            if "decision_session_date" in names
            else batch.column(names.index("rebalance_date"))
        )
        normalized = pc.utf8_slice_codeunits(pc.cast(date_column, pa.string()), 0, 10)
        filtered = batch.filter(pc.is_in(normalized, value_set=selected))
        if filtered.num_rows:
            rows.extend(filtered.to_pylist())
    return rows


def _canonical_histories(
    root: Path, symbols: set[str], *, maximum_date: str
) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    minimum_year = int(maximum_date[:4]) - 3
    maximum_year = int(maximum_date[:4])
    for symbol in sorted(symbols):
        rows: list[dict[str, Any]] = []
        symbol_root = root / f"symbol={symbol}"
        for year in range(minimum_year, maximum_year + 1):
            path = symbol_root / f"year={year}" / "bars.parquet"
            if not path.exists():
                continue
            table = pq.read_table(
                path,
                columns=["session_date", "model_close"],
                filters=[("session_date", "<=", maximum_date)],
            )
            rows.extend(
                {"date": str(row["session_date"])[:10], "close": row["model_close"]}
                for row in table.to_pylist()
                if row.get("model_close") is not None
            )
        histories[symbol] = rows
    return histories


class _MemorySampler:
    def __init__(self) -> None:
        self.peak_parent_rss = 0
        self.peak_worker_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            psutil = None
        process = psutil.Process() if psutil is not None else None

        def sample() -> None:
            while not self._stop.wait(0.02):
                try:
                    self.peak_parent_rss = max(
                        self.peak_parent_rss,
                        process.memory_info().rss if process is not None else _windows_rss(),
                    )
                    if process is not None:
                        worker_rss = sum(
                            child.memory_info().rss
                            for child in process.children(recursive=True)
                            if child.is_running()
                        )
                        self.peak_worker_rss = max(self.peak_worker_rss, worker_rss)
                except (Exception, OSError):
                    pass

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _windows_rss() -> int:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        return 0
    return int(counters.WorkingSetSize)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("dataset_logical_partition_hash")
        or payload.get("logical_checksum")
        or ""
    )


def _feature_columns(payload: Mapping[str, Any]) -> list[str]:
    explicit = payload.get("feature_columns")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if str(value)]
    records = payload.get("features")
    if isinstance(records, list):
        return [
            str(record["name"])
            for record in records
            if isinstance(record, Mapping) and str(record.get("name", ""))
        ]
    return []


def _json(path: Path, blockers: list[str], blocker: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        blockers.append(blocker)
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _plan_identity_hash(plan: Mapping[str, Any]) -> str:
    return _hash_json(
        {
            key: value
            for key, value in plan.items()
            if key not in {"status", "blockers", "mutation_performed", "resume"}
        }
    )


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run an isolated selector parent-chain smoke."
    )
    parser.add_argument("action", choices=("preflight", "smoke", "build"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--canonical-daily-manifest", required=True, type=Path)
    parser.add_argument("--asset-registry-manifest", required=True, type=Path)
    parser.add_argument("--feature-schema", required=True, type=Path)
    parser.add_argument("--canonical-daily-root", type=Path)
    parser.add_argument("--labeled-spine-root", type=Path)
    parser.add_argument("--labeled-spine-manifest", type=Path)
    parser.add_argument("--inference-spine-manifest", type=Path)
    parser.add_argument("--fundamentals-artifact", type=Path)
    parser.add_argument("--fundamentals-manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-in-flight-tasks", type=int, default=12)
    parser.add_argument("--memory-budget-gib", type=float, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--decision-date", action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--path-length-limit", type=int, default=240)
    args = parser.parse_args(argv)
    inputs = ParentChainInputs(
        run_id=args.run_id,
        output_root=args.output_root,
        base_artifact=args.base_artifact,
        base_manifest=args.base_manifest,
        canonical_daily_manifest=args.canonical_daily_manifest,
        asset_registry_manifest=args.asset_registry_manifest,
        feature_schema=args.feature_schema,
        canonical_daily_root=args.canonical_daily_root,
        labeled_spine_root=args.labeled_spine_root,
        labeled_spine_manifest=args.labeled_spine_manifest,
        inference_spine_manifest=args.inference_spine_manifest,
        fundamentals_artifact=args.fundamentals_artifact,
        fundamentals_manifest=args.fundamentals_manifest,
        data_root=args.data_root,
        workers=args.workers,
        max_in_flight_tasks=args.max_in_flight_tasks,
        memory_budget_gib=args.memory_budget_gib,
        resume=args.resume,
        production=args.production,
        config_path=args.config,
        path_length_limit=args.path_length_limit,
    )
    if args.action == "preflight":
        result = prepare_parent_chain_plan(inputs)
    elif args.action == "smoke":
        result = publish_bounded_smoke(inputs, decision_dates=args.decision_date)
    else:
        if args.config is None:
            parser.error("build requires --config")
        result = build_production_enriched_child(inputs, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"READY", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
