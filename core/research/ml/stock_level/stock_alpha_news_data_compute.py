from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.research.compute.lease_storage import atomic_write_json
from core.research.compute.machine_profile import GIB, MachineProfile
from core.research.compute.resource_governor import LeaseStatus, ResourceRequest
from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
from core.research.compute.run_contracts import (
    build_item_status,
    build_result_record,
    build_run_manifest,
    checksum,
    metric_value,
)
from core.research.compute.run_storage import (
    initialise_run,
    publish_item_status,
    publish_results_snapshot,
    publish_summary,
    update_global_registry_snapshot,
    update_run_status,
)

EXECUTION_CONTRACT_VERSION = "stock_alpha_news_data_compute_execution.v1"
PIPELINE = "stock_alpha_news"
STAGE = "data_materialisation"
CORPUS = "CANONICAL_CORPUS_MATERIALISATION"
FEATURES = "PIT_NEWS_FEATURE_STORE_MATERIALISATION"
SUPPORTED_STAGES = (CORPUS, FEATURES)


@dataclass(frozen=True)
class NewsDataMaterialisationPlan:
    selected_stages: tuple[str, ...]
    source_inventory_identity: str
    source_inventory_checksum: str
    canonical_corpus_contract_identity: str
    canonical_output_compatibility_identity: str
    canonical_parent_identity: str
    canonical_parent_checksum: str
    date_boundary_identity: str
    universe_identity: str
    availability_policy_identity: str
    pit_feature_contract_identity: str
    feature_output_compatibility_identity: str
    configuration_checksum: str
    source_git_commit: str
    corpus_work_units: tuple[Mapping[str, Any], ...] = ()
    feature_work_units: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.selected_stages:
            raise ValueError("At least one news data stage is required")
        if any(stage not in SUPPORTED_STAGES for stage in self.selected_stages):
            raise ValueError("Unsupported news data materialisation stage")
        if len(set(self.selected_stages)) != len(self.selected_stages):
            raise ValueError("Duplicate news data materialisation stage")
        required = (
            self.source_inventory_identity,
            self.source_inventory_checksum,
            self.canonical_corpus_contract_identity,
            self.canonical_output_compatibility_identity,
            self.canonical_parent_identity,
            self.canonical_parent_checksum,
            self.date_boundary_identity,
            self.universe_identity,
            self.availability_policy_identity,
            self.configuration_checksum,
            self.source_git_commit,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("News data materialisation plan lineage is incomplete")
        if FEATURES in self.selected_stages and not all(
            (
                self.pit_feature_contract_identity,
                self.feature_output_compatibility_identity,
            )
        ):
            raise ValueError("PIT feature-store contract lineage is incomplete")


class NewsDataComputeAdapter(Protocol):
    def compatible(
        self, stage: str, work_unit: Mapping[str, Any],
        corpus_parent: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...

    def resolve_corpus_input(self, work_unit: Mapping[str, Any]) -> Any: ...
    def read_corpus_source(self, resolved: Any) -> Any: ...
    def canonicalise(self, source: Any) -> Any: ...
    def validate_corpus_identity(self, canonical: Any) -> Mapping[str, Any]: ...
    def validate_corpus_availability(self, canonical: Any) -> Mapping[str, Any]: ...
    def publish_corpus(self, canonical: Any) -> Mapping[str, Any]: ...

    def resolve_corpus_parent(
        self, parent: Mapping[str, Any], work_unit: Mapping[str, Any]
    ) -> Any: ...
    def prepare_pit_inputs(self, parent: Any, work_unit: Mapping[str, Any]) -> Any: ...
    def calculate_features(self, prepared: Any) -> Any: ...
    def validate_feature_store(self, features: Any) -> Mapping[str, Any]: ...
    def publish_feature_store(self, features: Any) -> Mapping[str, Any]: ...


def deterministic_news_data_run_id(plan: NewsDataMaterialisationPlan) -> str:
    payload = {
        "execution_contract": EXECUTION_CONTRACT_VERSION,
        "selected_stages": list(plan.selected_stages),
        "source_inventory_identity": plan.source_inventory_identity,
        "source_inventory_checksum": plan.source_inventory_checksum,
        "canonical_corpus_contract_identity":
            plan.canonical_corpus_contract_identity,
        "canonical_output_compatibility_identity":
            plan.canonical_output_compatibility_identity,
        "canonical_parent_identity": plan.canonical_parent_identity,
        "canonical_parent_checksum": plan.canonical_parent_checksum,
        "date_boundary_identity": plan.date_boundary_identity,
        "universe_identity": plan.universe_identity,
        "availability_policy_identity": plan.availability_policy_identity,
        "pit_feature_contract_identity": plan.pit_feature_contract_identity,
        "feature_output_compatibility_identity":
            plan.feature_output_compatibility_identity,
        "configuration_checksum": plan.configuration_checksum,
        "corpus_work_units": _units(plan.corpus_work_units),
        "feature_work_units": _units(plan.feature_work_units),
    }
    return f"news-data-{checksum(payload)[:24]}"


def deterministic_news_data_item_id(
    *, run_id: str, stage: str, work_unit: Mapping[str, Any],
    plan: NewsDataMaterialisationPlan,
) -> str:
    compatibility = (
        plan.canonical_output_compatibility_identity
        if stage == CORPUS else plan.feature_output_compatibility_identity
    )
    parent = (
        plan.source_inventory_identity
        if stage == CORPUS else plan.canonical_parent_identity
    )
    return stage.lower() + "-" + checksum({
        "run_id": run_id,
        "stage": stage,
        "parent_stage": CORPUS if stage == FEATURES else None,
        "authoritative_input_identity": parent,
        "authoritative_work_unit": dict(work_unit),
        "output_compatibility_identity": compatibility,
        "execution_contract": EXECUTION_CONTRACT_VERSION,
    })[:32]


def build_news_data_resource_request(
    *, run_id: str, item_id: str, stage: str, attempt_identity: str,
) -> ResourceRequest:
    if stage == CORPUS:
        ram, cpu = 4 * GIB, 1
    elif stage == FEATURES:
        ram, cpu = 6 * GIB, 2
    else:
        raise ValueError("Unsupported news data resource stage")
    return ResourceRequest(
        pipeline=PIPELINE, stage=STAGE, job_id=item_id, run_id=run_id,
        resource_class="MEDIUM" if stage == CORPUS else "LARGE",
        estimated_peak_ram_bytes=ram, cpu_weight=cpu, inner_threads=1,
        gpu_required=False, concurrency_group="SELECTOR_TREE",
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity=EXECUTION_CONTRACT_VERSION,
        attempt_identity=attempt_identity, lightweight=False,
        safe_to_colocate=True,
    )


def execute_news_data_compute_run(
    *, plan: NewsDataMaterialisationPlan, adapter: NewsDataComputeAdapter,
    machine_profile: MachineProfile, lease_ledger: ResourceLeaseLedger,
    runs_root: Path, registry_path: Path,
) -> dict[str, Any]:
    run_id = deterministic_news_data_run_id(plan)
    stage_units = []
    for stage in plan.selected_stages:
        configured = (
            plan.corpus_work_units if stage == CORPUS else plan.feature_work_units
        )
        units = _units(configured)
        for unit in units:
            stage_units.append((stage, unit))
    inventory = [
        {
            "item_id": deterministic_news_data_item_id(
                run_id=run_id, stage=stage, work_unit=unit, plan=plan
            ),
            "ordered_position": position,
            "stage_kind": stage,
            "authoritative_work_unit": unit,
            "canonical_parent_identity": (
                plan.canonical_parent_identity if stage == FEATURES else None
            ),
        }
        for position, (stage, unit) in enumerate(stage_units)
    ]
    manifest = build_run_manifest(
        run_id=run_id, pipeline=PIPELINE, stage=STAGE,
        run_purpose="Materialise authoritative canonical news data and PIT features",
        source_git_commit=plan.source_git_commit,
        configuration_identity=checksum({
            "contract": EXECUTION_CONTRACT_VERSION,
            "selected_stages": plan.selected_stages,
        }),
        configuration_checksum=plan.configuration_checksum,
        machine_profile_identity=machine_profile.logical_checksum,
        requested_resource_profile_identity=checksum({
            CORPUS: {"ram": 4 * GIB, "cpu": 1},
            FEATURES: {"ram": 6 * GIB, "cpu": 2},
            "estimate_source": "CONSERVATIVE_DEFAULT", "gpu": False,
        }),
        parent_input_artifacts=[
            {"identity": plan.source_inventory_identity,
             "checksum": plan.source_inventory_checksum,
             "type": "SOURCE_PROVIDER_INVENTORY"},
            {"identity": plan.canonical_parent_identity,
             "checksum": plan.canonical_parent_checksum,
             "type": "CANONICAL_CORPUS_PARENT"},
        ],
        expected_inventory=inventory,
    )
    run_root = initialise_run(manifest, runs_root=runs_root)
    initial = update_run_status(
        run_root, expected_revision=_revision(run_root), inputs_valid=True
    )
    requests: list[ResourceRequest] = []
    telemetry: list[dict[str, Any]] = []
    lease_outcomes = []
    results = []
    counts = {"planned": len(stage_units), "reused": 0, "materialised": 0,
              "failed": 0}
    stage_status = {CORPUS: "NOT_REQUESTED", FEATURES: "NOT_REQUESTED"}
    validation_status = {CORPUS: "NOT_RUN", FEATURES: "NOT_RUN"}
    corpus_parent: dict[str, Any] = {
        "canonical_corpus_identity": plan.canonical_parent_identity,
        "canonical_corpus_checksum": plan.canonical_parent_checksum,
        "compatibility_identity": plan.canonical_output_compatibility_identity,
    }
    corpus_resolved = bool(corpus_parent["canonical_corpus_identity"])

    for position, (stage, unit) in enumerate(stage_units):
        item_id = inventory[position]["item_id"]
        attempt = checksum({"run_id": run_id, "item_id": item_id,
                            "unit": unit,
                            "contract": EXECUTION_CONTRACT_VERSION})
        if stage == FEATURES and not corpus_resolved:
            item = _failed_item(
                manifest, item_id, position, attempt,
                "CANONICAL_CORPUS_PARENT_UNRESOLVED",
                "Feature-store materialisation requires an exact compatible corpus",
            )
            counts["failed"] += 1
            stage_status[stage] = "FAILED"
            publish_item_status(run_root, item)
            results.append(_result(manifest, item, stage, unit))
            continue
        try:
            compatible = _observe(
                telemetry, run_id, item_id, "compatibility_resume_resolution",
                lambda: adapter.compatible(stage, unit, corpus_parent),
            )
        except BaseException as exc:
            item = _failed_item(
                manifest, item_id, position, attempt,
                "COMPATIBILITY_RESOLUTION_FAILED", _error(exc),
            )
            counts["failed"] += 1
            stage_status[stage] = "FAILED"
            publish_item_status(run_root, item)
            results.append(_result(manifest, item, stage, unit))
            if stage == CORPUS:
                corpus_resolved = False
            continue
        if compatible is not None:
            if stage == FEATURES:
                _require_feature_ancestry(compatible, corpus_parent, plan)
            if stage == CORPUS:
                corpus_parent = _corpus_reference(compatible, plan)
                corpus_resolved = True
            counts["reused"] += 1
            stage_status[stage] = "SKIPPED_COMPATIBLE"
            validation_status[stage] = str(
                compatible.get("validation_status") or "AUTHORITATIVE_COMPLETE"
            )
            item = _complete_item(
                manifest, item_id, position, attempt, "SKIPPED_COMPATIBLE",
                compatible, None, None,
            )
            publish_item_status(run_root, item)
            results.append(_result(manifest, item, stage, unit))
            continue

        request = build_news_data_resource_request(
            run_id=run_id, item_id=item_id, stage=stage,
            attempt_identity=attempt,
        )
        requests.append(request)
        lease = None
        phase = "lease_acquisition"
        try:
            lease, _ = lease_ledger.request_persisted_lease(request)
            if lease.status != LeaseStatus.GRANTED:
                raise RuntimeError(
                    "News data resource request was not granted: "
                    + ",".join(lease.blocked_reasons)
                )
            guarded = _observe(
                telemetry, run_id, item_id,
                "guarded_compatibility_recheck",
                lambda: adapter.compatible(stage, unit, corpus_parent),
            )
            if guarded is not None:
                if stage == FEATURES:
                    _require_feature_ancestry(guarded, corpus_parent, plan)
                output = guarded
                status = "SKIPPED_COMPATIBLE"
                counts["reused"] += 1
            elif stage == CORPUS:
                phase = "input_inventory_resolution"
                resolved = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.resolve_corpus_input(unit),
                )
                phase = "source_reading"
                source = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.read_corpus_source(resolved),
                )
                phase = "canonicalisation"
                canonical = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.canonicalise(source),
                )
                phase = "identity_deduplication_validation"
                identity_validation = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.validate_corpus_identity(canonical),
                )
                _require_passed(identity_validation, "corpus identity")
                phase = "availability_time_validation"
                availability_validation = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.validate_corpus_availability(canonical),
                )
                _require_passed(availability_validation, "availability-time")
                validation_status[stage] = "PASSED"
                phase = "atomic_publication"
                output = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.publish_corpus(canonical),
                )
                status = "COMPLETE"
                counts["materialised"] += 1
            else:
                phase = "corpus_parent_resolution"
                parent = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.resolve_corpus_parent(corpus_parent, unit),
                )
                phase = "pit_filtering_join_preparation"
                prepared = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.prepare_pit_inputs(parent, unit),
                )
                phase = "feature_calculation"
                features = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.calculate_features(prepared),
                )
                phase = "feature_store_validation"
                feature_validation = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.validate_feature_store(features),
                )
                _require_passed(feature_validation, "PIT feature-store")
                validation_status[stage] = "PASSED"
                phase = "atomic_publication"
                output = _observe(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.publish_feature_store(features),
                )
                _require_feature_ancestry(output, corpus_parent, plan)
                status = "COMPLETE"
                counts["materialised"] += 1
            if stage == CORPUS:
                corpus_parent = _corpus_reference(output, plan)
                corpus_resolved = True
            stage_status[stage] = status
            item = _complete_item(
                manifest, item_id, position, attempt, status, output,
                request, lease.logical_identity,
            )
            lease_ledger.release_persisted_lease(
                lease.logical_identity, attempt_identity=attempt,
                reason="SUCCESS" if status == "COMPLETE"
                else "COMPATIBLE_SKIP_AFTER_ACQUISITION",
            )
            lease_outcomes.append({
                "item_id": item_id, "lease_identity": lease.logical_identity,
                "outcome": "RELEASED_SUCCESS" if status == "COMPLETE"
                else "RELEASED_COMPATIBLE_SKIP",
            })
        except BaseException as exc:
            counts["failed"] += 1
            stage_status[stage] = "FAILED"
            if stage == CORPUS:
                corpus_resolved = False
            if lease is not None:
                try:
                    lease_ledger.fail_persisted_lease(
                        lease.logical_identity, attempt_identity=attempt,
                        reason=f"{phase}:{type(exc).__name__}",
                        startup_failure=phase in {
                            "lease_acquisition", "input_inventory_resolution",
                            "corpus_parent_resolution",
                        },
                    )
                    lease_outcomes.append({
                        "item_id": item_id,
                        "lease_identity": lease.logical_identity,
                        "outcome": "FAILED_RELEASED", "phase": phase,
                    })
                except ValueError:
                    pass
            item = _failed_item(
                manifest, item_id, position, attempt,
                f"{stage}_{phase}_FAILED", _error(exc),
                request=request,
                lease_identity=lease.logical_identity if lease else None,
            )
        publish_item_status(run_root, item)
        results.append(_result(manifest, item, stage, unit))

    atomic_write_json(run_root / "resource_requests.json", {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "requests": [asdict(row) for row in requests],
    })
    atomic_write_json(run_root / "telemetry_spans.json", {
        "contract_version": EXECUTION_CONTRACT_VERSION, "spans": telemetry,
    })
    resource_summary = {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "requests_created": len(requests), "lease_outcomes": lease_outcomes,
        "active_lease_count": len(
            lease_ledger.read_ledger_status().get("active_leases", [])
        ),
        "resource_profiles": {
            CORPUS: {"ram_bytes": 4 * GIB, "cpu_weight": 1,
                     "inner_threads": 1, "gpu_required": False},
            FEATURES: {"ram_bytes": 6 * GIB, "cpu_weight": 2,
                       "inner_threads": 1, "gpu_required": False},
        },
    }
    atomic_write_json(run_root / "resource_summary.json", resource_summary)
    final = update_run_status(
        run_root, expected_revision=int(initial["state_revision"]),
        inputs_valid=True, resource_evidence={
            "reserved_ram_bytes": 0, "active_cpu_weight": 0,
        },
    )
    publish_results_snapshot(run_root, results)
    summary = {
        "requested_stages": list(plan.selected_stages),
        "planned_work_items": counts["planned"],
        "reused_skipped_items": counts["reused"],
        "newly_materialised_items": counts["materialised"],
        "failed_items": counts["failed"],
        "corpus_status": stage_status[CORPUS],
        "feature_store_status": stage_status[FEATURES],
        "validation_status": validation_status,
        "lease_outcomes": lease_outcomes,
        "resource_profiles": resource_summary["resource_profiles"],
        "corpus_parent_identity": corpus_parent.get(
            "canonical_corpus_identity"),
        "feature_store_parent_identities": {
            "canonical_corpus_identity": plan.canonical_parent_identity,
            "universe_identity": plan.universe_identity,
            "date_boundary_identity": plan.date_boundary_identity,
            "availability_policy_identity": plan.availability_policy_identity,
        },
        "final_run_status": final["current_status"],
    }
    atomic_write_json(run_root / "news_data_summary.json", summary)
    publish_summary(run_root)
    registry = update_global_registry_snapshot(
        run_root, registry_path=registry_path
    )
    return {
        "run_id": run_id, "run_identity": manifest["run_identity"],
        "run_root": str(run_root), "summary": summary,
        "resource_summary": resource_summary, "registry": registry,
        "resource_requests": requests,
    }


def _units(configured: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    units = [dict(row) for row in configured] or [{"work_unit_id": "stage"}]
    if any(not str(row.get("work_unit_id") or "").strip() for row in units):
        raise ValueError("Authoritative work-unit identity is required")
    if len({str(row["work_unit_id"]) for row in units}) != len(units):
        raise ValueError("Duplicate authoritative work-unit identity")
    return sorted(units, key=lambda row: str(row["work_unit_id"]))


def _complete_item(
    manifest: Mapping[str, Any], item_id: str, position: int, attempt: str,
    status: str, output: Mapping[str, Any],
    request: ResourceRequest | None, lease_identity: str | None,
) -> dict[str, Any]:
    reference = _bounded_reference(output)
    artifact = str(
        output.get("artifact_identity")
        or output.get("canonical_corpus_identity")
        or output.get("feature_store_artifact_checksum")
        or checksum(reference)
    )
    return build_item_status(
        run_identity=manifest["run_identity"], item_id=item_id,
        ordered_position=position, pipeline=PIPELINE, stage=STAGE,
        attempt_identity=attempt, status=status, required_artifact_kind="NONE",
        stage_artifact_identity=artifact,
        artifact_validation={"stage_artifact_valid": True,
                             "authoritative_reference": reference},
        resource_request_identity=request.logical_checksum if request else None,
        lease_identity=lease_identity,
        compatible_skip_evidence=reference
        if status == "SKIPPED_COMPATIBLE" else None,
    )


def _failed_item(
    manifest: Mapping[str, Any], item_id: str, position: int, attempt: str,
    code: str, reason: str, *, request: ResourceRequest | None = None,
    lease_identity: str | None = None,
) -> dict[str, Any]:
    return build_item_status(
        run_identity=manifest["run_identity"], item_id=item_id,
        ordered_position=position, pipeline=PIPELINE, stage=STAGE,
        attempt_identity=attempt, status="FAILED", failure_code=code,
        failure_reason=reason, retryable=True,
        resource_request_identity=request.logical_checksum if request else None,
        lease_identity=lease_identity,
    )


def _result(
    manifest: Mapping[str, Any], item: Mapping[str, Any], stage: str,
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(
        item.get("artifact_validation", {})
        .get("authoritative_reference", {}).get("row_count") or 0
    )
    metric = metric_value(
        "row_count", count, unit="rows",
        population_identity=str(unit["work_unit_id"]),
        direction="INFORMATIONAL", availability="AVAILABLE",
        source_artifact_identity=item.get("stage_artifact_identity"),
    )
    return build_result_record(
        result_identity=checksum({"item": item["item_id"],
                                  "status": item["status"]}),
        run_identity=manifest["run_identity"], item_identity=item["item_id"],
        result_kind="DATA_STAGE", pipeline=PIPELINE, stage=stage,
        status=item["status"],
        artifact_identities=[item["stage_artifact_identity"]]
        if item.get("stage_artifact_identity") else [],
        metrics={"row_count": metric}, counts={"rows": count},
        dimensions={"work_unit_identity": unit["work_unit_id"],
                    "non_model_stage": True},
    )


def _corpus_reference(
    output: Mapping[str, Any], plan: NewsDataMaterialisationPlan,
) -> dict[str, Any]:
    identity = output.get("canonical_corpus_identity")
    artifact_checksum = output.get("canonical_corpus_checksum")
    compatibility = (
        output.get("compatibility_identity")
        or output.get("logical_manifest_checksum")
        or plan.canonical_output_compatibility_identity
    )
    if not identity or not artifact_checksum or not compatibility:
        raise ValueError("Authoritative canonical corpus reference is incomplete")
    return {
        "canonical_corpus_identity": identity,
        "canonical_corpus_checksum": artifact_checksum,
        "compatibility_identity": compatibility,
    }


def _require_feature_ancestry(
    output: Mapping[str, Any], corpus_parent: Mapping[str, Any],
    plan: NewsDataMaterialisationPlan,
) -> None:
    required = {
        "canonical_corpus_identity":
            corpus_parent.get("canonical_corpus_identity"),
        "canonical_corpus_checksum":
            corpus_parent.get("canonical_corpus_checksum"),
        "pit_eligibility_policy_identity": plan.availability_policy_identity,
    }
    if any(output.get(key) != value for key, value in required.items()):
        raise ValueError("PIT feature-store canonical/PIT ancestry is incompatible")


def _require_passed(result: Mapping[str, Any], owner: str) -> None:
    if result.get("passed") is not True:
        raise ValueError(f"Authoritative {owner} validation failed")


def _observe(
    spans: list[dict[str, Any]], run_id: str, item_id: str, name: str,
    operation: Callable[[], Any],
) -> Any:
    try:
        value = operation()
    except BaseException as exc:
        spans.append(_span(run_id, item_id, name, False, {
            "exception_type": type(exc).__name__,
        }))
        raise
    metadata = value if isinstance(value, Mapping) else {}
    spans.append(_span(run_id, item_id, name, True, metadata))
    return value


def _span(
    run_id: str, item_id: str, name: str, success: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "input_rows", "output_rows", "row_count", "article_count",
        "provider_count", "duplicate_rows_removed", "duplicate_count",
        "invalid_rows", "symbol_count", "date_count", "partition_count",
        "publication_result", "manifest_publication_result",
        "validation_status", "passed", "exception_type",
    }
    bounded = {key: metadata[key] for key in sorted(allowed)
               if key in metadata and not isinstance(
                   metadata[key], (dict, list, tuple))}
    return {
        "span_identity": checksum({"run": run_id, "item": item_id,
                                   "phase": name}),
        "run_id": run_id, "item_id": item_id, "phase": name,
        "status": "SUCCESS" if success else "FAILED", "metadata": bounded,
    }


def _bounded_reference(output: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "artifact_identity", "canonical_corpus_identity",
        "canonical_corpus_checksum", "logical_manifest_checksum",
        "canonical_schema_version", "canonical_schema_checksum",
        "canonical_row_count", "source_row_count", "duplicate_group_count",
        "publication_result", "manifest_path", "artifact_path",
        "feature_store_artifact_checksum", "logical_checksum",
        "feature_store_contract", "feature_schema_checksum", "row_count",
        "canonical_corpus_identity", "canonical_corpus_checksum",
        "canonical_daily_spine_identity", "ticker_mapping_identity",
        "pit_eligibility_policy_identity", "manifest_publication_result",
        "validation_status", "provider_count", "symbol_count",
        "date_min", "date_max", "availability_min", "availability_max",
    }
    return {key: output[key] for key in sorted(allowed) if key in output}


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _revision(run_root: Path) -> int:
    path = run_root / "run_status.json"
    return (
        int(json.loads(path.read_text(encoding="utf-8"))["state_revision"])
        if path.exists() else -1
    )
