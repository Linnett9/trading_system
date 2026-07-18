from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from core.research.ml.stock_level.stock_alpha_finbert_score_store import (
    _validate_plan,
)

PIPELINE = "stock_alpha_news"
STAGE = "finbert_scoring"
EXECUTION_CONTRACT_VERSION = "stock_alpha_finbert_compute_execution.v1"
CERTIFICATION_STAGE = "score_store_certification"
MODEL_REFERENCE_TYPE = "EXTERNAL_PINNED_MODEL_REFERENCE"


@dataclass(frozen=True)
class FinBertExecutionPolicy:
    device: str = "cpu"
    allow_gpu: bool = False

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("FinBERT execution device must be cpu or cuda")
        if self.device == "cuda" and not self.allow_gpu:
            raise ValueError("CUDA execution requires explicit allow_gpu policy")


class FinBertChunkComputeAdapter(Protocol):
    """Boundary implemented by the authoritative scoring execution owner."""

    def compatible_output(
        self, chunk: Mapping[str, Any]
    ) -> Mapping[str, Any] | None: ...

    def load_model(
        self, model_reference: Mapping[str, Any], policy: FinBertExecutionPolicy
    ) -> Any: ...

    def tokenize(
        self, model: Any, chunk: Mapping[str, Any]
    ) -> Any: ...

    def infer(
        self, model: Any, tokenized: Any, chunk: Mapping[str, Any]
    ) -> Any: ...

    def publish(
        self, chunk: Mapping[str, Any], predictions: Any
    ) -> Mapping[str, Any]: ...


Certification = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def deterministic_run_id(
    scoring_plan: Mapping[str, Any],
    policy: FinBertExecutionPolicy = FinBertExecutionPolicy(),
) -> str:
    _validate_plan(scoring_plan)
    identity = checksum(
        {
            "execution_contract": EXECUTION_CONTRACT_VERSION,
            "scoring_plan_identity": scoring_plan["logical_checksum"],
            "scoring_plan_artifact_checksum": scoring_plan[
                "plan_artifact_checksum"
            ],
            "model": scoring_plan["finbert_model_identity"],
            "inference_contract": scoring_plan["inference_contract"],
            "configuration_checksum": scoring_plan["configuration_checksum"],
            "execution_policy": asdict(policy),
        }
    )
    return f"finbert-{identity[:24]}"


def deterministic_chunk_item_id(
    run_id: str, chunk: Mapping[str, Any], scoring_plan: Mapping[str, Any]
) -> str:
    return "chunk-" + checksum(
        {
            "run_id": run_id,
            "authoritative_chunk_id": chunk["chunk_id"],
            "authoritative_chunk_identity": chunk["identity"],
            "pinned_model": scoring_plan["finbert_model_identity"],
            "execution_contract": EXECUTION_CONTRACT_VERSION,
        }
    )[:32]


def build_chunk_resource_request(
    *,
    run_id: str,
    item_id: str,
    attempt_identity: str,
    policy: FinBertExecutionPolicy,
) -> ResourceRequest:
    return ResourceRequest(
        pipeline=PIPELINE,
        stage=STAGE,
        job_id=item_id,
        run_id=run_id,
        resource_class="LARGE",
        estimated_peak_ram_bytes=10 * GIB,
        cpu_weight=2,
        inner_threads=1,
        gpu_required=policy.device == "cuda",
        concurrency_group="NEWS_TRANSFORMER",
        estimate_source="CONSERVATIVE_DEFAULT",
        estimate_evidence_identity=EXECUTION_CONTRACT_VERSION,
        attempt_identity=attempt_identity,
        safe_to_colocate=False,
    )


def execute_finbert_compute_run(
    *,
    scoring_plan: Mapping[str, Any],
    adapter: FinBertChunkComputeAdapter,
    certify: Certification,
    machine_profile: MachineProfile,
    lease_ledger: ResourceLeaseLedger,
    runs_root: Path,
    registry_path: Path,
    policy: FinBertExecutionPolicy = FinBertExecutionPolicy(),
) -> dict[str, Any]:
    """Execute one exact plan without owning FinBERT scoring semantics.

    The adapter must use the existing scoring owner's compatibility and atomic
    publication rules.  This function owns only compute lifecycle records.
    """
    _validate_plan(scoring_plan)
    plan = dict(scoring_plan)
    run_id = deterministic_run_id(plan, policy)
    chunks = list(plan["expected_chunks"])
    model_reference = _external_model_reference(plan)
    inventory = [
        {
            "item_id": deterministic_chunk_item_id(run_id, chunk, plan),
            "ordered_position": index,
            "authoritative_chunk_id": chunk["chunk_id"],
            "authoritative_chunk_identity": chunk["identity"],
            "execution_contract": EXECUTION_CONTRACT_VERSION,
            "pinned_model_reference": model_reference["artifact_identity"],
            "item_kind": "MODEL_STAGE",
        }
        for index, chunk in enumerate(chunks)
    ]
    certification_item_id = "certification-" + checksum(
        {
            "run_id": run_id,
            "stage": CERTIFICATION_STAGE,
            "plan": plan["logical_checksum"],
        }
    )[:32]
    inventory.append(
        {
            "item_id": certification_item_id,
            "ordered_position": len(inventory),
            "item_kind": "NON_MODEL_CERTIFICATION",
            "execution_contract": EXECUTION_CONTRACT_VERSION,
        }
    )
    manifest = build_run_manifest(
        run_id=run_id,
        pipeline=PIPELINE,
        stage=STAGE,
        run_purpose="Execute and certify one authoritative FinBERT scoring plan",
        source_git_commit=str(
            plan.get("source_code_commit") or "plan-source-unspecified"
        ),
        configuration_identity=str(plan["logical_checksum"]),
        configuration_checksum=str(plan["configuration_checksum"]),
        machine_profile_identity=machine_profile.logical_checksum,
        requested_resource_profile_identity=checksum(
            {
                "ram_bytes": 10 * GIB,
                "cpu_weight": 2,
                "estimate_source": "CONSERVATIVE_DEFAULT",
                "execution_policy": asdict(policy),
            }
        ),
        parent_input_artifacts=[
            {
                "artifact_type": "FINBERT_SCORING_PLAN",
                "identity": plan["logical_checksum"],
                "checksum": plan["plan_artifact_checksum"],
            },
            model_reference,
        ],
        expected_inventory=inventory,
    )
    run_root = initialise_run(manifest, runs_root=runs_root)
    state = update_run_status(
        run_root, expected_revision=_status_revision(run_root), inputs_valid=True
    )
    requests: list[ResourceRequest] = []
    results = []
    telemetry: list[dict[str, Any]] = []
    counters = {"planned": len(chunks), "reused": 0, "scored": 0, "failed": 0}
    lease_outcomes: list[dict[str, Any]] = []

    for position, chunk in enumerate(chunks):
        item_id = inventory[position]["item_id"]
        attempt = checksum(
            {
                "run_id": run_id,
                "item_id": item_id,
                "chunk": chunk["identity"],
                "contract": EXECUTION_CONTRACT_VERSION,
            }
        )
        request = build_chunk_resource_request(
            run_id=run_id,
            item_id=item_id,
            attempt_identity=attempt,
            policy=policy,
        )
        requests.append(request)
        output = adapter.compatible_output(chunk)
        if output is not None:
            counters["reused"] += 1
            telemetry.append(
                _span(run_id, item_id, "compatible_resume_check", True, {
                    "reuse": True, "row_count": output.get("row_count")
                })
            )
            item = _completed_item(
                manifest, item_id, position, attempt, "SKIPPED_COMPATIBLE",
                output, request, None,
            )
        else:
            lease = None
            phase = "lease_acquisition"
            try:
                lease, _ = lease_ledger.request_persisted_lease(request)
                if lease.status != LeaseStatus.GRANTED:
                    raise RuntimeError(
                        "FinBERT resource request was not granted: "
                        + ",".join(lease.blocked_reasons)
                    )
                phase = "base_model_loading"
                model = _observed(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.load_model(model_reference, policy),
                )
                phase = "tokenisation"
                tokenized = _observed(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.tokenize(model, chunk),
                    {"input_row_count": chunk.get("article_count")},
                )
                phase = "inference"
                predictions = _observed(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.infer(model, tokenized, chunk),
                    {"input_row_count": chunk.get("article_count")},
                )
                phase = "atomic_chunk_publication"
                output = _observed(
                    telemetry, run_id, item_id, phase,
                    lambda: adapter.publish(chunk, predictions),
                    {"publication_requested": True},
                )
                counters["scored"] += 1
                item = _completed_item(
                    manifest, item_id, position, attempt, "COMPLETE",
                    output, request, lease.logical_identity,
                )
                lease_ledger.release_persisted_lease(
                    lease.logical_identity,
                    attempt_identity=attempt,
                    reason="SUCCESS",
                )
                lease_outcomes.append(
                    {"item_id": item_id, "lease_identity": lease.logical_identity,
                     "outcome": "RELEASED_SUCCESS"}
                )
            except BaseException as exc:
                counters["failed"] += 1
                if lease is not None:
                    try:
                        lease_ledger.fail_persisted_lease(
                            lease.logical_identity,
                            attempt_identity=attempt,
                            reason=f"{phase}:{type(exc).__name__}",
                            startup_failure=phase == "base_model_loading",
                        )
                        lease_outcomes.append(
                            {"item_id": item_id,
                             "lease_identity": lease.logical_identity,
                             "outcome": "FAILED_RELEASED", "phase": phase}
                        )
                    except ValueError:
                        pass
                item = build_item_status(
                    run_identity=manifest["run_identity"], item_id=item_id,
                    ordered_position=position, pipeline=PIPELINE, stage=STAGE,
                    attempt_identity=attempt, status="FAILED",
                    failure_code=f"FINBERT_{phase.upper()}_FAILED",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    retryable=True, resource_request_identity=request.logical_checksum,
                    lease_identity=lease.logical_identity if lease else None,
                )
        publish_item_status(run_root, item)
        results.append(_result(manifest, item, chunk))

    certification_status = "INCOMPLETE"
    certification_output: Mapping[str, Any] | None = None
    cert_attempt = checksum(
        {"run_id": run_id, "item_id": certification_item_id,
         "plan": plan["logical_checksum"]}
    )
    if counters["failed"]:
        cert_item = build_item_status(
            run_identity=manifest["run_identity"], item_id=certification_item_id,
            ordered_position=len(chunks), pipeline=PIPELINE, stage=STAGE,
            attempt_identity=cert_attempt, status="INCOMPLETE",
            blocker_code="SCORING_CHUNK_FAILURE",
            blocker_reason="Certification requires all planned scoring chunks",
        )
    else:
        try:
            certification_output = dict(certify(plan))
            certification_status = str(
                certification_output.get("status")
                or ("COMPLETE" if certification_output.get(
                    "production_scoring_complete") else "INCOMPLETE")
            )
            successful = (
                certification_status == "COMPLETE"
                and certification_output.get("production_scoring_complete", True)
            )
            cert_item = build_item_status(
                run_identity=manifest["run_identity"],
                item_id=certification_item_id, ordered_position=len(chunks),
                pipeline=PIPELINE, stage=STAGE, attempt_identity=cert_attempt,
                status="COMPLETE" if successful else "FAILED",
                required_artifact_kind="NONE",
                stage_artifact_identity=certification_output.get(
                    "score_store_identity"),
                artifact_validation={
                    "stage_artifact_valid": successful,
                    "authoritative_reference": _bounded_reference(
                        certification_output
                    ),
                },
                failure_code=None if successful else "CERTIFICATION_INCOMPLETE",
                failure_reason=None if successful else "Score-store certification failed",
            )
        except BaseException as exc:
            certification_status = "FAILED"
            cert_item = build_item_status(
                run_identity=manifest["run_identity"],
                item_id=certification_item_id, ordered_position=len(chunks),
                pipeline=PIPELINE, stage=STAGE, attempt_identity=cert_attempt,
                status="FAILED", failure_code="CERTIFICATION_FAILED",
                failure_reason=f"{type(exc).__name__}: {exc}", retryable=True,
            )
    publish_item_status(run_root, cert_item)
    results.append(_certification_result(manifest, cert_item, certification_output))

    atomic_write_json(run_root / "resource_requests.json", {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "requests": [asdict(row) for row in requests],
    })
    atomic_write_json(run_root / "telemetry_spans.json", {
        "contract_version": EXECUTION_CONTRACT_VERSION, "spans": telemetry,
    })
    resource_summary = {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "requests_created": len(requests),
        "lease_outcomes": lease_outcomes,
        "active_lease_count": len(lease_ledger.read_ledger_status().get(
            "active_leases", [])),
    }
    atomic_write_json(run_root / "resource_summary.json", resource_summary)
    final = update_run_status(
        run_root, expected_revision=int(state["state_revision"]),
        inputs_valid=True, resource_evidence={
            "reserved_ram_bytes": 0,
            "active_cpu_weight": 0,
        },
    )
    publish_results_snapshot(run_root, results)
    summary = {
        "planned_chunks": counters["planned"],
        "compatible_reused_chunks": counters["reused"],
        "newly_scored_chunks": counters["scored"],
        "failed_chunks": counters["failed"],
        "certification_status": certification_status,
        "lease_outcomes": lease_outcomes,
        "model_execution_policy": asdict(policy),
        "pinned_external_model_reference": model_reference,
        "final_run_status": final["current_status"],
    }
    atomic_write_json(run_root / "finbert_summary.json", summary)
    publish_summary(run_root)
    registry = update_global_registry_snapshot(
        run_root, registry_path=registry_path
    )
    return {
        "run_id": run_id,
        "run_identity": manifest["run_identity"],
        "run_root": str(run_root),
        "resource_requests": requests,
        "summary": summary,
        "resource_summary": resource_summary,
        "registry": registry,
        "model_reference": model_reference,
    }


def _external_model_reference(plan: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(plan["finbert_model_identity"])
    payload = {
        "artifact_type": MODEL_REFERENCE_TYPE,
        "provider": "huggingface",
        "repository": model["model_id"],
        "revision": model["model_revision"],
        "tokenizer_repository": model["tokenizer_id"],
        "tokenizer_revision": model["tokenizer_revision"],
        "configuration_checksum": plan["configuration_checksum"],
        "scoring_configuration": {
            "inference_contract": plan["inference_contract"],
            "maximum_token_length": plan["maximum_token_length"],
        },
        "resolution_policy": "reference_only_no_weights_copied",
    }
    return {**payload, "artifact_identity": checksum(payload)}


def _completed_item(
    manifest: Mapping[str, Any], item_id: str, position: int, attempt: str,
    status: str, output: Mapping[str, Any], request: ResourceRequest,
    lease_identity: str | None,
) -> dict[str, Any]:
    reference = _bounded_reference(output)
    identity = str(
        output.get("artifact_identity") or output.get("chunk_artifact_sha256")
        or checksum(reference)
    )
    return build_item_status(
        run_identity=manifest["run_identity"], item_id=item_id,
        ordered_position=position, pipeline=PIPELINE, stage=STAGE,
        attempt_identity=attempt, status=status,
        required_artifact_kind="NONE", stage_artifact_identity=identity,
        artifact_validation={
            "stage_artifact_valid": True,
            "authoritative_reference": reference,
        },
        resource_request_identity=request.logical_checksum,
        lease_identity=lease_identity,
        compatible_skip_evidence=reference if status == "SKIPPED_COMPATIBLE" else None,
    )


def _result(
    manifest: Mapping[str, Any], item: Mapping[str, Any],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = item.get("stage_artifact_identity")
    rows = int(chunk.get("article_count") or 0)
    metric = metric_value(
        "row_count", rows, unit="rows",
        population_identity=str(chunk["chunk_id"]),
        direction="INFORMATIONAL", availability="AVAILABLE",
        source_artifact_identity=artifact,
    )
    return build_result_record(
        result_identity=checksum({"item": item["item_id"], "status": item["status"]}),
        run_identity=manifest["run_identity"], item_identity=item["item_id"],
        result_kind="NEWS_SCORING", pipeline=PIPELINE, stage=STAGE,
        status=item["status"], artifact_identities=[artifact] if artifact else [],
        metrics={"row_count": metric}, counts={"rows": rows},
    )


def _certification_result(
    manifest: Mapping[str, Any], item: Mapping[str, Any],
    output: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact = item.get("stage_artifact_identity")
    return build_result_record(
        result_identity=checksum({"item": item["item_id"], "status": item["status"]}),
        run_identity=manifest["run_identity"], item_identity=item["item_id"],
        result_kind="GENERIC_STAGE", pipeline=PIPELINE,
        stage=CERTIFICATION_STAGE, status=item["status"],
        artifact_identities=[artifact] if artifact else [], metrics={},
        dimensions={"no_model_resource": True,
                    "certification_reference": _bounded_reference(output or {})},
    )


def _observed(
    spans: list[dict[str, Any]], run_id: str, item_id: str, name: str,
    operation: Callable[[], Any], metadata: Mapping[str, Any] | None = None,
) -> Any:
    try:
        result = operation()
    except BaseException as exc:
        spans.append(_span(run_id, item_id, name, False, {
            **dict(metadata or {}), "error_type": type(exc).__name__
        }))
        raise
    spans.append(_span(run_id, item_id, name, True, metadata or {}))
    return result


def _span(
    run_id: str, item_id: str, name: str, success: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    bounded = {
        key: value for key, value in metadata.items()
        if key in {
            "reuse", "row_count", "input_row_count", "scored_row_count",
            "batch_count", "publication_requested", "publication_result",
            "error_type",
        }
    }
    return {
        "span_identity": checksum(
            {"run_id": run_id, "item_id": item_id, "name": name,
             "ordinal": len(bounded)}
        ),
        "run_id": run_id, "item_id": item_id, "name": name,
        "status": "SUCCESS" if success else "FAILED", "metadata": bounded,
    }


def _bounded_reference(output: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "artifact_identity", "chunk_id", "chunk_path", "chunk_artifact_sha256",
        "scored_rows_logical_checksum", "chunk_metadata_logical_checksum",
        "row_count", "article_count", "publication_result", "status",
        "score_store_identity", "score_store_checksum",
        "logical_manifest_checksum", "certificate_artifact_checksum",
        "output_path",
    }
    return {key: output[key] for key in sorted(allowed) if key in output}


def _status_revision(run_root: Path) -> int:
    path = run_root / "run_status.json"
    if not path.exists():
        return -1
    return int(json.loads(path.read_text(encoding="utf-8"))["state_revision"])
