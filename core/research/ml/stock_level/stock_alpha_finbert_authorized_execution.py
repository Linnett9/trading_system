from __future__ import annotations

import math
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
    deterministic_run_id,
    logical_identity,
    validate_authorization,
    validate_model_package,
    validate_request,
)

COMPLETE_STATUS = "FINBERT_SCORING_COMPLETE_AWAITING_SCORE_STORE_CERTIFICATION"
SCORE_LABELS = {"positive", "negative", "neutral"}


def immutable_model_activation_options(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": request["model_name"],
        "tokenizer_id": request["model_name"],
        "model_revision": request["model_revision"],
        "tokenizer_revision": request["model_revision"],
        "model_path": request["model_package_root"],
        "tokenizer_path": request["model_package_root"],
        "device": "cpu",
        "max_token_length": request["maximum_sequence_length"],
        "local_files_only": True,
        "cache_dir": request["model_package_root"],
    }


class Lease(Protocol):
    def release_success(self) -> None: ...
    def close_failure(self, code: str) -> None: ...


@dataclass(frozen=True)
class ExecutionCallbacks:
    persist_resource_request: Callable[[str], Any]
    acquire_lease: Callable[[str], Lease]
    activate_source: Callable[[], Mapping[str, Any]]
    activate_tokenizer: Callable[[], Any]
    activate_model: Callable[[], Any]
    compatible_chunk: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
    infer_chunk: Callable[
        [Mapping[str, Any], Mapping[str, Any], Any, Any], Sequence[Mapping[str, Any]]
    ]
    publish_chunk: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]
    ]
    publish_failure: Callable[[str, str], None]
    publish_completion: Callable[[Mapping[str, Any]], Mapping[str, Any]]


def execute_authorized_scoring(
    *,
    request: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_checksum: str,
    authorization: Mapping[str, Any],
    chunk_plan: Mapping[str, Any],
    callbacks: ExecutionCallbacks,
    plan_only: bool = False,
) -> dict[str, Any]:
    validate_request(request)
    if logical_identity(runtime) != runtime_checksum:
        raise ValueError("SCORING_RUNTIME_CHECKSUM_MISMATCH")
    run_id = deterministic_run_id(request, runtime_checksum)
    auth = validate_authorization(
        authorization, request, runtime_checksum=runtime_checksum,
        expected_run_id=run_id, execution_required=not plan_only,
    )
    package = validate_model_package(
        _path(request["model_package_root"]),
        expected_model_name=request["model_name"],
        expected_revision=request["model_revision"],
    )
    _validate_plan(request, chunk_plan)
    output_root = _path(request["score_output_root"])
    _validate_output_boundary(output_root, request)
    if plan_only:
        return {
            "status": "PLAN_VALIDATED_NOT_AUTHORIZED_FOR_EXECUTION",
            "run_id": run_id,
            "authorization_identity": auth["authorization_identity"],
            "model_package_identity": package["package_logical_identity"],
        }
    callbacks.persist_resource_request(run_id)
    lease = callbacks.acquire_lease(run_id)
    completed: list[Mapping[str, Any]] = []
    owners: set[tuple[str, str]] = set()
    try:
        source = callbacks.activate_source()
        if (
            source.get("canonical_row_count") != request["canonical_row_count"]
            or source.get("eligible_row_count")
            != request["expected_eligible_row_count"]
            or source.get("excluded_row_count")
            != request["expected_excluded_row_count"]
        ):
            raise ExecutionFailure("ELIGIBILITY_RECONCILIATION_FAILED")
        exclusions = list(source.get("exclusions") or [])
        _validate_exclusions(exclusions, request)
        tokenizer = callbacks.activate_tokenizer()
        model = callbacks.activate_model()
        for chunk in chunk_plan["chunks"]:
            reusable = callbacks.compatible_chunk(chunk)
            if reusable is not None:
                _validate_chunk_result(reusable, chunk, request, owners)
                completed.append(reusable)
                continue
            try:
                rows = list(
                    callbacks.infer_chunk(chunk, source, tokenizer, model)
                )
                _validate_rows(rows, chunk, request, owners)
                result = callbacks.publish_chunk(chunk, rows)
                _validate_chunk_result(result, chunk, request, set())
                completed.append(result)
            except BaseException as exc:
                code = (
                    exc.code if isinstance(exc, ExecutionFailure)
                    else "FINBERT_CHUNK_EXECUTION_FAILED"
                )
                callbacks.publish_failure(chunk["chunk_identity"], code)
                raise ExecutionFailure(code) from exc
        _validate_completion(completed, chunk_plan, request, owners, exclusions)
        completion = callbacks.publish_completion({
            "status": COMPLETE_STATUS,
            "run_id": run_id,
            "scored_rows": request["expected_eligible_row_count"],
            "governed_exclusions": request["expected_excluded_row_count"],
            "failed_rows": 0,
            "chunk_count": len(completed),
            "certified": False,
        })
        lease.release_success()
        return {"status": COMPLETE_STATUS, "run_id": run_id,
                "completion": completion}
    except BaseException as exc:
        code = exc.code if isinstance(exc, ExecutionFailure) else (
            "FINBERT_SCORING_EXECUTION_FAILED"
        )
        lease.close_failure(code)
        raise


class ExecutionFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _validate_plan(request, plan):
    if (
        plan.get("scoring_request_identity")
        != request["scoring_request_identity"]
        or plan.get("expected_eligible_row_count")
        != request["expected_eligible_row_count"]
        or plan.get("chunk_size") != request["chunk_size"]
        or plan.get("expected_chunk_count") != len(plan.get("chunks") or [])
    ):
        raise ValueError("FROZEN_CHUNK_PLAN_BINDING_MISMATCH")
    cursor = 0
    for ordinal, chunk in enumerate(plan["chunks"], 1):
        if (
            chunk["ordinal"] != ordinal
            or chunk["start_inclusive"] != cursor
            or chunk["stop_exclusive"] <= cursor
            or chunk["row_count"]
            != chunk["stop_exclusive"] - chunk["start_inclusive"]
            or chunk["status"] != "PLANNED"
        ):
            raise ValueError("FROZEN_CHUNK_PLAN_BOUNDARY_INVALID")
        cursor = chunk["stop_exclusive"]
    if cursor != request["expected_eligible_row_count"]:
        raise ValueError("FROZEN_CHUNK_PLAN_COVERAGE_MISMATCH")


def _validate_exclusions(exclusions, request):
    if len(exclusions) != request["expected_excluded_row_count"]:
        raise ExecutionFailure("GOVERNED_EXCLUSION_COUNT_MISMATCH")
    allowed = {
        "owner_identity", "provider_article_identity_hash", "symbol",
        "reason_code", "canonical_identity", "text_selection_contract",
        "exclusion_identity",
    }
    for row in exclusions:
        if (
            not {"owner_identity", "reason_code"} <= set(row)
            or not set(row) <= allowed
        ):
            raise ExecutionFailure("GOVERNED_EXCLUSION_SCOPE_INVALID")
        if not row["owner_identity"] or not str(row["reason_code"]).isupper():
            raise ExecutionFailure("GOVERNED_EXCLUSION_INVALID")


def _validate_rows(rows, chunk, request, owners):
    if len(rows) != chunk["row_count"]:
        raise ExecutionFailure("CHUNK_OUTPUT_COUNT_MISMATCH")
    for row in rows:
        owner = (str(row.get("article_id") or ""), str(row.get("symbol") or ""))
        if not all(owner) or owner in owners:
            raise ExecutionFailure("DUPLICATE_OR_BLANK_ARTICLE_SYMBOL_OWNER")
        owners.add(owner)
        probabilities = [
            row.get("positive_probability"), row.get("negative_probability"),
            row.get("neutral_probability"),
        ]
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or value > 1 for value in probabilities
        ):
            raise ExecutionFailure("INVALID_FINBERT_PROBABILITY")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise ExecutionFailure("FINBERT_PROBABILITY_NORMALIZATION_FAILED")
        if row.get("sentiment_label") not in SCORE_LABELS:
            raise ExecutionFailure("FINBERT_SENTIMENT_LABEL_INVALID")
        bindings = {
            "model_package_identity": request["model_package_identity"],
            "canonical_identity": request["canonical_identity"],
            "chunk_identity": chunk["chunk_identity"],
            "scored_at": request["scored_at_utc"],
        }
        if any(row.get(key) != value for key, value in bindings.items()):
            raise ExecutionFailure("SCORE_ROW_LINEAGE_MISMATCH")


def _validate_chunk_result(result, chunk, request, owners):
    if (
        result.get("status") != "COMPLETE"
        or result.get("chunk_identity") != chunk["chunk_identity"]
        or not result.get("output_sha256")
        or not result.get("logical_checksum")
        or result.get("row_count") != chunk["row_count"]
        or result.get("scoring_request_identity")
        != request["scoring_request_identity"]
    ):
        raise ExecutionFailure("CHUNK_COMPLETION_EVIDENCE_INVALID")
    for owner in result.get("owners") or []:
        pair = tuple(owner)
        if pair in owners:
            raise ExecutionFailure("DUPLICATE_ARTICLE_SYMBOL_ACROSS_CHUNKS")
        owners.add(pair)


def _validate_completion(completed, plan, request, owners, exclusions):
    if (
        len(completed) != plan["expected_chunk_count"]
        or sum(row["row_count"] for row in completed)
        != request["expected_eligible_row_count"]
        or len(exclusions) != request["expected_excluded_row_count"]
    ):
        raise ExecutionFailure("SCORE_STORE_COMPLETION_BOUNDARY_FAILED")


def _validate_output_boundary(output_root: Path, request):
    if not output_root.exists():
        return
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("PRODUCTION_SCORE_OUTPUT_ROOT_INVALID")
    if (output_root / "score_store_completion.json").exists():
        raise ValueError("PRODUCTION_SCORE_STORE_ALREADY_COMPLETE")
    ownership_path = output_root / "scoring_ownership.json"
    if not ownership_path.is_file():
        raise ValueError("PRODUCTION_SCORE_OUTPUT_ROOT_UNOWNED")
    ownership = json.loads(ownership_path.read_text(encoding="utf-8"))
    expected = {
        "scoring_request_identity": request["scoring_request_identity"],
        "canonical_identity": request["canonical_identity"],
        "canonical_csv_sha256": request["canonical_csv_sha256"],
        "model_package_identity": request["model_package_identity"],
        "model_revision": request["model_revision"],
        "scored_at_selection_identity":
            request["scored_at_selection_identity"],
    }
    if ownership != expected:
        raise ValueError("PRODUCTION_SCORE_OUTPUT_ROOT_INCOMPATIBLE")


def _path(value):
    return Path(value)


class PersistedLease:
    def __init__(self, ledger, lease, attempt_identity):
        self.ledger = ledger
        self.lease = lease
        self.attempt_identity = attempt_identity

    def release_success(self):
        self.ledger.release_persisted_lease(
            self.lease.logical_identity,
            attempt_identity=self.attempt_identity, reason="SUCCESS",
        )

    def close_failure(self, code):
        self.ledger.fail_persisted_lease(
            self.lease.logical_identity,
            attempt_identity=self.attempt_identity, reason=code,
            startup_failure=False,
        )


class ProductionExecutionFactory:
    """Concrete post-authorization binding to authoritative scoring owners."""

    def __init__(
        self, *, request: Mapping[str, Any], runtime: Mapping[str, Any],
        chunk_plan: Mapping[str, Any], runs_root: Path, ledger_path: Path,
        registry_path: Path, dependencies: Mapping[str, Any] | None = None,
    ):
        self.request = dict(request)
        self.runtime = dict(runtime)
        self.plan = dict(chunk_plan)
        self.runs_root = Path(runs_root)
        self.ledger_path = Path(ledger_path)
        self.registry_path = Path(registry_path)
        self.dependencies = dict(dependencies or {})
        self.events: list[str] = []
        self.source = None
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.run_root = None
        self.manifest = None
        self.results = []
        self._lease_attempt = logical_identity({
            "run": deterministic_run_id(request, logical_identity(runtime)),
            "request": request["scoring_request_identity"],
        })

    def callbacks(self) -> ExecutionCallbacks:
        return ExecutionCallbacks(
            persist_resource_request=self.persist_resource_request,
            acquire_lease=self.acquire_lease,
            activate_source=self.activate_source,
            activate_tokenizer=self.activate_tokenizer,
            activate_model=self.activate_model,
            compatible_chunk=self.compatible_chunk,
            infer_chunk=self.infer_chunk,
            publish_chunk=self.publish_chunk,
            publish_failure=self.publish_failure,
            publish_completion=self.publish_completion,
        )

    def persist_resource_request(self, run_id):
        self.events.append("resource_persistence")
        injected = self.dependencies.get("persist_resource_request")
        if injected:
            return injected(run_id)
        from core.research.compute.machine_profile import dell_i5_10500_profile
        from core.research.compute.run_contracts import build_run_manifest, checksum
        from core.research.compute.run_storage import initialise_run
        profile = dell_i5_10500_profile(source_git_commit="frozen-scoring-plan")
        inventory = [
            {
                "item_id": "chunk-" + row["chunk_identity"][:32],
                "ordered_position": index, "item_kind": "MODEL_STAGE",
            }
            for index, row in enumerate(self.plan["chunks"])
        ]
        self.manifest = build_run_manifest(
            run_id=run_id, pipeline="stock_alpha_news",
            stage="finbert_scoring",
            run_purpose="Authorized frozen FinBERT scoring",
            source_git_commit="frozen-scoring-plan",
            configuration_identity=self.request["scoring_request_identity"],
            configuration_checksum=logical_identity(self.runtime),
            machine_profile_identity=profile.logical_checksum,
            requested_resource_profile_identity=checksum({
                "memory": self.request["memory_request_bytes"],
                "cpu_weight": 2, "inner_threads": 1,
            }),
            parent_input_artifacts=[{
                "artifact_type": "CANONICAL_NEWS",
                "identity": self.request["canonical_identity"],
                "checksum": self.request["canonical_csv_sha256"],
            }],
            expected_inventory=inventory,
        )
        self.run_root = initialise_run(self.manifest, runs_root=self.runs_root)
        from core.research.compute.lease_storage import atomic_write_json
        atomic_write_json(self.run_root / "resource_request.json", {
            "run_id": run_id, "memory_request_bytes":
                self.request["memory_request_bytes"],
            "cpu_weight": 2, "inner_threads": 1, "gpu_required": False,
            "attempt_identity": self._lease_attempt,
        })

    def acquire_lease(self, run_id):
        self.events.append("lease_acquisition")
        injected = self.dependencies.get("acquire_lease")
        if injected:
            return injected(run_id)
        from core.research.compute.machine_profile import dell_i5_10500_profile
        from core.research.compute.resource_governor import LeaseStatus, ResourceRequest
        from core.research.compute.resource_lease_ledger import ResourceLeaseLedger
        profile = dell_i5_10500_profile(source_git_commit="frozen-scoring-plan")
        ledger = ResourceLeaseLedger(
            profile=profile, path=self.ledger_path,
            available_memory=lambda: profile.total_ram_bytes,
            available_gpus=lambda: (),
        )
        ledger.initialise_ledger()
        resource = ResourceRequest(
            pipeline="stock_alpha_news", stage="finbert_scoring",
            job_id="campaign", run_id=run_id, resource_class="LARGE",
            estimated_peak_ram_bytes=self.request["memory_request_bytes"],
            cpu_weight=2, inner_threads=1, gpu_required=False,
            concurrency_group="NEWS_TRANSFORMER",
            estimate_source="CONSERVATIVE_DEFAULT",
            estimate_evidence_identity=self.request["scoring_request_identity"],
            attempt_identity=self._lease_attempt, safe_to_colocate=False,
        )
        lease, _ = ledger.request_persisted_lease(resource)
        if lease.status != LeaseStatus.GRANTED:
            raise ExecutionFailure("FINBERT_RESOURCE_LEASE_NOT_GRANTED")
        return PersistedLease(ledger, lease, self._lease_attempt)

    def activate_source(self):
        self.events.append("source_activation")
        injected = self.dependencies.get("activate_source")
        if injected:
            self.source = injected()
            return self.source
        from core.research.ml.stock_level.stock_alpha_finbert_scoring_plan import (
            build_eligible_scoring_inventory,
        )
        row_reader = self.dependencies.get("read_canonical_rows")
        if row_reader:
            rows = [dict(row) for row in row_reader()]
        else:
            csv_path = Path(self.request["canonical_root"]) / (
                "stock_alpha_news_canonical_corpus.csv"
            )
            from core.research.ml.stock_level.stock_alpha_finbert_scoring_foundation import (
                file_sha256,
            )
            if file_sha256(csv_path) != self.request["canonical_csv_sha256"]:
                raise ExecutionFailure("CANONICAL_CSV_CHECKSUM_MISMATCH")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        eligible, exclusions = build_eligible_scoring_inventory(
            rows, max_characters=10_000,
            canonical_identity=self.request["canonical_identity"],
        )
        self.events.append("eligible_inventory_construction")
        if (
            len(rows) != self.request["canonical_row_count"]
            or len(eligible) != self.request["expected_eligible_row_count"]
            or len(exclusions) != self.request["expected_excluded_row_count"]
        ):
            raise ExecutionFailure("ELIGIBLE_INVENTORY_PLAN_MISMATCH")
        by_chunk = {}
        for chunk in self.plan["chunks"]:
            items = eligible[
                chunk["start_inclusive"]:chunk["stop_exclusive"]
            ]
            if len(items) != chunk["row_count"]:
                raise ExecutionFailure("ELIGIBLE_INVENTORY_PLAN_MISMATCH")
            by_chunk[chunk["chunk_identity"]] = items
        inventory_evidence = [{
            "article_id": item["article_id"], "symbol": item["symbol"],
            "selected_text_hash": item["text"].text_hash,
        } for item in eligible]
        self.source = {
            "canonical_row_count": len(rows),
            "eligible_row_count": len(eligible),
            "excluded_row_count": len(exclusions),
            "exclusions": exclusions, "items_by_chunk": by_chunk,
            "eligible_inventory_identity": logical_identity({
                "canonical_identity": self.request["canonical_identity"],
                "canonical_csv_sha256": self.request["canonical_csv_sha256"],
                "text_selection_contract": self.request[
                    "text_selection_contract"
                ],
                "ordering_contract":
                    "article_id,symbol,selected_text_hash ascending",
                "items": inventory_evidence,
                "exclusions": exclusions,
            }),
        }
        from core.research.compute.lease_storage import atomic_write_json
        output_root = Path(self.request["score_output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        ownership = {
            "scoring_request_identity":
                self.request["scoring_request_identity"],
            "canonical_identity": self.request["canonical_identity"],
            "canonical_csv_sha256": self.request["canonical_csv_sha256"],
            "model_package_identity": self.request["model_package_identity"],
            "model_revision": self.request["model_revision"],
            "scored_at_selection_identity":
                self.request["scored_at_selection_identity"],
        }
        ownership_path = output_root / "scoring_ownership.json"
        if ownership_path.exists():
            if json.loads(ownership_path.read_text(encoding="utf-8")) != ownership:
                raise ExecutionFailure("PRODUCTION_SCORE_OUTPUT_ROOT_INCOMPATIBLE")
        else:
            atomic_write_json(ownership_path, ownership)
        atomic_write_json(
            output_root / "governed_exclusions.json",
            {
                "contract": "stock_alpha_finbert_governed_exclusions.v1",
                "canonical_identity": self.request["canonical_identity"],
                "eligible_rows": len(eligible),
                "excluded_rows": len(exclusions),
                "exclusions": exclusions,
                "certified": False,
            },
        )
        if self.run_root:
            atomic_write_json(
                self.run_root / "eligible_inventory.json",
                {
                    "contract":
                        "stock_alpha_finbert_eligible_inventory.v1",
                    "canonical_rows": len(rows),
                    "eligible_rows": len(eligible),
                    "excluded_rows": len(exclusions),
                    "eligible_inventory_identity":
                        self.source["eligible_inventory_identity"],
                    "ordering_contract":
                        "article_id,symbol,selected_text_hash ascending",
                },
            )
        return self.source

    def activate_tokenizer(self):
        self.events.append("tokenizer_activation")
        injected = self.dependencies.get("activate_tokenizer")
        if injected:
            self.tokenizer = injected()
            return self.tokenizer
        from core.research.ml.stock_level.stock_alpha_finbert_news import (
            load_local_finbert_tokenizer,
        )
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.tokenizer = load_local_finbert_tokenizer(
            tokenizer_path=self.request["model_package_root"],
            revision=self.request["model_revision"],
            cache_dir=self.request["model_package_root"],
        )
        return self.tokenizer

    def activate_model(self):
        self.events.append("model_activation")
        injected = self.dependencies.get("activate_model")
        if injected:
            self.model = injected()
            return self.model
        from core.research.ml.stock_level.stock_alpha_finbert_compute import (
            FinBertExecutionPolicy,
        )
        from core.research.ml.stock_level.stock_alpha_finbert_news import (
            HuggingFaceFinBertAdapter,
        )
        import torch
        torch.set_num_threads(int(self.request["inner_thread_count"]))
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.model = HuggingFaceFinBertAdapter(
            **immutable_model_activation_options(self.request),
            tokenizer_instance=self.tokenizer,
        )
        return self.model

    def _authoritative_adapter(self):
        if self.adapter is not None:
            return self.adapter
        from core.research.ml.stock_level.stock_alpha_news_compute_adapters import (
            AuthoritativeFinBertChunkAdapter,
        )
        converted = self._authoritative_plan()
        self.adapter = AuthoritativeFinBertChunkAdapter(
            scoring_plan=converted, source_rows=None,
            prepared_items_by_chunk=self.source["items_by_chunk"],
            output_dir=Path(self.request["score_output_root"]),
            scoring_config=self.runtime,
            model_factory=lambda reference, policy: self.model,
            scored_at=self.request["scored_at_utc"],
        )
        self.adapter._active_identity = self.model.identity
        return self.adapter

    def _authoritative_plan(self):
        from core.research.ml.stock_level.stock_alpha_finbert_news import (
            FINBERT_INFERENCE_CONTRACT_VERSION,
            FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
        )
        chunks = []
        for row in self.plan["chunks"]:
            items = self.source["items_by_chunk"][row["chunk_identity"]]
            identity = {
                "article_identities": [{
                    "article_id": item["article_id"], "symbol": item["symbol"],
                    "selected_text_hash": item["text"].text_hash,
                } for item in items],
                "model_id": self.request["model_name"],
                "model_revision": self.request["model_revision"],
                "tokenizer_id": self.request["model_name"],
                "tokenizer_revision": self.request["model_revision"],
                "inference_contract_version":
                    FINBERT_INFERENCE_CONTRACT_VERSION,
                "text_selection_contract_version":
                    FINBERT_TEXT_SELECTION_CONTRACT_VERSION,
                "max_token_length": self.request["maximum_sequence_length"],
                "configuration_hash": logical_identity(self.runtime),
                "chunk_id": row["chunk_identity"],
            }
            chunks.append({
                "ordinal": row["ordinal"],
                "chunk_id": row["chunk_identity"],
                "article_count": row["row_count"], "identity": identity,
            })
        return {
            "logical_checksum": self.request["scoring_request_identity"],
            "plan_artifact_checksum": self.plan["chunk_plan_identity"],
            "maximum_token_length": self.request["maximum_sequence_length"],
            "maximum_selected_text_characters": 10_000,
            "expected_chunks": chunks,
            "finbert_model_identity": {
                "model_id": self.request["model_name"],
                "model_revision": self.request["model_revision"],
                "tokenizer_id": self.request["model_name"],
                "tokenizer_revision": self.request["model_revision"],
            },
        }

    def _converted_chunk(self, chunk):
        return next(
            row for row in self._authoritative_plan()["expected_chunks"]
            if row["chunk_id"] == chunk["chunk_identity"]
        )

    def compatible_chunk(self, chunk):
        self.events.append("compatibility:" + chunk["chunk_identity"])
        injected = self.dependencies.get("compatible_chunk")
        if injected:
            return injected(chunk)
        output = self._authoritative_adapter().compatible_output(
            self._converted_chunk(chunk)
        )
        if not output:
            return None
        chunk_payload = json.loads(
            Path(output["chunk_path"]).read_text(encoding="utf-8")
        )
        resume_bindings = {
            "model_package_identity": self.request["model_package_identity"],
            "canonical_identity": self.request["canonical_identity"],
            "scoring_request_identity":
                self.request["scoring_request_identity"],
            "chunk_identity": chunk["chunk_identity"],
            "scored_at": self.request["scored_at_utc"],
        }
        if any(
            any(row.get(key) != value for key, value in resume_bindings.items())
            for row in chunk_payload["rows"]
        ):
            raise ExecutionFailure("INCOMPATIBLE_COMPLETED_CHUNK_LINEAGE")
        return {
            "status": "COMPLETE",
            "chunk_identity": chunk["chunk_identity"],
            "output_sha256": output["chunk_artifact_sha256"],
            "logical_checksum": output["scored_rows_logical_checksum"],
            "row_count": output["row_count"],
            "scoring_request_identity":
                self.request["scoring_request_identity"],
            "owners": [
                [row["article_id"], row["symbol"]]
                for row in chunk_payload["rows"]
            ],
        }

    def infer_chunk(self, chunk, source, tokenizer, model):
        del source, tokenizer
        self.events.append("inference:" + chunk["chunk_identity"])
        injected = self.dependencies.get("infer_chunk")
        if injected:
            return injected(chunk, self.source, None, model)
        adapter = self._authoritative_adapter()
        converted = self._converted_chunk(chunk)
        texts = adapter.tokenize(model, converted)
        predictions = []
        batch_size = int(self.request["inference_batch_size"])
        for start in range(0, len(texts), batch_size):
            predictions.extend(
                adapter.infer(model, texts[start:start + batch_size], converted)
            )
        rows = adapter.build_rows(converted, predictions)
        for row in rows:
            row.update({
                "model_package_identity":
                    self.request["model_package_identity"],
                "canonical_identity": self.request["canonical_identity"],
                "scoring_request_identity":
                    self.request["scoring_request_identity"],
                "chunk_identity": chunk["chunk_identity"],
            })
        return rows

    def publish_chunk(self, chunk, rows):
        self.events.append("chunk_publication:" + chunk["chunk_identity"])
        injected = self.dependencies.get("publish_chunk")
        if injected:
            return injected(chunk, rows)
        destination = (
            Path(self.request["score_output_root"]) / "chunks"
            / f"{chunk['chunk_identity']}.json"
        )
        if destination.exists():
            raise ExecutionFailure("INCOMPATIBLE_CHUNK_DESTINATION_EXISTS")
        output = self._authoritative_adapter().publish_rows(
            self._converted_chunk(chunk), rows
        )
        return {
            "status": "COMPLETE",
            "chunk_identity": chunk["chunk_identity"],
            "output_sha256": output["chunk_artifact_sha256"],
            "logical_checksum": output["scored_rows_logical_checksum"],
            "row_count": output["row_count"],
            "scoring_request_identity":
                self.request["scoring_request_identity"],
        }

    def publish_failure(self, chunk_identity, code):
        self.events.append("failure_publication:" + chunk_identity)
        injected = self.dependencies.get("publish_failure")
        if injected:
            return injected(chunk_identity, code)
        if self.run_root:
            from core.research.compute.lease_storage import atomic_write_json
            atomic_write_json(self.run_root / "terminal_failure.json", {
                "chunk_identity": chunk_identity, "failure_code": code,
                "retryable": False,
            })

    def publish_completion(self, payload):
        self.events.append("shared_completion_publication")
        injected = self.dependencies.get("publish_completion")
        if injected:
            return injected(payload)
        from core.research.compute.lease_storage import atomic_write_json
        output = Path(self.request["score_output_root"])
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            **dict(payload), "certified": False,
            "certification_status": "NOT_RUN",
            "pit_generation_status": "BLOCKED_AWAITING_CERTIFICATION",
            "scoring_request_identity":
                self.request["scoring_request_identity"],
            "canonical_identity": self.request["canonical_identity"],
            "model_package_identity": self.request["model_package_identity"],
        }
        pending_completion = output / "score_store_completion.pending.json"
        atomic_write_json(pending_completion, manifest)
        if self.run_root:
            atomic_write_json(self.run_root / "scoring_completion.json", manifest)
            from core.research.compute.run_contracts import (
                build_item_status, build_result_record, metric_value,
            )
            from core.research.compute.run_storage import (
                publish_item_status, publish_results_snapshot, publish_summary,
                update_global_registry_snapshot, update_run_status,
            )
            for index, chunk in enumerate(self.plan["chunks"]):
                item_id = "chunk-" + chunk["chunk_identity"][:32]
                status = build_item_status(
                    run_identity=self.manifest["run_identity"],
                    item_id=item_id, ordered_position=index,
                    pipeline="stock_alpha_news", stage="finbert_scoring",
                    attempt_identity=self._lease_attempt, status="COMPLETE",
                    required_artifact_kind="NONE",
                    artifact_validation={"not_applicable": True},
                )
                publish_item_status(self.run_root, status)
                metric = metric_value(
                    "scored_rows", chunk["row_count"], unit="rows",
                    population_identity=chunk["chunk_identity"],
                    direction="INFORMATIONAL", availability="AVAILABLE",
                    source_artifact_identity=chunk["chunk_identity"],
                )
                self.results.append(build_result_record(
                    result_identity=logical_identity({
                        "run": payload["run_id"], "chunk":
                            chunk["chunk_identity"]
                    }),
                    run_identity=self.manifest["run_identity"],
                    item_identity=item_id, result_kind="NEWS_SCORING",
                    pipeline="stock_alpha_news", stage="finbert_scoring",
                    status="COMPLETE",
                    artifact_identities=[chunk["chunk_identity"]],
                    metrics={"scored_rows": metric},
                ))
            publish_results_snapshot(self.run_root, self.results)
            update_run_status(
                self.run_root, expected_revision=None, inputs_valid=True
            )
            publish_summary(self.run_root)
            registry = update_global_registry_snapshot(
                self.run_root, registry_path=self.registry_path
            )
            if registry["health"] != "HEALTHY":
                raise ExecutionFailure("SHARED_REGISTRY_PUBLICATION_FAILED")
        os.replace(
            pending_completion, output / "score_store_completion.json"
        )
        return manifest
