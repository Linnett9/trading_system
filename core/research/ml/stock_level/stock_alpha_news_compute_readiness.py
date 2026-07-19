from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

AUDIT_CONTRACT = "stock_alpha_news_compute_readiness.v1"
READ_ONLY_MARKER = "READ_ONLY_AUDIT"
STAGES = (
    "CANONICAL_CORPUS", "FINBERT_SCORING_PLAN",
    "EXTERNAL_MODEL_REFERENCE", "FINBERT_SCORE_STORE",
    "SCORE_STORE_CERTIFICATION", "PIT_PARENT_LINEAGE",
    "PIT_FEATURE_STORE", "RESOURCE_LEDGER", "RUN_REGISTRY",
    "OPERATOR_CONTROLS",
)
SUPPORTED_SELECTIONS = ("CORPUS", "SCORING", "CERTIFICATION", "PIT")
MAX_METADATA_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class NewsComputeReadinessRequest:
    selected_stages: tuple[str, ...]
    canonical_source_path: str
    canonical_corpus_root: str
    canonical_manifest_path: str
    scoring_plan_path: str
    score_store_root: str
    chunk_manifest_path: str
    certification_path: str
    pit_feature_store_root: str
    pit_feature_manifest_path: str
    daily_spine_manifest_path: str
    ticker_mapping_manifest_path: str
    alias_parent_path: str
    runtime_config_path: str
    shared_run_root: str
    resource_ledger_path: str
    run_registry_path: str
    model_cache_root: str
    audit_output_path: str
    cpu_gpu_policy: str = "CPU_SAFE_DEFAULT"
    offline_model_resolution_required: bool = True
    allow_empty_corpus: bool = False
    minimum_free_disk_bytes_warning: int = 20 * 1024**3
    contract_version: str = AUDIT_CONTRACT

    def __post_init__(self) -> None:
        if self.contract_version != AUDIT_CONTRACT:
            raise ValueError("Unsupported readiness request contract")
        if not self.selected_stages or any(
            stage not in SUPPORTED_SELECTIONS for stage in self.selected_stages
        ):
            raise ValueError("Explicit supported selected_stages are required")
        if len(set(self.selected_stages)) != len(self.selected_stages):
            raise ValueError("Duplicate selected stage")
        required = [
            "canonical_source_path", "canonical_corpus_root",
            "canonical_manifest_path", "runtime_config_path",
            "shared_run_root", "resource_ledger_path", "run_registry_path",
            "audit_output_path",
        ]
        if "SCORING" in self.selected_stages or "CERTIFICATION" in self.selected_stages:
            required += [
                "scoring_plan_path", "score_store_root",
                "chunk_manifest_path", "model_cache_root",
            ]
        if "CERTIFICATION" in self.selected_stages or "PIT" in self.selected_stages:
            required += ["certification_path"]
        if "PIT" in self.selected_stages:
            required += [
                "pit_feature_store_root", "pit_feature_manifest_path",
                "daily_spine_manifest_path", "ticker_mapping_manifest_path",
                "alias_parent_path",
            ]
        missing = [name for name in required if not str(getattr(self, name)).strip()]
        if missing:
            raise ValueError("Missing explicit readiness paths: " + ",".join(missing))
        if self.cpu_gpu_policy not in {
            "CPU_SAFE_DEFAULT", "GPU_EXPLICITLY_PERMITTED"
        }:
            raise ValueError("Explicit CPU/GPU policy is required")
        roots = {
            Path(self.canonical_corpus_root).resolve(),
            Path(self.score_store_root).resolve() if self.score_store_root else None,
            Path(self.pit_feature_store_root).resolve()
            if self.pit_feature_store_root else None,
        }
        if Path(self.audit_output_path).resolve() in roots:
            raise ValueError("Audit output must not alias a production artifact root")

    @property
    def identity(self) -> str:
        return _hash(asdict(self))


def audit_news_compute_readiness(
    request: NewsComputeReadinessRequest, *, max_blocker_examples: int = 10,
    max_chunk_examples: int = 10, repository_root: Path | None = None,
) -> dict[str, Any]:
    if min(max_blocker_examples, max_chunk_examples) < 1:
        raise ValueError("Bounded example limits must be positive")
    root = (repository_root or Path.cwd()).resolve()
    context: dict[str, Any] = {"request": request, "root": root}
    results = []
    for stage, function in (
        ("CANONICAL_CORPUS", _audit_corpus),
        ("FINBERT_SCORING_PLAN", _audit_plan),
        ("EXTERNAL_MODEL_REFERENCE", _audit_model_cache),
        ("FINBERT_SCORE_STORE", _audit_chunks),
        ("SCORE_STORE_CERTIFICATION", _audit_certification),
        ("PIT_PARENT_LINEAGE", _audit_pit_lineage),
        ("PIT_FEATURE_STORE", _audit_feature_store),
        ("RESOURCE_LEDGER", _audit_ledger),
        ("RUN_REGISTRY", _audit_registry_and_run_root),
        ("OPERATOR_CONTROLS", _audit_operator_controls),
    ):
        try:
            result = function(context, max_chunk_examples)
        except Exception as exc:
            result = _stage(
                stage, "BLOCKED", ["MANIFEST_MALFORMED"], {},
                f"Inspect and repair {stage.lower()} metadata ({type(exc).__name__}).",
                True,
            )
        results.append(result)
        context[stage] = result
    blockers = [
        {"stage": row["stage"], "code": code,
         "operator_action": row["required_operator_action"]}
        for row in results if row["blocks_production"]
        for code in row["reason_codes"]
    ][:max_blocker_examples]
    warnings = [
        {"stage": row["stage"], "code": code,
         "operator_action": row["required_operator_action"]}
        for row in results if row["status"] == "READY_WITH_CONDITIONS"
        for code in row["reason_codes"]
    ][:max_blocker_examples]
    overall = (
        "BLOCKED" if blockers else
        "READY_WITH_CONDITIONS" if warnings else "READY"
    )
    volatile = {
        "free_disk_bytes": _disk_free(Path(request.audit_output_path)),
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report = {
        "audit_contract_version": AUDIT_CONTRACT,
        "audit_mode": READ_ONLY_MARKER,
        "audit_timestamp": volatile["audit_timestamp"],
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "request_identity": request.identity,
        "selected_stages": list(request.selected_stages),
        "exact_input_references": _bounded_request(request),
        "stage_results": results,
        "scoring_resume_inventory": context.get(
            "FINBERT_SCORE_STORE", {}
        ).get("evidence", {}).get("resume_inventory", _empty_inventory()),
        "parent_lineage_graph": _lineage(context),
        "resource_path_checks": {
            "free_disk_bytes": volatile["free_disk_bytes"],
            "minimum_free_disk_bytes_warning":
                request.minimum_free_disk_bytes_warning,
            "resource_profiles": {
                "canonical_corpus": {"ram_bytes": 4 * 1024**3,
                                     "cpu_weight": 1},
                "finbert_scoring": {"ram_bytes": 10 * 1024**3,
                                    "cpu_weight": 2},
                "pit_feature_store": {"ram_bytes": 6 * 1024**3,
                                      "cpu_weight": 2},
            },
            "cpu_gpu_policy": request.cpu_gpu_policy,
        },
        "model_cache_reference_state": context.get(
            "EXTERNAL_MODEL_REFERENCE", {}
        ).get("evidence", {}).get(
            "cache_state", "MODEL_CACHE_REFERENCE_UNVERIFIED"
        ),
        "blockers": blockers, "warnings": warnings,
        "overall_readiness": overall,
        "production_execution_performed": False,
        "execution_lease_acquired": False,
        "network_access_performed": False,
        "model_activation_performed": False,
    }
    report["logical_checksum"] = _hash({
        key: value for key, value in report.items()
        if key not in {"audit_timestamp", "resource_path_checks"}
    })
    _publish_outputs(request, report)
    return report


def _audit_corpus(ctx, _):
    request = ctx["request"]
    if "CORPUS" not in request.selected_stages and not any(
        stage in request.selected_stages for stage in ("SCORING", "PIT")
    ):
        return _not_selected("CANONICAL_CORPUS")
    source = Path(request.canonical_source_path)
    manifest_path = Path(request.canonical_manifest_path)
    if not source.exists():
        return _blocked("CANONICAL_CORPUS", "MISSING_REQUIRED_PATH",
                        "Supply the explicit source assembly or inventory.")
    if not manifest_path.is_file():
        return _blocked("CANONICAL_CORPUS", "MANIFEST_NOT_FOUND",
                        "Supply the canonical corpus manifest.")
    manifest = _read_json(manifest_path)
    required = (
        "canonical_corpus_identity", "canonical_corpus_checksum",
        "logical_manifest_checksum", "canonical_schema_checksum",
        "canonical_row_count", "source_assembly_identity",
        "duplicate_group_count", "ingested_at_utc",
    )
    missing = [name for name in required if manifest.get(name) in (None, "")]
    if missing:
        return _blocked("CANONICAL_CORPUS", "MISSING_ARTIFACT_IDENTITY",
                        "Regenerate authoritative inventory evidence.",
                        {"missing_fields": missing})
    if manifest.get("schema_version") not in {
        "stock_alpha_news.historical_canonical_corpus.v2",
        "stock_alpha_news.historical_canonical_corpus.v1",
    }:
        return _blocked("CANONICAL_CORPUS", "UNSUPPORTED_CONTRACT_VERSION",
                        "Use a supported canonical corpus contract.")
    rows = int(manifest["canonical_row_count"])
    if rows <= 0 and not request.allow_empty_corpus:
        return _blocked("CANONICAL_CORPUS", "CORPUS_INVENTORY_EMPTY",
                        "Provide a non-empty certified canonical corpus.")
    artifact = Path(str(
        manifest.get("canonical_artifact_path")
        or Path(request.canonical_corpus_root) / "stock_alpha_news_canonical_corpus.csv"
    ))
    if not artifact.is_file() or not _beneath(artifact, Path(request.canonical_corpus_root)):
        return _blocked("CANONICAL_CORPUS", "MISSING_REQUIRED_PATH",
                        "Restore the referenced corpus beneath its explicit root.")
    temporary = list(Path(request.canonical_corpus_root).glob(".*.tmp"))
    if temporary:
        return _blocked("CANONICAL_CORPUS", "INCOMPLETE_PUBLICATION_MARKER",
                        "Review interrupted canonical publication.",
                        {"temporary_marker_count": len(temporary)})
    ctx["corpus_manifest"] = manifest
    return _ready("CANONICAL_CORPUS", {
        "identity": manifest["canonical_corpus_identity"],
        "checksum": manifest["canonical_corpus_checksum"], "row_count": rows,
        "provider_inventory_present": bool(
            manifest.get("source_metadata") or manifest.get("source_assembly_identity")),
        "duplicate_group_count": manifest["duplicate_group_count"],
        "availability_evidence": manifest["ingested_at_utc"],
    })


def _audit_plan(ctx, _):
    request = ctx["request"]
    if "SCORING" not in request.selected_stages and not any(
        stage in request.selected_stages for stage in ("CERTIFICATION", "PIT")
    ):
        return _not_selected("FINBERT_SCORING_PLAN")
    path = Path(request.scoring_plan_path)
    if not path.is_file():
        return _blocked("FINBERT_SCORING_PLAN", "MANIFEST_NOT_FOUND",
                        "Supply the exact production scoring plan.")
    plan = _read_json(path)
    if plan.get("scoring_plan_contract") != (
        "stock_alpha_finbert_production_scoring_plan.v1"
    ):
        return _blocked("FINBERT_SCORING_PLAN",
                        "UNSUPPORTED_CONTRACT_VERSION",
                        "Use the supported production scoring plan.")
    corpus = ctx.get("corpus_manifest", {})
    if (
        plan.get("canonical_corpus_identity") != corpus.get(
            "canonical_corpus_identity")
        or plan.get("canonical_corpus_checksum") != corpus.get(
            "canonical_corpus_checksum")
    ):
        return _blocked("FINBERT_SCORING_PLAN", "CORPUS_ANCESTRY_MISMATCH",
                        "Build or select a plan for the audited corpus.")
    model = plan.get("finbert_model_identity") or {}
    missing = [name for name in (
        "model_id", "model_revision", "tokenizer_id", "tokenizer_revision"
    ) if not str(model.get(name) or "").strip()]
    if missing:
        return _blocked("FINBERT_SCORING_PLAN", "MODEL_REVISION_MISMATCH",
                        "Pin exact model and tokenizer revisions.",
                        {"missing_model_fields": missing})
    if any(str(model[name]).lower() in {"main", "master", "latest"}
           for name in ("model_revision", "tokenizer_revision")):
        return _blocked("FINBERT_SCORING_PLAN", "MODEL_REVISION_MISMATCH",
                        "Replace floating model/tokenizer revisions.")
    chunks = list(plan.get("expected_chunks") or [])
    ids = [row.get("chunk_id") for row in chunks]
    if (not chunks or len(ids) != len(set(ids))
            or plan.get("expected_chunk_count") != len(chunks)):
        return _blocked("FINBERT_SCORING_PLAN", "SCORING_CHUNKS_INCOMPLETE",
                        "Repair the authoritative planned chunk inventory.")
    ctx["scoring_plan"] = plan
    return _ready("FINBERT_SCORING_PLAN", {
        "identity": plan.get("logical_checksum"), "planned_chunks": len(chunks),
        "model_reference": {key: model[key] for key in model},
        "configuration_checksum": plan.get("configuration_checksum"),
        "cpu_safe_default": request.cpu_gpu_policy == "CPU_SAFE_DEFAULT",
        "gpu_requires_permission": True,
    })


def _audit_model_cache(ctx, _):
    request = ctx["request"]
    if not any(stage in request.selected_stages for stage in (
        "SCORING", "CERTIFICATION", "PIT"
    )):
        return _not_selected("EXTERNAL_MODEL_REFERENCE")
    plan = ctx.get("scoring_plan")
    if not plan:
        return _blocked("EXTERNAL_MODEL_REFERENCE",
                        "MODEL_CACHE_REFERENCE_UNVERIFIED",
                        "Resolve the scoring plan before checking its cache.")
    root = Path(request.model_cache_root)
    if not root.is_dir():
        return _conditional(
            "EXTERNAL_MODEL_REFERENCE", "MODEL_CACHE_REFERENCE_INCOMPLETE",
            "Populate the exact pinned snapshots locally without network fallback.",
            {"cache_state": "MODEL_CACHE_REFERENCE_INCOMPLETE"})
    model = plan["finbert_model_identity"]
    revisions = (model["model_revision"], model["tokenizer_revision"])
    represented = {
        revision: any(
            child.name == revision for child in root.glob("**/snapshots/*")
            if child.is_dir()
        ) for revision in revisions
    }
    metadata = any(root.glob("**/config.json")) and any(
        root.glob("**/tokenizer_config.json")
    )
    if not all(represented.values()) or not metadata:
        return _conditional(
            "EXTERNAL_MODEL_REFERENCE", "MODEL_CACHE_REFERENCE_INCOMPLETE",
            "Populate pinned model/tokenizer metadata in the offline cache.",
            {"cache_state": "MODEL_CACHE_REFERENCE_INCOMPLETE",
             "revision_presence": represented})
    return _conditional(
        "EXTERNAL_MODEL_REFERENCE", "MODEL_EXECUTION_NOT_PROVEN",
        "Keep offline mode enabled; activation remains a separate operation.",
        {"cache_state": "MODEL_CACHE_REFERENCE_READY",
         "revision_presence": represented,
         "offline_resolution_required":
             request.offline_model_resolution_required})


def _audit_chunks(ctx, max_examples):
    request = ctx["request"]
    if not any(stage in request.selected_stages for stage in (
        "SCORING", "CERTIFICATION", "PIT"
    )):
        return _not_selected("FINBERT_SCORE_STORE")
    plan = ctx.get("scoring_plan")
    if not plan:
        return _blocked("FINBERT_SCORE_STORE", "SCORING_CHUNKS_INCOMPLETE",
                        "Resolve a valid scoring plan first.",
                        {"resume_inventory": _empty_inventory()})
    expected = {row["chunk_id"]: row for row in plan["expected_chunks"]}
    manifest_path = Path(request.chunk_manifest_path)
    rows = _read_csv(manifest_path) if manifest_path.is_file() else []
    by_id = {row.get("chunk_id"): row for row in rows if row.get("chunk_id")}
    inventory = _empty_inventory()
    examples = []
    for chunk_id, chunk in expected.items():
        row = by_id.get(chunk_id)
        if row is None:
            inventory["missing"] += 1
            examples.append(chunk_id)
            continue
        path = Path(str(row.get("chunk_path") or ""))
        if row.get("status") != "completed":
            inventory["incomplete"] += 1
        elif not path.is_file() or not _beneath(path, Path(request.score_store_root)):
            inventory["malformed"] += 1
        elif (
            row.get("scoring_plan_identity") != plan.get("logical_checksum")
            or not row.get("chunk_artifact_sha256")
            or not row.get("scored_rows_logical_checksum")
            or int(row.get("article_count") or -1)
            != int(chunk.get("article_count") or -2)
        ):
            inventory["incompatible"] += 1
        else:
            inventory["compatible"] += 1
    unexpected = sorted(set(by_id) - set(expected))
    inventory["unexpected"] = len(unexpected)
    inventory["planned"] = len(expected)
    examples = (examples + unexpected)[:max_examples]
    ctx["chunk_inventory"] = inventory
    evidence = {"resume_inventory": inventory, "bounded_examples": examples}
    if inventory["malformed"] or inventory["incompatible"]:
        return _blocked("FINBERT_SCORE_STORE", "SCORING_CHUNKS_INCOMPLETE",
                        "Isolate incompatible/malformed chunk outputs.", evidence)
    if inventory["missing"] or inventory["incomplete"]:
        return _conditional(
            "FINBERT_SCORE_STORE", "SCORING_CHUNKS_INCOMPLETE",
            "Resume scoring with the same logical request.", evidence)
    return _ready("FINBERT_SCORE_STORE", evidence)


def _audit_certification(ctx, _):
    request = ctx["request"]
    if "CERTIFICATION" not in request.selected_stages and "PIT" not in request.selected_stages:
        return _not_selected("SCORE_STORE_CERTIFICATION")
    path = Path(request.certification_path)
    if not path.is_file():
        return _blocked("SCORE_STORE_CERTIFICATION", "CERTIFICATION_MISSING",
                        "Run certification after all scoring chunks complete.")
    cert = _read_json(path)
    plan = ctx.get("scoring_plan", {})
    if cert.get("score_store_contract") != (
        "stock_alpha_finbert_production_score_store.v1"
    ):
        return _blocked("SCORE_STORE_CERTIFICATION",
                        "UNSUPPORTED_CONTRACT_VERSION",
                        "Use a supported score-store certificate.")
    if (
        cert.get("production_scoring_plan_identity")
        != plan.get("logical_checksum")
        or cert.get("canonical_corpus_identity")
        != plan.get("canonical_corpus_identity")
        or cert.get("finbert_model_identity")
        != plan.get("finbert_model_identity")
    ):
        return _blocked("SCORE_STORE_CERTIFICATION",
                        "CORPUS_ANCESTRY_MISMATCH",
                        "Certify the exact audited plan/model/corpus.")
    if cert.get("production_scoring_complete") is not True or cert.get(
        "status") != "COMPLETE":
        return _blocked("SCORE_STORE_CERTIFICATION", "CERTIFICATION_FAILED",
                        "Complete scoring and publish a successful certificate.")
    ctx["certificate"] = cert
    return _ready("SCORE_STORE_CERTIFICATION", {
        "score_store_identity": cert.get("score_store_identity"),
        "completed_chunks": cert.get("certified_completed_chunk_count"),
        "scored_rows": cert.get("certified_scored_row_count"),
    })


def _audit_pit_lineage(ctx, _):
    request = ctx["request"]
    if "PIT" not in request.selected_stages:
        return _not_selected("PIT_PARENT_LINEAGE")
    cert = ctx.get("certificate")
    if not cert:
        return _blocked("PIT_PARENT_LINEAGE", "CERTIFICATION_MISSING",
                        "Supply valid score-store certification.")
    for stage, path_value, code in (
        ("daily_spine", request.daily_spine_manifest_path,
         "DAILY_SPINE_PARENT_MISSING"),
        ("symbol_mapping", request.ticker_mapping_manifest_path,
         "SYMBOL_MAPPING_PARENT_MISSING"),
        ("alias_parent", request.alias_parent_path,
         "SYMBOL_MAPPING_PARENT_MISSING"),
    ):
        if not Path(path_value).is_file():
            return _blocked("PIT_PARENT_LINEAGE", code,
                            f"Supply the exact {stage} parent.")
    spine = _read_json(Path(request.daily_spine_manifest_path))
    mapping = _read_json(Path(request.ticker_mapping_manifest_path))
    if not all((
        spine.get("daily_spine_identity"), spine.get("daily_spine_checksum"),
        mapping.get("ticker_mapping_identity"),
        mapping.get("ticker_mapping_checksum"),
    )):
        return _blocked("PIT_PARENT_LINEAGE", "MISSING_ARTIFACT_IDENTITY",
                        "Repair PIT parent identity/checksum evidence.")
    ctx["pit_parents"] = {"spine": spine, "mapping": mapping}
    return _ready("PIT_PARENT_LINEAGE", {
        "canonical_corpus_identity": cert["canonical_corpus_identity"],
        "score_store_identity": cert["score_store_identity"],
        "daily_spine_identity": spine["daily_spine_identity"],
        "ticker_mapping_identity": mapping["ticker_mapping_identity"],
    })


def _audit_feature_store(ctx, _):
    request = ctx["request"]
    if "PIT" not in request.selected_stages:
        return _not_selected("PIT_FEATURE_STORE")
    path = Path(request.pit_feature_manifest_path)
    if not path.exists():
        if ctx.get("pit_parents"):
            return _conditional(
                "PIT_FEATURE_STORE", "FEATURE_STORE_ABSENT_BUILD_READY",
                "Build only after all other readiness blockers are cleared.",
                {"classification": "absent_and_ready_to_build"})
        return _blocked("PIT_FEATURE_STORE", "FEATURE_STORE_INCOMPATIBLE",
                        "Resolve PIT parents before feature generation.")
    manifest = _read_json(path)
    if manifest.get("feature_store_contract") != (
        "canonical_partitioned_pit_news_feature_store.v1"
    ):
        return _blocked("PIT_FEATURE_STORE", "UNSUPPORTED_CONTRACT_VERSION",
                        "Use a supported PIT feature-store manifest.")
    cert = ctx.get("certificate", {})
    parents = ctx.get("pit_parents", {})
    expected = {
        "canonical_corpus_identity": cert.get("canonical_corpus_identity"),
        "canonical_corpus_checksum": cert.get("canonical_corpus_checksum"),
        "score_store_identity": cert.get("score_store_identity"),
        "canonical_daily_spine_identity":
            parents.get("spine", {}).get("daily_spine_identity"),
        "ticker_mapping_identity":
            parents.get("mapping", {}).get("ticker_mapping_identity"),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return _blocked("PIT_FEATURE_STORE", "FEATURE_STORE_INCOMPATIBLE",
                        "Isolate and rebuild against exact audited parents.")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or any(
        not row.get("relative_path") or not row.get("artifact_checksum")
        for row in partitions
    ):
        return _blocked("PIT_FEATURE_STORE", "FEATURE_STORE_INCOMPATIBLE",
                        "Repair authoritative partition inventory.")
    root = Path(request.pit_feature_store_root)
    missing = sum(not (root / row["relative_path"]).is_file()
                  for row in partitions)
    if missing:
        return _blocked("PIT_FEATURE_STORE", "FEATURE_STORE_INCOMPATIBLE",
                        "Restore missing feature-store partitions.",
                        {"missing_partition_count": missing})
    return _ready("PIT_FEATURE_STORE", {
        "classification": "compatible_and_reusable",
        "row_count": manifest.get("row_count"),
        "partition_count": len(partitions),
        "feature_schema_checksum": manifest.get("feature_schema_checksum"),
    })


def _audit_ledger(ctx, _):
    request = ctx["request"]
    path = Path(request.resource_ledger_path)
    if _aliases_data(path, request):
        return _blocked("RESOURCE_LEDGER", "MISSING_REQUIRED_PATH",
                        "Choose a ledger path outside production data roots.")
    if path.exists():
        payload = _read_json(path)
        if payload.get("contract_version") != "compute_resource_lease_ledger.v1":
            return _blocked("RESOURCE_LEDGER", "LEDGER_MALFORMED",
                            "Repair or isolate the malformed ledger.")
        active = list(payload.get("active_leases") or [])
        if active:
            return _blocked(
                "RESOURCE_LEDGER", "STALE_ACTIVE_LEASES_DETECTED",
                "Reconcile active leases manually before launch.",
                {"active_lease_count": len(active)})
    probe = _probe_parent(path)
    if not probe["passed"]:
        return _blocked("RESOURCE_LEDGER", "LEDGER_UNWRITABLE",
                        "Grant operator write access to the ledger parent.", probe)
    return _ready("RESOURCE_LEDGER", {
        "existing_ledger": path.exists(), "non_mutating_probe": probe})


def _audit_registry_and_run_root(ctx, _):
    request = ctx["request"]
    registry = Path(request.run_registry_path)
    run_root = Path(request.shared_run_root)
    if any(_aliases_data(path, request) for path in (registry, run_root)):
        return _blocked("RUN_REGISTRY", "RUN_ROOT_UNWRITABLE",
                        "Choose shared-compute paths outside production data roots.")
    if registry.exists():
        payload = _read_json(registry)
        if payload.get("contract_version") != "compute_run_registry.v1":
            return _blocked("RUN_REGISTRY", "MANIFEST_MALFORMED",
                            "Repair or isolate the malformed registry.")
    registry_probe = _probe_parent(registry)
    run_probe = _probe_parent(run_root / "probe")
    if not registry_probe["passed"] or not run_probe["passed"]:
        return _blocked("RUN_REGISTRY", "REGISTRY_UNWRITABLE",
                        "Grant write access to registry and run-root parents.",
                        {"registry_probe": registry_probe,
                         "run_root_probe": run_probe})
    return _ready("RUN_REGISTRY", {
        "registry_probe": registry_probe, "run_root_probe": run_probe})


def _audit_operator_controls(ctx, _):
    root = ctx["root"]
    scripts = (
        root / "scripts/run_stock_alpha_finbert_compute.py",
        root / "scripts/run_stock_alpha_news_data_compute.py",
    )
    evidence = {}
    required = (
        "--request", "--run-root", "--resource-ledger", "--registry",
        "--plan-only",
    )
    for path in scripts:
        if not path.is_file():
            return _blocked("OPERATOR_CONTROLS", "OPERATOR_CONTROL_MISSING",
                            "Restore the committed shared-compute scripts.")
        text = path.read_text(encoding="utf-8")
        missing = [flag for flag in required if flag not in text]
        if "finbert" in path.name:
            missing += [flag for flag in ("--device", "--allow-gpu")
                        if flag not in text]
        if missing:
            return _blocked("OPERATOR_CONTROLS", "OPERATOR_CONTROL_MISSING",
                            "Restore mandatory explicit script controls.",
                            {"script": path.name, "missing": missing})
        evidence[path.name] = {"required_controls_present": True}
    return _ready("OPERATOR_CONTROLS", evidence)


def _publish_outputs(request, report):
    root = Path(request.audit_output_path)
    root.mkdir(parents=True, exist_ok=True)
    request_payload = asdict(request) | {"request_identity": request.identity}
    stage_results = report["stage_results"]
    outputs = {
        "request.json": request_payload,
        "readiness_report.json": report,
        "stage_results.json": {"stage_results": stage_results},
        "blockers.json": {"blockers": report["blockers"]},
        "warnings.json": {"warnings": report["warnings"]},
        "chunk_inventory_summary.json": report["scoring_resume_inventory"],
    }
    for name, payload in outputs.items():
        _atomic_json(root / name, payload)
    _atomic_text(root / "operator_runbook.md", _runbook(request, report))


def _runbook(request, report):
    blockers = report["blockers"] or [{"code": "NONE",
                                       "operator_action": "None"}]
    finbert = (
        f"python scripts/run_stock_alpha_finbert_compute.py --request "
        f"\"{request.runtime_config_path}\" --run-root "
        f"\"{request.shared_run_root}\" --resource-ledger "
        f"\"{request.resource_ledger_path}\" --registry "
        f"\"{request.run_registry_path}\""
    )
    data = (
        f"python scripts/run_stock_alpha_news_data_compute.py --request "
        f"\"{request.runtime_config_path}\" --run-root "
        f"\"{request.shared_run_root}\" --resource-ledger "
        f"\"{request.resource_ledger_path}\" --registry "
        f"\"{request.run_registry_path}\""
    )
    lines = [
        "# Shared News Compute Operator Runbook", "",
        f"Current readiness: **{report['overall_readiness']}**", "",
        "## Blockers and remediation", "",
        *(f"- `{row['code']}`: {row['operator_action']}" for row in blockers),
        "", "## Verified immutable identities", "",
        f"- Audit request: `{request.identity}`",
        f"- Repository commit: `{report['repository_commit']}`",
        "", "## Required production ordering", "",
        "1. Canonical corpus", "2. FinBERT scoring plan",
        "3. FinBERT scoring", "4. Score-store certification",
        "5. PIT feature store", "", "## Plan-only commands", "",
        f"```powershell\n{data} --plan-only\n```",
        f"```powershell\n{finbert} --plan-only --device cpu\n```",
        "", "## DO NOT RUN UNTIL READINESS IS READY", "",
        f"CPU-safe template:\n```powershell\n{finbert} --device cpu\n```",
        "Optional GPU template (requires explicit permission):",
        f"```powershell\n{finbert} --device cuda --allow-gpu\n```",
        f"Data template:\n```powershell\n{data}\n```",
        "", "## Resume behavior", "",
        "Rerun the same immutable request. Compatible corpus, chunks, and "
        "feature-store outputs are reused before expensive activation.",
        "", "## Summaries and registry", "",
        f"Inspect summaries beneath `{request.shared_run_root}` and registry "
        f"`{request.run_registry_path}`.",
        "", "## Failure response", "",
        "1. Stop new launches.", "2. Inspect the failed item.",
        "3. Verify persisted lease state.",
        "4. Preserve authoritative outputs.",
        "5. Do not delete compatible chunks.",
        "6. Remediate and rerun the same logical request.",
        "", "## Rollback boundaries", "",
        "- Shared run records may be isolated.",
        "- Authoritative outputs must not be deleted automatically.",
        "- Incompatible replacement requires a separate reviewed operation.",
        "", "The news-transformer trainer is not part of this workflow.", "",
    ]
    return "\n".join(lines)


def _stage(stage, status, codes, evidence, action, blocks):
    return {
        "stage": stage, "status": status, "reason_codes": list(codes),
        "evidence": evidence, "required_operator_action": action,
        "blocks_production": blocks,
    }


def _ready(stage, evidence):
    return _stage(stage, "READY", [], evidence, "No action required.", False)


def _conditional(stage, code, action, evidence=None):
    return _stage(stage, "READY_WITH_CONDITIONS", [code],
                  evidence or {}, action, False)


def _blocked(stage, code, action, evidence=None):
    return _stage(stage, "BLOCKED", [code], evidence or {}, action, True)


def _not_selected(stage):
    return _stage(stage, "NOT_SELECTED", [], {},
                  "Stage was explicitly omitted.", False)


def _read_json(path):
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Metadata file exceeds bounded audit limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object")
    return payload


def _read_csv(path):
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Chunk inventory exceeds bounded audit limit")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _probe_parent(path):
    parent = path.parent
    ancestor = parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        with tempfile.TemporaryDirectory(
            prefix=".news-readiness-probe-", dir=ancestor
        ):
            pass
        return {"passed": True, "probe_parent": str(ancestor)}
    except Exception as exc:
        return {"passed": False, "exception_type": type(exc).__name__,
                "probe_parent": str(ancestor)}


def _aliases_data(path, request):
    resolved = path.resolve()
    roots = [
        Path(request.canonical_corpus_root).resolve(),
        Path(request.score_store_root).resolve() if request.score_store_root else None,
        Path(request.pit_feature_store_root).resolve()
        if request.pit_feature_store_root else None,
    ]
    return any(root and (resolved == root or _beneath(resolved, root))
               for root in roots)


def _beneath(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _disk_free(path):
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def _bounded_request(request):
    return {
        key: value for key, value in asdict(request).items()
        if key not in {"minimum_free_disk_bytes_warning"}
    }


def _lineage(ctx):
    corpus = ctx.get("corpus_manifest", {})
    plan = ctx.get("scoring_plan", {})
    cert = ctx.get("certificate", {})
    return {
        "canonical_corpus": corpus.get("canonical_corpus_identity"),
        "scoring_plan": plan.get("logical_checksum"),
        "score_store": cert.get("score_store_identity"),
        "pit_feature_store_parent": cert.get("canonical_corpus_identity"),
    }


def _empty_inventory():
    return {key: 0 for key in (
        "planned", "compatible", "missing", "incomplete", "incompatible",
        "malformed", "unexpected",
    )}


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
