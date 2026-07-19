from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.stock_level.news_sources.historical_canonical_corpus import (
    CANONICAL_CORPUS_AUDIT_JSON,
    CANONICAL_CORPUS_CSV,
    CANONICAL_CORPUS_MANIFEST_JSON,
    CANONICAL_CORPUS_SUMMARY_MD,
    HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
    HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
)

EVIDENCE_CONTRACT = "stock_alpha_news_corpus_evidence_request.v1"
MATERIALISATION_CONTRACT = (
    "stock_alpha_news_canonical_materialisation_request.v1"
)
NOTICE = "READ-ONLY PLAN VALIDATION - NOT PRODUCTION EXECUTION AUTHORIZATION"
CANONICAL_STAGE = "CANONICAL_CORPUS_MATERIALISATION"


@dataclass(frozen=True)
class CorpusEvidenceRequest:
    external_roots: tuple[str, ...]
    expected_provider_scope: tuple[str, ...]
    runtime_config_path: str
    canonical_output_root: str
    shared_run_root: str
    resource_ledger_path: str
    run_registry_path: str
    readiness_output_root: str
    source_git_commit: str
    source_git_branch: str
    contract_version: str = EVIDENCE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract_version != EVIDENCE_CONTRACT:
            raise ValueError("Unsupported corpus evidence contract")
        values = (
            self.external_roots, self.expected_provider_scope,
            self.runtime_config_path, self.canonical_output_root,
            self.shared_run_root, self.resource_ledger_path,
            self.run_registry_path, self.readiness_output_root,
            self.source_git_commit, self.source_git_branch,
        )
        if not all(values) or any(not str(value).strip() for value in (
            *self.external_roots, *self.expected_provider_scope,
        )):
            raise ValueError("All evidence and output paths must be explicit")

    @property
    def identity(self) -> str:
        payload = asdict(self)
        payload["external_roots"] = sorted(
            str(Path(path).resolve()) for path in self.external_roots
        )
        for key in (
            "runtime_config_path", "canonical_output_root", "shared_run_root",
            "resource_ledger_path", "run_registry_path",
            "readiness_output_root",
        ):
            payload[key] = str(Path(payload[key]).resolve())
        return _identity(payload)


def resolve_corpus_evidence(
    request: CorpusEvidenceRequest, *, output_root: Path,
    selection: Mapping[str, str] | None = None,
    approve_selection: bool = False,
    emit_materialisation_request: bool = False,
    run_plan_only: bool = False,
    strict: bool = False,
    max_depth: int = 5,
    max_candidates: int = 300,
    max_metadata_bytes: int = 4 * 1024 * 1024,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    if min(max_depth, max_candidates, max_metadata_bytes) < 1:
        raise ValueError("Discovery bounds must be positive")
    candidates, unavailable = discover_external_evidence(
        request, max_depth=max_depth, max_candidates=max_candidates,
        max_metadata_bytes=max_metadata_bytes,
    )
    corpus = _canonical_decision(candidates, request)
    assembly = _assembly_decision(candidates, request)
    config = _configuration_analysis(request.runtime_config_path)
    blockers = []
    warnings = [
        {"code": "EXTERNAL_ROOT_UNAVAILABLE", "path": path}
        for path in unavailable
    ]
    if corpus["status"] not in {
        "EXISTING_CANONICAL_CORPUS_READY",
        "EXISTING_CANONICAL_CORPUS_READY_WITH_CONDITIONS",
    }:
        blockers.append({
            "code": "CANONICAL_CORPUS_REQUIRED",
            "evidence": corpus["status"],
        })
    if corpus["status"] not in {
        "EXISTING_CANONICAL_CORPUS_READY",
        "EXISTING_CANONICAL_CORPUS_READY_WITH_CONDITIONS",
    }:
        if assembly["status"] == "SOURCE_ASSEMBLY_NOT_FOUND":
            blockers.append({"code": "SOURCE_ASSEMBLY_NOT_FOUND"})
        elif assembly["status"] == "AMBIGUOUS_SOURCE_ASSEMBLY":
            blockers.append({"code": "AMBIGUOUS_SOURCE_ASSEMBLY"})
    if config["write_enabled"] is False:
        warnings.append({"code": "WRITE_ENABLED_FALSE_SAFETY_GATE"})
    if config["production_validated"] is False:
        blockers.append({"code": "PRODUCTION_VALIDATION_FALSE"})

    chosen_assembly = _approved_candidate(
        candidates, selection, "HISTORICAL_SOURCE_ASSEMBLY"
    )
    materialisation_draft = (
        _materialisation_request(request, chosen_assembly)
        if chosen_assembly else None
    )
    boundary = _validate_output_boundary(request, chosen_assembly)
    if chosen_assembly and not boundary["valid"]:
        blockers.extend(
            {"code": code} for code in boundary["reason_codes"]
        )
    approved = None
    if approve_selection and emit_materialisation_request:
        if chosen_assembly is None:
            blockers.append({"code": "APPROVED_SOURCE_ASSEMBLY_REQUIRED"})
        elif assembly["status"] == "AMBIGUOUS_SOURCE_ASSEMBLY":
            blockers.append({"code": "AMBIGUOUS_APPROVAL_REJECTED"})
        elif boundary["valid"]:
            approved = materialisation_draft
    elif emit_materialisation_request:
        blockers.append({"code": "MATERIALISATION_APPROVAL_REQUIRED"})

    plan_result = None
    repeat_plan_result = None
    if run_plan_only:
        if approved is None:
            blockers.append({"code": "PLAN_ONLY_REQUIRES_APPROVED_REQUEST"})
        else:
            plan_result = _run_plan_only(
                approved, output_root,
                repository_root or Path.cwd(),
            )
            repeat_plan_result = _run_plan_only(
                approved, output_root,
                repository_root or Path.cwd(),
            )
            if plan_result["status"] != "PLAN_VALIDATED_NOT_EXECUTED":
                blockers.append({"code": "PLAN_ONLY_VALIDATION_FAILED"})
            if (
                repeat_plan_result["status"] != "PLAN_VALIDATED_NOT_EXECUTED"
                or repeat_plan_result["run_id"] != plan_result["run_id"]
            ):
                blockers.append({"code": "PLAN_ONLY_IDENTITY_UNSTABLE"})

    decisions = {
        "canonical_corpus": corpus,
        "source_assembly": assembly,
        "timestamp_selection_authority": False,
    }
    _publish(
        output_root, request, candidates, decisions, config, blockers,
        warnings, materialisation_draft, approved, plan_result,
        repeat_plan_result, boundary,
    )
    status = "BLOCKED" if blockers else (
        "READY_WITH_CONDITIONS" if warnings else "READY"
    )
    return {
        "status": status,
        "strict_failure": strict and status != "READY",
        "evidence_identity": request.identity,
        "candidate_count": len(candidates),
        "canonical_status": corpus["status"],
        "assembly_status": assembly["status"],
        "blockers": blockers,
        "warnings": warnings,
        "approved_request_emitted": approved is not None,
        "plan_only_invoked": plan_result is not None,
        "plan_only_result": plan_result,
        "plan_only_repeat_result": repeat_plan_result,
        "external_workspace_mutated": False,
        "execution_authorized": False,
        "lease_acquired": False,
        "network_access_performed": False,
        "model_activation_performed": False,
    }


def discover_external_evidence(
    request: CorpusEvidenceRequest, *, max_depth: int,
    max_candidates: int, max_metadata_bytes: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    unavailable = []
    markers = ("canonical", "corpus", "assembly", "inventory", "manifest")
    for root_text in request.external_roots:
        root = Path(root_text).resolve()
        if not root.exists():
            unavailable.append(str(root))
            continue
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if depth >= max_depth:
                directories[:] = []
            directories[:] = sorted(
                name for name in directories
                if not name.startswith(".")
                and not _temporary(name)
            )
            for name in sorted(files):
                if len(candidates) >= max_candidates:
                    return candidates, unavailable
                lower = name.lower()
                if not any(marker in lower for marker in markers):
                    continue
                if Path(name).suffix.lower() not in {
                    ".json", ".csv", ".jsonl", ".md",
                }:
                    continue
                candidates.append(_classify(
                    current_path / name,
                    max_metadata_bytes=max_metadata_bytes,
                ))
    return sorted(candidates, key=lambda row: row["path"]), unavailable


def _classify(path: Path, *, max_metadata_bytes: int) -> dict[str, Any]:
    lower = str(path).lower()
    row = {
        "notice": NOTICE,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "modified_timestamp_low_trust": path.stat().st_mtime_ns,
        "artifact_class": "UNRECOGNISED",
        "classification": "REJECTED",
        "reason_codes": ["CONTRACT_NOT_RECOGNISED"],
        "logical_identity": None,
        "checksum_identity": None,
        "providers": [],
        "row_count": None,
        "date_range": {},
    }
    if _temporary(lower):
        row.update(
            artifact_class="PARTIAL_OR_TEMPORARY_ARTIFACT",
            reason_codes=["PARTIAL_OR_TEMPORARY_MARKER"],
        )
        return row
    if any(word in lower for word in ("smoke", "probe", "tiny_fixture", "\\dev\\")):
        row.update(
            artifact_class="DEVELOPMENT_OR_SMOKE_ARTIFACT",
            reason_codes=["DEVELOPMENT_OR_SMOKE_PATH"],
        )
    if path.suffix.lower() in {".csv", ".jsonl"}:
        if "canonical_corpus" in path.name.lower():
            row["artifact_class"] = "CANONICAL_CORPUS_BUNDLE"
            row["reason_codes"] = ["MANIFEST_VERIFICATION_REQUIRED"]
        elif "assembly" in path.name.lower():
            row["artifact_class"] = "HISTORICAL_SOURCE_ASSEMBLY"
            row["reason_codes"] = ["SOURCE_METADATA_REQUIRED"]
        return row
    if path.stat().st_size > max_metadata_bytes:
        row["reason_codes"] = ["METADATA_SIZE_LIMIT_EXCEEDED"]
        return row
    if path.suffix.lower() != ".json":
        return row
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("metadata is not an object")
    except Exception as exc:
        row["reason_codes"] = ["MALFORMED_METADATA"]
        row["parse_error_type"] = type(exc).__name__
        return row
    schema = str(payload.get("schema_version") or "")
    if schema.startswith("stock_alpha_news.historical_canonical_corpus."):
        row.update(
            artifact_class="CANONICAL_CORPUS_MANIFEST",
            logical_identity=payload.get("canonical_corpus_identity"),
            checksum_identity=payload.get("canonical_corpus_checksum"),
            row_count=payload.get("canonical_row_count"),
            date_range={
                "published_min": payload.get("published_at_min"),
                "published_max": payload.get("published_at_max"),
                "available_min": payload.get("available_at_min"),
                "available_max": payload.get("available_at_max"),
            },
            manifest=payload,
            reason_codes=[],
        )
    elif (
        "historical_backfill" in schema
        or payload.get("assembly_checksum")
        or payload.get("source_assembly_checksum")
    ):
        providers = _providers(payload)
        row.update(
            artifact_class="SOURCE_ASSEMBLY_METADATA",
            logical_identity=(
                payload.get("source_assembly_identity")
                or payload.get("assembly_checksum")
                or payload.get("checksum")
            ),
            checksum_identity=(
                payload.get("assembly_checksum")
                or payload.get("checksum")
                or payload.get("source_assembly_checksum")
            ),
            providers=providers,
            row_count=payload.get("row_count"),
            date_range={
                "min": payload.get("min_published_at_utc"),
                "max": payload.get("max_published_at_utc"),
            },
            metadata=payload,
            reason_codes=[],
        )
    if any(word in lower for word in (
        "smoke", "probe", "tiny_fixture", "\\dev\\",
    )):
        row["reason_codes"] = sorted(set(
            row["reason_codes"] + ["DEVELOPMENT_OR_SMOKE_PATH"]
        ))
    if (
        row["artifact_class"] in {
            "CANONICAL_CORPUS_MANIFEST", "SOURCE_ASSEMBLY_METADATA",
        }
        and not row["reason_codes"]
    ):
        row["classification"] = "ELIGIBLE"
    return row


def _canonical_decision(candidates, request):
    results = []
    for row in candidates:
        if row["artifact_class"] != "CANONICAL_CORPUS_MANIFEST":
            continue
        manifest = row["manifest"]
        root = Path(row["path"]).parent
        corpus = root / CANONICAL_CORPUS_CSV
        audit = root / CANONICAL_CORPUS_AUDIT_JSON
        summary = root / CANONICAL_CORPUS_SUMMARY_MD
        reasons = []
        if any(marker in row["path"].lower() for marker in (
            "smoke", "probe", "tiny_fixture",
        )):
            reasons.append("DEVELOPMENT_OR_SMOKE_PATH")
        required = (
            "canonical_corpus_identity", "canonical_corpus_checksum",
            "canonical_rows_logical_checksum", "canonical_schema_checksum",
            "logical_manifest_checksum", "source_assembly_checksum",
        )
        if any(not manifest.get(field) for field in required):
            reasons.append("STRENGTHENED_INVENTORY_IDENTITY_MISSING")
        if not corpus.is_file():
            reasons.append("CANONICAL_CORPUS_DATA_MISSING")
        if not audit.is_file() or audit.stat().st_size == 0:
            reasons.append("CANONICAL_CORPUS_AUDIT_MISSING_OR_EMPTY")
        if not summary.is_file():
            reasons.append("CANONICAL_CORPUS_SUMMARY_MISSING")
        if (manifest.get("source_metadata") or {}).get(
            "production_validated"
        ) is False:
            reasons.append("PRODUCTION_VALIDATION_FALSE")
        results.append({
            "path": row["path"],
            "corpus_path": str(corpus),
            "identity": manifest.get("canonical_corpus_identity"),
            "checksum": manifest.get("canonical_corpus_checksum"),
            "row_count": manifest.get("canonical_row_count"),
            "provider_evidence": _providers(
                manifest.get("source_metadata") or {}
            ),
            "reasons": reasons,
        })
    if not results:
        return {"status": "NO_CANONICAL_CORPUS_FOUND", "candidates": []}
    ready = [row for row in results if not row["reasons"]]
    if len(ready) == 1:
        status = "EXISTING_CANONICAL_CORPUS_READY"
    elif ready:
        status = "EXISTING_CANONICAL_CORPUS_INCOMPATIBLE"
    else:
        status = "EXISTING_CANONICAL_CORPUS_INCOMPLETE"
    return {"status": status, "candidates": results}


def _assembly_decision(candidates, request):
    metadata_rows = [
        row for row in candidates
        if row["artifact_class"] == "SOURCE_ASSEMBLY_METADATA"
    ]
    eligible = []
    rejected = []
    expected = {value.lower() for value in request.expected_provider_scope}
    for row in metadata_rows:
        metadata = row["metadata"]
        csv_value = (
            metadata.get("assembly_csv_path")
            or metadata.get("source_assembly_path")
        )
        csv_path = (
            Path(csv_value) if csv_value and Path(csv_value).is_absolute()
            else Path(row["path"]).parent / Path(csv_value or "").name
        )
        reasons = []
        if any(marker in row["path"].lower() for marker in (
            "smoke", "probe", "tiny_fixture",
        )):
            reasons.append("DEVELOPMENT_OR_SMOKE_PATH")
        providers = {value.lower() for value in row["providers"]}
        if not csv_path.is_file():
            reasons.append("SOURCE_ASSEMBLY_DATA_MISSING")
        if not row["checksum_identity"]:
            reasons.append("SOURCE_ASSEMBLY_CHECKSUM_MISSING")
        if not expected.issubset(providers):
            reasons.append("EXPECTED_PROVIDER_SCOPE_MISSING")
        candidate = {**row, "assembly_path": str(csv_path.resolve()),
                     "reason_codes": reasons}
        (eligible if not reasons else rejected).append(candidate)
    if len(eligible) > 1:
        status = "AMBIGUOUS_SOURCE_ASSEMBLY"
    elif len(eligible) == 1:
        status = "SOURCE_ASSEMBLY_READY_FOR_MATERIALISATION"
    elif metadata_rows:
        status = "SOURCE_ASSEMBLY_INCOMPLETE"
    else:
        status = "SOURCE_ASSEMBLY_NOT_FOUND"
    return {"status": status, "eligible": eligible, "rejected": rejected}


def _configuration_analysis(path_text):
    path = Path(path_text)
    payload = {}
    if path.is_file():
        import yaml
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ml = payload.get("ml") or {}
    config = ml.get("stock_alpha_news_canonical_corpus") or {}
    return {
        "notice": NOTICE,
        "path": str(path.resolve()),
        "write_enabled": config.get("write_enabled"),
        "write_gate_semantics": (
            "HARD_OWNER_GATE: materialize_historical_canonical_corpus raises "
            "before reading the source when false"
        ),
        "production_validated": ml.get("production_validated"),
        "production_validation_semantics": (
            "OPERATOR/WORKFLOW_DECLARATION: not consumed by the canonical "
            "materialisation owner; remains readiness evidence"
        ),
        "runtime_configuration_identity": _identity(config),
    }


def _validate_output_boundary(request, assembly):
    output = Path(request.canonical_output_root).resolve()
    reasons = []
    source_parent = (
        Path(assembly.get("assembly_path") or assembly["path"]).resolve().parent
        if assembly else None
    )
    aliases = {
        Path(request.shared_run_root).resolve(),
        Path(request.resource_ledger_path).resolve(),
        Path(request.run_registry_path).resolve(),
        Path(request.readiness_output_root).resolve(),
    }
    if source_parent is not None and (
        output == source_parent or source_parent in output.parents
    ):
        reasons.append("CANONICAL_OUTPUT_NESTED_IN_SOURCE")
    if output in aliases:
        reasons.append("CANONICAL_OUTPUT_ALIASES_OPERATOR_PATH")
    if output.exists():
        reasons.append("CANONICAL_OUTPUT_ALREADY_EXISTS")
    return {
        "notice": NOTICE,
        "path": str(output),
        "valid": not reasons,
        "reason_codes": reasons,
        "exists_before": output.exists(),
        "expected_paths": {
            "csv": str(output / CANONICAL_CORPUS_CSV),
            "manifest": str(output / CANONICAL_CORPUS_MANIFEST_JSON),
            "audit": str(output / CANONICAL_CORPUS_AUDIT_JSON),
            "summary_markdown": str(output / CANONICAL_CORPUS_SUMMARY_MD),
        },
    }


def _approved_candidate(candidates, selection, artifact_class):
    if not selection:
        return None
    selected = selection.get(artifact_class)
    if not selected:
        return None
    resolved = str(Path(selected).resolve())
    selected_checksum = selection.get("__assembly_checksum")
    for row in candidates:
        metadata = row.get("metadata") or {}
        csv_value = (
            metadata.get("assembly_csv_path")
            or metadata.get("source_assembly_path")
        )
        assembly_path = (
            str((Path(row["path"]).parent / Path(csv_value).name).resolve())
            if csv_value else ""
        )
        if (
            row["artifact_class"] == "SOURCE_ASSEMBLY_METADATA"
            and assembly_path == resolved
            and (
                not selected_checksum
                or row.get("checksum_identity") == selected_checksum
            )
        ):
            return {**row, "assembly_path": assembly_path}
    for row in candidates:
        if (
            row["artifact_class"] == artifact_class
            and row["path"] == resolved
            and (
                not selected_checksum
                or row.get("checksum_identity") == selected_checksum
            )
        ):
            return row
    return None


def _materialisation_request(request, assembly):
    metadata = assembly.get("metadata") or {}
    source_path = assembly.get("assembly_path") or assembly["path"]
    output = Path(request.canonical_output_root)
    payload = {
        "contract_version": MATERIALISATION_CONTRACT,
        "notice": NOTICE,
        "execution_authorized": False,
        "state": "PREFLIGHT_ONLY",
        "source_assembly_path": source_path,
        "source_assembly_identity": assembly["logical_identity"],
        "source_assembly_checksum": assembly["checksum_identity"],
        "source_metadata_path": assembly["path"],
        "source_metadata_identity": _identity(metadata),
        "provider_inventory": assembly["providers"],
        "expected_provider_scope": list(request.expected_provider_scope),
        "expected_row_count": metadata.get("row_count"),
        "expected_unique_article_count": (
            metadata.get("unique_provider_article_count")
            or metadata.get("unique_article_count")
        ),
        "expected_symbol_count": metadata.get("symbol_count"),
        "expected_published_min": metadata.get("min_published_at_utc"),
        "expected_published_max": metadata.get("max_published_at_utc"),
        "complete_partition_count": metadata.get("complete_partition_count"),
        "incomplete_partition_count": metadata.get("incomplete_partition_count"),
        "canonical_output_root": str(output.resolve()),
        "expected_corpus_path": str(
            (output / CANONICAL_CORPUS_CSV).resolve()
        ),
        "expected_manifest_path": str(
            (output / CANONICAL_CORPUS_MANIFEST_JSON).resolve()
        ),
        "expected_inventory_path": str(
            (output / CANONICAL_CORPUS_AUDIT_JSON).resolve()
        ),
        "expected_audit_path": str(
            (output / CANONICAL_CORPUS_AUDIT_JSON).resolve()
        ),
        "expected_summary_path": str(
            (output / CANONICAL_CORPUS_SUMMARY_MD).resolve()
        ),
        "canonical_contract_version":
            HISTORICAL_CANONICAL_CORPUS_SCHEMA_VERSION,
        "availability_time_policy_identity":
            "canonical_availability_precedence.v1",
        "duplicate_identity_contract":
            HISTORICAL_CANONICAL_TRANSFORMATION_VERSION,
        "runtime_configuration_identity": _configuration_analysis(
            request.runtime_config_path
        )["runtime_configuration_identity"],
        "shared_compute_run_root": request.shared_run_root,
        "resource_ledger_path": request.resource_ledger_path,
        "run_registry_path": request.run_registry_path,
        "readiness_output_root": request.readiness_output_root,
        "resource_profile": {
            "estimated_peak_ram_bytes": 4 * 1024**3,
            "cpu_weight": 1, "inner_threads": 1,
            "gpu_required": False,
            "estimate_source": "CONSERVATIVE_DEFAULT",
        },
        "execution_policy": "CPU_ONLY_NON_MODEL",
        "source_git_commit_evidence": request.source_git_commit,
        "source_git_branch_evidence": request.source_git_branch,
    }
    payload["logical_request_identity"] = _identity({
        key: value for key, value in payload.items()
        if key not in {
            "source_git_commit_evidence", "source_git_branch_evidence",
            "logical_request_identity",
        }
    })
    return payload


def _run_plan_only(approved, output_root, repository_root):
    from core.research.ml.stock_level.stock_alpha_news_data_compute import (
        NewsDataMaterialisationPlan,
        build_news_data_resource_request,
        deterministic_news_data_run_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    plan_request = {
        "plan": {
            "selected_stages": [CANONICAL_STAGE],
            "source_inventory_identity":
                approved["source_assembly_identity"],
            "source_inventory_checksum":
                approved["source_assembly_checksum"],
            "canonical_corpus_contract_identity":
                approved["canonical_contract_version"],
            "canonical_output_compatibility_identity":
                approved["logical_request_identity"],
            "canonical_parent_identity":
                approved["source_assembly_identity"],
            "canonical_parent_checksum":
                approved["source_assembly_checksum"],
            "date_boundary_identity": "source_metadata_date_boundary",
            "universe_identity": _identity(approved["provider_inventory"]),
            "availability_policy_identity":
                approved["availability_time_policy_identity"],
            "pit_feature_contract_identity": "",
            "feature_output_compatibility_identity": "",
            "configuration_checksum":
                approved["runtime_configuration_identity"],
            "source_git_commit": approved["source_git_commit_evidence"],
            "corpus_work_units": [],
            "feature_work_units": [],
        }
    }
    request_path = output_root / "plan_only_input.json"
    _atomic_json(request_path, {"notice": NOTICE, **plan_request})
    command = [
        sys.executable,
        str(repository_root / "scripts/run_stock_alpha_news_data_compute.py"),
        "--request", str(request_path),
        "--run-root", approved["shared_compute_run_root"],
        "--resource-ledger", approved["resource_ledger_path"],
        "--registry", approved["run_registry_path"],
        "--plan-only",
    ]
    completed = subprocess.run(
        command, cwd=repository_root, capture_output=True,
        text=True, timeout=30, check=False,
    )
    try:
        bounded_stdout = json.loads(completed.stdout)
    except json.JSONDecodeError:
        bounded_stdout = {"status": "MALFORMED_PLAN_OUTPUT"}
    plan = NewsDataMaterialisationPlan(**{
        **plan_request["plan"],
        "selected_stages": tuple(plan_request["plan"]["selected_stages"]),
        "corpus_work_units": tuple(plan_request["plan"]["corpus_work_units"]),
        "feature_work_units": tuple(plan_request["plan"]["feature_work_units"]),
    })
    run_id = deterministic_news_data_run_id(plan)
    resource = build_news_data_resource_request(
        run_id=run_id, item_id=f"{run_id}-corpus-plan",
        stage=CANONICAL_STAGE, attempt_identity="plan-only",
    )
    return {
        "notice": NOTICE,
        "status": (
            "PLAN_VALIDATED_NOT_EXECUTED"
            if completed.returncode == 0
            and bounded_stdout.get("status") == "PLAN_VALID"
            else "PLAN_VALIDATION_FAILED"
        ),
        "exit_code": completed.returncode,
        "run_id": bounded_stdout.get("run_id"),
        "selected_stages": list(plan.selected_stages),
        "planned_work_item_count": 1,
        "resource_request": asdict(resource),
        "output_boundary": approved["canonical_output_root"],
        "lease_acquired": False,
        "source_assembly_opened": False,
        "execution_performed": False,
    }


def _publish(output_root, request, candidates, decisions, config, blockers,
             warnings, draft, approved, plan_result, repeat_plan_result,
             boundary):
    output_root.mkdir(parents=True, exist_ok=True)
    rejected = [
        row for row in candidates if row["classification"] != "ELIGIBLE"
    ]
    payloads = {
        "evidence_request.json": {
            "notice": NOTICE, **asdict(request),
            "evidence_identity": request.identity,
        },
        "candidate_inventory.json": {
            "notice": NOTICE, "candidates": candidates,
        },
        "candidate_rejections.json": {
            "notice": NOTICE, "rejections": rejected,
        },
        "canonical_corpus_evidence.json": {
            "notice": NOTICE, **decisions["canonical_corpus"],
        },
        "source_assembly_evidence.json": {
            "notice": NOTICE, **decisions["source_assembly"],
        },
        "configuration_gate_analysis.json": config,
        "selection_decisions.json": {"notice": NOTICE, **decisions},
        "blockers.json": {"notice": NOTICE, "blockers": blockers},
        "warnings.json": {"notice": NOTICE, "warnings": warnings},
        "lineage_graph.json": {
            "notice": NOTICE,
            "source_assembly": (
                draft.get("source_assembly_identity") if draft else None
            ),
            "expected_canonical_output": request.canonical_output_root,
        },
        "output_boundary_validation.json": boundary,
        "readiness_update.json": {
            "notice": NOTICE,
            "canonical_stage":
                "SOURCE_READY_PLAN_VALIDATED_MATERIALISATION_NOT_AUTHORIZED"
                if plan_result
                and plan_result["status"] == "PLAN_VALIDATED_NOT_EXECUTED"
                else "SOURCE_EVIDENCE_REVIEW_REQUIRED",
            "resolved": [
                "SOURCE_ASSEMBLY_NOT_FOUND",
                "ASSEMBLY_IDENTITY_UNCERTAINTY",
                "PROVIDER_SCOPE_UNCERTAINTY",
            ] if approved else [],
            "execution_authorized": False,
        },
    }
    if draft:
        payloads["canonical_materialisation_request.draft.json"] = draft
    if approved:
        payloads["canonical_materialisation_request.json"] = approved
    if plan_result:
        payloads["plan_only_result.json"] = plan_result
    if repeat_plan_result:
        payloads["plan_only_repeat_result.json"] = repeat_plan_result
        payloads["request_identity_comparison.json"] = {
            "notice": NOTICE,
            "request_identity": (
                approved["logical_request_identity"] if approved else None
            ),
            "first_run_id": plan_result["run_id"],
            "repeat_run_id": repeat_plan_result["run_id"],
            "request_identity_stable": True,
            "run_identity_stable":
                plan_result["run_id"] == repeat_plan_result["run_id"],
        }
    for name, payload in payloads.items():
        _assert_private(payload)
        _atomic_json(output_root / name, payload)
    review = _review(
        decisions, config, blockers, request, approved, plan_result,
        repeat_plan_result, boundary,
    )
    _assert_private(review)
    _atomic_text(output_root / "operator_review.md", review)


def _review(decisions, config, blockers, request, approved, plan_result,
            repeat_plan_result, boundary):
    corpus = decisions["canonical_corpus"]
    assembly = decisions["source_assembly"]
    return "\n".join([
        "# Canonical Corpus Evidence Review", "", NOTICE, "",
        f"- Canonical corpus status: `{corpus['status']}`",
        f"- Source assembly status: `{assembly['status']}`",
        f"- Approved source path: "
        f"`{approved['source_assembly_path'] if approved else 'NOT_APPROVED'}`",
        f"- Approved source checksum: "
        f"`{approved['source_assembly_checksum'] if approved else 'NOT_APPROVED'}`",
        f"- Source metadata: "
        f"`{approved['source_metadata_path'] if approved else 'NOT_APPROVED'}`",
        f"- Provider evidence: "
        f"`{approved['provider_inventory'] if approved else 'NOT_APPROVED'}`",
        f"- Proposed output root: `{request.canonical_output_root}`",
        f"- Output root absent and isolated: "
        f"`{boundary['valid'] and not boundary['exists_before']}`",
        f"- write_enabled: `{config['write_enabled']}` "
        f"({config['write_gate_semantics']})",
        f"- production_validated: `{config['production_validated']}` "
        f"({config['production_validation_semantics']})", "",
        "The 1.5 GB CSV is a canonical-corpus candidate only; its manifest and "
        "publication sidecars determine readiness.",
        "The expected approximately 317 MB historical assembly is authoritative "
        "input only when its metadata, checksum and provider scope are present.",
        "Timestamps were not used as selection authority.", "",
        "## Blockers", "",
        *([f"- `{row['code']}`" for row in blockers] or ["- None"]),
        "", "## Plan-only", "",
        f"- Result: `{plan_result['status'] if plan_result else 'NOT_RUN'}`",
        f"- Exit code: `{plan_result['exit_code'] if plan_result else 'NOT_RUN'}`",
        f"- Deterministic request identity: "
        f"`{approved['logical_request_identity'] if approved else 'NOT_EMITTED'}`",
        f"- First run ID: `{plan_result['run_id'] if plan_result else 'NOT_RUN'}`",
        f"- Repeat run ID: "
        f"`{repeat_plan_result['run_id'] if repeat_plan_result else 'NOT_RUN'}`",
        f"- Resource request: "
        f"`{plan_result['resource_request'] if plan_result else 'NOT_RUN'}`",
        "- Command: `python scripts/resolve_stock_alpha_news_corpus_evidence.py "
        "--evidence-request <request> --output-root <new-root> --selection "
        "<reviewed-selection> --approve-selection "
        "--emit-materialisation-request --run-plan-only --strict --json`", "",
        "## Eventual production boundary", "",
        "DO NOT RUN - REQUIRES SEPARATE OPERATOR AUTHORIZATION",
        "Preserve the source assembly. Never overwrite an incompatible canonical "
        "bundle. Publication must remain atomic. Preserve partial evidence for "
        "diagnosis, do not delete compatible artifacts, inspect lease state "
        "before retrying, and reuse the same logical request after remediation.",
        "", "Pre-execution checklist: reviewed write-enabled configuration; "
        "isolated output still absent or empty; assembly checksum reverified; "
        "disk capacity checked; ledger and registry paths approved; no "
        "competing materialisation run; operator authorization recorded.",
        "", f"Approved request emitted: `{approved is not None}`",
        "No production execution occurred.",
        "No scoring, certification, PIT, model, or network operation occurred.",
        "The news-transformer trainer remains out of scope.", "",
    ])


def _providers(payload):
    explicit = payload.get("providers") or payload.get("provider_inventory")
    providers = set()
    if isinstance(explicit, list):
        providers.update(str(value) for value in explicit)
    distribution = payload.get("source_distribution") or {}
    if isinstance(distribution, Mapping):
        providers.update(str(value) for value in distribution)
    if "alpaca_benzinga" in str(payload.get("schema_version", "")).lower():
        providers.update(("Alpaca", "Benzinga"))
    return sorted(providers)


def _temporary(value):
    lower = str(value).lower()
    return any(marker in lower for marker in (
        ".tmp", ".partial", ".incomplete", "~",
    ))


def _assert_private(value):
    forbidden_keys = {
        "headline", "summary", "body", "body_or_full_text",
        "raw_provider_payload", "api_key", "token", "secret",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden_keys:
                raise ValueError("Private or row-level evidence is prohibited")
            _assert_private(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_private(child)
    elif isinstance(value, str) and len(value) > 32_768:
        raise ValueError("Unbounded evidence string is prohibited")


def _identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def _atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
