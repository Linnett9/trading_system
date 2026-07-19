from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.stock_level.stock_alpha_news_compute_readiness import (
    AUDIT_CONTRACT,
    NewsComputeReadinessRequest,
    audit_news_compute_readiness,
)

DISCOVERY_CONTRACT = "stock_alpha_news_compute_readiness_discovery.v1"
READ_ONLY_NOTICE = "READ_ONLY — NOT PRODUCTION EXECUTION AUTHORIZATION"
CLASSIFICATIONS = {
    "ELIGIBLE", "INELIGIBLE_DEVELOPMENT_ONLY",
    "INELIGIBLE_SMOKE_OR_PROBE", "INELIGIBLE_INCOMPLETE",
    "INELIGIBLE_MALFORMED", "INELIGIBLE_WRONG_CONTRACT",
    "INELIGIBLE_LINEAGE_MISMATCH", "AMBIGUOUS_MULTIPLE_CANDIDATES",
}


@dataclass(frozen=True)
class NewsReadinessDiscoveryRequest:
    discovery_roots: tuple[str, ...]
    selected_stages: tuple[str, ...]
    expected_provider_scope: str
    expected_model_id: str
    expected_model_revision: str
    expected_tokenizer_id: str
    expected_tokenizer_revision: str
    canonical_source_path: str
    runtime_config_path: str
    candidate_canonical_corpus_root: str
    candidate_score_store_root: str
    candidate_pit_feature_store_root: str
    candidate_model_cache_root: str
    candidate_run_root: str
    candidate_resource_ledger: str
    candidate_registry: str
    final_audit_output_root: str
    contract_version: str = DISCOVERY_CONTRACT

    def __post_init__(self) -> None:
        if self.contract_version != DISCOVERY_CONTRACT:
            raise ValueError("Unsupported discovery request contract")
        required = (
            self.discovery_roots, self.selected_stages,
            self.expected_provider_scope, self.expected_model_id,
            self.expected_model_revision, self.expected_tokenizer_id,
            self.expected_tokenizer_revision, self.canonical_source_path,
            self.runtime_config_path, self.candidate_canonical_corpus_root,
            self.candidate_score_store_root,
            self.candidate_pit_feature_store_root,
            self.candidate_model_cache_root, self.candidate_run_root,
            self.candidate_resource_ledger, self.candidate_registry,
            self.final_audit_output_root,
        )
        if not all(required):
            raise ValueError("Explicit discovery roots, pins, and paths are required")
        if any(not str(root).strip() for root in self.discovery_roots):
            raise ValueError("Discovery roots cannot be blank")

    @property
    def identity(self) -> str:
        return _hash(_normalised_request(self))


def build_news_readiness_request(
    discovery: NewsReadinessDiscoveryRequest, *, output_root: Path,
    selection: Mapping[str, str] | None = None, approve_selection: bool = False,
    run_audit: bool = False, strict: bool = False, max_depth: int = 6,
    max_candidates: int = 500, max_file_bytes: int = 8 * 1024 * 1024,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    if min(max_depth, max_candidates, max_file_bytes) < 1:
        raise ValueError("Discovery bounds must be positive")
    repository = (repository_root or Path.cwd()).resolve()
    candidates, unavailable = discover_candidates(
        discovery, max_depth=max_depth, max_candidates=max_candidates,
        max_file_bytes=max_file_bytes,
    )
    candidates.append(_inspect_model_cache(discovery))
    candidates = sorted(candidates, key=lambda row: row["path"])
    decisions = _decisions(candidates, discovery)
    blockers = _request_blockers(decisions, discovery)
    warnings = [
        {"code": "DISCOVERY_ROOT_UNAVAILABLE", "path": path}
        for path in unavailable
    ]
    selected: dict[str, dict[str, Any]] = {}
    if selection:
        by_path = {row["path"]: row for row in candidates}
        for artifact_type, path in selection.items():
            candidate = by_path.get(str(Path(path).resolve()))
            if candidate is None:
                blockers.append({
                    "code": "SELECTED_CANDIDATE_NOT_DISCOVERED",
                    "artifact_type": artifact_type, "path": path,
                })
            elif candidate["artifact_type"] != artifact_type:
                blockers.append({
                    "code": "SELECTED_CANDIDATE_TYPE_MISMATCH",
                    "artifact_type": artifact_type, "path": path,
                })
            elif candidate["classification"] != "ELIGIBLE":
                blockers.append({
                    "code": "SELECTED_CANDIDATE_INELIGIBLE",
                    "artifact_type": artifact_type, "path": path,
                    "reason_codes": candidate["reason_codes"],
                })
            else:
                selected[artifact_type] = candidate
    draft = _draft_request(discovery, selected, output_root)
    approved_request = None
    audit_report = None
    if approve_selection:
        if not selection:
            blockers.append({"code": "APPROVED_SELECTION_FILE_REQUIRED"})
        required = _required_candidate_types(discovery.selected_stages)
        missing = sorted(required - set(selected))
        if missing:
            blockers.append({
                "code": "APPROVED_SELECTION_INCOMPLETE",
                "missing_artifact_types": missing,
            })
        ambiguous = [
            row for row in decisions
            if row["decision"] == "AMBIGUOUS_MULTIPLE_CANDIDATES"
            and row["artifact_type"] in required
        ]
        if ambiguous:
            blockers.append({
                "code": "AMBIGUOUS_APPROVAL_REJECTED",
                "artifact_types": [row["artifact_type"] for row in ambiguous],
            })
        if not blockers:
            approved_request = NewsComputeReadinessRequest(**draft)
            if run_audit:
                audit_report = audit_news_compute_readiness(
                    approved_request, repository_root=repository
                )
    elif run_audit:
        blockers.append({"code": "AUDIT_REQUIRES_APPROVED_SELECTION"})
    remediation = _remediation(blockers, decisions)
    result = {
        "contract_version": DISCOVERY_CONTRACT,
        "notice": READ_ONLY_NOTICE,
        "discovery_identity": discovery.identity,
        "bounds": {
            "max_depth": max_depth, "max_candidates": max_candidates,
            "max_file_bytes": max_file_bytes,
        },
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(
            row["classification"] == "ELIGIBLE" for row in candidates
        ),
        "blockers": blockers, "warnings": warnings,
        "approved_request_emitted": approved_request is not None,
        "audit_invoked": audit_report is not None,
        "audit_result": (
            audit_report["overall_readiness"] if audit_report else None
        ),
        "read_only": True,
        "production_execution_performed": False,
        "model_activation_performed": False,
        "network_access_performed": False,
        "execution_lease_acquired": False,
    }
    _publish(
        output_root, discovery, candidates, decisions, draft, blockers,
        warnings, selected, remediation, approved_request, audit_report,
    )
    result["status"] = (
        "BLOCKED" if blockers else
        "READY_WITH_CONDITIONS" if warnings or not approve_selection else
        (audit_report["overall_readiness"] if audit_report else "READY")
    )
    if strict and result["status"] != "READY":
        result["strict_failure"] = True
    return result


def discover_candidates(
    request: NewsReadinessDiscoveryRequest, *, max_depth: int,
    max_candidates: int, max_file_bytes: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = []
    unavailable = []
    accepted = {
        ".json", ".yaml", ".yml", ".csv",
    }
    for root_text in request.discovery_roots:
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
                if not name.startswith(".") and name not in {"__pycache__"}
            )
            for name in sorted(files):
                path = current_path / name
                if path.suffix.lower() not in accepted:
                    continue
                if len(candidates) >= max_candidates:
                    return candidates, unavailable
                candidates.append(_classify(
                    path, request, max_file_bytes=max_file_bytes
                ))
    return sorted(candidates, key=lambda row: row["path"]), unavailable


def _inspect_model_cache(
    request: NewsReadinessDiscoveryRequest,
) -> dict[str, Any]:
    root = Path(request.candidate_model_cache_root).resolve()
    required_groups = {
        "config": ("config.json",),
        "tokenizer_config": ("tokenizer_config.json",),
        "tokenizer_assets": ("vocab.txt", "tokenizer.json"),
        "weights": ("model.safetensors", "pytorch_model.bin"),
    }
    present = []
    missing = []
    incomplete = []
    if root.is_dir():
        names = {entry.name for entry in root.iterdir()}
        present = sorted(names)
        incomplete = sorted(
            name for name in names
            if name.endswith((".incomplete", ".lock", ".tmp"))
        )
        for group, alternatives in required_groups.items():
            if not any(name in names for name in alternatives):
                missing.append(group)
    else:
        missing = sorted(required_groups)
    revision_matches = root.name == request.expected_model_revision
    ready = root.is_dir() and revision_matches and not missing and not incomplete
    return {
        "notice": READ_ONLY_NOTICE,
        "path": str(root),
        "modified_timestamp_low_trust": None,
        "size_bytes": None,
        "artifact_type": "MODEL_CACHE",
        "contract": "huggingface_snapshot_reference.v1",
        "logical_identity": (
            f"{request.expected_model_id}@{request.expected_model_revision}"
        ),
        "checksum_identity": request.expected_model_revision,
        "parent_identities": {},
        "date_range": {},
        "counts": {"bounded_top_level_entries": len(present)},
        "complete": ready,
        "classification": (
            "ELIGIBLE" if ready else "INELIGIBLE_INCOMPLETE"
        ),
        "reference_state": (
            "MODEL_CACHE_REFERENCE_READY" if ready else
            "MODEL_CACHE_REFERENCE_INCOMPLETE" if root.exists() else
            "MODEL_CACHE_REFERENCE_UNVERIFIED"
        ),
        "reason_codes": [] if ready else sorted({
            *(["MODEL_CACHE_PATH_UNAVAILABLE"] if not root.exists() else []),
            *(["MODEL_CACHE_REVISION_MISMATCH"] if not revision_matches else []),
            *(["MODEL_CACHE_REQUIRED_FILE_MISSING"] if missing else []),
            *(["MODEL_CACHE_INCOMPLETE_MARKER"] if incomplete else []),
        }),
        "bounded_file_names": present[:50],
        "missing_asset_groups": missing,
        "incomplete_markers": incomplete[:20],
        "model_or_tokenizer_activated": False,
        "network_access_performed": False,
        "file_presence_proves_successful_inference": False,
    }


def _classify(
    path: Path, request: NewsReadinessDiscoveryRequest, *,
    max_file_bytes: int,
) -> dict[str, Any]:
    base = {
        "notice": READ_ONLY_NOTICE, "path": str(path.resolve()),
        "modified_timestamp_low_trust":
            datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "size_bytes": path.stat().st_size, "artifact_type": "UNKNOWN",
        "contract": None, "logical_identity": None, "checksum_identity": None,
        "parent_identities": {}, "date_range": {}, "counts": {},
        "complete": False, "classification": "INELIGIBLE_WRONG_CONTRACT",
        "reason_codes": ["CONTRACT_NOT_RECOGNISED"],
    }
    lower = str(path).lower()
    if any(marker in lower for marker in ("smoke", "probe", "tiny_fixture")):
        base.update(
            classification="INELIGIBLE_SMOKE_OR_PROBE",
            reason_codes=["SMOKE_OR_PROBE_PATH"],
        )
    elif any(marker in lower for marker in ("development", "\\dev\\", "/dev/")):
        base.update(
            classification="INELIGIBLE_DEVELOPMENT_ONLY",
            reason_codes=["DEVELOPMENT_ONLY_PATH"],
        )
    if path.stat().st_size > max_file_bytes:
        base.update(
            classification="INELIGIBLE_INCOMPLETE",
            reason_codes=["METADATA_SIZE_LIMIT_EXCEEDED"],
        )
        return base
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        return base
    try:
        payload = _metadata(path)
    except Exception as exc:
        base.update(
            classification="INELIGIBLE_MALFORMED",
            reason_codes=["MANIFEST_MALFORMED"],
            parse_error_type=type(exc).__name__,
        )
        return base
    details = _contract_details(payload)
    base.update(details)
    if details["artifact_type"] == "UNKNOWN":
        return base
    reasons = []
    if base["classification"] in {
        "INELIGIBLE_DEVELOPMENT_ONLY", "INELIGIBLE_SMOKE_OR_PROBE"
    }:
        reasons.extend(base["reason_codes"])
    canonical_config = (
        (payload.get("ml") or {}).get("stock_alpha_news_canonical_corpus")
        if isinstance(payload.get("ml"), Mapping) else None
    )
    production_validated = payload.get("production_validated")
    if production_validated is None and isinstance(payload.get("ml"), Mapping):
        production_validated = payload["ml"].get("production_validated")
    if production_validated is False:
        reasons.append("PRODUCTION_VALIDATION_FALSE")
    write_enabled = payload.get("write_enabled")
    if write_enabled is None and isinstance(canonical_config, Mapping):
        write_enabled = canonical_config.get("write_enabled")
    if write_enabled is False and details[
        "artifact_type"] == "CANONICAL_CONFIG":
        reasons.append("CANONICAL_WRITE_DISABLED")
    if not details["complete"]:
        reasons.append("INCOMPLETE_AUTHORITATIVE_EVIDENCE")
    if details["artifact_type"] == "SCORING_PLAN":
        model = payload.get("finbert_model_identity") or {}
        if (
            model.get("model_id") != request.expected_model_id
            or model.get("model_revision") != request.expected_model_revision
            or model.get("tokenizer_id") != request.expected_tokenizer_id
            or model.get("tokenizer_revision")
            != request.expected_tokenizer_revision
        ):
            reasons.append("MODEL_OR_TOKENIZER_PIN_MISMATCH")
    if reasons:
        if base["classification"] not in {
            "INELIGIBLE_DEVELOPMENT_ONLY", "INELIGIBLE_SMOKE_OR_PROBE"
        }:
            base["classification"] = "INELIGIBLE_INCOMPLETE"
        base["reason_codes"] = sorted(set(reasons))
    else:
        base["classification"] = "ELIGIBLE"
        base["reason_codes"] = []
    return base


def _contract_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") in {
        "stock_alpha_news.historical_canonical_corpus.v1",
        "stock_alpha_news.historical_canonical_corpus.v2",
    }:
        return _details(
            "CANONICAL_CORPUS", payload["schema_version"],
            payload.get("canonical_corpus_identity"),
            payload.get("canonical_corpus_checksum"),
            {"source_assembly": payload.get("source_assembly_identity")},
            {"min": payload.get("published_at_min"),
             "max": payload.get("published_at_max")},
            {"rows": payload.get("canonical_row_count"),
             "providers": payload.get("provider_count")},
            all(payload.get(key) not in (None, "") for key in (
                "canonical_corpus_identity", "canonical_corpus_checksum",
                "logical_manifest_checksum", "canonical_schema_checksum",
                "canonical_row_count", "source_assembly_identity",
                "duplicate_group_count", "ingested_at_utc",
            )),
        )
    if payload.get("scoring_plan_contract") == (
        "stock_alpha_finbert_production_scoring_plan.v1"
    ):
        return _details(
            "SCORING_PLAN", payload["scoring_plan_contract"],
            payload.get("logical_checksum"), payload.get("plan_artifact_checksum"),
            {"canonical_corpus_identity":
                 payload.get("canonical_corpus_identity"),
             "canonical_corpus_checksum":
                 payload.get("canonical_corpus_checksum")},
            {}, {"chunks": payload.get("expected_chunk_count"),
                 "rows": payload.get("expected_article_count")},
            bool(payload.get("logical_checksum")
                 and payload.get("finbert_model_identity")
                 and payload.get("expected_chunks")),
        )
    if payload.get("score_store_contract") == (
        "stock_alpha_finbert_production_score_store.v1"
    ):
        return _details(
            "SCORE_STORE_CERTIFICATE", payload["score_store_contract"],
            payload.get("score_store_identity"), payload.get(
                "score_store_checksum"),
            {"scoring_plan": payload.get(
                "production_scoring_plan_identity"),
             "canonical_corpus": payload.get("canonical_corpus_identity")},
            {}, {"chunks": payload.get("certified_completed_chunk_count"),
                 "rows": payload.get("certified_scored_row_count")},
            payload.get("production_scoring_complete") is True
            and payload.get("status") == "COMPLETE",
        )
    if payload.get("feature_store_contract") == (
        "canonical_partitioned_pit_news_feature_store.v1"
    ):
        return _details(
            "PIT_FEATURE_STORE", payload["feature_store_contract"],
            payload.get("logical_checksum"),
            payload.get("feature_store_artifact_checksum"),
            {key: payload.get(key) for key in (
                "canonical_corpus_identity", "score_store_identity",
                "canonical_daily_spine_identity", "ticker_mapping_identity",
                "pit_eligibility_policy_identity",
            )}, payload.get("coverage") or {},
            {"rows": payload.get("row_count"),
             "partitions": len(payload.get("partitions") or [])},
            bool(payload.get("logical_checksum")
                 and payload.get("feature_schema_checksum")
                 and isinstance(payload.get("partitions"), list)),
        )
    if payload.get("daily_spine_identity"):
        return _details(
            "DAILY_SPINE", "canonical_daily_spine",
            payload.get("daily_spine_identity"),
            payload.get("daily_spine_checksum"), {}, payload.get(
                "coverage") or {}, {"rows": payload.get("row_count")},
            bool(payload.get("daily_spine_checksum")),
        )
    if payload.get("ticker_mapping_identity"):
        return _details(
            "TICKER_MAPPING", "canonical_ticker_mapping",
            payload.get("ticker_mapping_identity"),
            payload.get("ticker_mapping_checksum"), {}, {},
            {"aliases": payload.get("alias_count")},
            bool(payload.get("ticker_mapping_checksum")),
        )
    if payload.get("alias_mapping_identity") or payload.get("aliases"):
        return _details(
            "ALIAS_MAPPING", "alias_mapping",
            payload.get("alias_mapping_identity") or _hash(payload),
            payload.get("logical_checksum") or _hash(payload), {}, {},
            {"aliases": len(payload.get("aliases") or payload)}, True,
        )
    canonical_config = (
        (payload.get("ml") or {}).get("stock_alpha_news_canonical_corpus")
        if isinstance(payload.get("ml"), Mapping) else None
    )
    if canonical_config:
        return _details(
            "CANONICAL_CONFIG", "canonical_corpus_config",
            _hash(canonical_config), canonical_config.get(
                "expected_source_checksum"), {}, {},
            {}, bool(canonical_config.get("source_assembly_csv_path")
                     and canonical_config.get("output_dir")),
        )
    return _details("UNKNOWN", None, None, None, {}, {}, {}, False)


def _details(artifact_type, contract, identity, checksum_value, parents,
             date_range, counts, complete):
    return {
        "artifact_type": artifact_type, "contract": contract,
        "logical_identity": identity, "checksum_identity": checksum_value,
        "parent_identities": parents, "date_range": date_range,
        "counts": counts, "complete": complete,
    }


def _decisions(candidates, request):
    decisions = []
    for artifact_type in sorted({
        row["artifact_type"] for row in candidates if row["artifact_type"] != "UNKNOWN"
    } | _required_candidate_types(request.selected_stages)):
        eligible = [
            row for row in candidates
            if row["artifact_type"] == artifact_type
            and row["classification"] == "ELIGIBLE"
        ]
        if len(eligible) == 1:
            decision = "UNIQUE_ELIGIBLE_CANDIDATE"
        elif len(eligible) > 1:
            decision = "AMBIGUOUS_MULTIPLE_CANDIDATES"
            for row in eligible:
                row["classification"] = "AMBIGUOUS_MULTIPLE_CANDIDATES"
                row["reason_codes"] = ["MULTIPLE_CONTENT_ELIGIBLE_CANDIDATES"]
        else:
            decision = "NO_ELIGIBLE_CANDIDATE"
        decisions.append({
            "notice": READ_ONLY_NOTICE, "artifact_type": artifact_type,
            "decision": decision,
            "eligible_paths": [row["path"] for row in eligible],
            "selection_basis": "contract_identity_lineage_not_mtime",
        })
    return decisions


def _request_blockers(decisions, request):
    required = _required_candidate_types(request.selected_stages)
    blockers = []
    for row in decisions:
        if row["artifact_type"] not in required:
            continue
        if row["decision"] == "NO_ELIGIBLE_CANDIDATE":
            code = (
                "PRODUCTION_SCORING_PLAN_REQUIRED"
                if row["artifact_type"] == "SCORING_PLAN"
                else f"{row['artifact_type']}_REQUIRED"
            )
            blockers.append({
                "code": code, "artifact_type": row["artifact_type"],
                "authoritative_owner": _owner(row["artifact_type"]),
                "operator_action": _action(row["artifact_type"]),
            })
        elif row["decision"] == "AMBIGUOUS_MULTIPLE_CANDIDATES":
            blockers.append({
                "code": "AMBIGUOUS_MULTIPLE_CANDIDATES",
                "artifact_type": row["artifact_type"],
                "operator_action":
                    "Review immutable identities and provide a selection file.",
            })
    return blockers


def _required_candidate_types(selected_stages: Sequence[str]) -> set[str]:
    required = {"CANONICAL_CORPUS"}
    if any(stage in selected_stages for stage in ("SCORING", "CERTIFICATION", "PIT")):
        required |= {"SCORING_PLAN", "MODEL_CACHE"}
    if any(stage in selected_stages for stage in ("CERTIFICATION", "PIT")):
        required.add("SCORE_STORE_CERTIFICATE")
    if "PIT" in selected_stages:
        required |= {"DAILY_SPINE", "TICKER_MAPPING", "ALIAS_MAPPING"}
    return required


def _draft_request(discovery, selected, output_root):
    def path(kind):
        return selected.get(kind, {}).get("path", "")
    corpus_root = discovery.candidate_canonical_corpus_root
    score_root = discovery.candidate_score_store_root
    feature_root = discovery.candidate_pit_feature_store_root
    return {
        "selected_stages": tuple(discovery.selected_stages),
        "canonical_source_path": discovery.canonical_source_path,
        "canonical_corpus_root": corpus_root,
        "canonical_manifest_path": path("CANONICAL_CORPUS"),
        "scoring_plan_path": path("SCORING_PLAN"),
        "score_store_root": score_root,
        "chunk_manifest_path": (
            str(Path(score_root) / "finbert_chunk_manifest.csv")
            if score_root else ""
        ),
        "certification_path": path("SCORE_STORE_CERTIFICATE"),
        "pit_feature_store_root": feature_root,
        "pit_feature_manifest_path": (
            path("PIT_FEATURE_STORE")
            or str(Path(feature_root) / "manifest.json")
        ),
        "daily_spine_manifest_path": path("DAILY_SPINE"),
        "ticker_mapping_manifest_path": path("TICKER_MAPPING"),
        "alias_parent_path": path("ALIAS_MAPPING"),
        "runtime_config_path": discovery.runtime_config_path,
        "shared_run_root": discovery.candidate_run_root,
        "resource_ledger_path": discovery.candidate_resource_ledger,
        "run_registry_path": discovery.candidate_registry,
        "model_cache_root": discovery.candidate_model_cache_root,
        "audit_output_path": discovery.final_audit_output_root,
        "cpu_gpu_policy": "CPU_SAFE_DEFAULT",
        "offline_model_resolution_required": True,
        "allow_empty_corpus": False,
        "minimum_free_disk_bytes_warning": 20 * 1024**3,
        "contract_version": AUDIT_CONTRACT,
    }


def _remediation(blockers, decisions):
    order = {
        "CANONICAL_CORPUS": 1, "SCORING_PLAN": 2,
        "MODEL_CACHE": 3, "SCORE_STORE": 4,
        "SCORE_STORE_CERTIFICATE": 5, "DAILY_SPINE": 6,
        "TICKER_MAPPING": 6, "ALIAS_MAPPING": 6,
        "PIT_FEATURE_STORE": 7, "FINAL_AUDIT": 8,
    }
    rows = []
    for blocker in blockers:
        artifact = blocker.get("artifact_type", "FINAL_AUDIT")
        rows.append({
            "notice": READ_ONLY_NOTICE,
            "dependency_order": order.get(artifact, 8),
            "blocker_code": blocker["code"], "affected_stage": artifact,
            "missing_or_incompatible_evidence": blocker,
            "authoritative_owner": _owner(artifact),
            "read_only_verification_command":
                "python scripts/build_stock_alpha_news_compute_readiness_request.py "
                "--discovery-request <reviewed.json> --output-root <new-root> "
                "--draft-only --strict --json",
            "eventual_mutating_command_owner": _action(artifact),
            "eventual_command_status":
                "DO NOT RUN — REQUIRES SEPARATE AUTHORIZATION",
            "prerequisite_blockers": [
                row["blocker_code"] for row in rows
                if row["dependency_order"] < order.get(artifact, 8)
            ],
            "expected_artifact": artifact,
            "separate_implementation_ticket_required": False,
            "separate_operator_authorization_required": True,
        })
    return sorted(rows, key=lambda row: row["dependency_order"])


def _owner(artifact_type):
    return {
        "CANONICAL_CORPUS":
            "materialize_historical_canonical_corpus",
        "SCORING_PLAN": "publish_finbert_scoring_plan",
        "MODEL_CACHE": "operator-managed offline model cache",
        "SCORE_STORE_CERTIFICATE": "certify_finbert_score_store",
        "DAILY_SPINE": "canonical daily-spine owner",
        "TICKER_MAPPING": "canonical ticker-mapping owner",
        "ALIAS_MAPPING": "canonical alias-mapping owner",
        "PIT_FEATURE_STORE": "publish_pit_news_feature_store",
    }.get(artifact_type, "operator-reviewed owner")


def _action(artifact_type):
    return {
        "SCORING_PLAN":
            "DO NOT RUN — build_stock_alpha_finbert_scoring_plan.py requires "
            "separate authorization",
        "MODEL_CACHE":
            "DO NOT RUN — populate and verify the pinned offline snapshot "
            "under separate authorization",
        "CANONICAL_CORPUS":
            "DO NOT RUN — canonical materialisation requires separate authorization",
        "SCORE_STORE_CERTIFICATE":
            "DO NOT RUN — certification requires separate authorization",
        "PIT_FEATURE_STORE":
            "DO NOT RUN — PIT publication requires separate authorization",
    }.get(artifact_type, "Supply and review the exact authoritative metadata.")


def _publish(output_root, discovery, candidates, decisions, draft, blockers,
             warnings, selected, remediation, approved, audit_report):
    output_root.mkdir(parents=True, exist_ok=True)
    rejected = [row for row in candidates if row["classification"] != "ELIGIBLE"]
    lineage = {
        "notice": READ_ONLY_NOTICE,
        "canonical_corpus": selected.get("CANONICAL_CORPUS", {}).get(
            "logical_identity"),
        "scoring_plan_parent": selected.get("SCORING_PLAN", {}).get(
            "parent_identities"),
        "certificate_parent": selected.get(
            "SCORE_STORE_CERTIFICATE", {}).get("parent_identities"),
        "pit_feature_parents": selected.get(
            "PIT_FEATURE_STORE", {}).get("parent_identities"),
    }
    payloads = {
        "candidate_inventory.json": {"notice": READ_ONLY_NOTICE,
                                     "candidates": candidates},
        "candidate_rejections.json": {"notice": READ_ONLY_NOTICE,
                                      "rejections": rejected},
        "selection_decisions.json": {"notice": READ_ONLY_NOTICE,
                                     "decisions": decisions},
        "readiness_request.draft.json": {"notice": READ_ONLY_NOTICE,
                                         "request": draft,
                                         "draft_identity": _hash(draft)},
        "request_blockers.json": {"notice": READ_ONLY_NOTICE,
                                  "blockers": blockers},
        "request_warnings.json": {"notice": READ_ONLY_NOTICE,
                                  "warnings": warnings},
        "lineage_graph.json": lineage,
        "remediation_plan.json": {"notice": READ_ONLY_NOTICE,
                                  "steps": remediation},
    }
    if approved is not None:
        payloads["readiness_request.json"] = {
            "notice": READ_ONLY_NOTICE, **asdict(approved),
            "request_identity": approved.identity,
        }
    if audit_report is not None:
        payloads["real_audit_result.json"] = {
            "notice": READ_ONLY_NOTICE,
            "overall_readiness": audit_report["overall_readiness"],
            "report_path": str(
                Path(approved.audit_output_path) / "readiness_report.json"
            ),
        }
    for name, payload in payloads.items():
        _atomic_json(output_root / name, payload)
    _atomic_text(
        output_root / "operator_review.md",
        _review_markdown(discovery, candidates, decisions, blockers,
                         remediation, approved, audit_report),
    )


def _review_markdown(discovery, candidates, decisions, blockers, remediation,
                     approved, audit_report):
    blocker_lines = (
        [f"- `{row['code']}`: `{row.get('artifact_type')}`"
         for row in blockers] or ["- None"]
    )
    remediation_lines = (
        [f"{row['dependency_order']}. `{row['blocker_code']}` - "
         f"{row['authoritative_owner']}" for row in remediation] or ["- None"]
    )
    lines = [
        "# News Compute Readiness Request Review", "",
        READ_ONLY_NOTICE, "",
        f"- Discovery identity: `{discovery.identity}`",
        f"- Selected stages: `{list(discovery.selected_stages)}`",
        f"- Candidates considered: `{len(candidates)}`", "",
        "## Candidate decisions", "",
        *(f"- `{row['artifact_type']}`: `{row['decision']}`"
          for row in decisions),
        "", "Modification timestamps were recorded as low-trust metadata and "
        "were not used as the primary selection authority. Contract identity, "
        "checksums, and lineage control selection.", "",
        "## Ambiguities and blockers", "",
        *blocker_lines,
        "", "## Immutable identities", "",
        *(f"- `{row['artifact_type']}` `{row['logical_identity']}`"
          for row in candidates if row["classification"] == "ELIGIBLE"),
        "", "## Ordered remediation", "",
        *remediation_lines,
        "", f"Approved request emitted: `{approved is not None}`",
        f"Real audit result: `{audit_report['overall_readiness'] if audit_report else 'NOT_RUN'}`",
        "", "No production execution occurred.",
        "The news-transformer trainer remains out of scope.", "",
    ]
    return "\n".join(lines)


def _metadata(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        import yaml
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError("Metadata must be an object")
    return dict(payload)


def _normalised_request(request):
    payload = asdict(request)
    payload["discovery_roots"] = sorted(
        str(Path(path).resolve()) for path in request.discovery_roots
    )
    for key in (
        "canonical_source_path", "runtime_config_path", "candidate_run_root",
        "candidate_canonical_corpus_root", "candidate_score_store_root",
        "candidate_pit_feature_store_root", "candidate_model_cache_root",
        "candidate_resource_ledger", "candidate_registry",
        "final_audit_output_root",
    ):
        payload[key] = str(Path(payload[key]).resolve())
    return payload


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


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
