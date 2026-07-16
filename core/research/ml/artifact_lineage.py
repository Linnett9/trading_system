from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.registries.io import canonical_hash


ARTIFACT_LINK_CONTRACT_VERSION = "artifact_link_contract_v1"
STRICT_OOS_EVIDENCE_VERSION = "strict_oos_evidence_v1"
PROMOTION_CONTRACT_VERSION = "promotion_eligibility_v1"

VERIFIED_STRICT_OOS = "VERIFIED_STRICT_OOS"
DECLARED_STRICT_OOS_UNVERIFIED = "DECLARED_STRICT_OOS_UNVERIFIED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
NOT_STRICT_OOS = "NOT_STRICT_OOS"
INELIGIBLE_ARTIFACT_KIND = "INELIGIBLE_ARTIFACT_KIND"
REJECTED_ARTIFACT = "REJECTED_ARTIFACT"
INVALIDATED_ARTIFACT = "INVALIDATED_ARTIFACT"
NOT_APPLICABLE = "NOT_APPLICABLE"

SELECTOR_KINDS = frozenset({"BOUNDED_SELECTOR_PREDICTION", "ORDINARY_SELECTOR_PREDICTION"})
DIAGNOSTIC_KINDS = frozenset({"RESEARCH_DIAGNOSTIC", "TREE_DIAGNOSTIC"})
DOWNSTREAM_KINDS = frozenset({"PORTFOLIO_REPLAY", "POLICY_SWEEP", "EXPOSURE_DATASET", "EXPOSURE_EXPERIMENT"})

IDENTITY_FIELDS = (
    "artifact_kind", "artifact_id", "artifact_checksum", "immutable_run_id",
    "experiment_spec_hash", "experiment_run_id", "registry_identity_version",
    "canonical_model_or_policy_id", "model_or_policy_entry_hash", "dataset_id",
    "dataset_checksum", "row_population_hash", "feature_schema_hash",
    "target_contract_hash", "decision_start", "decision_end", "training_cutoff",
    "maximum_label_available_timestamp", "strict_oos_evidence_version",
)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason_codes: tuple[str, ...]
    failing_edge: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED_STRICT_OOS

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_status": self.status,
            "verification_reasons": list(self.reason_codes),
            "failing_edge": self.failing_edge,
            "warnings": list(self.warnings),
        }


def canonical_repo_path(value: str | Path | None, repo_root: Path = Path(".")) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def artifact_link_hash(link: Mapping[str, Any]) -> str:
    upstream = []
    for item in link.get("upstream_links", ()) or ():
        if isinstance(item, Mapping):
            upstream.append({
                "artifact_id": item.get("artifact_id"),
                "artifact_link_hash": item.get("artifact_link_hash"),
                "artifact_checksum": item.get("artifact_checksum"),
            })
    upstream.sort(key=lambda row: (str(row.get("artifact_id")), str(row.get("artifact_link_hash"))))
    identity = {key: link.get(key) for key in IDENTITY_FIELDS}
    identity["upstream_links"] = upstream
    identity["artifact_link_contract_version"] = ARTIFACT_LINK_CONTRACT_VERSION
    return canonical_hash(identity)


def build_artifact_link(**fields: Any) -> dict[str, Any]:
    link = {key: None for key in (
        "artifact_manifest_path", "immutable_run_id", "experiment_spec_hash",
        "experiment_run_id", "created_at", "source_commit", "registry_identity_version",
        "requested_model_or_policy_id", "canonical_model_or_policy_id", "dataset_id",
        "dataset_checksum", "feature_schema_hash", "target_contract_hash", "decision_start",
        "decision_end", "training_cutoff", "maximum_label_available_timestamp",
    )}
    link.update({
        "artifact_link_contract_version": ARTIFACT_LINK_CONTRACT_VERSION,
        "strict_oos_evidence_version": STRICT_OOS_EVIDENCE_VERSION,
        "strict_oos_claim": False, "strict_oos_evidence": {}, "upstream_links": [],
        "verification_status": INSUFFICIENT_EVIDENCE, "verification_reasons": [],
    })
    link.update(fields)
    link["artifact_manifest_path"] = canonical_repo_path(link.get("artifact_manifest_path"))
    if "artifact_path" in link:
        link["artifact_path"] = canonical_repo_path(link.get("artifact_path"))
    link["upstream_links"] = list(link.get("upstream_links") or [])
    link["artifact_link_hash"] = artifact_link_hash(link)
    return link


def read_artifact_link(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    link = payload.get("artifact_link", payload)
    if not isinstance(link, dict):
        raise ValueError(f"Artifact manifest has no artifact link: {path}")
    result = dict(link)
    result.setdefault("artifact_manifest_path", canonical_repo_path(path))
    if isinstance(payload.get("promotion"), Mapping):
        result["promotion"] = dict(payload["promotion"])
    return result


def verify_selector_artifact(link: Mapping[str, Any], *, manifest_path: Path | None = None) -> VerificationResult:
    kind = link.get("artifact_kind")
    if kind in DIAGNOSTIC_KINDS:
        return VerificationResult(INELIGIBLE_ARTIFACT_KIND, ("DIAGNOSTIC_NOT_PREDICTION",))
    if kind not in SELECTOR_KINDS:
        return VerificationResult(INELIGIBLE_ARTIFACT_KIND, ("INVALID_ARTIFACT_KIND",))
    state = str(link.get("completion_status") or link.get("run_status") or "").lower()
    if state in {"rejected", "failed"}:
        return VerificationResult(REJECTED_ARTIFACT, ("REJECTED_ARTIFACT",))
    if state == "invalidated":
        return VerificationResult(INVALIDATED_ARTIFACT, ("ARTIFACT_INVALIDATED",))
    reasons: list[str] = []
    required = {
        "artifact_id": "MANIFEST_INCOMPLETE", "artifact_checksum": "UPSTREAM_CHECKSUM_MISSING",
        "dataset_id": "DATASET_IDENTITY_MISSING", "dataset_checksum": "DATASET_CHECKSUM_MISSING",
        "row_population_hash": "ROW_POPULATION_UNVERIFIED", "feature_schema_hash": "FEATURE_SCHEMA_MISSING",
        "target_contract_hash": "TARGET_CONTRACT_MISSING", "training_cutoff": "TRAINING_CUTOFF_MISSING",
        "maximum_label_available_timestamp": "LABEL_AVAILABILITY_MISSING", "decision_start": "DECISION_DATE_MISSING",
        "canonical_model_or_policy_id": "MODEL_IDENTITY_MISSING", "experiment_run_id": "RUN_IDENTITY_MISSING",
    }
    for field, reason in required.items():
        if link.get(field) in (None, ""):
            reasons.append(reason)
    if manifest_path is not None and link.get("artifact_checksum") and manifest_path.exists():
        if _sha256(manifest_path) != str(link["artifact_checksum"]).upper():
            reasons.append("UPSTREAM_CHECKSUM_MISMATCH")
    decision = _timestamp(link.get("decision_start"))
    cutoff = _timestamp(link.get("training_cutoff"))
    label_max = _timestamp(link.get("maximum_label_available_timestamp"))
    if decision and cutoff and cutoff >= decision:
        reasons.append("TRAINING_CUTOFF_NOT_BEFORE_DECISION")
    if decision and label_max and label_max > decision:
        reasons.append("LABEL_AVAILABILITY_AFTER_DECISION")
    evidence = link.get("strict_oos_evidence") if isinstance(link.get("strict_oos_evidence"), Mapping) else {}
    if evidence.get("prediction_quality_passed") is not True:
        reasons.append("PREDICTION_QUALITY_FAILED" if evidence.get("prediction_quality_passed") is False else "PREDICTION_QUALITY_MISSING")
    if evidence.get("row_population_verified") is not True:
        reasons.append("ROW_POPULATION_UNVERIFIED")
    reasons = sorted(set(reasons))
    if reasons:
        conflicting = any(reason in reasons for reason in (
            "UPSTREAM_CHECKSUM_MISMATCH", "TRAINING_CUTOFF_NOT_BEFORE_DECISION",
            "LABEL_AVAILABILITY_AFTER_DECISION", "PREDICTION_QUALITY_FAILED",
        ))
        status = CONFLICTING_EVIDENCE if conflicting else (DECLARED_STRICT_OOS_UNVERIFIED if link.get("strict_oos_claim") else INSUFFICIENT_EVIDENCE)
        return VerificationResult(status, tuple(reasons))
    if link.get("strict_oos_claim") is not True:
        return VerificationResult(NOT_STRICT_OOS, ("STRICT_OOS_NOT_CLAIMED",))
    return VerificationResult(VERIFIED_STRICT_OOS, ())


def verify_upstream_set(links: Sequence[Mapping[str, Any]], *, promotion_mode: bool, unique_decisions: bool = True) -> VerificationResult:
    if not links:
        return VerificationResult(INSUFFICIENT_EVIDENCE, ("UPSTREAM_LINK_MISSING",))
    reasons: list[str] = []
    targets, schemas, datasets, decisions = set(), set(), set(), set()
    for link in links:
        result = verify_selector_artifact(link)
        if result.status != VERIFIED_STRICT_OOS:
            reasons.extend(result.reason_codes or ("UPSTREAM_NOT_VERIFIED_STRICT_OOS",))
            reasons.append("UPSTREAM_NOT_VERIFIED_STRICT_OOS")
        for field, bucket in (("target_contract_hash", targets), ("feature_schema_hash", schemas), ("dataset_id", datasets)):
            if link.get(field): bucket.add(link[field])
        decision = link.get("decision_start")
        if unique_decisions and decision in decisions:
            reasons.append("DUPLICATE_DECISION_DATE_OWNERSHIP")
        decisions.add(decision)
    if len(targets) > 1: reasons.append("TARGET_CONTRACT_MISMATCH")
    if len(schemas) > 1: reasons.append("FEATURE_SCHEMA_MISMATCH")
    if len(datasets) > 1: reasons.append("DATASET_IDENTITY_MISMATCH")
    reasons = sorted(set(reasons))
    if reasons:
        return VerificationResult(CONFLICTING_EVIDENCE if promotion_mode else INSUFFICIENT_EVIDENCE, tuple(reasons), warnings=("PROMOTION_INELIGIBLE",) if not promotion_mode else ())
    return VerificationResult(VERIFIED_STRICT_OOS, ())


def promotion_eligibility(artifact: Mapping[str, Any], lineage: VerificationResult, *, quality_passed: bool = True) -> dict[str, Any]:
    reasons = list(lineage.reason_codes)
    kind = artifact.get("artifact_kind")
    if kind in DIAGNOSTIC_KINDS or kind not in SELECTOR_KINDS | DOWNSTREAM_KINDS:
        reasons.append("INVALID_ARTIFACT_KIND")
    state = str(artifact.get("completion_status") or artifact.get("run_status") or "").lower()
    if state not in {"complete", "completed"}: reasons.append("ARTIFACT_NOT_COMPLETED")
    if not artifact.get("artifact_checksum"): reasons.append("UPSTREAM_CHECKSUM_MISSING")
    if not quality_passed: reasons.append("QUALITY_GATE_FAILED")
    evidence = artifact.get("strict_oos_evidence") if isinstance(artifact.get("strict_oos_evidence"), Mapping) else {}
    if kind == "EXPOSURE_DATASET" and evidence.get("target_maturity_guard_passed") is not True:
        reasons.append("EXPOSURE_TARGET_MATURITY_UNVERIFIED")
    reasons = sorted(set(reasons))
    return {
        "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
        "promotion_eligible": not reasons and lineage.verified,
        "artifact_kind": kind, "strict_oos_status": lineage.status,
        "lineage_status": lineage.status, "quality_status": "PASSED" if quality_passed else "FAILED",
        "source_cleanliness_status": "RECORDED", "dependency_status": "RECORDED",
        "blocking_reasons": reasons, "warnings": list(lineage.warnings),
    }


def verify_lineage_graph(root_manifest: Path, *, expected_artifact_kind: str | None = None, require_promotion_grade: bool = False) -> dict[str, Any]:
    seen_ids: dict[str, str] = {}
    active: set[Path] = set()
    nodes: list[dict[str, Any]] = []

    def visit(path: Path, edge: str) -> VerificationResult:
        resolved = path.resolve()
        if resolved in active:
            return VerificationResult(CONFLICTING_EVIDENCE, ("LINEAGE_CYCLE",), edge)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return VerificationResult(INSUFFICIENT_EVIDENCE, ("UPSTREAM_LINK_MISSING",), edge)
        link = payload.get("artifact_link", payload)
        if not isinstance(link, Mapping):
            return VerificationResult(INSUFFICIENT_EVIDENCE, ("MANIFEST_INCOMPLETE",), edge)
        if payload is not link and payload.get("manifest_checksum"):
            manifest_identity = {key: value for key, value in payload.items() if key != "manifest_checksum"}
            if canonical_hash(manifest_identity) != payload.get("manifest_checksum"):
                return VerificationResult(CONFLICTING_EVIDENCE, ("MANIFEST_CHECKSUM_MISMATCH",), edge)
        link_hash = artifact_link_hash(link)
        if link.get("artifact_link_hash") != link_hash:
            return VerificationResult(CONFLICTING_EVIDENCE, ("ARTIFACT_LINK_HASH_MISMATCH",), edge)
        artifact_id = str(link.get("artifact_id") or "")
        if artifact_id in seen_ids and seen_ids[artifact_id] != link_hash:
            return VerificationResult(CONFLICTING_EVIDENCE, ("CONFLICTING_ARTIFACT_ID",), edge)
        seen_ids[artifact_id] = link_hash
        active.add(resolved)
        nodes.append({"artifact_id": artifact_id, "artifact_kind": link.get("artifact_kind"), "artifact_link_hash": link_hash, "manifest_path": canonical_repo_path(resolved)})
        child_results: list[VerificationResult] = []
        for index, upstream in enumerate(link.get("upstream_links", ()) or ()):
            if not isinstance(upstream, Mapping) or not upstream.get("artifact_manifest_path"):
                child_results.append(VerificationResult(INSUFFICIENT_EVIDENCE, ("UPSTREAM_LINK_MISSING",), f"{artifact_id}[{index}]"))
                continue
            child_path = Path(str(upstream["artifact_manifest_path"]))
            if not child_path.is_absolute():
                repo_candidate = Path.cwd() / child_path
                child_path = repo_candidate if repo_candidate.exists() else resolved.parent / child_path
            manifest_checksum = upstream.get("artifact_manifest_checksum")
            if manifest_checksum and child_path.exists() and _sha256(child_path) != str(manifest_checksum).upper():
                child_results.append(VerificationResult(CONFLICTING_EVIDENCE, ("UPSTREAM_CHECKSUM_MISMATCH",), f"{artifact_id}->{upstream.get('artifact_id')}"))
                continue
            child_results.append(visit(child_path, f"{artifact_id}->{upstream.get('artifact_id')}"))
        active.remove(resolved)
        bad = next((result for result in child_results if not result.verified), None)
        if bad: return bad
        if link.get("artifact_kind") in SELECTOR_KINDS | DIAGNOSTIC_KINDS:
            artifact_path_value = link.get("artifact_path")
            if artifact_path_value and link.get("artifact_checksum"):
                artifact_path = Path(str(artifact_path_value))
                if not artifact_path.is_absolute():
                    repo_candidate = Path.cwd() / artifact_path
                    artifact_path = repo_candidate if repo_candidate.exists() else resolved.parent / artifact_path
                if not artifact_path.exists():
                    return VerificationResult(INSUFFICIENT_EVIDENCE, ("ARTIFACT_FILE_MISSING",), edge)
                if _sha256(artifact_path) != str(link["artifact_checksum"]).upper():
                    return VerificationResult(CONFLICTING_EVIDENCE, ("UPSTREAM_CHECKSUM_MISMATCH",), edge)
            return verify_selector_artifact(link)
        status = link.get("verification_status")
        if status != VERIFIED_STRICT_OOS:
            return VerificationResult(str(status or INSUFFICIENT_EVIDENCE), tuple(link.get("verification_reasons") or ("UPSTREAM_NOT_VERIFIED_STRICT_OOS",)), edge)
        return VerificationResult(VERIFIED_STRICT_OOS, ())

    result = visit(root_manifest, str(root_manifest))
    try:
        root_payload = json.loads(root_manifest.read_text(encoding="utf-8"))
        root_link = root_payload.get("artifact_link", root_payload)
    except (OSError, ValueError, TypeError):
        root_link = {}
    reasons = list(result.reason_codes)
    if expected_artifact_kind and root_link.get("artifact_kind") != expected_artifact_kind:
        reasons.append("EXPECTED_ARTIFACT_KIND_MISMATCH")
        result = VerificationResult(CONFLICTING_EVIDENCE, tuple(sorted(set(reasons))), str(root_manifest))
    promotion = promotion_eligibility(root_link, result)
    if isinstance(root_payload, Mapping) and isinstance(root_payload.get("promotion"), Mapping):
        recorded = dict(root_payload["promotion"])
        recorded_reasons = sorted(set(recorded.get("blocking_reasons", ())) | set(promotion["blocking_reasons"]))
        promotion.update(recorded); promotion["blocking_reasons"] = recorded_reasons
        promotion["promotion_eligible"] = bool(recorded.get("promotion_eligible")) and not recorded_reasons and result.verified
    if require_promotion_grade and not promotion["promotion_eligible"] and "PROMOTION_GRADE_REQUIRED" not in promotion["blocking_reasons"]:
        promotion["blocking_reasons"].append("PROMOTION_GRADE_REQUIRED")
    return {"artifact_lineage_verification_version": "artifact_lineage_verification_v1", **result.to_dict(), "artifact_link_hash": root_link.get("artifact_link_hash"), "nodes": nodes, "promotion": promotion}


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""): return None
    try: return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError: return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest().upper()
