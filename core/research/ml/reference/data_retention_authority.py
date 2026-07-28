from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MANIFEST_PATH = Path("config/data_retention_authority_manifest.v1.json")
MANIFEST_CONTRACT_VERSION = "data_retention_authority_manifest.v1"

REQUIRED_FAMILY_FIELDS = (
    "family_id",
    "path_pattern",
    "data_domain",
    "storage_layer",
    "producer_or_provider",
    "known_consumers",
    "authority_role",
    "canonical_status",
    "retention_classification",
    "pit_or_knowledge_time_significance",
    "expected_schema_or_manifest_reference",
    "rebuildability",
    "rebuild_prerequisites",
    "reproducibility_requirement",
    "retention_requirement",
    "expiry_or_rotation_policy",
    "authority_precedence",
    "conflict_policy",
    "cleanup_eligibility",
    "cleanup_confidence",
    "evidence_reference",
    "owner_or_responsible_component",
    "notes",
    "unresolved_questions",
)

ALLOWED_RETENTION_CLASSIFICATIONS = frozenset(
    {
        "RAW_IMMUTABLE_AUTHORITY",
        "RAW_RETAIN_FOR_REPROCESSING",
        "CANONICAL_NORMALIZED",
        "DERIVED_REQUIRED",
        "FROZEN_EXPERIMENT_INPUT",
        "REGENERABLE_CACHE",
        "REPORT_ACCEPTANCE_EVIDENCE",
        "REPORT_REGENERABLE",
        "MODEL_REQUIRED",
        "EXACT_DUPLICATE_PENDING_CANONICAL_PATH",
        "SEMANTIC_OVERLAP_REQUIRES_DIFF",
        "LEGACY_CONSUMER_DEPENDENT",
        "ARCHIVE_PENDING_REVIEW",
        "UNKNOWN_DO_NOT_DELETE",
    }
)

ALLOWED_CLEANUP_ELIGIBILITY = frozenset({"NOT_ELIGIBLE", "REVIEW_ONLY", "FUTURE_PROPOSAL_ALLOWED"})
ALLOWED_CLEANUP_CONFIDENCE = frozenset({"FAIL_CLOSED", "LOW", "MEDIUM", "HIGH"})
ALLOWED_REBUILDABILITY = frozenset(
    {
        "UNKNOWN",
        "NOT_APPLICABLE",
        "NOT_REBUILDABLE_FROM_LOCAL_STATE",
        "REBUILDABLE_WITH_PREREQUISITES",
        "REGENERABLE",
        "EXTERNAL_PROVIDER_DEPENDENT",
    }
)
ALLOWED_PIT_SIGNIFICANCE = frozenset(
    {
        "NONE",
        "PIT_CRITICAL",
        "KNOWLEDGE_TIME_CRITICAL",
        "STATIC_CURRENT_ONLY",
        "POSSIBLE_LOOKAHEAD_RISK",
        "FROZEN_REPRODUCIBILITY",
    }
)

RAW_CLASSIFICATIONS = frozenset({"RAW_IMMUTABLE_AUTHORITY", "RAW_RETAIN_FOR_REPROCESSING"})
FROZEN_CLASSIFICATIONS = frozenset({"FROZEN_EXPERIMENT_INPUT", "REPORT_ACCEPTANCE_EVIDENCE"})
PIT_CRITICAL_STATES = frozenset({"PIT_CRITICAL", "KNOWLEDGE_TIME_CRITICAL"})


class RetentionManifestValidationError(ValueError):
    """Raised when the data-retention authority manifest is unsafe or malformed."""


def load_retention_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_retention_manifest(manifest)
    return manifest


def validate_retention_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("manifest_contract_version") == MANIFEST_CONTRACT_VERSION, "invalid manifest_contract_version")
    _require(manifest.get("cleanup_executed") is False, "manifest must not record cleanup execution")
    families = manifest.get("families")
    _require(isinstance(families, list) and families, "families must be a non-empty list")

    family_ids: set[str] = set()
    for family in families:
        _validate_family(family, family_ids)

    _validate_authority_domains(manifest, family_ids)
    _validate_exact_duplicate_reviews(manifest.get("exact_duplicate_reviews", ()))
    _validate_semantic_overlap_reviews(manifest.get("semantic_overlap_reviews", ()))


def stable_manifest_serialization(manifest: Mapping[str, Any]) -> str:
    validate_retention_manifest(manifest)
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def retention_manifest_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_manifest_serialization(manifest).encode("utf-8")).hexdigest()


def _validate_family(family: Mapping[str, Any], seen_ids: set[str]) -> None:
    _require(isinstance(family, Mapping), "family entries must be mappings")
    missing = [field for field in REQUIRED_FAMILY_FIELDS if field not in family]
    _require(not missing, f"family missing required fields: {', '.join(missing)}")

    family_id = _non_empty_text(family, "family_id")
    _require(family_id not in seen_ids, f"duplicate family_id: {family_id}")
    seen_ids.add(family_id)
    _validate_path_pattern(_non_empty_text(family, "path_pattern"), family_id)

    classification = _non_empty_text(family, "retention_classification")
    _require(classification in ALLOWED_RETENTION_CLASSIFICATIONS, f"unknown retention_classification for {family_id}: {classification}")
    cleanup = _non_empty_text(family, "cleanup_eligibility")
    _require(cleanup in ALLOWED_CLEANUP_ELIGIBILITY, f"unknown cleanup_eligibility for {family_id}: {cleanup}")
    confidence = _non_empty_text(family, "cleanup_confidence")
    _require(confidence in ALLOWED_CLEANUP_CONFIDENCE, f"unknown cleanup_confidence for {family_id}: {confidence}")
    rebuildability = _non_empty_text(family, "rebuildability")
    _require(rebuildability in ALLOWED_REBUILDABILITY, f"unknown rebuildability for {family_id}: {rebuildability}")
    pit_state = _non_empty_text(family, "pit_or_knowledge_time_significance")
    _require(pit_state in ALLOWED_PIT_SIGNIFICANCE, f"unknown PIT significance for {family_id}: {pit_state}")

    for field in ("known_consumers", "rebuild_prerequisites", "evidence_reference", "unresolved_questions"):
        _require(isinstance(family[field], list), f"{family_id}.{field} must be a list")

    if classification == "UNKNOWN_DO_NOT_DELETE":
        _require(cleanup == "NOT_ELIGIBLE", f"{family_id} is unknown and must fail closed")
        _require(confidence == "FAIL_CLOSED", f"{family_id} unknown cleanup confidence must be FAIL_CLOSED")

    if classification in RAW_CLASSIFICATIONS:
        _require(cleanup != "FUTURE_PROPOSAL_ALLOWED", f"{family_id} raw authority cannot be cleanup-proposal eligible")

    if classification in FROZEN_CLASSIFICATIONS:
        _require(cleanup == "NOT_ELIGIBLE", f"{family_id} frozen evidence must be explicitly retained")
        _require("retain" in _non_empty_text(family, "retention_requirement").lower(), f"{family_id} retention requirement must be explicit")

    if pit_state in PIT_CRITICAL_STATES:
        _require(cleanup == "NOT_ELIGIBLE" or classification == "SEMANTIC_OVERLAP_REQUIRES_DIFF", f"{family_id} PIT/knowledge-time data cannot be collapsed without equivalence evidence")
        _require("proof" in family["reproducibility_requirement"].lower() or "checksum" in family["reproducibility_requirement"].lower() or "manifest" in family["reproducibility_requirement"].lower(), f"{family_id} PIT/knowledge-time family needs manifest/checksum/proof language")


def _validate_authority_domains(manifest: Mapping[str, Any], family_ids: set[str]) -> None:
    domains = manifest.get("authority_domains")
    _require(isinstance(domains, Mapping) and domains, "authority_domains must be a non-empty mapping")
    for domain, payload in domains.items():
        _require(isinstance(domain, str) and domain.strip(), "authority domain names must be non-empty")
        _require(isinstance(payload, Mapping), f"authority domain {domain} must be a mapping")
        canonical = payload.get("canonical_family_id")
        _require(isinstance(canonical, str) and canonical in family_ids, f"authority domain {domain} references unknown canonical family: {canonical}")
        for field in ("intended_authority", "fallback_policy", "conflict_policy"):
            _require(isinstance(payload.get(field), str) and payload[field].strip(), f"authority domain {domain} missing {field}")


def _validate_exact_duplicate_reviews(reviews: Any) -> None:
    _require(isinstance(reviews, list), "exact_duplicate_reviews must be a list")
    for review in reviews:
        _require(isinstance(review, Mapping), "exact duplicate review entries must be mappings")
        for field in ("group_id", "paths", "intended_canonical_path", "known_consumers", "cleanup_can_be_proposed_later"):
            _require(field in review, f"exact duplicate review missing {field}")
        _require(isinstance(review["paths"], list) and len(review["paths"]) >= 2, f"{review['group_id']} must include duplicate paths")
        _require(review["cleanup_can_be_proposed_later"] in {True, False}, f"{review['group_id']} cleanup flag must be boolean")


def _validate_semantic_overlap_reviews(reviews: Any) -> None:
    _require(isinstance(reviews, list) and reviews, "semantic_overlap_reviews must be a non-empty list")
    for review in reviews:
        _require(isinstance(review, Mapping), "semantic overlap review entries must be mappings")
        _require(review.get("consolidation_allowed") is False, f"{review.get('overlap_id')} must not allow consolidation in this manifest")
        proof = review.get("proof_required")
        _require(isinstance(proof, list) and proof, f"{review.get('overlap_id')} requires proof_required entries")


def _validate_path_pattern(value: str, family_id: str) -> None:
    for part in value.split(";"):
        part = part.strip()
        _require(part, f"{family_id} has an empty path pattern component")
        _require("\\" not in part, f"{family_id} path pattern must use forward slashes")
        _require(not Path(part).is_absolute(), f"{family_id} path pattern must be repository-relative")
        _require(".." not in Path(part).parts, f"{family_id} path pattern cannot escape the repository")
        _require("*" not in part and "?" not in part, f"{family_id} path pattern must use deterministic placeholders, not globs")


def _non_empty_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    _require(isinstance(value, str) and value.strip(), f"{key} must be a non-empty string")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RetentionManifestValidationError(message)
