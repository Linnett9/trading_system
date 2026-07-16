from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifact_lineage import (
    INSUFFICIENT_EVIDENCE, build_artifact_link, verify_selector_artifact,
)


IMPORT_CONTRACT_VERSION = "legacy_artifact_evidence_import_v1"
SUPPORTED_IDENTITY_VERSIONS = frozenset({
    "bounded_selector_identity_v3_registry", "ordinary_selector_identity_v2_registry",
    "portfolio_replay_identity_v2_registry", "exposure_experiment_identity_v2_registry",
})


def import_legacy_evidence(path: Path, *, expected_artifact_kind: str | None = None) -> dict[str, Any]:
    before = path.stat().st_mtime_ns
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("identity_version") or payload.get("registry_provenance", {}).get("identity_version")
    if version not in SUPPORTED_IDENTITY_VERSIONS:
        raise ValueError(f"Unsupported legacy identity version: {version}")
    existing_link = payload.get("artifact_link")
    if isinstance(existing_link, Mapping):
        link = dict(existing_link)
    else:
        kind = expected_artifact_kind or _legacy_kind(payload, version)
        link = build_artifact_link(
            artifact_kind=kind, artifact_id=payload.get("artifact_id") or f"legacy:{version}:{path.name}",
            artifact_manifest_path=path, artifact_checksum=payload.get("prediction_checksum") or payload.get("dataset_hash"),
            experiment_run_id=payload.get("experiment_run_id"),
            canonical_model_or_policy_id=payload.get("canonical_model_id") or payload.get("model_id"),
            dataset_id=payload.get("dataset_id"), dataset_checksum=payload.get("source_dataset_checksum"),
            row_population_hash=payload.get("row_population_hash"), feature_schema_hash=payload.get("feature_schema_hash"),
            target_contract_hash=(payload.get("target_identity") or {}).get("target_entry_hash"),
            decision_start=payload.get("decision_timestamp") or payload.get("decision_date"),
            decision_end=payload.get("decision_timestamp") or payload.get("decision_date"),
            training_cutoff=payload.get("training_decision_timestamp_max"),
            maximum_label_available_timestamp=payload.get("training_label_available_timestamp_max"),
            strict_oos_claim=bool(payload.get("strict_oos_claim")),
            strict_oos_evidence={"prediction_quality_passed": payload.get("prediction_quality_accepted"), "row_population_verified": bool(payload.get("row_population_hash"))},
            completion_status=payload.get("completion_status") or payload.get("run_status"),
        )
    if expected_artifact_kind and link.get("artifact_kind") != expected_artifact_kind:
        status, reasons = "CONFLICTING_EVIDENCE", ["EXPECTED_ARTIFACT_KIND_MISMATCH"]
    elif link.get("artifact_kind") in {"BOUNDED_SELECTOR_PREDICTION", "ORDINARY_SELECTOR_PREDICTION", "RESEARCH_DIAGNOSTIC"}:
        verification = verify_selector_artifact(link); status, reasons = verification.status, list(verification.reason_codes)
    else:
        status = str(link.get("verification_status") or INSUFFICIENT_EVIDENCE)
        reasons = list(link.get("verification_reasons") or ["LEGACY_IDENTITY_INSUFFICIENT"])
    if path.stat().st_mtime_ns != before: raise RuntimeError("Legacy evidence import mutated its source")
    return {
        "legacy_evidence_import_contract_version": IMPORT_CONTRACT_VERSION,
        "artifact_kind": "LEGACY_ARTIFACT_EVIDENCE_IMPORT", "source_manifest_path": str(path),
        "source_identity_version": version, "source_untouched": True,
        "verification_status": status, "verification_reasons": reasons,
        "promotion_eligible": status == "VERIFIED_STRICT_OOS", "imported_artifact_link": link,
    }


def _legacy_kind(payload: Mapping[str, Any], version: str) -> str:
    if version.startswith("bounded_selector"): return "BOUNDED_SELECTOR_PREDICTION"
    if version.startswith("ordinary_selector"): return "ORDINARY_SELECTOR_PREDICTION"
    if version.startswith("portfolio_replay"): return "PORTFOLIO_REPLAY"
    return "EXPOSURE_EXPERIMENT"
