from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.research.ml.artifact_lineage import VERIFIED_STRICT_OOS, verify_lineage_graph
from core.research.ml.experiment_ledger import append_ledger_event, experiment_spec_hash, new_experiment_run_id
from core.research.ml.provenance import source_provenance
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.legacy_evidence import import_legacy_evidence
from core.research.ml.reference.canonical_assets import (
    file_sha256,
    read_aliases_csv,
    read_assets_csv,
    registry_content_hash,
)


def run_artifact_lineage_verify(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if not args.artifact_manifest:
        raise ValueError("--artifact-manifest is required for ml-artifact-lineage-verify")
    manifest = Path(args.artifact_manifest)
    result = verify_lineage_graph(
        manifest, expected_artifact_kind=args.expected_artifact_kind,
        require_promotion_grade=bool(args.require_promotion_grade),
    )
    if getattr(args, "expected_decision_date", None):
        root = json.loads(manifest.read_text(encoding="utf-8")); link = root.get("artifact_link", root)
        if link.get("decision_start") != args.expected_decision_date and root.get("decision_date") != args.expected_decision_date:
            result["verification_status"] = "CONFLICTING_EVIDENCE"; result["verification_reasons"].append("EXPECTED_DECISION_DATE_MISMATCH")
    if getattr(args, "expected_replay_link", None):
        root = json.loads(manifest.read_text(encoding="utf-8")); link = root.get("artifact_link", root)
        replay_hashes = {row.get("artifact_link_hash") for row in link.get("upstream_links", []) if row.get("artifact_kind") == "PORTFOLIO_REPLAY"}
        if args.expected_replay_link not in replay_hashes:
            result["verification_status"] = "CONFLICTING_EVIDENCE"; result["verification_reasons"].append("EXPECTED_REPLAY_LINK_MISMATCH")
    report_paths: list[str] = []
    if args.verification_output:
        report_paths = _write_reports(Path(args.verification_output), result)
    if config.get("ml", {}).get("lineage_audit_ledger_enabled", False):
        _append_audit_event(config, manifest, result, report_paths)
    print(json.dumps({
        "verification_status": result["verification_status"],
        "verification_reasons": result["verification_reasons"],
        "promotion_eligible": result["promotion"]["promotion_eligible"],
        "failing_edge": result["failing_edge"],
    }, sort_keys=True))
    if result["verification_status"] != VERIFIED_STRICT_OOS or (args.require_promotion_grade and not result["promotion"]["promotion_eligible"]):
        raise SystemExit(2)
    return result


def run_registry_verify(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "artifact_manifest", None):
        return _run_canonical_registry_publication_verify(args)
    first = load_registry_bundle()
    second = load_registry_bundle()
    if first.registry_set_hash != second.registry_set_hash:
        raise RuntimeError("Registry hashes are not reproducible")
    resolver = RegistryResolver(first)
    aliases = 0
    for kind, document in first.documents.items():
        for entry in document.entries:
            for alias in entry.aliases:
                resolution = resolver.resolve(kind, alias)
                if resolution.canonical_id != entry.canonical_id:
                    raise RuntimeError(f"Alias resolution mismatch: {kind}:{alias}")
                aliases += 1
    result = {
        "registry_verification_version": "registry_verification_v1", "status": "VERIFIED",
        "registry_set_hash": first.registry_set_hash, "registry_hashes": {
            kind: document.registry_hash for kind, document in first.documents.items()
        }, "entry_count": sum(len(document.entries) for document in first.documents.values()),
        "alias_count": aliases,
    }
    if args.verification_output: _write_reports(Path(args.verification_output), result)
    print(json.dumps(result, sort_keys=True))
    return result


def _run_canonical_registry_publication_verify(args: Any) -> dict[str, Any]:
    manifest_path = Path(args.artifact_manifest)
    expected_run_id = str(getattr(args, "registry_run_id", None) or "")
    blockers: list[str] = []
    manifest = _read_json(manifest_path, blockers, "MANIFEST_MISSING_OR_INVALID")
    audit = _read_json(manifest_path.parent / "registry_audit.json", blockers, "AUDIT_MISSING_OR_INVALID")
    if not expected_run_id or manifest_path.parent.name != f"run={expected_run_id}":
        blockers.append("RUN_ID_MISMATCH")
    registry_path = Path(str(manifest.get("registry_path") or "")) if manifest.get("registry_path") else None
    alias_path = Path(str(manifest.get("alias_registry_path") or "")) if manifest.get("alias_registry_path") else None
    assets, aliases = [], []
    for path, checksum_key, missing, mismatch, invalid, reader, destination in (
        (registry_path, "registry_content_checksum", "REGISTRY_ARTIFACT_MISSING", "REGISTRY_ARTIFACT_CHECKSUM_MISMATCH", "REGISTRY_ARTIFACT_INVALID", read_assets_csv, assets),
        (alias_path, "alias_registry_checksum", "ALIAS_ARTIFACT_MISSING", "ALIAS_ARTIFACT_CHECKSUM_MISMATCH", "ALIAS_ARTIFACT_INVALID", read_aliases_csv, aliases),
    ):
        if path is None or not path.is_file():
            blockers.append(missing)
            continue
        if file_sha256(path) != str(manifest.get(checksum_key) or ""):
            blockers.append(mismatch)
        try:
            destination.extend(reader(path))
        except Exception:
            blockers.append(invalid)
    calculated_hash = registry_content_hash(assets, aliases) if assets and aliases else None
    published_hash = str(manifest.get("registry_content_hash") or "")
    if not calculated_hash or calculated_hash != published_hash or audit.get("registry_content_hash") != published_hash:
        blockers.append("REGISTRY_CONTENT_HASH_MISMATCH")
    checks = (
        (manifest.get("publication_status"), "complete", "PUBLICATION_INCOMPLETE"),
        (manifest.get("validation_status"), "VERIFIED", "MANIFEST_NOT_VERIFIED"),
        (manifest.get("row_count"), 514, "MANIFEST_ASSET_COUNT_NOT_514"),
        (manifest.get("symbol_count"), 514, "MANIFEST_SYMBOL_COUNT_NOT_514"),
        (manifest.get("row_identity_checksum"), published_hash, "MANIFEST_IDENTITY_MISMATCH"),
        (audit.get("canonical_asset_count"), 514, "CANONICAL_ASSET_COUNT_NOT_514"),
        (audit.get("resolved_collection_symbol_count"), 514, "RESOLVED_SYMBOL_COUNT_NOT_514"),
    )
    blockers.extend(reason for actual, expected, reason in checks if actual != expected)
    if audit.get("unresolved_collection_symbols"):
        blockers.append("UNRESOLVED_COLLECTION_SYMBOLS")
    if audit.get("ambiguous_aliases"):
        blockers.append("AMBIGUOUS_ALIASES")
    blockers = sorted(set(blockers))
    result = {
        "registry_verification_version": "canonical_registry_publication_verification_v1",
        "status": "READY" if not blockers else "BLOCKED",
        "run_id": expected_run_id,
        "manifest_path": str(manifest_path),
        "manifest_checksum": file_sha256(manifest_path) if manifest_path.is_file() else None,
        "registry_content_hash": calculated_hash,
        "canonical_asset_count": len(assets),
        "resolved_collection_symbol_count": audit.get("resolved_collection_symbol_count"),
        "unresolved_collection_symbols": audit.get("unresolved_collection_symbols", []),
        "ambiguous_aliases": audit.get("ambiguous_aliases", []),
        "blockers": blockers,
        "feedless": True,
        "publication_modified": False,
    }
    if args.verification_output:
        _write_reports(Path(args.verification_output), result)
    print(json.dumps(result, sort_keys=True))
    if blockers:
        raise SystemExit(2)
    return result


def _read_json(path: Path, blockers: list[str], blocker: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload
    except (OSError, ValueError):
        blockers.append(blocker)
        return {}


def run_legacy_evidence_import(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if not args.legacy_manifest: raise ValueError("--legacy-manifest is required")
    if not args.verification_output: raise ValueError("--verification-output is required for a separate legacy evidence report")
    result = import_legacy_evidence(Path(args.legacy_manifest), expected_artifact_kind=args.expected_artifact_kind)
    _write_reports(Path(args.verification_output), result)
    print(json.dumps({"verification_status": result["verification_status"], "verification_reasons": result["verification_reasons"], "source_untouched": result["source_untouched"]}, sort_keys=True))
    if args.require_promotion_grade and not result["promotion_eligible"]: raise SystemExit(2)
    return result


def _write_reports(path: Path, payload: Mapping[str, Any]) -> list[str]:
    json_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Artifact verification", "", f"- Status: `{payload.get('verification_status', payload.get('status'))}`"]
    if "promotion" in payload: lines.append(f"- Promotion eligible: `{payload['promotion'].get('promotion_eligible')}`")
    for reason in payload.get("verification_reasons", []): lines.append(f"- Reason: `{reason}`")
    if payload.get("failing_edge"): lines.append(f"- Failing edge: `{payload['failing_edge']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(json_path), str(md_path)]


def _append_audit_event(config: Mapping[str, Any], manifest: Path, result: Mapping[str, Any], reports: list[str]) -> None:
    specification = {"artifact_manifest": str(manifest), "artifact_link_hash": result.get("artifact_link_hash")}
    spec_hash = experiment_spec_hash(specification)
    source = source_provenance()
    append_ledger_event(
        Path(config.get("ml", {}).get("experiment_ledger_path", "reports/ml/experiments/experiment_ledger.jsonl")),
        experiment_spec_hash_value=spec_hash, experiment_run_id=new_experiment_run_id(spec_hash),
        event_status="COMPLETED", artifact_kind="ARTIFACT_LINEAGE_AUDIT",
        canonical_model_id=None, requested_model_id=None, registry_hashes={},
        source_commit=source["git_commit"], artifact_paths=[str(manifest), *reports],
        metadata={"artifact_link_hash": result.get("artifact_link_hash"), "verification_status": result["verification_status"], "reason_codes": result["verification_reasons"]},
    )
