from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.research.ml.artifact_lineage import (
    ARTIFACT_LINK_CONTRACT_VERSION,
    VERIFIED_STRICT_OOS,
    read_artifact_link,
    verify_lineage_graph,
)
from core.research.ml.registries import RegistryResolver, load_registry_bundle
from core.research.ml.registries.io import canonical_hash
from core.research.ml.selector_evaluation import (
    COST_CONTRACT,
    COST_SCENARIOS_BPS,
    DATE_PANEL_CONTRACT,
    EVALUATION_CONTRACT,
    PORTFOLIO_METRICS_CONTRACT,
)


PANEL_FREEZE_CONTRACT = "authoritative_selector_panel_v1"
PREFLIGHT_CONTRACT = "selector_evaluation_preflight_v1"
MATCHED_POPULATION_CONTRACT = "selector_matched_population_v1"
PRIMARY_MODEL = "ordered_logit_ranker"
CHALLENGERS = ("ridge", "elastic_net")
RANKING_METRICS_CONTRACT = "ranking_metric_contract_v1"


def discover_selector_components(manifest_roots: Sequence[Path], *, models: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Read registered manifests only; filenames never establish eligibility."""
    expected = set(models)
    eligible: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: set[tuple[str, str]] = set()
    for root in manifest_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("manifest.json")):
            try:
                link = read_artifact_link(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            model = str(link.get("canonical_model_or_policy_id") or "")
            decision = str(link.get("decision_start") or "")[:10]
            if model not in expected or not decision:
                continue
            verification = verify_lineage_graph(path, require_promotion_grade=True)
            evidence = _component_evidence(path, link, verification)
            key = (decision, model)
            if key in eligible and eligible[key]["artifact_link_hash"] != evidence["artifact_link_hash"]:
                conflicts.add(key)
            else:
                eligible[key] = evidence
    for key in conflicts:
        eligible[key]["eligible"] = False
        eligible[key]["rejection_reasons"].append("CONFLICTING_COMPONENT_OWNERS")
    return eligible


def resolve_authoritative_panel(
    *, panel_name: str, panel_version: str, requested_dates: Sequence[str],
    components: Mapping[tuple[str, str], Mapping[str, Any]], primary_model: str,
    challengers: Sequence[str], registry_set_hash: str, policy_registry_hash: str,
    resolution: str = "forward_then_backward",
) -> dict[str, Any]:
    models = (primary_model, *challengers)
    dates = sorted({date for date, model in components if model in models})
    shared_dates = [date for date in dates if _shared_date_eligible(date, models, components)]
    mappings: list[dict[str, Any]] = []
    resolved: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for requested in requested_dates:
        chosen, method = _resolve_one(str(requested), shared_dates, resolution)
        if chosen is None:
            row = {"requested": str(requested), "resolved": None, "method": "rejected", "rejection_reason": "NO_SHARED_ELIGIBLE_SELECTOR_DATE"}
            mappings.append(row); exclusions.append(row); continue
        if chosen in resolved:
            row = {"requested": str(requested), "resolved": chosen, "method": "rejected", "rejection_reason": "DUPLICATE_RESOLVED_DATE"}
            mappings.append(row); exclusions.append(row); continue
        resolved.append(chosen); mappings.append({"requested": str(requested), "resolved": chosen, "method": method})
    evidence = {
        date: {model: dict(components[(date, model)]) for model in models}
        for date in resolved
    }
    population_hashes = sorted({row["row_population_hash"] for date in evidence.values() for row in date.values()})
    dataset_checksums = sorted({row["dataset_checksum"] for date in evidence.values() for row in date.values()})
    feature_hashes = {model: sorted({date[model]["feature_schema_hash"] for date in evidence.values()}) for model in models}
    payload: dict[str, Any] = {
        "panel_freeze_contract_version": PANEL_FREEZE_CONTRACT,
        "date_panel_contract_version": DATE_PANEL_CONTRACT,
        "evaluation_contract_version": EVALUATION_CONTRACT,
        "panel_name": panel_name, "panel_version": panel_version,
        "requested_dates": list(requested_dates), "resolved_dates": resolved,
        "requested_to_resolved": mappings, "resolution_rule": resolution,
        "eligibility_evidence": evidence, "exclusions": exclusions,
        "primary_selector_identity": primary_model, "expected_challenger_identities": list(challengers),
        "expected_policy_registry_hash": policy_registry_hash,
        "expected_metric_versions": {"ranking": RANKING_METRICS_CONTRACT, "portfolio": PORTFOLIO_METRICS_CONTRACT},
        "expected_cost_contract": COST_CONTRACT, "expected_cost_scenarios_bps": list(COST_SCENARIOS_BPS),
        "expected_population_identity": {"contract_version": MATCHED_POPULATION_CONTRACT, "row_population_hashes": population_hashes, "dataset_checksums": dataset_checksums, "feature_schema_hashes_by_model": feature_hashes},
        "source_registry_set_hash": registry_set_hash,
        "lineage_contract_version": ARTIFACT_LINK_CONTRACT_VERSION,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if len(resolved) == len(requested_dates) and not exclusions else "BLOCKED",
    }
    checksum_payload = {key: value for key, value in payload.items() if key not in {"creation_timestamp", "panel_checksum"}}
    payload["panel_checksum"] = canonical_hash(checksum_payload)
    return payload


def freeze_panel(path: Path, panel: Mapping[str, Any]) -> str:
    """Publish once. Identical content is resumable; changed identity fails closed."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("panel_checksum") == panel.get("panel_checksum"):
            return "skipped_identical"
        raise FileExistsError(f"Frozen panel conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(panel, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)
    return "published"


def run_preflight(
    *, frozen_panel: Path, output_root: Path, current_registry_set_hash: str | None = None,
    mutate_reports: bool = False, report_root: Path | None = None,
) -> dict[str, Any]:
    before = _tree_identity(output_root)
    panel = json.loads(frozen_panel.read_text(encoding="utf-8"))
    bundle = load_registry_bundle(); resolver = RegistryResolver(bundle)
    registry_hash = current_registry_set_hash or bundle.registry_set_hash
    reasons: list[str] = []
    if panel.get("panel_freeze_contract_version") != PANEL_FREEZE_CONTRACT: reasons.append("PANEL_CONTRACT_MISMATCH")
    expected_checksum = canonical_hash({key: value for key, value in panel.items() if key not in {"creation_timestamp", "panel_checksum"}})
    if panel.get("panel_checksum") != expected_checksum: reasons.append("PANEL_CHECKSUM_MISMATCH")
    if panel.get("source_registry_set_hash") != registry_hash: reasons.append("STALE_REGISTRY_SET_HASH")
    policy_hash = bundle.documents["portfolio_policies"].registry_hash
    if panel.get("expected_policy_registry_hash") != policy_hash: reasons.append("STALE_POLICY_REGISTRY_HASH")
    if panel.get("expected_metric_versions") != {"ranking": RANKING_METRICS_CONTRACT, "portfolio": PORTFOLIO_METRICS_CONTRACT}: reasons.append("METRIC_VERSION_MISMATCH")
    if panel.get("expected_cost_scenarios_bps") != list(COST_SCENARIOS_BPS): reasons.append("COST_SCENARIO_MISMATCH")
    evidence = panel.get("eligibility_evidence", {})
    required_models = [panel.get("primary_selector_identity"), *panel.get("expected_challenger_identities", [])]
    for model in required_models:
        try: resolver.resolve("selector_models", str(model))
        except KeyError: reasons.append(f"UNKNOWN_SELECTOR:{model}")
    for date in panel.get("resolved_dates", []):
        date_rows = evidence.get(date, {})
        populations = set(); datasets = set()
        for model in required_models:
            row = date_rows.get(model)
            if not row: reasons.append(f"MISSING_COMPONENT:{date}:{model}"); continue
            if not row.get("eligible") or row.get("verification_status") != VERIFIED_STRICT_OOS: reasons.append(f"STRICT_OOS_REJECTED:{date}:{model}")
            populations.add(row.get("row_population_hash")); datasets.add((row.get("dataset_id"), row.get("dataset_checksum")))
        if len(populations) != 1: reasons.append(f"ROW_POPULATION_MISMATCH:{date}")
        if len(datasets) != 1: reasons.append(f"DATASET_VERSION_MISMATCH:{date}")
    if panel.get("status") != "READY" or panel.get("exclusions"): reasons.append("PANEL_RESOLUTION_BLOCKED")
    writable = _probe_writable_parent(output_root)
    if not writable: reasons.append("OUTPUT_ROOT_NOT_WRITABLE")
    partitions = inspect_evaluation_partitions(panel, output_root)
    if any(row["status"] == "conflicting" for row in partitions): reasons.append("CONFLICTING_EVALUATION_PARTITION")
    report = {
        "preflight_contract_version": PREFLIGHT_CONTRACT,
        "status": "READY" if not reasons else "BLOCKED", "exit_code": 0 if not reasons else 2,
        "frozen_panel": str(frozen_panel), "panel_checksum": panel.get("panel_checksum"),
        "registry_set_hash": registry_hash, "output_root": str(output_root), "output_root_writable": writable,
        "partition_plan": partitions, "would_run": sum(row["action"] == "run" for row in partitions),
        "would_resume": sum(row["action"] == "resume" for row in partitions), "would_skip": sum(row["action"] == "skip" for row in partitions),
        "blocking_reasons": sorted(set(reasons)), "selector_fitting_performed": False,
        "historical_evaluation_performed": False, "evaluation_partitions_mutated": False,
    }
    if mutate_reports and report_root:
        _write_reports(report_root, report)
    if _tree_identity(output_root) != before:
        raise RuntimeError("Preflight mutated evaluation output root")
    return report


def inspect_evaluation_partitions(panel: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    result = []
    policies = _registered_evaluation_policies()
    for date in panel.get("resolved_dates", []):
        for policy in policies:
            for bps in COST_SCENARIOS_BPS:
                owner = output_root / f"model={panel['primary_selector_identity']}" / f"date={date}" / f"policy={policy}" / f"cost_bps={bps}"
                manifest_path = owner / "manifest.json"; manifest = _read_json(manifest_path)
                if not owner.exists(): status, action = "missing", "run"
                elif not manifest: status, action = "incomplete", "resume"
                elif manifest.get("status") != "complete": status, action = "failed" if manifest.get("status") == "failed" else "incomplete", "resume"
                elif manifest.get("identity", {}).get("date_panel_checksum") != panel.get("panel_checksum"): status, action = "conflicting", "block"
                else: status, action = "complete", "skip"
                result.append({"date": date, "policy_id": policy, "cost_bps": bps, "owner": str(owner), "status": status, "action": action})
    return result


def powershell_commands(panel_path: Path, output_root: Path, log_root: Path, *, concurrency: int = 2, failure_threshold: int = 1) -> dict[str, str]:
    preflight = f'python main.py --mode ml-selector-evaluation-preflight --frozen-panel "{panel_path}" --evaluation-output-root "{output_root}" --verification-output "{log_root / "preflight.json"}"'
    run = f'python scripts/run_selector_evaluation_panel.py --frozen-panel "{panel_path}" --model ordered_logit_ranker --output-root "{output_root}" --log-root "{log_root}" --concurrency {concurrency} --failure-threshold {failure_threshold} --resume --no-promotion'
    return {
        "preflight": preflight, "first_run": run,
        "monitor": f'Get-Content "{log_root / "orchestration_manifest.json"}" -Raw | ConvertFrom-Json',
        "resume": run,
        "post_run_verify": f'python main.py --mode ml-selector-evaluation-preflight --frozen-panel "{panel_path}" --evaluation-output-root "{output_root}" --verification-output "{log_root / "post_run_verification.json"}"',
        "confirm_no_fitting": f'Select-String -Path "{log_root / "**/*.log"}" -Pattern "fit|training started|retrain" -CaseSensitive:$false',
        "aggregate_publication_blocked": f'python scripts/publish_selector_evaluation_aggregate.py --frozen-panel "{panel_path}" --input-root "{output_root}" --require-all-components --no-promotion',
    }


def _component_evidence(path: Path, link: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(path) or {}
    publication_identity = manifest.get("publication_identity", {}) if isinstance(manifest.get("publication_identity"), Mapping) else {}
    ranking_identity = publication_identity.get("ranking_problem_contract_hash") or manifest.get("ranking_contract_hash") or manifest.get("ranking_contract_version")
    required = ("artifact_link_hash", "canonical_model_or_policy_id", "registry_identity_version", "training_cutoff", "decision_start", "dataset_id", "dataset_checksum", "feature_schema_hash", "target_contract_hash", "row_population_hash")
    reasons = list(verification.get("verification_reasons", []))
    for field in required:
        if link.get(field) in (None, ""): reasons.append(f"MISSING_{field.upper()}")
    evidence = link.get("strict_oos_evidence", {})
    if evidence.get("temporal_guard_passed") is False: reasons.append("TRAINING_POPULATION_OVERLAP")
    if evidence.get("temporal_guard_passed") is not True and evidence.get("temporal_legality_checked") is not True: reasons.append("TRAINING_POPULATION_OVERLAP_UNVERIFIED")
    if not ranking_identity: reasons.append("RANKING_CONTRACT_MISSING")
    return {
        "manifest_path": str(path), "artifact_link_hash": link.get("artifact_link_hash"),
        "model_id": link.get("canonical_model_or_policy_id"), "selector_version": link.get("registry_identity_version"),
        "training_cutoff": link.get("training_cutoff"), "prediction_date": str(link.get("decision_start"))[:10],
        "dataset_id": link.get("dataset_id"), "dataset_checksum": link.get("dataset_checksum"),
        "feature_schema_hash": link.get("feature_schema_hash"), "target_contract_hash": link.get("target_contract_hash"),
        "ranking_contract_version": ranking_identity, "row_population_hash": link.get("row_population_hash"),
        "verification_status": verification.get("verification_status"), "lineage_contract_version": link.get("artifact_link_contract_version"),
        "eligible": verification.get("verification_status") == VERIFIED_STRICT_OOS and not reasons,
        "rejection_reasons": sorted(set(reasons)),
    }


def _shared_date_eligible(date: str, models: Sequence[str], components: Mapping[tuple[str, str], Mapping[str, Any]]) -> bool:
    rows = [components.get((date, model)) for model in models]
    if any(not row or not row.get("eligible") for row in rows): return False
    return len({row["row_population_hash"] for row in rows}) == 1 and len({(row["dataset_id"], row["dataset_checksum"]) for row in rows}) == 1


def _resolve_one(requested: str, available: Sequence[str], resolution: str) -> tuple[str | None, str]:
    if requested in available: return requested, "exact"
    forward = next((value for value in available if value > requested), None)
    backward = next((value for value in reversed(available) if value < requested), None)
    chosen = (forward or backward) if resolution == "forward_then_backward" else (backward or forward)
    return chosen, "forward" if chosen and chosen == forward else ("backward" if chosen else "rejected")


def _registered_evaluation_policies() -> list[str]:
    document = load_registry_bundle().documents["portfolio_policies"]
    return [entry.canonical_id for entry in document.entries if entry.payload.get("output_owner") == "core.research.ml.selector_evaluation"]


def _probe_writable_parent(path: Path) -> bool:
    parent = path if path.exists() else next((candidate for candidate in (path, *path.parents) if candidate.exists()), None)
    return bool(parent and parent.is_dir() and os.access(parent, os.W_OK))


def _tree_identity(path: Path) -> tuple[tuple[str, int, int], ...]:
    if not path.exists(): return ()
    return tuple(sorted((str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns) for item in path.rglob("*") if item.is_file()))


def _read_json(path: Path) -> dict[str, Any] | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError): return None


def _write_reports(root: Path, report: Mapping[str, Any]) -> None:
    json_path = root if root.suffix == ".json" else root.with_suffix(".json")
    md_path = json_path.with_suffix(".md"); json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Selector evaluation preflight", "", f"- Status: `{report['status']}`", f"- Panel checksum: `{report.get('panel_checksum')}`", f"- Would run: `{report['would_run']}`", f"- Would resume: `{report['would_resume']}`", f"- Would skip: `{report['would_skip']}`", "- Selector fitting performed: `false`", "- Historical evaluation performed: `false`"]
    lines.extend(f"- Blocker: `{reason}`" for reason in report["blocking_reasons"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
